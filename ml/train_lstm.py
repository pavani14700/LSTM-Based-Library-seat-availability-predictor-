"""
Training script for Library Seat Availability Prediction using LSTM.
Builds, trains, and evaluates a TensorFlow/Keras LSTM model on chronological data.
Saves model, scaler, configuration, and evaluation metrics locally.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Suppress excessive TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from ml.preprocess import load_capacity_config, create_sliding_windows


# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


FEATURE_COLUMNS = [
    'occupancy_rate',
    'floor_1_rate',
    'floor_2_rate',
    'floor_3_rate',
    'sin_hour',
    'cos_hour',
    'sin_dow',
    'cos_dow',
    'is_weekend',
    'is_open'
]

TARGET_COLUMNS = [
    'occupancy_rate',
    'floor_1_rate',
    'floor_2_rate',
    'floor_3_rate'
]

LOOKBACK_WINDOW = 24


def train_lstm_pipeline(
    processed_csv_path: str = "data/processed/hourly_occupancy.csv",
    models_dir: str = "models",
    epochs: int = 40,
    batch_size: int = 32
) -> Dict[str, Any]:
    """Full training pipeline with chronological split and evaluation."""
    os.makedirs(models_dir, exist_ok=True)
    config = load_capacity_config()

    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"Processed file '{processed_csv_path}' not found. Run preprocess.py first.")

    df = pd.read_csv(processed_csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    total_steps = len(df)
    train_end = int(total_steps * 0.70)
    val_end = int(total_steps * 0.85)

    print("=" * 65)
    print("LSTM MODEL TRAINING PIPELINE")
    print("=" * 65)
    print(f"Total time-series steps: {total_steps} hours")
    print(f"Train split:        0 to {train_end} ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[train_end-1]})")
    print(f"Validation split:   {train_end} to {val_end} ({df['timestamp'].iloc[train_end]} -> {df['timestamp'].iloc[val_end-1]})")
    print(f"Test split:         {val_end} to {total_steps} ({df['timestamp'].iloc[val_end]} -> {df['timestamp'].iloc[-1]})")

    # Fit scaler ONLY on training data
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(train_df[FEATURE_COLUMNS].values)

    # Transform splits
    scaled_train = scaler.transform(train_df[FEATURE_COLUMNS].values)
    scaled_val = scaler.transform(val_df[FEATURE_COLUMNS].values)
    scaled_test = scaler.transform(test_df[FEATURE_COLUMNS].values)

    target_indices = [FEATURE_COLUMNS.index(col) for col in TARGET_COLUMNS]

    # Create sliding windows
    X_train, y_train = create_sliding_windows(scaled_train, target_indices, lookback=LOOKBACK_WINDOW)
    X_val, y_val = create_sliding_windows(scaled_val, target_indices, lookback=LOOKBACK_WINDOW)
    X_test, y_test = create_sliding_windows(scaled_test, target_indices, lookback=LOOKBACK_WINDOW)

    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
    print(f"X_test shape:  {X_test.shape}, y_test shape:  {y_test.shape}")

    # Build Keras LSTM Architecture
    model = Sequential([
        Input(shape=(LOOKBACK_WINDOW, len(FEATURE_COLUMNS))),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(TARGET_COLUMNS), activation='sigmoid')  # Outputs bounded in [0, 1]
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join(models_dir, "lstm_best_weights.keras"),
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
    ]

    print("\nTraining LSTM network...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on unseen chronological test data
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0.0, 1.0)

    # Baseline: 24-hour persistence baseline (predict value from 24 hours ago)
    # Since lookback is 24, X_test[:, 0, target_idx] represents the observation 24 hours ago!
    y_baseline = X_test[:, 0, target_indices]

    # Calculate actual metrics
    metrics = {
        "test_mae_overall": float(mean_absolute_error(y_test[:, 0], y_pred[:, 0])),
        "test_rmse_overall": float(np.sqrt(mean_squared_error(y_test[:, 0], y_pred[:, 0]))),
        "baseline_mae_overall": float(mean_absolute_error(y_test[:, 0], y_baseline[:, 0])),
        "baseline_rmse_overall": float(np.sqrt(mean_squared_error(y_test[:, 0], y_baseline[:, 0]))),
        "floors": {}
    }

    floor_names = ['floor_1', 'floor_2', 'floor_3']
    for idx, fl in enumerate(floor_names, start=1):
        fl_mae = float(mean_absolute_error(y_test[:, idx], y_pred[:, idx]))
        fl_rmse = float(np.sqrt(mean_squared_error(y_test[:, idx], y_pred[:, idx])))
        base_mae = float(mean_absolute_error(y_test[:, idx], y_baseline[:, idx]))
        base_rmse = float(np.sqrt(mean_squared_error(y_test[:, idx], y_baseline[:, idx])))
        metrics["floors"][fl] = {
            "test_mae": fl_mae,
            "test_rmse": fl_rmse,
            "baseline_mae": base_mae,
            "baseline_rmse": base_rmse
        }

    print("\n" + "=" * 65)
    print("EVALUATION RESULTS (UNSEEN TEST DATA)")
    print("=" * 65)
    print(f"Overall Library Occupancy:")
    print(f"  LSTM Model  -> MAE: {metrics['test_mae_overall']:.4f} ({metrics['test_mae_overall']*100:.2f}%), RMSE: {metrics['test_rmse_overall']:.4f}")
    print(f"  24h Baseline-> MAE: {metrics['baseline_mae_overall']:.4f} ({metrics['baseline_mae_overall']*100:.2f}%), RMSE: {metrics['baseline_rmse_overall']:.4f}")

    for fl in floor_names:
        print(f"\n{fl.upper()}:")
        print(f"  LSTM Model  -> MAE: {metrics['floors'][fl]['test_mae']:.4f}, RMSE: {metrics['floors'][fl]['test_rmse']:.4f}")
        print(f"  24h Baseline-> MAE: {metrics['floors'][fl]['baseline_mae']:.4f}, RMSE: {metrics['floors'][fl]['baseline_rmse']:.4f}")

    # Save Model, Scaler, and Configuration
    model_save_path = os.path.join(models_dir, "lstm_model.keras")
    model.save(model_save_path)
    print(f"\n[OK] Model saved to '{model_save_path}'.")

    scaler_save_path = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_save_path)
    print(f"[OK] Scaler saved to '{scaler_save_path}'.")

    model_config = {
        "model_architecture": "LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(32) -> Dense(4, sigmoid)",
        "lookback_window": LOOKBACK_WINDOW,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
        "training_metadata": {
            "total_steps": total_steps,
            "train_steps": train_end,
            "val_steps": val_end - train_end,
            "test_steps": total_steps - val_end,
            "train_start": str(df['timestamp'].iloc[0]),
            "train_end": str(df['timestamp'].iloc[train_end-1]),
            "test_start": str(df['timestamp'].iloc[val_end]),
            "test_end": str(df['timestamp'].iloc[-1])
        },
        "capacities": config
    }

    config_save_path = os.path.join(models_dir, "model_config.json")
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"[OK] Model config saved to '{config_save_path}'.")

    metrics_save_path = os.path.join(models_dir, "metrics.json")
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[OK] Evaluation metrics saved to '{metrics_save_path}'.")

    return metrics


if __name__ == "__main__":
    train_lstm_pipeline()
