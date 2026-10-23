"""
Evaluation module for trained LSTM model on unseen chronological test data.
Prints comparison between LSTM model and baseline benchmarks.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf

from ml.preprocess import create_sliding_windows


def evaluate_model(
    model_path: str = "models/lstm_model.keras",
    scaler_path: str = "models/scaler.pkl",
    config_path: str = "models/model_config.json",
    data_path: str = "data/processed/hourly_occupancy.csv"
):
    """Loads saved model, computes comprehensive test metrics, and prints evaluation report."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run train_lstm.py first.")

    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)

    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    feature_cols = model_config["feature_columns"]
    target_cols = model_config["target_columns"]
    lookback = model_config["lookback_window"]

    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    total_steps = len(df)
    val_end = int(total_steps * 0.85)

    test_df = df.iloc[val_end:]
    scaled_test = scaler.transform(test_df[feature_cols].values)
    target_indices = [feature_cols.index(col) for col in target_cols]

    X_test, y_test = create_sliding_windows(scaled_test, target_indices, lookback=lookback)
    y_pred = np.clip(model.predict(X_test, verbose=0), 0.0, 1.0)
    y_baseline = X_test[:, 0, target_indices]

    results = {}
    print("\n" + "=" * 70)
    print("DETAILED MODEL EVALUATION REPORT (UNSEEN TEST DATA)")
    print("=" * 70)
    print(f"Test period: {df['timestamp'].iloc[val_end]} -> {df['timestamp'].iloc[-1]}")
    print(f"Test samples: {len(y_test)} windowed sequences\n")

    for i, target in enumerate(target_cols):
        y_true_col = y_test[:, i]
        y_pred_col = y_pred[:, i]
        y_base_col = y_baseline[:, i]

        mae = mean_absolute_error(y_true_col, y_pred_col)
        rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
        r2 = r2_score(y_true_col, y_pred_col)

        b_mae = mean_absolute_error(y_true_col, y_base_col)
        b_rmse = np.sqrt(mean_squared_error(y_true_col, y_base_col))

        results[target] = {
            "lstm_mae": float(mae),
            "lstm_rmse": float(rmse),
            "lstm_r2": float(r2),
            "baseline_mae": float(b_mae),
            "baseline_rmse": float(b_rmse)
        }

        print(f"Target: {target.upper()}")
        print(f"  LSTM Model  -> MAE: {mae:.4f} ({mae*100:.2f}%), RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"  24h Baseline-> MAE: {b_mae:.4f} ({b_mae*100:.2f}%), RMSE: {b_rmse:.4f}")
        improvement = ((b_mae - mae) / b_mae) * 100 if b_mae > 0 else 0
        print(f"  MAE Improvement over Baseline: {improvement:+.1f}%\n")

    print("=" * 70)
    return results


if __name__ == "__main__":
    evaluate_model()
