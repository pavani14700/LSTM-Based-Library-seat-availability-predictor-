"""
Preprocessing Pipeline for Library Seat Availability Prediction
Reads raw dataset without modifying it, generates continuous hourly time-series,
extracts temporal and cyclical features, and prepares sliding windows.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List


def load_capacity_config(config_path: str = "config/capacities.json") -> Dict[str, Any]:
    """Loads capacity and threshold configuration."""
    if not os.path.exists(config_path):
        # Fallback to default
        return {
            "total_capacity": 150,
            "floors": {
                "floor_1": {"name": "Floor 1", "capacity": 50},
                "floor_2": {"name": "Floor 2", "capacity": 50},
                "floor_3": {"name": "Floor 3", "capacity": 50}
            },
            "operating_hours": {
                "open_hour": 8,
                "close_hour": 22
            }
        }
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def inspect_raw_dataset(raw_path: str = "final dataset.csv") -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Inspects the raw CSV file without modifying or overwriting it.
    Extracts data types, counts, date ranges, and distributions.
    """
    if not os.path.exists(raw_path):
        alt_path = os.path.join("data", raw_path)
        if os.path.exists(alt_path):
            raw_path = alt_path
        else:
            raise FileNotFoundError(f"Dataset file '{raw_path}' not found.")

    df = pd.read_csv(raw_path)
    total_rows = len(df)
    total_cols = len(df.columns)
    missing_counts = df.isnull().sum().to_dict()
    exact_duplicates = int(df.duplicated().sum())

    # Parse datetimes strictly for inspection
    df['entry_dt'] = pd.to_datetime(df['entry_datetime'], format="%d-%m-%Y %H:%M", errors='coerce')
    df['exit_dt'] = pd.to_datetime(df['exit_datetime'], format="%d-%m-%Y %H:%M", errors='coerce')

    min_entry = str(df['entry_dt'].min())
    max_entry = str(df['entry_dt'].max())
    min_exit = str(df['exit_dt'].min())
    max_exit = str(df['exit_dt'].max())

    floor_distribution = df['library_zone'].value_counts().to_dict() if 'library_zone' in df.columns else {}
    crowd_distribution = df['hourly_crowd_level'].value_counts().to_dict() if 'hourly_crowd_level' in df.columns else {}

    summary = {
        "dataset_file": raw_path,
        "total_records": total_rows,
        "total_columns": total_cols,
        "columns": list(df.columns),
        "missing_values": missing_counts,
        "exact_duplicates": exact_duplicates,
        "date_range": {
            "first_entry": min_entry,
            "last_entry": max_entry,
            "first_exit": min_exit,
            "last_exit": max_exit
        },
        "floor_distribution": floor_distribution,
        "crowd_distribution": crowd_distribution,
        "avg_duration_minutes": float(df['duration_minutes'].mean()) if 'duration_minutes' in df.columns else 0.0,
        "min_duration_minutes": float(df['duration_minutes'].min()) if 'duration_minutes' in df.columns else 0.0,
        "max_duration_minutes": float(df['duration_minutes'].max()) if 'duration_minutes' in df.columns else 0.0,
        "limitations": [
            "Dataset records sessions from 08:00 to 19:00 entry hours with exits extending up to 23:36.",
            "Night hours (00:00 - 07:00) have zero sessions because the facility is closed.",
            "Long-term forecasting (>14 days ahead) is subject to multi-step recursive error accumulation."
        ]
    }

    return df, summary


def generate_hourly_time_series(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Computes hourly active occupancy by counting students whose sessions overlap each hour.
    Calculates overall and floor-wise occupancy rates based on configured capacities.
    """
    min_hour = df['entry_dt'].min().floor('h')
    max_hour = df['exit_dt'].max().ceil('h')
    hourly_index = pd.date_range(start=min_hour, end=max_hour, freq='h', name='timestamp')

    # Expand sessions into hourly bins
    expanded_rows = []
    for _, row in df.iterrows():
        cur_hour = row['entry_dt'].floor('h')
        exit_dt = row['exit_dt']
        zone = row['library_zone']
        while cur_hour < exit_dt:
            expanded_rows.append((cur_hour, zone))
            cur_hour += pd.Timedelta(hours=1)

    exp_df = pd.DataFrame(expanded_rows, columns=['timestamp', 'library_zone'])

    # Aggregate counts per timestamp
    total_counts = exp_df.groupby('timestamp').size().reindex(hourly_index, fill_value=0).rename('occupied_seats')
    floor_counts = exp_df.groupby(['timestamp', 'library_zone']).size().unstack(fill_value=0).reindex(hourly_index, fill_value=0)

    for fl in ['floor_1', 'floor_2', 'floor_3']:
        if fl not in floor_counts.columns:
            floor_counts[fl] = 0

    total_cap = config.get("total_capacity", 150)
    fl1_cap = config.get("floors", {}).get("floor_1", {}).get("capacity", 50)
    fl2_cap = config.get("floors", {}).get("floor_2", {}).get("capacity", 50)
    fl3_cap = config.get("floors", {}).get("floor_3", {}).get("capacity", 50)

    # Compile unified time series DataFrame
    ts_df = pd.DataFrame({'timestamp': hourly_index})
    ts_df['occupied_seats'] = total_counts.values
    ts_df['occupancy_rate'] = np.clip(ts_df['occupied_seats'] / total_cap, 0.0, 1.0)
    ts_df['available_seats'] = np.maximum(0, total_cap - ts_df['occupied_seats'])

    # Floor metrics
    ts_df['floor_1_occupied'] = floor_counts['floor_1'].values
    ts_df['floor_1_rate'] = np.clip(ts_df['floor_1_occupied'] / fl1_cap, 0.0, 1.0)
    ts_df['floor_1_available'] = np.maximum(0, fl1_cap - ts_df['floor_1_occupied'])

    ts_df['floor_2_occupied'] = floor_counts['floor_2'].values
    ts_df['floor_2_rate'] = np.clip(ts_df['floor_2_occupied'] / fl2_cap, 0.0, 1.0)
    ts_df['floor_2_available'] = np.maximum(0, fl2_cap - ts_df['floor_2_occupied'])

    ts_df['floor_3_occupied'] = floor_counts['floor_3'].values
    ts_df['floor_3_rate'] = np.clip(ts_df['floor_3_occupied'] / fl3_cap, 0.0, 1.0)
    ts_df['floor_3_available'] = np.maximum(0, fl3_cap - ts_df['floor_3_occupied'])

    # Temporal & Cyclical features
    ts_df['hour'] = ts_df['timestamp'].dt.hour
    ts_df['day_of_week'] = ts_df['timestamp'].dt.dayofweek
    ts_df['day_name'] = ts_df['timestamp'].dt.day_name()
    ts_df['is_weekend'] = ts_df['day_of_week'].isin([5, 6]).astype(float)
    
    # Cyclical sin/cos encodings
    ts_df['sin_hour'] = np.sin(2 * np.pi * ts_df['hour'] / 24.0)
    ts_df['cos_hour'] = np.cos(2 * np.pi * ts_df['hour'] / 24.0)
    ts_df['sin_dow'] = np.sin(2 * np.pi * ts_df['day_of_week'] / 7.0)
    ts_df['cos_dow'] = np.cos(2 * np.pi * ts_df['day_of_week'] / 7.0)

    # Library operational status flag
    open_h = config.get("operating_hours", {}).get("open_hour", 8)
    close_h = config.get("operating_hours", {}).get("close_hour", 22)
    ts_df['is_open'] = ((ts_df['hour'] >= open_h) & (ts_df['hour'] <= close_h)).astype(float)

    return ts_df


def save_processed_data(ts_df: pd.DataFrame, summary: Dict[str, Any], output_dir: str = "data/processed") -> None:
    """Saves processed files into data/processed/ directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save overall hourly occupancy
    overall_path = os.path.join(output_dir, "hourly_occupancy.csv")
    ts_df.to_csv(overall_path, index=False)

    # Save floor-wise hourly occupancy
    floor_cols = ['timestamp', 'hour', 'day_of_week', 'is_weekend',
                  'floor_1_occupied', 'floor_1_rate', 'floor_1_available',
                  'floor_2_occupied', 'floor_2_rate', 'floor_2_available',
                  'floor_3_occupied', 'floor_3_rate', 'floor_3_available']
    floor_path = os.path.join(output_dir, "hourly_floor_occupancy.csv")
    ts_df[floor_cols].to_csv(floor_path, index=False)

    # Save summary metadata
    summary_path = os.path.join(output_dir, "dataset_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[OK] Processed data saved successfully to '{output_dir}'.")
    print(f"     - {overall_path} ({len(ts_df)} hourly steps)")
    print(f"     - {floor_path}")
    print(f"     - {summary_path}")


def create_sliding_windows(
    data: np.ndarray,
    target_indices: List[int],
    lookback: int = 24
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates (X, y) sliding window sequences for LSTM training.
    X shape: (N - lookback, lookback, num_features)
    y shape: (N - lookback, len(target_indices))
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, :])
        y.append(data[i, target_indices])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def run_pipeline() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Executes the full preprocessing pipeline."""
    print("=" * 60)
    print("LIBRARY SEAT PREDICTION: PREPROCESSING PIPELINE")
    print("=" * 60)

    config = load_capacity_config()
    print(f"[1/4] Loaded configuration: Total Capacity = {config['total_capacity']} seats.")

    df, summary = inspect_raw_dataset("final dataset.csv")
    print(f"[2/4] Raw dataset inspected: {summary['total_records']} rows, {summary['total_columns']} columns.")
    print(f"      Date span: {summary['date_range']['first_entry']} -> {summary['date_range']['last_exit']}")
    print(f"      Missing values: {sum(summary['missing_values'].values())}, Duplicates: {summary['exact_duplicates']}")

    print("[3/4] Generating hourly time-series via session overlap...")
    ts_df = generate_hourly_time_series(df, config)
    print(f"      Constructed continuous timeline: {len(ts_df)} hourly observations.")

    print("[4/4] Saving processed datasets to data/processed/...")
    save_processed_data(ts_df, summary)

    return ts_df, summary


if __name__ == "__main__":
    run_pipeline()
