"""
Prediction and Inference Engine for Library Seat Availability.
Loads the trained LSTM model and historical data, performs multi-step recursive forecasting,
calculates occupied/available seats and crowd availability status for overall library and floors.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf


class LibraryPredictor:
    """Manages model loading and inference for future library occupancy."""

    def __init__(
        self,
        model_path: str = "models/lstm_model.keras",
        scaler_path: str = "models/scaler.pkl",
        config_path: str = "models/model_config.json",
        historical_path: str = "data/processed/hourly_occupancy.csv"
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.historical_path = historical_path

        self.model: Optional[tf.keras.Model] = None
        self.scaler = None
        self.config: Dict[str, Any] = {}
        self.historical_df: Optional[pd.DataFrame] = None
        self._load_resources()

    def _load_resources(self) -> None:
        """Loads model, scaler, configuration and historical data once."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Trained model '{self.model_path}' not found. Train the model first.")

        self.model = tf.keras.models.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.historical_df = pd.read_csv(self.historical_path)
        self.historical_df['timestamp'] = pd.to_datetime(self.historical_df['timestamp'])
        self.historical_df = self.historical_df.sort_values('timestamp').reset_index(drop=True)

        # Precompute calendar diurnal profiles (day_of_week, hour) for fast future anchoring
        temp_df = self.historical_df.copy()
        temp_df['dow'] = temp_df['timestamp'].dt.dayofweek
        temp_df['hour'] = temp_df['timestamp'].dt.hour
        self.diurnal_profile = temp_df.groupby(['dow', 'hour'])[['occupancy_rate', 'floor_1_rate', 'floor_2_rate', 'floor_3_rate']].mean().to_dict(orient='index')

    @property
    def last_historical_time(self) -> datetime:
        """Returns the most recent timestamp present in the historical dataset."""
        return self.historical_df['timestamp'].iloc[-1].to_pydatetime()

    def determine_status(self, available_seats: int, total_capacity: int = 150, **kwargs) -> Dict[str, str]:
        """
        Permanent Availability Rules:
        - Available: At least one seat is vacant (available_seats > 0).
        - Occupied: All seats in the library or the relevant floor are occupied (available_seats <= 0).
        Zero capacity is handled safely to prevent division-by-zero errors.
        """
        safe_cap = max(0, int(total_capacity))
        avail = max(0, int(available_seats))

        if safe_cap <= 0:
            return {
                "status": "Occupied",
                "level": "occupied",
                "color": "#DC2626",  # Red
                "badge": "OCCUPIED",
                "display_text": "Occupied — 0 seats available.",
                "recommendation": "Zero capacity configured. 0 seats are available."
            }

        if avail > 0:
            if avail == 1:
                display_text = "Available — 1 seat remaining."
                rec = "Available — exactly 1 seat remaining."
            else:
                display_text = f"Available — {avail} seats remaining."
                rec = f"Available — {avail} seats available."

            return {
                "status": "Available",
                "level": "available",
                "color": "#059669",  # Emerald Green
                "badge": "AVAILABLE",
                "display_text": display_text,
                "recommendation": rec
            }
        else:
            return {
                "status": "Occupied",
                "level": "occupied",
                "color": "#DC2626",  # Red
                "badge": "OCCUPIED",
                "display_text": "Occupied — 0 seats available.",
                "recommendation": "Occupied — all seats are currently occupied."
            }

    def _generate_time_features(self, dt: datetime, open_h: int = 8, close_h: int = 22) -> Dict[str, float]:
        """Computes deterministic calendar and cyclical features for a given datetime."""
        hour = dt.hour
        dow = dt.weekday()
        is_weekend = 1.0 if dow in [5, 6] else 0.0
        is_open = 1.0 if (open_h <= hour <= close_h) else 0.0

        return {
            "sin_hour": float(np.sin(2 * np.pi * hour / 24.0)),
            "cos_hour": float(np.cos(2 * np.pi * hour / 24.0)),
            "sin_dow": float(np.sin(2 * np.pi * dow / 7.0)),
            "cos_dow": float(np.cos(2 * np.pi * dow / 7.0)),
            "is_weekend": is_weekend,
            "is_open": is_open
        }

    def forecast_range(
        self,
        target_datetime: datetime,
        custom_capacities: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates forecast for target datetime and the entire target date (00:00 to 23:00).
        Uses recursive rolling window LSTM inference.
        """
        caps = custom_capacities or self.config.get("capacities", {})
        total_cap = max(0, int(caps.get("total_capacity", 150)))
        open_h = caps.get("operating_hours", {}).get("open_hour", 8)
        close_h = caps.get("operating_hours", {}).get("close_hour", 22)

        fl1_cap = max(0, int(caps.get("floors", {}).get("floor_1", {}).get("capacity", 50)))
        fl2_cap = max(0, int(caps.get("floors", {}).get("floor_2", {}).get("capacity", 50)))
        fl3_cap = max(0, int(caps.get("floors", {}).get("floor_3", {}).get("capacity", 50)))

        feature_cols = self.config["feature_columns"]
        lookback = self.config["lookback_window"]

        last_hist_dt = self.last_historical_time
        target_hour = target_datetime.replace(minute=0, second=0, microsecond=0)

        # Warning for horizon > 14 days (336 hours)
        hours_ahead = int((target_hour - last_hist_dt).total_seconds() / 3600)
        horizon_warning = None
        if hours_ahead > 336:
            days_ahead = round(hours_ahead / 24, 1)
            horizon_warning = (
                f"Selected date is {days_ahead} days ahead ({hours_ahead} hours beyond historical records). "
                f"Multi-step recursive forecasting carries increasing uncertainty for distant horizons."
            )
        elif hours_ahead < 0:
            horizon_warning = "Selected date/time lies within the historical training period."

        # Target day full hours (00:00 to 23:00)
        day_start = target_hour.replace(hour=0)
        day_end = target_hour.replace(hour=23)
        forecast_end_dt = max(target_hour, day_end)

        # If target hour is already in historical data, retrieve it directly
        hist_match = self.historical_df[self.historical_df['timestamp'] == target_hour]
        if not hist_match.empty and hours_ahead <= 0:
            row = hist_match.iloc[0]
            occ_rate = float(row['occupancy_rate'])
            fl1_rate = float(row['floor_1_rate'])
            fl2_rate = float(row['floor_2_rate'])
            fl3_rate = float(row['floor_3_rate'])
            
            occ_seats = int(round(occ_rate * total_cap))
            avail_seats = max(0, total_cap - occ_seats)

            status_info = self.determine_status(avail_seats, total_cap)

            fl1_occupied = int(round(fl1_rate * fl1_cap))
            fl1_available = max(0, fl1_cap - fl1_occupied)
            fl1_status = self.determine_status(fl1_available, fl1_cap)

            fl2_occupied = int(round(fl2_rate * fl2_cap))
            fl2_available = max(0, fl2_cap - fl2_occupied)
            fl2_status = self.determine_status(fl2_available, fl2_cap)

            fl3_occupied = int(round(fl3_rate * fl3_cap))
            fl3_available = max(0, fl3_cap - fl3_occupied)
            fl3_status = self.determine_status(fl3_available, fl3_cap)

            # Also get 24h of that day
            day_hist = self.historical_df[self.historical_df['timestamp'].dt.date == target_hour.date()].copy()
            hourly_curve = []
            for _, d_row in day_hist.iterrows():
                h_rate = float(d_row['occupancy_rate'])
                h_occ = int(round(h_rate * total_cap))
                h_avail = max(0, total_cap - h_occ)
                h_status_info = self.determine_status(h_avail, total_cap)
                hourly_curve.append({
                    "timestamp": d_row['timestamp'].strftime("%Y-%m-%d %H:%M"),
                    "hour": d_row['timestamp'].hour,
                    "occupancy_rate": h_rate,
                    "occupied_seats": h_occ,
                    "available_seats": h_avail,
                    "status": h_status_info["status"],
                    "display_text": h_status_info["display_text"],
                    "floor_1_rate": float(d_row['floor_1_rate']),
                    "floor_2_rate": float(d_row['floor_2_rate']),
                    "floor_3_rate": float(d_row['floor_3_rate'])
                })

            return {
                "target_datetime": target_hour.strftime("%Y-%m-%d %H:00"),
                "is_historical": True,
                "warning": horizon_warning,
                "overall": {
                    "occupancy_rate": occ_rate,
                    "occupancy_percent": round(occ_rate * 100, 1),
                    "total_capacity": total_cap,
                    "occupied_seats": occ_seats,
                    "available_seats": avail_seats,
                    **status_info
                },
                "floors": {
                    "floor_1": {
                        "name": caps.get("floors", {}).get("floor_1", {}).get("name", "Floor 1"),
                        "rate": fl1_rate,
                        "percent": round(fl1_rate * 100, 1),
                        "capacity": fl1_cap,
                        "occupied": fl1_occupied,
                        "available": fl1_available,
                        **fl1_status
                    },
                    "floor_2": {
                        "name": caps.get("floors", {}).get("floor_2", {}).get("name", "Floor 2"),
                        "rate": fl2_rate,
                        "percent": round(fl2_rate * 100, 1),
                        "capacity": fl2_cap,
                        "occupied": fl2_occupied,
                        "available": fl2_available,
                        **fl2_status
                    },
                    "floor_3": {
                        "name": caps.get("floors", {}).get("floor_3", {}).get("name", "Floor 3"),
                        "rate": fl3_rate,
                        "percent": round(fl3_rate * 100, 1),
                        "capacity": fl3_cap,
                        "occupied": fl3_occupied,
                        "available": fl3_available,
                        **fl3_status
                    }
                },
                "hourly_curve": hourly_curve
            }

        # Otherwise: Run Recursive Multi-Step Forecast
        target_record = None
        day_records = []

        if hours_ahead > 72:
            # Fast Calendar-Anchored Initialization:
            # For dates far in the future (weeks/months ahead), stepping through thousands of intermediate
            # hours takes minutes and causes compounding error drift.
            # We construct an anchor lookback window from the 24 hours immediately preceding the target date
            # using empirical day-of-week & hourly calendar profiles, then simulate only the target day (24 steps).
            anchor_rows = []
            for h in range(lookback):
                adt = day_start - timedelta(hours=lookback - h)
                rates = self.diurnal_profile.get(
                    (adt.weekday(), adt.hour),
                    {"occupancy_rate": 0.0, "floor_1_rate": 0.0, "floor_2_rate": 0.0, "floor_3_rate": 0.0}
                )
                tf_dict = self._generate_time_features(adt, open_h, close_h)
                row = {**rates, **tf_dict}
                anchor_rows.append([row[col] for col in feature_cols])

            current_window = self.scaler.transform(np.array(anchor_rows))
            sim_dt = day_start
        else:
            # Near horizon (within 3 days): roll forward sequentially from the last historical observation
            recent_window_df = self.historical_df.tail(lookback).copy()
            rolling_features = recent_window_df[feature_cols].values.copy()
            current_window = self.scaler.transform(rolling_features)
            sim_dt = last_hist_dt + timedelta(hours=1)

        while sim_dt <= forecast_end_dt:
            # 1. Prepare batch of 1
            input_seq = current_window[np.newaxis, :, :]  # Shape: (1, 24, n_features)

            # 2. Predict next hour [occupancy_rate, floor_1, floor_2, floor_3]
            pred_targets = self.model.predict(input_seq, verbose=0)[0]
            pred_targets = np.clip(pred_targets, 0.0, 1.0)

            # Enforce zero occupancy if library is closed at night (00:00 to 07:00)
            if sim_dt.hour < open_h or sim_dt.hour > 23:
                pred_targets = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

            pred_occ, pred_fl1, pred_fl2, pred_fl3 = pred_targets

            # 3. Form unscaled feature vector for this step
            time_feats = self._generate_time_features(sim_dt, open_h, close_h)
            next_feat_dict = {
                "occupancy_rate": pred_occ,
                "floor_1_rate": pred_fl1,
                "floor_2_rate": pred_fl2,
                "floor_3_rate": pred_fl3,
                **time_feats
            }
            next_vec = np.array([[next_feat_dict[col] for col in feature_cols]])
            next_scaled = self.scaler.transform(next_vec)

            # 4. Roll window forward
            current_window = np.vstack([current_window[1:, :], next_scaled])

            pred_occ_seats = int(round(pred_occ * total_cap))
            pred_avail_seats = max(0, total_cap - pred_occ_seats)
            rec_status = self.determine_status(pred_avail_seats, total_cap)

            rec = {
                "timestamp": sim_dt.strftime("%Y-%m-%d %H:%M"),
                "hour": sim_dt.hour,
                "occupancy_rate": float(pred_occ),
                "occupied_seats": pred_occ_seats,
                "available_seats": pred_avail_seats,
                "status": rec_status["status"],
                "display_text": rec_status["display_text"],
                "floor_1_rate": float(pred_fl1),
                "floor_2_rate": float(pred_fl2),
                "floor_3_rate": float(pred_fl3)
            }

            if sim_dt.date() == target_hour.date():
                day_records.append(rec)

            if sim_dt == target_hour:
                target_record = rec

            sim_dt += timedelta(hours=1)

        # Fallback if target record not captured directly
        if target_record is None:
            target_record = day_records[-1] if day_records else rec

        tgt_rate = target_record["occupancy_rate"]
        tgt_fl1 = target_record["floor_1_rate"]
        tgt_fl2 = target_record["floor_2_rate"]
        tgt_fl3 = target_record["floor_3_rate"]

        tgt_avail = target_record["available_seats"]
        tgt_occ = target_record["occupied_seats"]

        overall_status = self.determine_status(tgt_avail, total_cap)

        fl1_occupied = int(round(tgt_fl1 * fl1_cap))
        fl1_available = max(0, fl1_cap - fl1_occupied)
        fl1_status = self.determine_status(fl1_available, fl1_cap)

        fl2_occupied = int(round(tgt_fl2 * fl2_cap))
        fl2_available = max(0, fl2_cap - fl2_occupied)
        fl2_status = self.determine_status(fl2_available, fl2_cap)

        fl3_occupied = int(round(tgt_fl3 * fl3_cap))
        fl3_available = max(0, fl3_cap - fl3_occupied)
        fl3_status = self.determine_status(fl3_available, fl3_cap)

        return {
            "target_datetime": target_hour.strftime("%Y-%m-%d %H:00"),
            "is_historical": False,
            "hours_ahead": hours_ahead,
            "warning": horizon_warning,
            "overall": {
                "occupancy_rate": tgt_rate,
                "occupancy_percent": round(tgt_rate * 100, 1),
                "total_capacity": total_cap,
                "occupied_seats": tgt_occ,
                "available_seats": tgt_avail,
                **overall_status
            },
            "floors": {
                "floor_1": {
                    "name": caps.get("floors", {}).get("floor_1", {}).get("name", "Floor 1"),
                    "rate": tgt_fl1,
                    "percent": round(tgt_fl1 * 100, 1),
                    "capacity": fl1_cap,
                    "occupied": fl1_occupied,
                    "available": fl1_available,
                    **fl1_status
                },
                "floor_2": {
                    "name": caps.get("floors", {}).get("floor_2", {}).get("name", "Floor 2"),
                    "rate": tgt_fl2,
                    "percent": round(tgt_fl2 * 100, 1),
                    "capacity": fl2_cap,
                    "occupied": fl2_occupied,
                    "available": fl2_available,
                    **fl2_status
                },
                "floor_3": {
                    "name": caps.get("floors", {}).get("floor_3", {}).get("name", "Floor 3"),
                    "rate": tgt_fl3,
                    "percent": round(tgt_fl3 * 100, 1),
                    "capacity": fl3_cap,
                    "occupied": fl3_occupied,
                    "available": fl3_available,
                    **fl3_status
                }
            },
            "hourly_curve": day_records
        }


if __name__ == "__main__":
    predictor = LibraryPredictor()
    test_dt = predictor.last_historical_time + timedelta(days=1, hours=4)
    print(f"Testing prediction for future date: {test_dt}")
    res = predictor.forecast_range(test_dt)
    print(json.dumps(res["overall"], indent=2))
    print("\nFloor 1:")
    print(json.dumps(res["floors"]["floor_1"], indent=2))
