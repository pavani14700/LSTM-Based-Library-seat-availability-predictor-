# 📚 SeatSense AI – Library Seat Availability Predictor

A complete, production-ready, local deep learning web application that forecasts future university library occupancy and seat availability using a genuine **TensorFlow/Keras Long Short-Term Memory (LSTM) neural network**.

Built with **Python**, **Streamlit**, **TensorFlow/Keras**, **Pandas**, **NumPy**, **Scikit-learn**, and **Plotly**.

---

## 🔒 Strict Zero-API Guarantee

This application runs **100% locally and offline**:
- **No external APIs, cloud services, or API keys** required.
- **No REST endpoints, FastAPI, or Flask servers** needed.
- Predictions are generated locally by the trained TensorFlow LSTM model running on your computer.
- Preserves the original `final dataset.csv` without modifying or overwriting it.

---

## 🌟 Key Features

1. **Student Login Portal**: Secure student authentication supporting student IDs (from 8,966 registered students in the dataset or custom guest access), demo accounts, personal study metrics, and attendance tracking.
2. **Side Navigation Menu Bar**: All analytical output features organized into an intuitive sidebar menu:
   - 🔮 **Seat Availability Predictor**: Real-time and future forecasting for selected date and hourly slots.
   - 🏢 **Floor-Wise Breakdown**: Detailed metrics for Floor 1 (Quiet Study), Floor 2 (Collaborative & Tech), and Floor 3 (Reference).
   - 📈 **Interactive Forecast Charts**: 24-hour horizon curve, floor comparisons, and historical timelines.
   - 🎓 **My Student Dashboard**: Personal attendance history, cumulative study hours, favorite floor, and smart study recommendations.
   - 🎯 **Model Accuracy & Benchmarks**: LSTM test metrics (MAE, RMSE, R²) vs 24-hour persistence baseline.
   - ⚙️ **Capacities & Settings**: Dynamic seating capacity controls.
   - 📖 **System Architecture**: Technical methodology, session overlap formulation, and offline guarantees.
3. **Genuine Deep Learning Architecture**: Multi-output 2-layer LSTM with Dropout regularization and Dense projection.
4. **Multi-Step Recursive Forecasting**: Allows users to select any future date and hourly time slot (e.g., 08:00 AM to 10:00 PM).
5. **Overall & Floor-Wise Breakdown**: Displays predicted occupancy percentage, estimated occupied seats, and estimated available seats for the overall library as well as **Floor 1**, **Floor 2**, and **Floor 3**.
6. **Permanent Availability Rules (Two-State Binary Classification)**:
   - 🟢 **Available:** At least one seat is vacant (`available_seats > 0`). When 1 seat remains, displays **“Available — 1 seat remaining.”**
   - 🔴 **Occupied:** All seats in the library or floor are occupied (`available_seats == 0`). Displays **“Occupied — 0 seats available.”**
7. **Interactive Plotly Visualizations**:
   - 24-Hour Forecast Horizon curve with highlighted selected slot.
   - Floor-by-floor occupancy comparison.
   - Historical vs. Predicted time-series continuation.
   - Model Evaluation vs. 24-hour persistence baseline benchmark.
8. **Configurable Capacities**: Real-time configuration of total library seating capacity and per-floor seat allocations via the sidebar.

---

## 🏗️ Project Structure

```text
library-seat-prediction/
├── app.py                      # Interactive Streamlit dashboard application
├── final dataset.csv           # Original dataset (preserved intact)
├── data/
│   ├── final dataset.csv       # Working copy of dataset
│   └── processed/
│       ├── hourly_occupancy.csv         # 4,313 continuous hourly observations
│       ├── hourly_floor_occupancy.csv   # Floor-wise hourly time series
│       └── dataset_summary.json         # Dataset metadata and statistics
├── ml/
│   ├── __init__.py
│   ├── preprocess.py           # Session overlap processing & window generation
│   ├── train_lstm.py           # TensorFlow/Keras LSTM model training
│   ├── predict.py              # Recursive multi-step inference engine
│   └── evaluate.py             # Evaluation on unseen chronological test data
├── models/
│   ├── lstm_model.keras        # Trained Keras LSTM model artifact
│   ├── scaler.pkl              # Feature scaler fit only on training split
│   ├── model_config.json       # Lookback window and feature configuration
│   └── metrics.json            # Actual test evaluation metrics (MAE, RMSE)
├── config/
│   └── capacities.json         # Seating capacity and threshold configuration
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Complete documentation
```

---

## 📊 Dataset Summary

The primary dataset `final dataset.csv` contains:
- **50,000 student session records** spanning from **January 1, 2026** to **June 29, 2026** (180 consecutive days).
- **Zero missing values** and **zero duplicate records**.
- Sessions record entry datetime, exit datetime, duration (minutes), department, academic year, and `library_zone`:
  - `floor_1`: 16,784 records
  - `floor_2`: 16,540 records
  - `floor_3`: 16,676 records
- **Hourly Presence Calculation**: Computed by counting students whose sessions overlap each hourly interval $[h, h+1)$:
  $$\text{Active Students at hour } h = \sum_{i} \mathbb{I}(\text{entry}_i < h+1 \text{ and } \text{exit}_i > h)$$

---

## 🎯 Model Architecture & Evaluation Results

### Model Specifications
- **Input Window**: 24 consecutive hourly observations (24-hour lookback).
- **Features (10)**: `occupancy_rate`, `floor_1_rate`, `floor_2_rate`, `floor_3_rate`, `sin_hour`, `cos_hour`, `sin_dow`, `cos_dow`, `is_weekend`, `is_open`.
- **Layers**:
  - `LSTM(64, return_sequences=True)` + `Dropout(0.2)`
  - `LSTM(32, return_sequences=False)` + `Dropout(0.2)`
  - `Dense(32, activation='relu')`
  - `Dense(4, activation='sigmoid')` (multi-output bounded in $[0, 1]$)
- **Data Split**: Chronological (70% Train, 15% Validation, 15% Test) — **Zero Data Leakage**.
- **Scaler**: `MinMaxScaler` fit **only** on the training split.

### Actual Evaluation on Unseen Chronological Test Data (June 3 to June 30, 2026)

| Target | LSTM MAE | LSTM RMSE | 24h Baseline MAE | 24h Baseline RMSE | MAE Improvement |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Overall Library Occupancy** | **3.01%** | **0.0467** | 3.96% | 0.0638 | **+24.1%** |
| **Floor 1 (Quiet Study)** | **4.53%** | **0.0706** | 5.65% | 0.0957 | **+19.8%** |
| **Floor 2 (Collaborative)** | **4.16%** | **0.0650** | 5.64% | 0.0909 | **+26.3%** |
| **Floor 3 (Reference Room)** | **4.74%** | **0.0757** | 6.05% | 0.1001 | **+21.7%** |

*R² Score on Overall Library Occupancy:* **0.9722**

---

## 🚀 Setup & Execution Guide (Windows)

### 1. Prerequisites
- **Python 3.10, 3.11, or 3.12** installed on Windows. (Check via `python --version` or `py --list`).

### 2. Open PowerShell or Command Prompt
Open PowerShell in the project directory:
```powershell
cd "c:\Users\thota\OneDrive\Documents\dl pro"
```

### 3. Create and Activate Virtual Environment (Optional but Recommended)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 5. Verify Dataset Location
Ensure `final dataset.csv` is located in the project root directory:
```powershell
Test-Path "final dataset.csv"
```
*(The original CSV is never modified or overwritten).*

### 6. Run Preprocessing Pipeline
Processes sessions, calculates hourly active overlaps, and creates `data/processed/`:
```powershell
python ml/preprocess.py
```

### 7. Train the LSTM Deep Learning Model
Trains the multi-output LSTM model, compares with baseline, and saves model artifacts:
```powershell
python ml/train_lstm.py
```

### 8. Run Standalone Model Evaluation (Optional)
```powershell
python ml/evaluate.py
```

### 9. Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

---

## 💡 How to Use the Application

1. **Select a Future Date**: Use the date picker to choose an upcoming date (e.g., July 1, 2026 or any date).
2. **Select an Hourly Time Slot**: Choose a time slot between 08:00 AM and 10:00 PM.
3. **Click "Predict Seat Availability"**: The app runs the local LSTM model to project occupancy.
4. **View Results**:
   - Check the **Predicted Occupancy %**, **Occupied Seats**, and **Available Seats**.
   - Review the color-coded **Availability Status** and recommendation.
   - Inspect individual **Floor 1**, **Floor 2**, and **Floor 3** availability cards.
   - Interact with the Plotly 24-hour forecast curve and floor comparisons.
5. **Adjust Seating Capacities**: Use the sidebar to tune total library seating capacity (e.g. 150 seats) or floor capacities to match your campus library.

---

## ⚙️ Configuration (`config/capacities.json`)

You can customize library seating capacities and status thresholds:

```json
{
  "total_capacity": 150,
  "floors": {
    "floor_1": {
      "name": "Floor 1 (Quiet Study & Digital Library)",
      "capacity": 50
    },
    "floor_2": {
      "name": "Floor 2 (Collaborative & Tech Zone)",
      "capacity": 50
    },
    "floor_3": {
      "name": "Floor 3 (Reference & Reading Room)",
      "capacity": 50
    }
  },
  "operating_hours": {
    "open_hour": 8,
    "close_hour": 22
  }
}
```

---

## ⚠️ Known Limitations

1. **Operating Hours**: The library operating schedule is between 08:00 AM and 10:00 PM. Outside these hours (00:00 to 07:00), the library is closed and seat occupancy is 0.
2. **Recursive Multi-Step Horizon**: Forecasting dates far beyond the historical range (>14 days ahead) is subject to multi-step recursive error accumulation. A friendly warning is shown on the dashboard for long horizons.
3. **Special Events**: Unannounced campus events or unscheduled library maintenance are not reflected in historical patterns.
4. **Estimates, Not Guarantees**: Predicted seat numbers are statistical projections designed to help students plan study sessions, not live sensor hardware counts.

---

## 📜 License

Created for academic and university library occupancy forecasting research. Runs completely locally without third-party subscriptions or cloud services.
