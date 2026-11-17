"""
TrafficSense AI - Model Training Script
=======================================
Project: Fundamentals of Artificial Intelligence (FAI) Mini-Project
Algorithm: Random Forest Regressor (Scikit-Learn)
Task: Predict vehicle_count based on temporal, weather, and road features.

Congestion Classification Rules (Demonstration Standards):
- Low:    < 300 vehicles
- Medium: 300 to 700 vehicles
- High:   > 700 vehicles
* Note: These thresholds are academic demonstration boundaries, not official
  traffic department regulations.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "traffic_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "traffic_model.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "model_metrics.json")

# Features & Target
TARGET = "vehicle_count"
CATEGORICAL_FEATURES = ["day_of_week", "weather", "road_name"]
NUMERIC_FEATURES = ["hour", "is_weekend"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Congestion Thresholds
CONGESTION_THRESHOLDS = {
    "low_max": 299,
    "medium_min": 300,
    "medium_max": 700,
    "high_min": 701,
    "description": "Educational thresholds: Low (<300), Medium (300-700), High (>700)"
}


def classify_congestion(vehicle_count):
    """Classifies vehicle count into congestion category."""
    if vehicle_count < 300:
        return "Low"
    elif vehicle_count <= 700:
        return "Medium"
    else:
        return "High"


def load_and_validate_data(filepath):
    """Loads dataset and performs validation and cleaning."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at '{filepath}'. Please run 'python generate_data.py' first!"
        )

    df = pd.read_csv(filepath)
    print(f"[DATA] Successfully loaded {len(df)} records from {filepath}")

    # Check required columns
    required_cols = ALL_FEATURES + [TARGET]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    # Check for nulls and drop invalid rows
    initial_len = len(df)
    df = df.dropna(subset=required_cols)
    if len(df) < initial_len:
        print(f"[CLEAN] Dropped {initial_len - len(df)} rows with missing values.")

    # Data type enforcement
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df["is_weekend"] = pd.to_numeric(df["is_weekend"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["hour", "is_weekend", TARGET])

    df["hour"] = df["hour"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df[TARGET] = df[TARGET].astype(int)

    # Validate value ranges
    df = df[(df["hour"] >= 0) & (df["hour"] <= 23)]
    df = df[df["is_weekend"].isin([0, 1])]
    df = df[df[TARGET] >= 0]

    print(f"[VALID] Verified {len(df)} clean records for ML training.")
    return df


def build_pipeline():
    """
    Constructs an end-to-end Scikit-Learn Pipeline combining
    feature transformation and RandomForestRegressor.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES)
        ],
        remainder="drop"
    )

    model = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])

    return pipeline


def train_and_evaluate():
    """Main training routine."""
    print("=" * 65)
    print(" TrafficSense AI: Machine Learning Model Training (Random Forest)")
    print("=" * 65)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load Data
    df = load_and_validate_data(DATA_PATH)

    X = df[ALL_FEATURES]
    y = df[TARGET]

    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"[SPLIT] Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

    # 3. Build & Train Pipeline
    print("[TRAIN] Fitting Random Forest Regressor pipeline...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("[TRAIN] Model training completed.")

    # 4. Evaluation
    y_pred = pipeline.predict(X_test)
    y_train_pred = pipeline.predict(X_train)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    print("\n" + "-" * 65)
    print(" MODEL PERFORMANCE EVALUATION (TEST SET)")
    print("-" * 65)
    print(f" Mean Absolute Error (MAE):      {mae:.2f} vehicles")
    print(f" Root Mean Squared Error (RMSE): {rmse:.2f} vehicles")
    print(f" R-Squared Score (Test R²):      {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f" R-Squared Score (Train R²):     {train_r2:.4f}")
    print("-" * 65)

    # 5. Save Model Pipeline & Metadata
    metadata = {
        "model_type": "RandomForestRegressor",
        "algorithm": "Scikit-Learn Pipeline with ColumnTransformer (OneHotEncoder + StandardScaler)",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": len(X_train),
        "test_samples": len(X_test),
        "metrics": {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "r2": round(float(r2), 4),
            "train_r2": round(float(train_r2), 4)
        },
        "features": {
            "categorical": CATEGORICAL_FEATURES,
            "numeric": NUMERIC_FEATURES,
            "all": ALL_FEATURES
        },
        "congestion_thresholds": CONGESTION_THRESHOLDS,
        "sample_prediction_test": {
            "input": {
                "hour": 9,
                "day_of_week": "Monday",
                "weather": "Rainy",
                "road_name": "Silk Board",
                "is_weekend": 0
            },
            "predicted_count": int(round(pipeline.predict(pd.DataFrame([{
                "hour": 9,
                "day_of_week": "Monday",
                "weather": "Rainy",
                "road_name": "Silk Board",
                "is_weekend": 0
            }]))[0])),
        }
    }
    sample_vol = metadata["sample_prediction_test"]["predicted_count"]
    metadata["sample_prediction_test"]["congestion"] = classify_congestion(sample_vol)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"[SAVE] Serialized model pipeline to: {MODEL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[SAVE] Saved performance metrics & metadata to: {METRICS_PATH}")

    print("\nDemonstration Prediction Test:")
    print(f"  Input: Silk Board | Monday 9:00 AM | Rainy | Weekday")
    print(f"  Predicted Volume: {sample_vol} vehicles")
    print(f"  Congestion Level: {metadata['sample_prediction_test']['congestion']}")
    print("=" * 65)
    print(" SUCCESS: Model ready for Flask backend integration!")
    print("=" * 65)


if __name__ == "__main__":
    train_and_evaluate()
