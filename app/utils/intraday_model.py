import os
import joblib
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from app.utils.intraday_fetcher import fetch_intraday_data
from app.utils.intraday_features import build_intraday_features


MODEL_PATH = "artifacts/intraday_xgb_model.joblib"
TRAIN_DATA_PATH = "data/intraday_training.csv"

def train_intraday_model():
    """Train an XGBoost model for intraday close prediction."""
    if not os.path.exists(TRAIN_DATA_PATH):
        raise FileNotFoundError("Historical training dataset not found!")

    df = pd.read_csv(TRAIN_DATA_PATH)
    df["final_close"] = pd.to_numeric(df["final_close"], errors="coerce")
    df = df.dropna(subset=["final_close"])

    drop_cols = ["final_close", "Datetime", "ticker", "date"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    X = X.select_dtypes(include=["number"]).copy()
    y = df["final_close"]

    if X.empty:
        raise ValueError("No numeric features found for training!")

    print(f"✅ Training model with {X.shape[1]} numeric features...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

    preds = model.predict(X)
    print(f"Training metrics: R²={r2_score(y,preds):.3f}, MAE={mean_absolute_error(y,preds):.3f}, RMSE={np.sqrt(mean_squared_error(y,preds)):.3f}")

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "features": X.columns.tolist()}, MODEL_PATH)
    print(f"💾 Model saved to {MODEL_PATH}")
    return model

def predict_today_final_close(ticker="NVDA"):
    """Predict today's intraday close using saved model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Train the model first!")

    data = joblib.load(MODEL_PATH)
    model = data["model"]
    model_features = data["features"]

    historical = fetch_intraday_data(ticker)
    features = build_intraday_features(historical)

    drop_cols = [c for c in ["Datetime", "ticker", "date"] if c in features.columns]
    X_today = features.drop(columns=drop_cols, errors="ignore")
    X_today = X_today.apply(pd.to_numeric, errors='coerce').ffill().fillna(0)

    for col in model_features:
        if col not in X_today.columns:
            X_today[col] = 0.0
    X_today = X_today[model_features]

    pred = model.predict(X_today)
    features["predicted_close"] = pred

    predicted_day = pd.to_datetime(features["Datetime"].iloc[-1]).date()
    print(f"[{datetime.now()}] Predicted intraday close for {ticker} on {predicted_day}: {pred[-1]:.2f}")
    return features


