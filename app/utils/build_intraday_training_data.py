# scripts/build_intraday_training_data.py
import os
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from app.utils.intraday_features import build_intraday_features

def build_training_data(ticker="NVDA", days=60):
    records = []

    for i in range(days, 0, -1):
        start_day = datetime.now() - timedelta(days=i)
        end_day = start_day + timedelta(days=1)

        data = yf.download(
            tickers=ticker,
            start=start_day.strftime("%Y-%m-%d"),
            end=end_day.strftime("%Y-%m-%d"),
            interval="15m",
            progress=False,
            auto_adjust=True
        )

        if data.empty:
            continue

        # --- Fix MultiIndex columns if they exist ---
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] for col in data.columns]

        data = data.reset_index()
        data["ticker"] = ticker

        # Ensure column names are consistent
        data.rename(columns={"datetime": "Datetime"}, inplace=True)

        # --- build features ---
        features = build_intraday_features(data.copy())

        features["date"] = pd.to_datetime(features["Datetime"]).dt.date

        # --- assign final close per day ---
        features["final_close"] = features.groupby("date")["close"].transform("last")

        records.append(features)

    if not records:
        print("No valid data downloaded.")
        return

    df = pd.concat(records, ignore_index=True)

    # Drop rows where final_close couldn't be computed
    df = df.dropna(subset=["final_close"])

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/intraday_training.csv", index=False)

    print(f"✅ Saved training data with {df['date'].nunique()} days, {len(df)} rows.")
    print(df[["Datetime", "close", "final_close"]].tail())


if __name__ == "__main__":
    build_training_data()


