# Placeholder for utility functions: preprocessing, scaling, etc.

import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Tuple


def save_forecast(forecast_df: pd.DataFrame, file_path: str):
    """
    Save forecast dataframe to CSV, appending with timestamp.
    :param forecast_df: DataFrame with Chronos forecast results
    :param file_path: path to CSV file to save/append
    """
    forecast_df = forecast_df.copy()
    forecast_df['generated_at'] = datetime.now()

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if os.path.exists(file_path):
        forecast_df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        forecast_df.to_csv(file_path, index=False)


def load_latest_forecast_backtest(n=5):
    df = pd.read_csv("data/forecasts_backtest.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=False).head(n)
    return df.to_dict(orient="records")

def load_latest_forecast_future(n=5):
    df = pd.read_csv("data/forecasts_future.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date", ascending=False).head(n)
    return df.to_dict(orient="records")



def detect_anomalies_zscore(
    df: pd.DataFrame,
    column: str,
    window: int = 20,
    threshold: float = 3.0,
    min_periods: int = 5
) -> pd.DataFrame:
    """
    Detect anomalies in `column` using rolling z-score.
    Returns a copy of df with two new columns:
      - <column>_zscore : rolling z-score
      - <column>_anomaly : boolean (True when abs(z) > threshold)
    Parameters:
      window: rolling window size in rows
      threshold: absolute z-score threshold to flag anomaly
    """
    out = df.copy()
    # rolling mean/std
    roll_mean = out[column].rolling(window=window, min_periods=min_periods).mean()
    roll_std = out[column].rolling(window=window, min_periods=min_periods).std().replace(0, np.nan)

    zscore = (out[column] - roll_mean) / (roll_std + 1e-12)
    zcol = f"{column}_zscore"
    acol = f"{column}_anomaly"
    out[zcol] = zscore
    out[acol] = out[zcol].abs() > threshold
    return out

def detect_anomalies_iqr(
    df: pd.DataFrame,
    column: str,
    multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Detect anomalies using the IQR method across the whole series (non-rolling).
    Returns a copy of df with:
      - <column>_iqr_lower
      - <column>_iqr_upper
      - <column>_iqr_anomaly (True if outside [lower,upper])
    """
    out = df.copy()
    q1 = out[column].quantile(0.25)
    q3 = out[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    out[f"{column}_iqr_lower"] = lower
    out[f"{column}_iqr_upper"] = upper
    out[f"{column}_iqr_anomaly"] = (out[column] < lower) | (out[column] > upper)
    return out

def summarize_anomaly_stats(df: pd.DataFrame, anomaly_col: str) -> dict:
    """
    Small summary of anomaly boolean column:
      - n_rows, n_anomalies, pct_anomalies, first/last anomaly index & value (if any)
    """
    total = len(df)
    if anomaly_col not in df.columns:
        return {"error": f"{anomaly_col} not found in DataFrame"}

    n_anom = int(df[anomaly_col].sum())
    pct = (n_anom / total) if total > 0 else 0.0

    res = {"n_rows": total, "n_anomalies": n_anom, "pct_anomalies": float(pct)}
    if n_anom > 0:
        idxs = df.index[df[anomaly_col]].tolist()
        first_idx = idxs[0]
        last_idx = idxs[-1]
        res.update({
            "first_anomaly_index": int(first_idx),
            "last_anomaly_index": int(last_idx),
            "first_anomaly_value": float(df.loc[first_idx].get("close", df.loc[first_idx].iloc[0])),
            "last_anomaly_value": float(df.loc[last_idx].get("close", df.loc[last_idx].iloc[0])),
        })
    return res

