from fastapi import APIRouter
import pandas as pd
import os
from datetime import datetime

from app.utils.helpers import detect_anomalies_zscore, summarize_anomaly_stats
from app.utils.llm_client import summarize_analysis

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INTRADAY_FILE = os.path.join(DATA_DIR, "intraday_features_all.csv")
FORECAST_FILE = os.path.join(DATA_DIR, "forecast_future.csv")
HISTORICAL_FILE = os.path.join(DATA_DIR, "nvdastocks.csv")
INSIGHTS_LOG_FILE = os.path.join(DATA_DIR, "insights_log.csv")


@router.get("/")
def analyze_data(ticker: str = "NVDA", column: str = "close", history_days: int = 5):
    """
    Combine intraday, historical, and forecast data for analysis and anomaly detection.
    """
    # --- Load datasets safely ---
    def safe_read(path):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception as e:
                print(f"⚠️ Failed to read {path}: {e}")
        return pd.DataFrame()

    intraday = safe_read(INTRADAY_FILE)
    forecasts = safe_read(FORECAST_FILE)
    history = safe_read(HISTORICAL_FILE)

    if intraday.empty and history.empty:
        return {"error": "No data available. Run data fetchers first."}

    # --- Detect anomalies on historical close values ---
    anomalies_df = pd.DataFrame()
    anomaly_stats = {"n_anomalies": 0}
    if not history.empty and column in history.columns:
        anomalies_df = detect_anomalies_zscore(history, column=column)
        anomaly_stats = summarize_anomaly_stats(anomalies_df, f"{column}_anomaly")

    top_anomalies = []
    if not anomalies_df.empty and f"{column}_anomaly" in anomalies_df.columns:
        top_anomalies = anomalies_df[anomalies_df[f"{column}_anomaly"]].tail(5).to_dict(orient="records")

    # --- Get latest intraday snapshot ---
    latest_snapshot = intraday.iloc[-1].to_dict() if not intraday.empty else {}

    # --- Get last few daily rows for trend context ---
    historical_rows = []
    if not history.empty:
        historical_rows = history.tail(history_days).to_dict(orient="records")

    # --- Get forecast summary (next few days) ---
    forecast_summary = []
    if not forecasts.empty:
        forecast_summary = forecasts.tail(5).to_dict(orient="records")

    # --- Build combined LLM summary ---
    llm_result = summarize_analysis(
        ticker=ticker,
        latest_snapshot=latest_snapshot,
        anomaly_stats=anomaly_stats,
        top_anomalies=top_anomalies,
        historical_rows=historical_rows,
        forecast_rows=forecast_summary  # new param
    )

    # --- Log insights ---
    os.makedirs(DATA_DIR, exist_ok=True)
    log_entry = {
        "datetime": datetime.utcnow().isoformat(),
        "ticker": ticker,
        "n_anomalies": anomaly_stats.get("n_anomalies", 0),
        "engine": llm_result["engine"],
        "summary": llm_result["summary"].replace("\n", " "),
    }

    if os.path.exists(INSIGHTS_LOG_FILE):
        log_df = pd.read_csv(INSIGHTS_LOG_FILE)
        log_df = pd.concat([log_df, pd.DataFrame([log_entry])], ignore_index=True)
        log_df = log_df.tail(100)
    else:
        log_df = pd.DataFrame([log_entry])

    log_df.to_csv(INSIGHTS_LOG_FILE, index=False)

    return {
    "ticker": ticker,
    "timestamp_utc": datetime.utcnow().isoformat() + "Z",
    "data": {
        "intraday_latest": {
            "datetime_local": latest_snapshot.get("Datetime"),
            "close": round(float(latest_snapshot.get("close", 0)), 2),
            "predicted_close": round(float(latest_snapshot.get("predicted_close", 0)), 2),
            "RSI_14": round(float(latest_snapshot.get("RSI_14", 0)), 2),
            "MACD": round(float(latest_snapshot.get("MACD", 0)), 2),
            "volume": int(latest_snapshot.get("volume", 0))
        } if latest_snapshot else None,
        "historical_recent": [
            {
                "date": str(row.get("Date"))[:10],
                "close": round(float(row.get("Close", 0)), 2),
                "SMA_5": round(float(row.get("SMA_5", 0)), 2),
                "RSI_14": round(float(row.get("RSI_14", 0)), 2),
                "MACD": round(float(row.get("MACD", 0)), 2),
            } for row in historical_rows
        ],
        "forecast_upcoming": [
            {
                "date": str(row.get("Date"))[:10],
                "predicted": round(float(row.get("predictions", 0)), 2),
                "p05": round(float(row.get("0.05", 0)), 2),
                "p50": round(float(row.get("0.5", 0)), 2),
                "p95": round(float(row.get("0.95", 0)), 2),
            } for row in forecast_summary
        ]
    },
    "anomalies": {
        "count": anomaly_stats.get("n_anomalies", 0),
        "recent": top_anomalies
    },
    "insights": {
        "summary": llm_result["summary"],
        "engine": llm_result["engine"]
    }
}
