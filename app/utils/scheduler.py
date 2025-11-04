from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from datetime import timedelta
import pandas as pd
import os
import logging

from app.utils.chronos_predictor import forecast_nvda
from app.utils.intraday_model import train_intraday_model, predict_today_final_close
from app.utils.build_intraday_training_data import build_training_data

# Paths
DATA_DIR = "data"
INTRADAY_CACHE_FILE = os.path.join(DATA_DIR, "intraday_latest.csv")
INTRADAY_FEATURES_FILE = os.path.join(DATA_DIR, "intraday_features_all.csv")
TRAIN_DATA_PATH = os.path.join(DATA_DIR, "intraday_training.csv")
TEMP_DATA_PATH = os.path.join(DATA_DIR, "intraday_training_temp.csv")

# Track last modified timestamp
last_train_timestamp = None

# ----------------------------------------------------------
# Setup logging clarity
# ----------------------------------------------------------
logging.getLogger("apscheduler.executors.default").setLevel(logging.ERROR)
logging.getLogger("apscheduler.scheduler").setLevel(logging.ERROR)

# ----------------------------------------------------------
# Helper for logging with timestamps
# ----------------------------------------------------------
def log(msg: str):
    print(f"[{datetime.now()}] {msg}")

# ----------------------------------------------------------
# 1️⃣ Chronos forecast
# ----------------------------------------------------------
def chronos_job():
    """Run Chronos forecasts."""
    try:
        log("🧭 Starting scheduled Chronos forecast job...")
        forecast_backtest, forecast_future, *_ = forecast_nvda()
        log(
            f"✅ Chronos forecasts saved "
            f"({len(forecast_backtest)} backtest rows, {len(forecast_future)} future rows)."
        )
    except Exception as e:
        log(f"❌ Error during Chronos forecast: {e}")

# ----------------------------------------------------------
# 2️⃣ Rebuild or extend training data if new days exist
# ----------------------------------------------------------
def check_and_update_training_data(ticker="NVDA", days=3):
    """
    Build temporary training data and merge if new dates exist.
    Returns True if new data was added.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(TRAIN_DATA_PATH):
        log("⚠️ No training data found. Building from scratch...")
        build_training_data(ticker=ticker, days=60)
        return True

    old_df = pd.read_csv(TRAIN_DATA_PATH)
    old_dates = set(old_df["date"].astype(str))

    log("🔍 Checking for new intraday data...")
    build_training_data(ticker=ticker, days=days)

    if not os.path.exists(TRAIN_DATA_PATH):
        log("⚠️ Failed to build training data for comparison.")
        return False

    new_df = pd.read_csv(TRAIN_DATA_PATH)
    new_dates = set(new_df["date"].astype(str))

    unseen_dates = sorted(list(new_dates - old_dates))
    if unseen_dates:
        log(f"🆕 Found new data for {len(unseen_dates)} days: {unseen_dates}")
        combined = pd.concat([old_df, new_df], ignore_index=True).drop_duplicates(subset=["Datetime"])
        combined.to_csv(TRAIN_DATA_PATH, index=False)
        log("✅ Training data updated and saved.")
        return True
    else:
        log("ℹ️ No new dates found. Skipping data update.")
        return False

# ----------------------------------------------------------
# 3️⃣ Retrain model if data updated
# ----------------------------------------------------------
def intraday_retrain_if_new_data():
    """Retrain intraday model if new training data is available."""
    global last_train_timestamp

    if not os.path.exists(TRAIN_DATA_PATH):
        log("⚠️ Training data not found. Skipping retraining.")
        return

    current_timestamp = os.path.getmtime(TRAIN_DATA_PATH)
    if last_train_timestamp is None or current_timestamp > last_train_timestamp:
        log("🧠 New training data detected. Retraining intraday model...")
        try:
            train_intraday_model()
            last_train_timestamp = current_timestamp
            log("✅ Intraday model retrained successfully.")
        except Exception as e:
            log(f"❌ Error during intraday retraining: {e}")
    else:
        log("⏳ No new training data. Skipping retraining.")

# ----------------------------------------------------------
# 4️⃣ Intraday prediction (XGBoost)
# ----------------------------------------------------------
def intraday_predict_job(ticker="NVDA"):
    """Run intraday prediction and cache results."""
    try:
        log("🕒 Running intraday XGBoost prediction...")
        features = predict_today_final_close(ticker)

        # Save latest prediction
        latest = features.iloc[-1].copy()
        os.makedirs(os.path.dirname(INTRADAY_CACHE_FILE), exist_ok=True)
        latest.to_frame().T.to_csv(INTRADAY_CACHE_FILE, index=False)

        # Save all features
        features.to_csv(INTRADAY_FEATURES_FILE, index=False)
        log("✅ Cached latest prediction and all historical features.")
    except Exception as e:
        log(f"❌ Error during intraday prediction: {e}")

# ----------------------------------------------------------
# 5️⃣ Master scheduler setup
# ----------------------------------------------------------
def start_scheduler(interval_minutes: int = 5):
    scheduler = BackgroundScheduler()

    job_defaults = {
        "misfire_grace_time": 60,  # tolerate up to 60s delay
        "coalesce": True,           # merge missed runs
        "max_instances": 1          # avoid overlap
    }

    # ----------------------------------------------------------
    # Ensure training data exists on startup
    # ----------------------------------------------------------
    os.makedirs(DATA_DIR, exist_ok=True)
    log("🧱 Ensuring training data is available before starting scheduler...")
    check_and_update_training_data(ticker="NVDA", days=60)

    # ----------------------------------------------------------
    # Chronos forecast job
    # ----------------------------------------------------------
    scheduler.add_job(
        chronos_job,
        trigger="interval",
        minutes=interval_minutes,
        #start_date = datetime.now() + timedelta(seconds = 30),
        next_run_time=datetime.now(),
        **job_defaults
    )

    # Daily training data rebuild (after market close)
    scheduler.add_job(
        check_and_update_training_data,
        trigger="cron",
        hour=1,
        minute=0,
        id="rebuild_training_data",
        **job_defaults
    )

    # Retrain model if new data exists
    scheduler.add_job(
        intraday_retrain_if_new_data,
        trigger="interval",
        minutes=interval_minutes,
        next_run_time=datetime.now(),
        **job_defaults
    )

    # Predict intraday price
    scheduler.add_job(
        intraday_predict_job,
        trigger="interval",
        minutes=interval_minutes,
        next_run_time=datetime.now(),
        **job_defaults
    )

    scheduler.start()
    log(f"⏰ Scheduler started. Forecasts & intraday model updated every {interval_minutes} minutes.")
    return scheduler
