# /home/arthur/flowoptim/front/dashboard.py
# Run with: streamlit run dashboard.py
# http://localhost:8501

import os
import time
import requests
import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# ------------------------------
# Paths to datasets
# ------------------------------
DATA_DIR = Path("../data")  # relative to front/
INTRADAY_FILE = DATA_DIR / "intraday_latest.csv"
HISTORICAL_FILE = DATA_DIR / "nvdastocks.csv"
FORECAST_BACKTEST_FILE = DATA_DIR / "forecasts_backtest.csv"
FORECAST_FUTURE_FILE = DATA_DIR / "forecasts_future.csv"

# ------------------------------
# File-sync helpers
# ------------------------------
@st.cache_resource
def init_file_timestamps():
    """Initialize with current modification times of all CSV files."""
    files = [INTRADAY_FILE, HISTORICAL_FILE, FORECAST_BACKTEST_FILE, FORECAST_FUTURE_FILE]
    timestamps = {}
    for f in files:
        if f.exists():
            timestamps[f] = os.path.getmtime(f)
    return timestamps

def check_file_updates(last_timestamps):
    """Check if any CSV file has been updated since last check."""
    updated = False
    for f, old_time in list(last_timestamps.items()):
        if f.exists():
            new_time = os.path.getmtime(f)
            if new_time > old_time:
                updated = True
                last_timestamps[f] = new_time
    return updated

# ------------------------------
# Helper functions
# ------------------------------
def load_csv(path: Path):
    if path.exists():
        df = pd.read_csv(path)
        # Convert datetimes safely
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True)
        if "Datetime" in df.columns:
            df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
        return df
    return pd.DataFrame()

def plot_forecast_interactive(full_df, context_df, pred_df, test_df,
                              target="Close", timeseries_id="1",
                              id_column="id", timestamp_column="Date", title_suffix=""):

    # Convert dates safely
    for df in [full_df, context_df, pred_df, test_df]:
        if timestamp_column in df.columns:
            df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)

    # Filter by timeseries_id safely
    ts_id_str = str(timeseries_id)
    df_full = full_df[full_df[id_column].astype(str) == ts_id_str]
    df_context = context_df[context_df[id_column].astype(str) == ts_id_str]
    df_test = test_df[test_df[id_column].astype(str) == ts_id_str]

    # Filter forecast by target_name safely
    df_pred = pd.DataFrame()
    if not pred_df.empty:
        pred_df["target_name_clean"] = pred_df["target_name"].astype(str).str.strip().str.lower()
        target_clean = target.strip().lower()
        df_pred = pred_df[
            (pred_df[id_column].astype(str) == ts_id_str) &
            (pred_df["target_name_clean"] == target_clean)
        ].copy()

    if df_pred.empty:
        st.write("No forecast data available for this selection.")
        return

    # Ensure numeric and continuous timestamps
    for col in ["predictions", "0.05", "0.5", "0.95"]:
        if col in df_pred.columns:
            df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")

    # Sort and aggregate by timestamp (to fix multiple forecast windows)
    df_pred = (
        df_pred.groupby(timestamp_column, as_index=False)[["predictions", "0.05", "0.5", "0.95"]]
        .mean()
        .sort_values(timestamp_column)
        .dropna(subset=["predictions"])
    )

    # ------------------------------
    # Plotly interactive chart
    # ------------------------------
    fig = go.Figure()

    # Full history
    fig.add_trace(go.Scatter(
        x=df_full[timestamp_column],
        y=pd.to_numeric(df_full[target], errors="coerce"),
        mode="lines",
        name="Full history",
        line=dict(color="gray", width=1),
        opacity=0.4
    ))

    # Context (training)
    fig.add_trace(go.Scatter(
        x=df_context[timestamp_column],
        y=pd.to_numeric(df_context[target], errors="coerce"),
        mode="lines",
        name="Context (train)",
        line=dict(color="blue", width=2)
    ))

    # Ground truth (testing)
    if not df_test.empty:
        fig.add_trace(go.Scatter(
            x=df_test[timestamp_column],
            y=pd.to_numeric(df_test[target], errors="coerce"),
            mode="lines",
            name="Ground truth (test)",
            line=dict(color="green", width=2)
        ))

    # Forecast median (continuous)
    fig.add_trace(go.Scatter(
        x=df_pred[timestamp_column],
        y=df_pred["predictions"],
        mode="lines",
        name="Forecast (median)",
        line=dict(color="purple", width=2)
    ))

    # Prediction interval
    fig.add_trace(go.Scatter(
        x=pd.concat([df_pred[timestamp_column], df_pred[timestamp_column][::-1]]),
        y=pd.concat([df_pred["0.05"], df_pred["0.95"][::-1]]),
        fill="toself",
        fillcolor="rgba(238,130,238,0.3)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Prediction interval (5–95%)",
        hoverinfo="skip"
    ))

    # Vertical line for context end
    if not df_context.empty:
        context_end = df_context[timestamp_column].max()
        fig.add_vline(x=context_end, line_width=1, line_dash="dash", line_color="black")

    # Layout
    fig.update_layout(
        title=f"{target} forecast for series {timeseries_id} {title_suffix}",
        xaxis_title="Date",
        yaxis_title=target.capitalize(),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
        hovermode="x unified",
        template="plotly_white",
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# Streamlit page
# ------------------------------
st.set_page_config(page_title="Chronos Dashboard", layout="wide")
st.title("📈 Chronos Live Dashboard")

# ----------------------------------------
# Auto-refresh synced with backend updates
# ----------------------------------------
refresh_sec = st.sidebar.number_input("Auto-refresh interval (seconds)", value=60, min_value=10, step=10)
timeseries_id = st.sidebar.text_input("Timeseries ID", value="1")

# Initialize timestamps once
timestamps = init_file_timestamps()

# If files updated → force refresh immediately
if check_file_updates(timestamps):
    st.experimental_rerun()
else:
    st_autorefresh(interval=refresh_sec * 1000, key="auto_refresh")

# Optional: show last timestamps
with st.sidebar.expander("Last update times"):
    for f, t in timestamps.items():
        st.write(f"{f.name}: {time.ctime(t)}")

# ------------------------------
# Load datasets
# ------------------------------
intraday_df = load_csv(INTRADAY_FILE)
historical_df = load_csv(HISTORICAL_FILE)
forecast_backtest_df = load_csv(FORECAST_BACKTEST_FILE)
forecast_future_df = load_csv(FORECAST_FUTURE_FILE)

# ------------------------------
# Display sections
# ------------------------------
st.subheader("Latest Intraday Snapshot")
if not intraday_df.empty:
    st.dataframe(intraday_df.tail(1))
else:
    st.write("No intraday data available.")

st.subheader("Historical Data (tail)")
if not historical_df.empty:
    st.dataframe(historical_df.tail(10))
else:
    st.write("No historical data available.")

st.subheader("Forecasts: Backtest")
if not forecast_backtest_df.empty and not historical_df.empty:
    plot_forecast_interactive(
        historical_df, historical_df.tail(20),
        forecast_backtest_df, historical_df.tail(5),
        target="Close", timeseries_id=timeseries_id, title_suffix="(Backtest)"
    )
else:
    st.write("No backtest forecast available.")

st.subheader("Forecasts: Future")
if not forecast_future_df.empty and not historical_df.empty:
    plot_forecast_interactive(
        historical_df, historical_df.tail(20),
        forecast_future_df, historical_df.tail(5),
        target="Close", timeseries_id=timeseries_id, title_suffix="(Future)"
    )
else:
    st.write("No future forecast available.")

# ------------------------------
# Insights Section (Ollama/OpenAI/HF)
# ------------------------------
st.subheader("💬 What can your forecast assistant tell you today?")

ticker = st.sidebar.text_input("Ticker for insights", value="NVDA")
ANALYZE_URL = f"http://127.0.0.1:8000/analyze/?ticker={ticker}"

try:
    with st.spinner(f"Fetching insights for {ticker}..."):
        response = requests.get(ANALYZE_URL, timeout=60)

    if response.status_code == 200:
        result = response.json()
        insights = result.get("insights", {})
        summary = insights.get("summary", "")
        engine = insights.get("engine", "unknown")
        anomalies = result.get("anomalies", {})

        st.markdown(f"**Engine used:** `{engine}`")
        st.markdown(f"**Detected anomalies:** {anomalies.get('count', 0)}")

        if summary:
            st.markdown("#### Key Insights:")
            #st.markdown(summary)
            st.text(summary)
        else:
            st.info("No summary generated yet. Try refreshing or check backend logs.")

        recent_anomalies = anomalies.get("recent", [])
        if recent_anomalies:
            st.markdown("#### Recent Anomalies:")
            st.dataframe(pd.DataFrame(recent_anomalies))
    else:
        st.error(f"Failed to fetch insights (HTTP {response.status_code})")

except requests.exceptions.ConnectionError:
    st.warning("⚠️ Could not connect to FastAPI backend at http://127.0.0.1:8000. "
               "Make sure it's running: `uvicorn app.main:app --reload --port 8000`")
except Exception as e:
    st.error(f"Error fetching insights: {e}")
