import yfinance as yf
import pandas as pd
from datetime import datetime
import numpy as np

def fetch_stock_data(
    ticker: str = "NVDA",
    period: str = "3y",
    interval: str = "1d",
    target: str = "Close"
) -> pd.DataFrame:
    # --- Fetch ---
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval).reset_index()

    # --- Chronos formatting ---
    data.insert(0, "id", ["1"] * data.shape[0])
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.set_index("Date").asfreq("B").reset_index()
    data = data.ffill()

    # --- Price features ---
    data["SMA_5"] = data["Close"].rolling(5).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["EMA_12"] = data["Close"].ewm(span=12, adjust=False).mean()
    data["EMA_26"] = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]
    data["Bollinger_upper"] = data["SMA_20"] + 2 * data["Close"].rolling(20).std()
    data["Bollinger_lower"] = data["SMA_20"] - 2 * data["Close"].rolling(20).std()
    data["Daily_return"] = data["Close"].pct_change()
    data["Momentum_5"] = data["Close"] - data["Close"].shift(5)

    # --- Volatility ---
    data["Volatility_20"] = data["Daily_return"].rolling(20).std()
    high_low = data["High"] - data["Low"]
    high_close = np.abs(data["High"] - data["Close"].shift(1))
    low_close = np.abs(data["Low"] - data["Close"].shift(1))
    data["ATR_14"] = pd.DataFrame([high_low, high_close, low_close]).max().rolling(14).mean()

    # --- RSI 14 ---
    delta = data["Close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    # --- Volume features ---
    data["Volume_SMA_5"] = data["Volume"].rolling(5).mean()
    data["Volume_SMA_20"] = data["Volume"].rolling(20).mean()
    data["Volume_change"] = data["Volume"].pct_change()

    # --- Cleanup ---
    data = data.ffill().bfill()
    print(f"[{datetime.now()}] Updated {ticker} data with indicators ({len(data)} rows).")
    return data