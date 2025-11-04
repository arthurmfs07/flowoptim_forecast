import yfinance as yf
import pandas as pd
from datetime import datetime

def fetch_intraday_data(ticker="NVDA", interval="15m", period="30d"):
    """
    Fetches intraday data for the given ticker.
    """
    data = yf.download(tickers=ticker, period=period, interval=interval, progress=False, auto_adjust=True)
    data.reset_index(inplace=True)
    data["Datetime"] = pd.to_datetime(data["Datetime"])
    data["ticker"] = ticker
    print(f"[{datetime.now()}] Fetched intraday data for {ticker} ({len(data)} rows)")
    return data
