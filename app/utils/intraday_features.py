import pandas as pd
import numpy as np

def build_intraday_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Builds intraday features from 15m OHLCV data for a single day,
    using only short-horizon features that can be computed without multi-day history.
    Includes extra features inspired by Chronos fetcher.
    """

    datetime_col = next((c for c in ["Datetime", "datetime", "index"] if c in data.columns), None)
    if datetime_col is None:
        raise ValueError("No datetime-like column found in data.")
    
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    if data[datetime_col].dt.tz is None:
        data[datetime_col] = data[datetime_col].dt.tz_localize("UTC")
    data[datetime_col] = data[datetime_col].dt.tz_convert("America/New_York")

    close = data["Close"].squeeze()
    high = data["High"].squeeze()
    low = data["Low"].squeeze()
    volume = data["Volume"].squeeze()

    returns = close.pct_change()
    data["returns"] = returns
    data["rolling_volatility"] = returns.rolling(3, min_periods=1).std()
    data["rolling_mean"] = close.rolling(3, min_periods=1).mean()
    data["rolling_momentum"] = close - close.shift(3)

    data["SMA_5"] = close.rolling(5, min_periods=1).mean()
    data["SMA_20"] = close.rolling(20, min_periods=1).mean()
    data["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    data["EMA_26"] = close.ewm(span=26, adjust=False).mean()
    data["MACD"] = data["EMA_12"] - data["EMA_26"]

    std_20 = close.rolling(20, min_periods=1).std()
    data["Bollinger_upper"] = data["SMA_20"] + 2 * std_20
    data["Bollinger_lower"] = data["SMA_20"] - 2 * std_20

    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["ATR_14"] = tr.rolling(14, min_periods=1).mean()

    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data["RSI_14"] = 100 - (100 / (1 + rs))

    data["Volume_SMA_5"] = volume.rolling(5, min_periods=1).mean()
    data["Volume_SMA_20"] = volume.rolling(20, min_periods=1).mean()
    data["Volume_change"] = volume.pct_change()


    latest = data.iloc[-1]
    dt_value = latest[datetime_col]
    if isinstance(dt_value, pd.Series):
        dt_value = dt_value.iloc[0]
    dt = pd.to_datetime(dt_value)
    hour_float = float(dt.hour) + float(dt.minute) / 60.0

    def scalar(x):
        return x.iloc[0] if isinstance(x, pd.Series) else x

    snapshot = {
        "Datetime": dt,
        "hour": hour_float,
        "close": float(scalar(latest["Close"])),
        "open": float(scalar(latest["Open"])),
        "high": float(scalar(latest["High"])),
        "low": float(scalar(latest["Low"])),
        "volume": float(scalar(latest["Volume"])),
        "rolling_volatility": float(scalar(latest["rolling_volatility"])),
        "rolling_mean": float(scalar(latest["rolling_mean"])),
        "rolling_momentum": float(scalar(latest["rolling_momentum"])),
        "daily_return": float(scalar(latest["returns"])),
        "SMA_5": float(scalar(latest["SMA_5"])),
        "SMA_20": float(scalar(latest["SMA_20"])),
        "EMA_12": float(scalar(latest["EMA_12"])),
        "EMA_26": float(scalar(latest["EMA_26"])),
        "MACD": float(scalar(latest["MACD"])),
        "Bollinger_upper": float(scalar(latest["Bollinger_upper"])),
        "Bollinger_lower": float(scalar(latest["Bollinger_lower"])),
        "ATR_14": float(scalar(latest["ATR_14"])),
        "RSI_14": float(scalar(latest["RSI_14"])),
        "Volume_SMA_5": float(scalar(latest["Volume_SMA_5"])),
        "Volume_SMA_20": float(scalar(latest["Volume_SMA_20"])),
        "Volume_change": float(scalar(latest["Volume_change"]))
    }

    return pd.DataFrame([snapshot])
