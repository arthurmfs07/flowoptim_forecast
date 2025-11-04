import os
import pandas as pd
from chronos import BaseChronosPipeline, Chronos2Pipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_pipeline = None

def get_chronos_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = BaseChronosPipeline.from_pretrained(
            "s3://autogluon/chronos-2/",
            device_map="cuda"
        )
    return _pipeline

def forecast_nvda(prediction_length: int = 5+1, future_days: int = 5+1):
    """
    :param prediction_length: Number of days for backtest (last days in dataset)
    :param future_days: Number of true future days to forecast
    """
    from .data_fetcher import fetch_stock_data
    from .helpers import save_forecast

    # Fetch historical daily data
    df = fetch_stock_data("NVDA", period="3y", interval="1d", target="Close")
    id_column = "id"
    timestamp_column = "Date"
    target = "Close"
    timeseries_id = "1"

    series = df[df[id_column] == timeseries_id].copy()
    series[timestamp_column] = pd.to_datetime(series[timestamp_column])

    # -------------------------
    # BACKTEST: predict last `prediction_length` days
    # -------------------------
    data_context = series.iloc[:-prediction_length]
    data_test = series.iloc[-prediction_length:]
    data_future_backtest = data_test.drop(columns=[target])

    pipeline = get_chronos_pipeline()
    forecast_backtest = pipeline.predict_df(
        df=data_context,
        future_df=data_future_backtest,
        prediction_length=prediction_length,
        quantile_levels=[0.05, 0.5, 0.95],
        id_column=id_column,
        timestamp_column=timestamp_column,
        target=target
    )

    # -------------------------
    # TRUE FUTURE: predict next `future_days` business days
    # -------------------------
    last_date = series[timestamp_column].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=future_days, freq="B")
    data_future_real = pd.DataFrame({
        timestamp_column: future_dates,
        id_column: [timeseries_id] * future_days
    })

    forecast_future = pipeline.predict_df(
        df=series,
        future_df=data_future_real,
        prediction_length=future_days,
        quantile_levels=[0.05, 0.5, 0.95],
        id_column=id_column,
        timestamp_column=timestamp_column,
        target=target
    )

    # Save both forecasts
    save_forecast(forecast_backtest, file_path="data/forecasts_backtest.csv")
    save_forecast(forecast_future, file_path="data/forecasts_future.csv")
    data_context.to_csv("data/data_context.csv", index=False)
    data_test.to_csv("data/data_test.csv", index=False)
    df.to_csv("data/nvdastocks.csv", index=False)

    return forecast_backtest, forecast_future, df, data_future_backtest, data_context, data_test