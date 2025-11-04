from fastapi import APIRouter
from app.utils.chronos_predictor import forecast_nvda
from app.utils.helpers import (
    load_latest_forecast_backtest,
    load_latest_forecast_future
)

router = APIRouter()


@router.get("/")
def get_forecast():
    """
    Run a fresh NVDA forecast using Chronos-2.
    Returns both backtest and future predictions.
    """
    forecast_backtest, forecast_future, *_ = forecast_nvda()

    result = {
        "backtest": {
            "rows": len(forecast_backtest),
            "columns": forecast_backtest.columns.tolist(),
            "last_forecast_date": str(forecast_backtest.iloc[-1]['Date'].date()),
            "forecast_sample": forecast_backtest.tail(5).to_dict(orient="records")
        },
        "future": {
            "rows": len(forecast_future),
            "columns": forecast_future.columns.tolist(),
            "last_forecast_date": str(forecast_future.iloc[-1]['Date'].date()),
            "forecast_sample": forecast_future.tail(5).to_dict(orient="records")
        }
    }

    return result


@router.get("/latest")
def get_latest_forecasts(n: int = 5):
    """
    Retrieve the latest saved forecasts from disk
    (both backtest and future versions).
    """
    latest_backtest = load_latest_forecast_backtest(n)
    latest_future = load_latest_forecast_future(n)

    return {
        "latest_backtest": latest_backtest,
        "latest_future": latest_future
    }





'''from fastapi import APIRouter, Query
from app.utils.chronos_predictor import forecast_nvda

# Router with no extra prefix
router = APIRouter()

@router.get("/")
def get_forecast():
    """
    Return NVDA forecast using Chronos-2.
    """
    # chronos_predictor function
    forecast_df = forecast_nvda()[0]

    # Return a concise JSON response
    result = {
        "rows": len(forecast_df),
        "columns": forecast_df.columns.tolist(),
        "last_forecast_date": str(forecast_df.iloc[-1]['Date'].date()),
        "forecast_sample": forecast_df.tail(5).to_dict(orient="records")
    }
    return result


@router.get("/latest")
def get_latest_forecasts():
    from app.utils.helpers import load_latest_forecast
    return {"latest": load_latest_forecast(5)}
'''