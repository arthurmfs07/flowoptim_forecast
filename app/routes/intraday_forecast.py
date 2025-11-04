from fastapi import APIRouter
from fastapi.responses import JSONResponse
import pandas as pd
import os
from app.utils.intraday_model import predict_today_final_close

router = APIRouter()
INTRADAY_CACHE_FILE = "data/intraday_latest.csv"  # same path used by scheduler


@router.get("/predict_close/{ticker}")
def predict_close(ticker: str, show_features: bool = True):
    """
    Return latest cached intraday prediction for a given ticker.
    Uses the cache file written by the scheduler (fast and safe).
    """
    try:
        if not os.path.exists(INTRADAY_CACHE_FILE):
            return JSONResponse(
                content={"error": "No intraday prediction available yet. Try again later."},
                status_code=404
            )

        latest_df = pd.read_csv(INTRADAY_CACHE_FILE)
        if latest_df.empty:
            return JSONResponse(
                content={"error": "Intraday cache file is empty."},
                status_code=500
            )

        latest = latest_df.iloc[-1]
        response = {
            "ticker": ticker,
            "predicted_close": float(latest["predicted_close"])
        }

        if show_features:
            features_dict = {
                k: (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in latest.items()
                if k not in ["predicted_close"]
            }
            response["features"] = features_dict

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@router.get("/intraday_features/{ticker}")
def get_intraday_features(ticker: str, n_rows: int = 10):
    """
    Return the latest intraday features (before prediction) used by the model.
    """
    try:
        features = predict_today_final_close(ticker)
        recent_features = features.tail(n_rows).copy()
        response = {
            "ticker": ticker,
            "features": recent_features.fillna(0).to_dict(orient="records")
        }
        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)



@router.post("/rebuild_training/{ticker}")
def rebuild_training(ticker: str = "NVDA", days: int = 60):
    """
    Manually rebuild the intraday training dataset and save it under data/intraday_training.csv.
    This triggers data fetching + feature building for the last N days.
    """
    from app.utils.build_intraday_training_data import build_training_data
    import traceback

    try:
        build_training_data(ticker=ticker, days=days)
        file_path = "data/intraday_training.csv"

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            info = {
                "ticker": ticker,
                "rows": len(df),
                "days": df["date"].nunique() if "date" in df.columns else None,
                "path": file_path,
            }
            return JSONResponse(
                content={
                    "message": "✅ Training data rebuilt successfully.",
                    "info": info,
                },
                status_code=200,
            )
        else:
            return JSONResponse(
                content={
                    "error": "Training file not found after rebuild attempt."
                },
                status_code=500,
            )
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            content={"error": f"Failed to rebuild training data: {str(e)}"},
            status_code=500,
        )



