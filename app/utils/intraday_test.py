import joblib

model = joblib.load("artifacts/intraday_xgb_model.joblib")
#print(model)

from intraday_model import predict_today_final_close

features = predict_today_final_close("NVDA")
print(features[["hour", "predicted_close"]].tail())


