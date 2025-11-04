from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import forecast, analyze, intraday_forecast
from app.utils.scheduler import start_scheduler

app = FastAPI(title="Forecast Assistant", version='0.1')

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=["*"]
)

# Routers
app.include_router(forecast.router, prefix="/forecast", tags=['Forecast'])
app.include_router(analyze.router, prefix="/analyze", tags=['Analyze'])
#app.include_router(ask.router, prefix="/ask", tags=['Ask'])
app.include_router(intraday_forecast.router, prefix="/intraday", tags=['Intraday'])

@app.on_event("startup")
def startup_event():
    start_scheduler()

@app.get("/")
def root():
    return {"message": "Welcome to Forecast Assistant API"}
