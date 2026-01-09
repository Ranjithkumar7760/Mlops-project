"""
FastAPI Prediction Service for SMS Spam Detection
MLOps-ready API with:
- MLflow Model Registry support
- Prometheus metrics for monitoring
- Kubernetes compatibility
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
import joblib
import os
import time

# ============================
# PROMETHEUS METRICS
# ============================
from prometheus_client import Counter, Histogram, start_http_server

REQUEST_COUNT = Counter(
    "spam_prediction_requests_total",
    "Total number of prediction requests"
)

REQUEST_LATENCY = Histogram(
    "spam_prediction_latency_seconds",
    "Latency for spam prediction requests"
)

# ============================
# MLFLOW IMPORTS
# ============================
import mlflow
import mlflow.pyfunc

# ============================
# FASTAPI APP
# ============================
app = FastAPI(
    title="SMS Spam Detection API",
    description="AI-powered text classification service for spam detection",
    version="1.0.0"
)

# ============================
# STATIC FILES
# ============================
app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================
# GLOBAL OBJECTS
# ============================
model = None
vectorizer = None

# ============================
# ENVIRONMENT VARIABLES
# ============================
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "SpamClassifier")
MLFLOW_MODEL_STAGE = os.getenv("MLFLOW_MODEL_STAGE", "Production")

MODEL_DIR = "model"

# ============================
# LOAD MODEL FUNCTION
# ============================
def load_model():
    global model, vectorizer

    # Attempt MLflow Model Registry loading
    if MLFLOW_TRACKING_URI:
        try:
            print("Loading model from MLflow Model Registry...")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{MLFLOW_MODEL_STAGE}"
            mlflow_model = mlflow.pyfunc.load_model(model_uri)

            model = mlflow_model
            vectorizer = None  # MLflow model includes preprocessing

            print("Model loaded successfully from MLflow Registry")
            return
        except Exception as e:
            print(f"MLflow model load failed: {e}")
            print("Falling back to local model files...")

    # Fallback: Load local model
    try:
        print("Loading local model and vectorizer...")
        model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        print("Local model and vectorizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load local model files: {e}")
        model = None
        vectorizer = None


# ============================
# STARTUP EVENTS
# ============================
@app.on_event("startup")
def startup_event():
    print("Starting Spam Detection API...")
    load_model()

    # Start Prometheus metrics server
    start_http_server(8001)
    print("Prometheus metrics exposed on port 8001")


# ============================
# REQUEST / RESPONSE SCHEMAS
# ============================
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text message to classify")

    class Config:
        schema_extra = {
            "example": {
                "text": "Congratulations! You've won a $1000 gift card. Click here!"
            }
        }


class PredictionResponse(BaseModel):
    prediction: Literal["Spam", "Not Spam"]
    confidence: float


# ============================
# HEALTH CHECK
# ============================
@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "healthy",
        "model_source": "mlflow" if vectorizer is None else "local",
        "service": "SMS Spam Detection API"
    }


# ============================
# PREDICTION ENDPOINT
# ============================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        # Case 1: MLflow model (pipeline included)
        if vectorizer is None:
            prediction = model.predict([text])[0]
            confidence = 0.95  # MLflow pyfunc does not expose proba easily

        # Case 2: Local model
        else:
            text_tfidf = vectorizer.transform([text])
            prediction_proba = model.predict_proba(text_tfidf)[0]
            prediction = model.predict(text_tfidf)[0]
            confidence = float(prediction_proba[prediction])

        result = "Spam" if prediction == 1 else "Not Spam"

        return PredictionResponse(
            prediction=result,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


# ============================
# UI & INFO ENDPOINTS
# ============================
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/api")
async def api_info():
    return {
        "message": "SMS Spam Detection API",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
            "metrics": ":8001/metrics"
        }
    }


# ============================
# LOCAL RUN SUPPORT
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
