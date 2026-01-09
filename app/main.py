"""
FastAPI Prediction Service for SMS Spam Detection
Production-ready API:
- Loads model from container (immutable image)
- Prometheus metrics enabled
- Kubernetes compatible
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
    "Total number of spam prediction requests"
)

REQUEST_LATENCY = Histogram(
    "spam_prediction_latency_seconds",
    "Latency for spam prediction requests"
)

# ============================
# FASTAPI APP
# ============================
app = FastAPI(
    title="SMS Spam Detection API",
    description="AI-powered SMS spam classification service",
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

MODEL_DIR = "model"

# ============================
# LOAD MODEL
# ============================
def load_model():
    global model, vectorizer
    try:
        print("📦 Loading model from local container files...")
        model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "vectorizer.pkl"))
        print("✅ Model and vectorizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        model = None
        vectorizer = None


# ============================
# STARTUP EVENT
# ============================
@app.on_event("startup")
def startup_event():
    print("🚀 Starting Spam Detection API...")
    load_model()

    # Prometheus metrics
    start_http_server(8001)
    print("📊 Prometheus metrics available on port 8001")


# ============================
# REQUEST / RESPONSE MODELS
# ============================
class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="SMS message text")


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
        "model_source": "local-container",
        "service": "SMS Spam Detection API"
    }


# ============================
# PREDICTION ENDPOINT
# ============================
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: TextRequest):
    if model is None or vectorizer is None:
        raise HTTPException(status_code=503, detail="Model not available")

    REQUEST_COUNT.inc()
    start_time = time.time()

    try:
        text = request.text.strip()
        if not text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")

        text_tfidf = vectorizer.transform([text])
        proba = model.predict_proba(text_tfidf)[0]
        prediction = model.predict(text_tfidf)[0]

        confidence = float(proba[prediction])
        result = "Spam" if prediction == 1 else "Not Spam"

        return PredictionResponse(
            prediction=result,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)


# ============================
# UI
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
# LOCAL RUN
# ============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
