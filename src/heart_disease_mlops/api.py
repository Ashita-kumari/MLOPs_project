"""FastAPI inference service with request logging and Prometheus metrics."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict

from heart_disease_mlops.schema import FEATURES

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("heart_disease_api")
REQUEST_COUNT = Counter(
    "prediction_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram("prediction_request_duration_seconds", "HTTP request duration")

app = FastAPI(title="Heart Disease Prediction API", version="0.1.0")


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: int


@lru_cache(maxsize=1)
def load_model():
    model_path = Path(os.environ.get("MODEL_PATH", "models/heart_disease_pipeline.joblib"))
    logger.info("Loading serving model")
    if not model_path.is_file():
        logger.error("Serving model file is missing")
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    model = joblib.load(model_path)
    logger.info("Serving model loaded successfully")
    return model


@app.middleware("http")
async def record_request(request: Request, call_next):
    start = perf_counter()
    request_id = uuid4().hex
    request.state.request_id = request_id
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    except Exception as error:
        logger.error(
            "Request failed request_id=%s error_type=%s",
            request_id,
            type(error).__name__,
        )
        raise
    finally:
        duration = perf_counter() - start
        REQUEST_COUNT.labels(request.method, request.url.path, str(status_code)).inc()
        REQUEST_LATENCY.observe(duration)
        logger.info(
            "request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            status_code,
            duration * 1000,
        )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def homepage():
    return """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Heart Disease Prediction API</title>
    <style>
        :root { color-scheme: light; color: #202824; background: #f6f8f7; }
        * { box-sizing: border-box; }
        body { margin: 0; font: 18px/1.6 Georgia, serif; }
        main { max-width: 760px; margin: 48px auto; padding: 0 24px; }
        h1 { font-size: 32px; line-height: 1.25; overflow-wrap: anywhere; }
        nav ul { list-style: none; padding: 0; border-top: 1px solid #cbd5cf; }
        nav li { border-bottom: 1px solid #cbd5cf; }
        nav a { display: block; padding: 18px 12px; color: #12634c; }
        nav a:hover { background: #e5eee9; }
        nav a:focus-visible { outline: 3px solid #b75e1d; outline-offset: 2px; }
        footer { margin-top: 32px; color: #535d57; font-size: 15px; }
    </style>
</head>
<body>
    <main>
        <h1>Heart Disease Prediction API</h1>
        <nav aria-label="Service navigation">
            <ul>
                <li><a href="/docs#/default/predict_predict_post">Make a prediction</a></li>
                <li><a href="/docs">API documentation</a></li>
                <li><a href="/health">Model health</a></li>
                <li><a href="/metrics">Prometheus metrics</a></li>
            </ul>
        </nav>
        <footer>Educational model only. Not for diagnosis or treatment.</footer>
    </main>
</body>
</html>"""


@app.get("/health")
def health():
    try:
        load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.warning("Health check failed: serving model unavailable")
        raise HTTPException(status_code=503, detail="Model is not available") from error
    return {"status": "ok"}


@app.post("/predict")
def predict(patient: PatientFeatures):
    try:
        model = load_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.warning("Prediction rejected: serving model unavailable")
        raise HTTPException(status_code=503, detail="Model is not available") from error
    row = pd.DataFrame([patient.model_dump()], columns=FEATURES)
    prediction = int(model.predict(row)[0])
    probabilities = model.predict_proba(row)[0]
    class_probabilities = {
        str(int(label)): float(probability)
        for label, probability in zip(model.classes_, probabilities)
    }
    return {
        "prediction": prediction,
        "label": "disease_present" if prediction == 1 else "disease_not_present",
        "confidence": float(max(probabilities)),
        "probabilities": class_probabilities,
    }


@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)