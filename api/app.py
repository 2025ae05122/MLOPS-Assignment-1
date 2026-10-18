import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)

from api.schemas import (
    HealthResponse,
    HeartDiseaseRequest,
    PredictionResponse,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger("heart-disease-api")


MODEL_PATH = Path(
    os.getenv(
        "MODEL_PATH",
        "artifacts/heart_disease_pipeline.joblib",
    )
)

model = None


REQUEST_COUNT = Counter(
    "heart_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "heart_api_predictions_total",
    "Total model predictions",
    ["prediction"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "named_steps"):
        raise RuntimeError(
            "Loaded model is not a sklearn pipeline."
        )

    required_steps = {"preprocessor", "model"}

    if not required_steps.issubset(
        model.named_steps.keys()
    ):
        raise RuntimeError(
            "Pipeline must contain preprocessor and model steps."
        )

    logger.info(
        "Model loaded successfully from %s",
        MODEL_PATH,
    )

    yield

    logger.info("API shutting down.")


app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0.0",
    description=(
        "Educational heart-disease prediction API. "
        "Not a medical diagnostic service."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def request_logging(
    request: Request,
    call_next,
):
    start_time = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    except Exception:
        logger.exception(
            "Unhandled API error: %s %s",
            request.method,
            request.url.path,
        )
        raise

    finally:
        duration = time.perf_counter() - start_time

        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        logger.info(
            "method=%s path=%s status=%s duration=%.6f",
            request.method,
            request.url.path,
            status_code,
            duration,
        )


@app.get("/")
def root():
    return {
        "service": "Heart Disease Prediction API",
        "status": "running",
        "documentation": "/docs",
        "health": "/health",
        "metrics": "/metrics",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    loaded = model is not None

    return HealthResponse(
        status="healthy" if loaded else "unhealthy",
        model_loaded=loaded,
        model_path=str(MODEL_PATH),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(payload: HeartDiseaseRequest):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    input_frame = pd.DataFrame(
        [payload.model_dump()]
    )

    try:
        prediction = int(
            model.predict(input_frame)[0]
        )

        confidence = float(
            model.predict_proba(input_frame)[0, 1]
        )

    except Exception as exc:
        logger.exception("Prediction failed.")

        raise HTTPException(
            status_code=500,
            detail="Model prediction failed.",
        ) from exc

    PREDICTION_COUNT.labels(
        prediction=str(prediction)
    ).inc()

    logger.info(
        "prediction=%s confidence=%.6f",
        prediction,
        confidence,
    )

    return PredictionResponse(
        prediction=prediction,
        prediction_label=(
            "Disease" if prediction == 1
            else "No disease"
        ),
        confidence=confidence,
    )


@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )