"""
app.py
======
FastAPI application exposing Person 3's ML pipeline as an HTTP API for the
n8n workflow to call.

Endpoints
---------
POST /predict   -> run Isolation Forest + LSTM + SHAP, return risk assessment
GET  /health    -> liveness / model-loaded check

Run locally with:
    uvicorn app:app --host 0.0.0.0 --port 8000

Models are loaded ONCE at process startup (see predict.get_bundle(), called
below) and reused for every request — they are never retrained inside a
request handler.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import predict
import preprocessing as pp

app = FastAPI(
    title="Railway Point Machine - Predictive Maintenance ML API",
    description=(
        "Person 3's ML system: Isolation Forest anomaly detection, LSTM "
        "degradation trend prediction, and SHAP explainability, combined "
        "into a single risk assessment for the n8n automation layer."
    ),
    version="1.0.0",
)

# --- CORS -------------------------------------------------------------------
# Open CORS for demo/testing convenience (n8n, browser-based testing, Colab).
# NOTE: this is intentionally permissive for the prototype. For a real
# deployment, replace allow_origins=["*"] with the specific origin(s) that
# are allowed to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- Request / response schemas ---------------------------------------------
class SensorReading(BaseModel):
    motor_current_A: float = Field(..., description="Motor current draw, amps")
    vibration_g: float = Field(..., description="Vibration amplitude, g")
    temperature_C: float = Field(..., description="Motor/mechanism temperature, Celsius")

    @field_validator("motor_current_A", "vibration_g", "temperature_C")
    @classmethod
    def must_be_finite(cls, v: float) -> float:
        if v is None or not isinstance(v, (int, float)):
            raise ValueError("sensor values must be numeric")
        return float(v)


class PredictRequest(BaseModel):
    asset_id: str = Field(..., min_length=1, description="Unique point machine identifier, e.g. PM-001")
    timestamp: str = Field(..., description="ISO-8601 timestamp of the latest reading")
    sequence: List[SensorReading] = Field(
        ...,
        description=(
            f"Historical sensor readings, OLDEST first, NEWEST last. "
            f"Must contain at least {pp.SEQUENCE_LENGTH} readings — the LSTM "
            f"consumes a fixed window of {pp.SEQUENCE_LENGTH} readings. If more "
            f"than {pp.SEQUENCE_LENGTH} are sent, only the most recent "
            f"{pp.SEQUENCE_LENGTH} are used."
        ),
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        try:
            # Accept standard ISO-8601, including a trailing 'Z'.
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception as exc:
            raise ValueError(f"timestamp must be a valid ISO-8601 string, got: {v!r}") from exc
        return v

    @field_validator("sequence")
    @classmethod
    def validate_sequence_length(cls, v: List[SensorReading]) -> List[SensorReading]:
        if len(v) < pp.SEQUENCE_LENGTH:
            raise ValueError(
                f"sequence must contain at least {pp.SEQUENCE_LENGTH} readings, got {len(v)}"
            )
        return v


class ShapContribution(BaseModel):
    motor_current_A: float
    vibration_g: float
    temperature_C: float


class PredictResponse(BaseModel):
    asset_id: str
    timestamp: str
    anomaly_score: float = Field(..., description="0 (normal) - 1 (highly anomalous), from Isolation Forest")
    is_anomaly: bool = Field(..., description="anomaly_score > 0.6 (demo threshold, see README)")
    trend_risk: float = Field(..., description="0 (healthy) - 1 (fully degraded), from LSTM")
    risk_level: str = Field(..., description="LOW / MEDIUM / HIGH — see README for threshold derivation")
    forecast_hours: Optional[float] = Field(
        None, description="Demo heuristic: estimated hours until trend_risk reaches 1.0"
    )
    shap: ShapContribution = Field(
        ..., description="Per-feature SHAP contribution to anomaly_score (see explainability.py)"
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


# --- Startup: load models once ----------------------------------------------
@app.on_event("startup")
def load_models_on_startup():
    """
    Load every model artifact once, at process startup, so the first
    request isn't slow and so we fail fast (clear error at boot) if
    `train.py` hasn't been run yet.
    """
    predict.get_bundle()


# --- Error handling -----------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Catch-all so unexpected errors still return clean JSON instead of an
    # HTML traceback page (helpful for n8n, which expects JSON).
    return JSONResponse(
        status_code=500,
        content={"error": "internal_server_error", "detail": str(exc)},
    )


# --- Endpoints ----------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health():
    try:
        predict.get_bundle()
        model_loaded = True
    except Exception:
        model_loaded = False
    return {"status": "ok" if model_loaded else "model_not_loaded", "model_loaded": model_loaded}


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    try:
        result = predict.run_prediction(
            asset_id=payload.asset_id,
            timestamp=payload.timestamp,
            sequence=[r.model_dump() for r in payload.sequence],
        )
    except FileNotFoundError as exc:
        # Models not trained yet.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"prediction failed: {exc}") from exc

    return result
