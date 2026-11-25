"""FastAPI service for the trained model.

    uvicorn mlpipe.app:app --app-dir src --port 8000
"""
from __future__ import annotations

import json
import time
from collections import Counter
from contextlib import asynccontextmanager

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from mlpipe import config
from mlpipe.drift import drift_report

STATE: dict = {}
STATS = Counter()
LATENCY_MS: list[float] = []


def load_artifacts():
    STATE["model"] = joblib.load(config.MODEL_PATH)
    STATE["meta"] = json.loads(config.METADATA_PATH.read_text())
    STATE["reference"] = np.load(config.REFERENCE_PATH)["X"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE.clear()


app = FastAPI(title="Wine classifier", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    instances: list[list[float]] = Field(..., min_length=1, max_length=1000,
                                         description="Rows of 13 numeric features, in metadata order")

    @field_validator("instances")
    @classmethod
    def check_width(cls, rows):
        for i, r in enumerate(rows):
            if len(r) != 13:
                raise ValueError(f"row {i} has {len(r)} features, expected 13")
            if not all(np.isfinite(r)):
                raise ValueError(f"row {i} contains NaN or infinity")
        return rows


class Prediction(BaseModel):
    label: int
    class_name: str
    probabilities: dict[str, float]


class PredictResponse(BaseModel):
    model_version: str
    predictions: list[Prediction]


@app.middleware("http")
async def track(request: Request, call_next):
    t = time.perf_counter()
    response = await call_next(request)
    if request.url.path == "/predict":
        LATENCY_MS.append((time.perf_counter() - t) * 1000)
        del LATENCY_MS[:-5000]
        STATS[f"predict_status_{response.status_code}"] += 1
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in STATE}


@app.get("/model-info")
def model_info():
    return STATE["meta"]


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    X = np.asarray(req.instances, dtype=float)
    proba = STATE["model"].predict_proba(X)
    names = STATE["meta"]["class_names"]
    STATS["rows_predicted"] += len(X)
    return PredictResponse(
        model_version=STATE["meta"]["model_version"],
        predictions=[Prediction(label=int(p.argmax()), class_name=names[int(p.argmax())],
                                probabilities={n: round(float(v), 4) for n, v in zip(names, p, strict=True)})
                     for p in proba])


@app.post("/drift")
def drift(req: PredictRequest):
    X = np.asarray(req.instances, dtype=float)
    if len(X) < 30:
        raise HTTPException(422, "send at least 30 rows for a drift check")
    return drift_report(STATE["reference"], X, STATE["meta"]["feature_names"])


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = [f"mlpipe_{k}_total {v}" for k, v in sorted(STATS.items())]
    if LATENCY_MS:
        lat = np.asarray(LATENCY_MS)
        lines += [f'mlpipe_predict_latency_ms{{quantile="{q}"}} {np.quantile(lat, q):.3f}'
                  for q in (0.5, 0.95, 0.99)]
    return "\n".join(lines) + "\n"
