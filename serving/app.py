import os
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from loguru import logger
from prometheus_client import Counter
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from serving.predict import load_model
from serving.predict import predict as run_predict

DB_PATH = os.getenv("PREDICTION_LOG_DB", "monitoring/predictions.db")

# Per-class prediction distribution to check model drift
PREDICTIONS_BY_LABEL = Counter(
    "predictions_by_label_total",
    "Count of predictions grouped by predicted emotion label",
    ["label"],
)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            text_hash TEXT,
            label TEXT,
            label_id INTEGER,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(text: str, label: str, label_id: int, latency_ms: float):
    import hashlib
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (ts, text_hash, label, label_id, latency_ms) VALUES (?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), text_hash, label, label_id, latency_ms),
    )
    conn.commit()
    conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(title="Sentiment Analysis API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    label_id: int
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    start = time.perf_counter()
    result = run_predict(req.text)
    latency_ms = (time.perf_counter() - start) * 1000

    log_prediction(req.text, result["label"], result["label_id"], latency_ms)
    PREDICTIONS_BY_LABEL.labels(label=result["label"]).inc()
    logger.info(f"predict | label={result['label']} latency={latency_ms:.1f}ms")

    return PredictResponse(latency_ms=round(latency_ms, 2), **result)
