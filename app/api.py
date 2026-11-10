"""REST API untuk scoring risiko kegagalan mesin.

Jalankan:
    uvicorn app.api:app --host 0.0.0.0 --port 8000
Dokumentasi interaktif (Swagger): http://localhost:8000/docs

Contoh:
    curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
      -d '{"type":"L","air_temp_k":302.4,"process_temp_k":310.9,
           "rot_speed_rpm":1320,"torque_nm":62.1,"tool_wear_min":205}'
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from mfp.data import SchemaError  # noqa: E402
from mfp.inference import ModelBundle  # noqa: E402

MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "model_bundle.joblib"))


class MachineReading(BaseModel):
    """Satu snapshot sensor mesin. Rentang = kontrak data (sama dengan mfp.data.SCHEMA)."""
    type: Literal["L", "M", "H"] = Field(description="Varian kualitas produk")
    air_temp_k: float = Field(ge=280, le=330, examples=[300.1])
    process_temp_k: float = Field(ge=280, le=340, examples=[310.1])
    rot_speed_rpm: float = Field(ge=500, le=4000, examples=[1500])
    torque_nm: float = Field(ge=0, le=120, examples=[40.0])
    tool_wear_min: float = Field(ge=0, le=400, examples=[108])
    unit_id: str | None = Field(default=None, description="ID unit (opsional, dikembalikan apa adanya)")


class Prediction(BaseModel):
    unit_id: str | None
    failure_probability: float
    predicted_failure: bool
    risk_level: Literal["LOW", "MEDIUM", "HIGH"]
    recommended_action: str
    likely_failure_mode: str | None = None
    mode_probabilities: dict[str, float] | None = None


@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model tidak ditemukan di {MODEL_PATH}. Jalankan `make all`.")
    return ModelBundle.load(MODEL_PATH)


app = FastAPI(title="Machine Failure Prediction API", version="1.0.0",
              description="Predictive maintenance — probabilitas kegagalan, level risiko, "
                          "dan dugaan mode kegagalan.")


def _score(readings: list[MachineReading]) -> list[Prediction]:
    bundle = get_bundle()
    df = pd.DataFrame([r.model_dump() for r in readings])
    try:
        res = bundle.predict(df)
    except SchemaError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    out = []
    for r, (_, row) in zip(readings, res.iterrows()):
        modes = {m: float(row[f"p_{m}"]) for m in bundle.mode_labels} if bundle.mode_labels else None
        lm = row.get("likely_failure_mode")
        out.append(Prediction(
            unit_id=r.unit_id,
            failure_probability=float(row["failure_probability"]),
            predicted_failure=bool(row["predicted_failure"]),
            risk_level=row["risk_level"],
            recommended_action=row["recommended_action"],
            likely_failure_mode=None if lm in (None, "-") else lm,
            mode_probabilities=modes,
        ))
    return out


@app.get("/health")
def health() -> dict:
    b = get_bundle()
    return {"status": "ok", "model_version": b.metadata.get("model_version"),
            "algorithm": b.metadata.get("algorithm"), "threshold": b.threshold}


@app.get("/model-info")
def model_info() -> dict:
    return get_bundle().metadata


@app.post("/predict", response_model=Prediction)
def predict(reading: MachineReading) -> Prediction:
    return _score([reading])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(readings: list[MachineReading]) -> list[Prediction]:
    if not readings:
        raise HTTPException(status_code=422, detail="Batch kosong")
    if len(readings) > 10_000:
        raise HTTPException(status_code=413, detail="Maksimum 10.000 baris per request")
    return _score(readings)
