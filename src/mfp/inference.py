"""Inference: satu artefak (`ModelBundle`) berisi pipeline, threshold
keputusan, model diagnosis mode kegagalan, dan metadata — sehingga yang
di-deploy persis sama dengan yang dievaluasi."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .data import validate_features

FEATURE_COLUMNS = ["type", "air_temp_k", "process_temp_k", "rot_speed_rpm",
                   "torque_nm", "tool_wear_min"]

ACTIONS = {
    "LOW": "Operasi normal — lanjutkan monitoring rutin.",
    "MEDIUM": "Jadwalkan inspeksi pada jendela maintenance berikutnya.",
    "HIGH": "Inspeksi segera / pertimbangkan menghentikan unit.",
}


@dataclass
class ModelBundle:
    pipeline: Any
    threshold: float
    risk_bands: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    mode_model: Any | None = None          # multi-label: TWF/HDF/PWF/OSF
    mode_labels: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)
        return path

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        obj = joblib.load(path)
        if not isinstance(obj, ModelBundle):
            raise TypeError(f"{path} bukan ModelBundle")
        return obj

    # ------------------------------------------------------------- predict
    def risk_level(self, proba: np.ndarray) -> np.ndarray:
        low, high = self.risk_bands["low"], self.risk_bands["high"]
        return np.where(proba >= max(high, self.threshold), "HIGH",
                        np.where(proba >= min(low, self.threshold), "MEDIUM", "LOW"))

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = validate_features(df.copy())[FEATURE_COLUMNS]
        X["type"] = X["type"].astype(str)
        num = FEATURE_COLUMNS[1:]
        X[num] = X[num].astype(float)
        proba = self.pipeline.predict_proba(X)[:, 1]
        out = pd.DataFrame(index=df.index)
        out["failure_probability"] = proba.round(4)
        out["predicted_failure"] = (proba >= self.threshold).astype(int)
        out["risk_level"] = self.risk_level(proba)
        out["recommended_action"] = out["risk_level"].map(ACTIONS)
        if self.mode_model is not None:
            mode_proba = np.column_stack(
                [p[:, 1] for p in self.mode_model.predict_proba(X)])
            for i, m in enumerate(self.mode_labels):
                out[f"p_{m}"] = mode_proba[:, i].round(4)
            out["likely_failure_mode"] = np.where(
                out["predicted_failure"] == 1,
                np.asarray(self.mode_labels)[mode_proba.argmax(axis=1)], "-")
        return out
