"""Monitoring data drift dengan Population Stability Index (PSI).

PSI = sum((p_new - p_ref) * ln(p_new / p_ref)) per bin (bin dari kuantil data
referensi/training).  Aturan praktis industri:
    PSI < 0.10  -> stabil
    0.10-0.25   -> pergeseran moderat, investigasi
    > 0.25      -> pergeseran signifikan, pertimbangkan retraining
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10,
        eps: float = 1e-4) -> float:
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    p_ref = np.histogram(reference, edges)[0] / len(reference)
    p_cur = np.histogram(current, edges)[0] / len(current)
    p_ref, p_cur = np.clip(p_ref, eps, None), np.clip(p_cur, eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def categorical_psi(reference: pd.Series, current: pd.Series, eps: float = 1e-4) -> float:
    cats = sorted(set(reference.astype(str)) | set(current.astype(str)))
    p_ref = reference.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    p_cur = current.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
    p_ref, p_cur = np.clip(p_ref.values, eps, None), np.clip(p_cur.values, eps, None)
    return float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 numeric: list[str], categorical: list[str]) -> pd.DataFrame:
    rows = []
    for c in numeric:
        rows.append((c, psi(reference[c], current[c])))
    for c in categorical:
        rows.append((c, categorical_psi(reference[c], current[c])))
    rep = pd.DataFrame(rows, columns=["feature", "psi"])
    rep["status"] = pd.cut(rep["psi"], [-np.inf, 0.10, 0.25, np.inf],
                           labels=["stabil", "moderat", "signifikan"])
    return rep.sort_values("psi", ascending=False).reset_index(drop=True)
