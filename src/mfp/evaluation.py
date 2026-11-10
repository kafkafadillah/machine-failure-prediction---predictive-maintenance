"""Metrik, interval kepercayaan bootstrap, optimasi threshold berbasis biaya,
dan uji statistik perbandingan model.

Catatan metodologi:
* Accuracy menyesatkan pada data 3,4% positif (model "selalu normal" = 96,6%).
  Metrik utama: PR-AUC (average precision); pendukung: ROC-AUC, recall,
  precision, F2 (recall lebih penting), MCC, Brier score (kalibrasi).
* Threshold TIDAK dipilih di test set. Threshold dipilih dari prediksi
  out-of-fold (OOF) pada data train, lalu dikunci sebelum test dievaluasi.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, fbeta_score, matthews_corrcoef, precision_score,
    recall_score, roc_auc_score,
)


def threshold_metrics(y: np.ndarray, proba: np.ndarray, threshold: float,
                      c_fn: float = 0.0, c_fp: float = 0.0) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": fbeta_score(y, pred, beta=1, zero_division=0),
        "f2": fbeta_score(y, pred, beta=2, zero_division=0),
        "mcc": matthews_corrcoef(y, pred) if len(np.unique(pred)) > 1 else 0.0,
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        # Setiap alarm (TP maupun FP) memicu inspeksi berbiaya c_fp;
        # setiap kegagalan yang lolos (FN) menimbulkan biaya kerusakan c_fn.
        "expected_cost": float(fn * c_fn + (tp + fp) * c_fp),
    }


def full_metrics(y: np.ndarray, proba: np.ndarray, threshold: float,
                 c_fn: float = 0.0, c_fp: float = 0.0) -> dict[str, float]:
    out = {
        "pr_auc": average_precision_score(y, proba),
        "roc_auc": roc_auc_score(y, proba),
        "brier": brier_score_loss(y, proba),
        "threshold": threshold,
    }
    out.update(threshold_metrics(y, proba, threshold, c_fn, c_fp))
    return out


def bootstrap_ci(y: np.ndarray, proba: np.ndarray, threshold: float,
                 n_boot: int = 1000, level: float = 0.95, seed: int = 42,
                 metrics=("pr_auc", "roc_auc", "recall", "precision", "f2", "mcc")
                 ) -> pd.DataFrame:
    """Stratified bootstrap: resampling positif & negatif terpisah supaya
    setiap sampel bootstrap tetap memiliki kasus kegagalan."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    records = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = full_metrics(y[idx], proba[idx], threshold)
        records.append({k: m[k] for k in metrics})
    boot = pd.DataFrame(records)
    point = full_metrics(y, proba, threshold)
    a = (1 - level) / 2
    return pd.DataFrame({
        "estimate": [point[k] for k in metrics],
        "ci_low": boot.quantile(a).values,
        "ci_high": boot.quantile(1 - a).values,
    }, index=list(metrics))


def cost_curve(y: np.ndarray, proba: np.ndarray, c_fn: float, c_fp: float,
               grid: np.ndarray | None = None) -> pd.DataFrame:
    if grid is None:  # lebih rapat di area probabilitas rendah (kelas langka)
        grid = np.unique(np.round(np.concatenate(
            [np.arange(0.005, 0.20, 0.005), np.arange(0.20, 0.99, 0.01)]), 4))
    rows = []
    for t in grid:
        m = threshold_metrics(y, proba, t, c_fn, c_fp)
        rows.append({"threshold": t, **m})
    return pd.DataFrame(rows)


def select_threshold(y: np.ndarray, proba: np.ndarray, c_fn: float, c_fp: float,
                     min_recall: float = 0.0) -> tuple[float, pd.DataFrame]:
    """Threshold dengan biaya total minimum, dengan syarat recall >= min_recall.
    Bila tidak ada threshold yang memenuhi syarat, kembali ke biaya minimum."""
    curve = cost_curve(y, proba, c_fn, c_fp)
    feasible = curve[curve["recall"] >= min_recall]
    pool = feasible if len(feasible) else curve
    best = pool.loc[pool["expected_cost"].idxmin(), "threshold"]
    return float(best), curve


def corrected_resampled_ttest(a: np.ndarray, b: np.ndarray, n_train: int,
                              n_test: int) -> tuple[float, float]:
    """Corrected resampled t-test (Nadeau & Bengio, 2003).

    Skor fold CV tidak independen (training set saling tumpang tindih),
    sehingga paired t-test biasa terlalu optimistis. Koreksi menambah
    varians dengan faktor n_test/n_train.
    """
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = d.mean() / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return float(t), float(p)
