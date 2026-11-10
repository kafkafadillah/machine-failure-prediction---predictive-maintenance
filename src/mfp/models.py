"""Model zoo, pipeline builder, dan ruang pencarian hyperparameter.

Setiap model dibungkus dalam Pipeline end-to-end:
    PhysicsFeatures -> ColumnTransformer -> Estimator
sehingga seluruh preprocessing ikut di-fit ulang di tiap fold CV
(mencegah leakage) dan artefak yang di-deploy adalah satu objek tunggal.

XGBoost bersifat opsional: bila paket tidak terpasang, model `xgboost`
dilewati dan `hist_gb` (HistGradientBoosting milik scikit-learn — algoritma
gradient boosting berbasis histogram yang serupa) tetap tersedia.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import PhysicsFeatures, build_preprocessor, feature_lists

try:  # pragma: no cover - tergantung environment
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:  # pragma: no cover
    XGBClassifier = None
    HAS_XGB = False

MODEL_LABELS = {
    "dummy": "Dummy (prior)",
    "logreg": "Logistic Regression",
    "logreg_spline": "Logistic Regression + Splines (GAM-like)",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "hist_gb": "Histogram Gradient Boosting",
}


def available_models(include_dummy: bool = True) -> list[str]:
    names = ["dummy"] if include_dummy else []
    names += ["logreg", "logreg_spline", "random_forest", "hist_gb"]
    if HAS_XGB:
        names.append("xgboost")
    else:
        warnings.warn("xgboost tidak terpasang -> model 'xgboost' dilewati "
                      "(pip install xgboost untuk mengaktifkan).", stacklevel=2)
    return names


def _estimator(name: str, seed: int, pos_weight: float):
    if name == "dummy":
        return DummyClassifier(strategy="prior")
    if name in ("logreg", "logreg_spline"):
        return LogisticRegression(class_weight="balanced", max_iter=5000, C=1.0)
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)
    if name == "hist_gb":
        return HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
            l2_regularization=1.0, class_weight="balanced", random_state=seed)
    if name == "xgboost":
        if not HAS_XGB:
            raise ImportError("xgboost belum terpasang: pip install xgboost")
        return XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=np.sqrt(pos_weight),
            eval_metric="aucpr", tree_method="hist", n_jobs=-1, random_state=seed)
    raise KeyError(f"Model tidak dikenal: {name}")


def build_pipeline(name: str, cfg, engineered: bool = True,
                   pos_weight: float = 28.5) -> Pipeline:
    numeric, categorical = feature_lists(cfg, engineered)
    return Pipeline([
        ("physics", PhysicsFeatures(enabled=engineered,
                                    overstrain_limits=cfg["features"]["overstrain_limits"])),
        ("prep", build_preprocessor(numeric, categorical,
                                    splines=(name == "logreg_spline"))),
        ("model", _estimator(name, cfg.seed, pos_weight)),
    ])


def search_space(name: str, pos_weight: float) -> dict:
    """Ruang pencarian RandomizedSearchCV (prefix `model__`)."""
    spaces = {
        "logreg": {"model__C": loguniform(1e-3, 1e2)},
        "logreg_spline": {"model__C": loguniform(1e-3, 1e2),
                          "prep__num__n_knots": randint(4, 10)},
        "random_forest": {
            "model__n_estimators": [300, 500],
            "model__max_depth": [None, 8, 12, 16, 24],
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", 0.3, 0.5, 0.7],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "hist_gb": {
            "model__learning_rate": loguniform(0.02, 0.2),
            "model__max_iter": randint(200, 800),
            "model__max_leaf_nodes": randint(8, 64),
            "model__min_samples_leaf": randint(10, 60),
            "model__l2_regularization": loguniform(1e-3, 10),
            "model__class_weight": [None, "balanced"],
        },
        "xgboost": {
            "model__n_estimators": randint(200, 800),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__max_depth": randint(3, 8),
            "model__subsample": uniform(0.6, 0.4),
            "model__colsample_bytree": uniform(0.5, 0.5),
            "model__min_child_weight": randint(1, 10),
            "model__reg_lambda": loguniform(1e-2, 10),
            "model__scale_pos_weight": [1.0, float(np.sqrt(pos_weight)), float(pos_weight)],
        },
    }
    return spaces.get(name, {})
