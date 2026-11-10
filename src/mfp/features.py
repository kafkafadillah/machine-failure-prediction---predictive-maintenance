"""Feature engineering berbasis fisika (domain-driven).

Mengapa fitur fisika? Mekanisme kegagalan mesin bergantung pada KOMBINASI
sensor, bukan sensor tunggal:

* Heat dissipation failure (HDF)  -> selisih suhu proses-udara kecil
                                     DAN putaran rendah.
* Power failure (PWF)             -> daya mekanik P = T * omega terlalu
                                     rendah atau terlalu tinggi.
* Overstrain failure (OSF)        -> keausan tool * torsi melewati batas
                                     yang bergantung pada varian produk.
* Tool wear failure (TWF)         -> keausan tool mendekati akhir umur pakai.

Semua transformasi dibungkus sebagai sklearn Transformer sehingga identik
saat training, cross-validation, dan inference (tidak ada training-serving skew).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

DEFAULT_OVERSTRAIN = {"L": 11000.0, "M": 12000.0, "H": 13000.0}

ENGINEERED = [
    "temp_diff_k",        # process - air: kapasitas pembuangan panas
    "power_w",            # torque * angular velocity
    "strain_minnm",       # tool wear * torque
    "strain_ratio",       # strain / batas overstrain varian produk
    "torque_per_krpm",    # beban relatif terhadap kecepatan
    "wear_fraction",      # umur tool yang sudah terpakai (asumsi umur 240 min)
]


class PhysicsFeatures(BaseEstimator, TransformerMixin):
    """Menambah fitur turunan fisika. Stateless (tidak belajar dari data),
    sehingga aman dari leakage di dalam cross-validation."""

    def __init__(self, enabled: bool = True, overstrain_limits: dict | None = None,
                 tool_life_min: float = 240.0):
        self.enabled = enabled
        self.overstrain_limits = overstrain_limits
        self.tool_life_min = tool_life_min

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        if not self.enabled:
            return X
        limits = self.overstrain_limits or DEFAULT_OVERSTRAIN
        omega = X["rot_speed_rpm"] * 2 * np.pi / 60.0  # rad/s
        X["temp_diff_k"] = X["process_temp_k"] - X["air_temp_k"]
        X["power_w"] = X["torque_nm"] * omega
        X["strain_minnm"] = X["tool_wear_min"] * X["torque_nm"]
        X["strain_ratio"] = X["strain_minnm"] / X["type"].astype(str).map(limits).astype(float)
        X["torque_per_krpm"] = X["torque_nm"] / (X["rot_speed_rpm"] / 1000.0)
        X["wear_fraction"] = X["tool_wear_min"] / self.tool_life_min
        return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names_in_)
        return np.asarray(base + (ENGINEERED if self.enabled else []), dtype=object)


def build_preprocessor(numeric: list[str], categorical: list[str],
                       splines: bool = False) -> ColumnTransformer:
    """ColumnTransformer: one-hot untuk `type`, scaling (atau spline basis
    untuk model linear agar bisa menangkap hubungan non-linear seperti
    'daya terlalu rendah ATAU terlalu tinggi')."""
    if splines:
        num_tf = SplineTransformer(n_knots=6, degree=3, include_bias=False)
    else:
        num_tf = StandardScaler()
    return ColumnTransformer(
        [
            ("num", num_tf, numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        verbose_feature_names_out=False,
    )


def feature_lists(cfg, engineered: bool = True) -> tuple[list[str], list[str]]:
    numeric = list(cfg["features"]["numeric_raw"]) + (ENGINEERED if engineered else [])
    return numeric, list(cfg["features"]["categorical"])
