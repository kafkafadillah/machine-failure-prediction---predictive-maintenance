"""Data loading, schema validation, quality audit, dan train/test split.

Prinsip:
* Nama kolom asli (mis. ``Torque [Nm]``) dinormalisasi ke snake_case karena
  karakter ``[ ] <`` ditolak beberapa library (XGBoost).
* Validasi skema eksplisit -> pipeline gagal cepat (fail fast) bila data baru
  menyimpang dari kontrak, bukan diam-diam menghasilkan prediksi keliru.
* Split dilakukan SEKALI, stratified, dan disimpan ke disk agar semua tahap
  memakai test set yang sama dan test set tidak pernah "diintip".
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import Config

COLUMN_MAP = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temp_k",
    "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rot_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
}

# Kontrak data: rentang fisik yang masuk akal (bukan sekadar min/max sampel).
SCHEMA: dict[str, tuple[float, float]] = {
    "air_temp_k": (280.0, 330.0),
    "process_temp_k": (280.0, 340.0),
    "rot_speed_rpm": (500.0, 4000.0),
    "torque_nm": (0.0, 120.0),
    "tool_wear_min": (0.0, 400.0),
}
VALID_TYPES = {"L", "M", "H"}


class SchemaError(ValueError):
    """Data tidak memenuhi kontrak skema."""


def file_md5(path: str | Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(path: str | Path) -> pd.DataFrame:
    """Baca CSV mentah dan normalisasi nama kolom."""
    df = pd.read_csv(path)
    df = df.rename(columns=COLUMN_MAP)
    df["type"] = df["type"].astype(str)
    return df


def validate_features(df: pd.DataFrame, require_target: bool = False,
                      target: str = "machine_failure") -> pd.DataFrame:
    """Validasi kolom fitur. Dipakai saat training DAN inference."""
    required = ["type", *SCHEMA]
    if require_target:
        required.append(target)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SchemaError(f"Kolom wajib tidak ada: {missing}")

    problems: list[str] = []
    if df[required].isna().any().any():
        cols = df[required].columns[df[required].isna().any()].tolist()
        problems.append(f"nilai kosong pada {cols}")
    bad_types = set(df["type"].astype(str).unique()) - VALID_TYPES
    if bad_types:
        problems.append(f"type tidak dikenal {sorted(bad_types)}")
    for col, (lo, hi) in SCHEMA.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        n_bad = int(((vals < lo) | (vals > hi) | vals.isna()).sum())
        if n_bad:
            problems.append(f"{col}: {n_bad} nilai di luar [{lo}, {hi}]")
    if require_target and not set(df[target].unique()) <= {0, 1}:
        problems.append("target harus biner 0/1")
    if problems:
        raise SchemaError("; ".join(problems))
    return df


@dataclass
class QualityReport:
    n_rows: int
    n_duplicates: int
    n_missing: int
    failure_rate: float
    failure_without_mode: list[int] = field(default_factory=list)
    mode_without_failure: dict[str, int] = field(default_factory=dict)
    product_prefix_matches_type: float = 1.0

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("Jumlah baris", self.n_rows),
            ("Baris duplikat", self.n_duplicates),
            ("Sel kosong", self.n_missing),
            ("Failure rate", f"{self.failure_rate:.2%}"),
            ("Failure=1 tanpa mode (UDI)", len(self.failure_without_mode)),
            ("Mode=1 tetapi failure=0", sum(self.mode_without_failure.values())),
            ("Prefix Product ID == Type", f"{self.product_prefix_matches_type:.0%}"),
        ]
        return pd.DataFrame(rows, columns=["Pemeriksaan", "Hasil"])


def audit_quality(df: pd.DataFrame, cfg: Config) -> QualityReport:
    target = cfg["data"]["target"]
    modes = cfg["data"]["failure_modes"]
    any_mode = df[modes].max(axis=1)
    return QualityReport(
        n_rows=len(df),
        n_duplicates=int(df.drop(columns=cfg["data"]["id_columns"]).duplicated().sum()),
        n_missing=int(df.isna().sum().sum()),
        failure_rate=float(df[target].mean()),
        failure_without_mode=df.loc[(df[target] == 1) & (any_mode == 0), "udi"].tolist(),
        mode_without_failure={
            m: int(((df[m] == 1) & (df[target] == 0)).sum()) for m in modes
            if ((df[m] == 1) & (df[target] == 0)).any()
        },
        product_prefix_matches_type=float((df["product_id"].str[0] == df["type"]).mean()),
    )


def split_data(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified hold-out split. Stratifikasi pada target agar proporsi
    kegagalan (3,4%) identik di train dan test."""
    target = cfg["data"]["target"]
    train, test = train_test_split(
        df, test_size=cfg["data"]["test_size"], stratify=df[target],
        random_state=cfg.seed,
    )
    return train.sort_values("udi").reset_index(drop=True), \
        test.sort_values("udi").reset_index(drop=True)


def load_split(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = cfg.path("processed_dir")
    train_p, test_p = d / "train.csv", d / "test.csv"
    if not train_p.exists():
        raise FileNotFoundError(
            "Split belum dibuat. Jalankan: python pipelines/03_data_preparation.py")
    return pd.read_csv(train_p), pd.read_csv(test_p)


def xy(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, np.ndarray]:
    """Pisahkan fitur mentah (tanpa ID & tanpa kolom leakage) dan target."""
    num = cfg["features"]["numeric_raw"]
    X = df[cfg["features"]["categorical"] + num].copy()
    X[num] = X[num].astype(float)          # dtype konsisten training == serving
    X["type"] = X["type"].astype(str)
    return X, df[cfg["data"]["target"]].to_numpy()
