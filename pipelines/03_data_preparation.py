"""CRISP-DM Tahap 3 — Data Preparation.

Keputusan (didokumentasikan di docs/03_data_preparation.md):
  * Kolom ID (UDI, Product ID) dibuang: tidak ada makna prediktif; huruf
    pertama Product ID identik 100% dengan `type` (redundan).
  * Kolom mode kegagalan (TWF..RNF) dibuang dari fitur: TARGET LEAKAGE.
    Disimpan terpisah untuk model diagnosis & error analysis.
  * Label anomali (9 failure tanpa mode, 18 RNF tanpa failure) TIDAK diubah:
    label `Machine failure` adalah ground truth yang dicatat operator;
    mengubahnya = memanipulasi target. Dampaknya dianalisis di tahap evaluasi.
  * Tidak ada imputasi/outlier removal: data lengkap & nilai ekstrem justru
    sinyal kegagalan (mis. torsi tinggi), bukan noise.
  * Feature engineering & scaling TIDAK dilakukan di sini, melainkan di dalam
    sklearn Pipeline (di-fit per fold) untuk mencegah leakage.
  * Split stratified 80/20 disimpan ke disk; test set dikunci.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from _bootstrap import banner

from mfp.config import load_config
from mfp.data import file_md5, load_raw, split_data, validate_features

cfg = load_config()
banner("Data preparation")
raw_path = cfg.path("raw_data")
df = load_raw(raw_path)
validate_features(df, require_target=True)
print("Validasi skema: OK")

keep = (cfg["data"]["id_columns"] + cfg["features"]["categorical"]
        + cfg["features"]["numeric_raw"] + [cfg["data"]["target"]]
        + cfg["data"]["failure_modes"])
df = df[keep]

train, test = split_data(df, cfg)
out = cfg.path("processed_dir")
train.to_csv(out / "train.csv", index=False)
test.to_csv(out / "test.csv", index=False)

target = cfg["data"]["target"]
manifest = {
    "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "raw_file": str(raw_path.relative_to(raw_path.parents[2])),
    "raw_md5": file_md5(raw_path),
    "random_state": cfg.seed,
    "test_size": cfg["data"]["test_size"],
    "n_train": len(train), "n_test": len(test),
    "failures_train": int(train[target].sum()), "failures_test": int(test[target].sum()),
    "failure_rate_train": round(float(train[target].mean()), 5),
    "failure_rate_test": round(float(test[target].mean()), 5),
    "feature_columns": cfg["features"]["categorical"] + cfg["features"]["numeric_raw"],
    "excluded_leakage_columns": cfg["data"]["failure_modes"],
}
(out / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
