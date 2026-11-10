"""CRISP-DM Tahap 6 — Deployment.

  6.1  Packaging: pipeline + threshold + model diagnosis + metadata
       -> models/model_bundle.joblib  (hanya jika lolos deployment gate)
  6.2  Smoke test: muat ulang bundle, prediksi contoh, cek konsistensi
  6.3  Batch scoring contoh -> reports/tables/06_batch_predictions.csv
  6.4  Monitoring: data referensi + simulasi drift (PSI)

Serving:   uvicorn app.api:app --port 8000      (REST API)
           python app/predict_cli.py --input data.csv --output pred.csv
"""
from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone

import _bootstrap  # noqa: F401
from _bootstrap import banner, to_markdown

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn

from mfp import __version__
from mfp.config import load_config
from mfp.data import load_split, xy
from mfp.inference import FEATURE_COLUMNS, ModelBundle
from mfp.monitoring import drift_report
from mfp.viz import INK, STATUS, save, set_style, subtitle

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true", help="paketkan walau gate gagal (tidak disarankan)")
args = ap.parse_args()

cfg = load_config()
set_style()
FIG, TAB, MODELS = cfg.path("figures_dir"), cfg.path("tables_dir"), cfg.path("models_dir")
summary = json.loads((MODELS / "training_summary.json").read_text())
evaluation = json.loads((MODELS / "evaluation_report.json").read_text())
manifest = json.loads((cfg.path("processed_dir") / "data_manifest.json").read_text())

# =========================================================================== 6.1
banner("6.1 Packaging model bundle")
if not evaluation["approved"] and not args.force:
    raise SystemExit("Deployment gate GAGAL — lihat models/evaluation_report.json")

bundle = ModelBundle(
    pipeline=joblib.load(MODELS / "final_pipeline.joblib"),
    threshold=float(summary["threshold"]),
    risk_bands=dict(cfg["risk_bands"]),
    mode_model=joblib.load(MODELS / "mode_model.joblib"),
    mode_labels=list(summary["mode_labels"]),
    metadata={
        "model_name": cfg["project"]["name"],
        "model_version": cfg["project"]["version"],
        "package_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": summary["final_model_label"],
        "calibration": summary["calibration"],
        "threshold": summary["threshold"],
        "feature_columns": FEATURE_COLUMNS,
        "training_data_md5": manifest["raw_md5"],
        "n_train": manifest["n_train"],
        "test_metrics": {k: evaluation["test_metrics"][k]
                         for k in ("pr_auc", "roc_auc", "recall", "precision", "f2", "brier")},
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    },
)
path = bundle.save(MODELS / "model_bundle.joblib")
print(f"Tersimpan: {path} ({path.stat().st_size / 1e6:.1f} MB)")
(MODELS / "model_metadata.json").write_text(json.dumps(bundle.metadata, indent=2))

# =========================================================================== 6.2
banner("6.2 Smoke test")
train, test = load_split(cfg)
loaded = ModelBundle.load(path)
X_te, y_te = xy(test, cfg)
p_direct = loaded.pipeline.predict_proba(X_te)[:, 1]
pred = loaded.predict(test)
assert np.allclose(pred["failure_probability"], p_direct.round(4)), "inkonsistensi prediksi"
assert set(pred["risk_level"]) <= {"LOW", "MEDIUM", "HIGH"}
assert (pred.loc[pred.predicted_failure == 1, "risk_level"] != "LOW").all(), \
    "unit yang di-flag tidak boleh berisiko LOW"
print("Smoke test OK — bundle konsisten dengan pipeline yang dievaluasi.")

# =========================================================================== 6.3
banner("6.3 Contoh batch scoring")
out = test[["udi", "product_id", *FEATURE_COLUMNS, "machine_failure"]].join(pred)
out.sort_values("failure_probability", ascending=False).to_csv(
    TAB / "06_batch_predictions.csv", index=False)
print(out["risk_level"].value_counts().to_string())
print("\nTop-5 unit berisiko tertinggi:")
print(out.sort_values("failure_probability", ascending=False)
      [["udi", "type", "failure_probability", "risk_level", "likely_failure_mode",
        "machine_failure"]].head().to_string(index=False))
rl = out.groupby("risk_level")["machine_failure"].agg(["count", "sum", "mean"]) \
        .rename(columns={"count": "n_unit", "sum": "n_failure", "mean": "failure_rate"}) \
        .reindex(["LOW", "MEDIUM", "HIGH"])
print(to_markdown(rl, TAB / "06_risk_band_validation", index=True))

# =========================================================================== 6.4
banner("6.4 Monitoring drift (PSI)")
num, cat = cfg["features"]["numeric_raw"], cfg["features"]["categorical"]
train[FEATURE_COLUMNS].to_csv(MODELS / "reference_data.csv", index=False)

rng = np.random.default_rng(cfg.seed)
drifted = test.copy()   # skenario: musim panas + jadwal ganti tool terlambat
drifted["air_temp_k"] = drifted["air_temp_k"] + 2.5
drifted["process_temp_k"] = drifted["process_temp_k"] + 2.0
drifted["tool_wear_min"] = np.clip(drifted["tool_wear_min"] + rng.normal(40, 10, len(drifted)), 0, 300)

rep_ok = drift_report(train, test, num, cat)
rep_drift = drift_report(train, drifted, num, cat)
to_markdown(rep_ok, TAB / "06_drift_train_vs_test")
print(to_markdown(rep_drift, TAB / "06_drift_simulated"))
print(f"Rata-rata P(failure): test={pred.failure_probability.mean():.4f} | "
      f"drifted={loaded.predict(drifted).failure_probability.mean():.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
color_of = {"stabil": STATUS["good"], "moderat": STATUS["warning"], "signifikan": STATUS["critical"]}
feat_order = rep_drift["feature"].tolist()[::-1]
for ax, rep, title in ((axes[0], rep_ok, "Train vs test (tanpa drift)"),
                       (axes[1], rep_drift, "Train vs batch simulasi drift")):
    r = rep.set_index("feature").loc[feat_order]
    ax.barh(r.index, r["psi"], color=[color_of[s] for s in r["status"]], height=0.55)
    for i, (v, s) in enumerate(zip(r["psi"], r["status"])):
        ax.text(v + 0.02, i, f"{v:.3f} · {s}", va="center", fontsize=8.5,
                bbox=dict(facecolor=INK["surface"], edgecolor="none", pad=1.5))
    for x in (0.10, 0.25):
        ax.axvline(x, color=INK["muted"], ls="--", lw=1, zorder=0)
    ax.set_xlim(0, max(0.4, rep_drift["psi"].max() * 1.35))
    ax.set_xlabel("Population Stability Index"); ax.set_title(title, fontsize=11)
    ax.grid(axis="y", visible=False)
subtitle(axes[0], "PSI < 0,10 stabil · 0,10–0,25 moderat · > 0,25 retrain")
fig.tight_layout()
save(fig, FIG / "06_drift_monitoring.png")
print("\nDeployment artefak siap di", MODELS)
