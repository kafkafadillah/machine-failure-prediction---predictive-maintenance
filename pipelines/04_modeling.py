"""CRISP-DM Tahap 4 — Modeling.

Desain eksperimen (SEMUA hanya memakai data train; test set tidak disentuh):

  4.1  Perbandingan model x set fitur (raw vs +physics) dengan
       Repeated Stratified K-Fold (5x3) + corrected resampled t-test.
  4.2  Hyperparameter tuning (RandomizedSearchCV, metrik PR-AUC).
  4.3  Prediksi out-of-fold (OOF) model tuned + soft-voting ensemble
       -> pemilihan model final berdasarkan OOF PR-AUC.
  4.4  Kalibrasi probabilitas (isotonic vs sigmoid vs none) via OOF Brier.
  4.5  Pemilihan threshold berbasis biaya bisnis pada OOF.
  4.6  Learning curve model final.
  4.7  Model sekunder: diagnosis mode kegagalan (multi-label).
  4.8  Fit final pada seluruh data train -> models/.

Pakai `--fast` untuk iterasi cepat (CV & search lebih kecil).
"""
from __future__ import annotations

import argparse
import json
import time

import _bootstrap  # noqa: F401
from _bootstrap import banner, to_markdown

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, cross_validate, learning_curve,
)
from sklearn.pipeline import Pipeline

from mfp.config import load_config
from mfp.data import load_split, xy
from mfp.evaluation import corrected_resampled_ttest, select_threshold
from mfp.features import PhysicsFeatures, build_preprocessor, feature_lists
from mfp.models import MODEL_LABELS, available_models, build_pipeline, search_space
from mfp.viz import INK, SERIES, save, set_style, subtitle

ap = argparse.ArgumentParser()
ap.add_argument("--fast", action="store_true", help="CV/search lebih kecil untuk debugging")
args = ap.parse_args()

cfg = load_config()
set_style()
FIG, TAB, MODELS = cfg.path("figures_dir"), cfg.path("tables_dir"), cfg.path("models_dir")
SEED = cfg.seed
N_SPLITS = cfg["cv"]["n_splits"]
N_REPEATS = 1 if args.fast else cfg["cv"]["n_repeats"]
N_ITER = 4 if args.fast else cfg["cv"]["search_iter"]
C_FN, C_FP = cfg["business"]["cost_false_negative"], cfg["business"]["cost_false_positive"]
MIN_RECALL = cfg["business"]["min_recall"]

train, _ = load_split(cfg)
X, y = xy(train, cfg)
POS_WEIGHT = float((y == 0).sum() / (y == 1).sum())
print(f"Train: {X.shape}, failure rate {y.mean():.3%}, neg/pos = {POS_WEIGHT:.1f}")
t0 = time.time()

# =========================================================================== 4.1
banner("4.1 Perbandingan model x set fitur (Repeated Stratified 5-Fold)")
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
models = available_models()
fold_scores: dict[tuple[str, str], np.ndarray] = {}
rows = []
for name in models:
    for fs, eng in (("raw", False), ("physics", True)):
        pipe = build_pipeline(name, cfg, engineered=eng, pos_weight=POS_WEIGHT)
        res = cross_validate(pipe, X, y, cv=rskf, n_jobs=1,
                             scoring={"pr_auc": "average_precision", "roc_auc": "roc_auc"})
        fold_scores[(name, fs)] = res["test_pr_auc"]
        rows.append({"model": MODEL_LABELS[name], "key": name, "features": fs,
                     "pr_auc_mean": res["test_pr_auc"].mean(), "pr_auc_std": res["test_pr_auc"].std(),
                     "roc_auc_mean": res["test_roc_auc"].mean(), "fit_time_s": res["fit_time"].mean()})
        print(f"  {name:15s} {fs:8s} PR-AUC {res['test_pr_auc'].mean():.4f} ± {res['test_pr_auc'].std():.4f}"
              f" | ROC-AUC {res['test_roc_auc'].mean():.4f}")
cmp_df = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
pd.DataFrame({f"{k[0]}__{k[1]}": v for k, v in fold_scores.items()}).to_csv(
    TAB / "04_cv_fold_scores.csv", index_label="fold")

# corrected t-test: tiap konfigurasi vs konfigurasi terbaik; dan physics vs raw per model
n_test_fold = len(y) // N_SPLITS
n_train_fold = len(y) - n_test_fold
best_key = (cmp_df.iloc[0]["key"], cmp_df.iloc[0]["features"])
cmp_df["p_vs_best"] = [
    np.nan if (r.key, r.features) == best_key else
    corrected_resampled_ttest(fold_scores[best_key], fold_scores[(r.key, r.features)],
                              n_train_fold, n_test_fold)[1]
    for r in cmp_df.itertuples()]
to_markdown(cmp_df.drop(columns="key"), TAB / "04_model_comparison")

ablation = []
for name in models:
    if name == "dummy":
        continue
    a, b = fold_scores[(name, "physics")], fold_scores[(name, "raw")]
    t, p = corrected_resampled_ttest(a, b, n_train_fold, n_test_fold)
    ablation.append({"model": MODEL_LABELS[name], "pr_auc_raw": b.mean(),
                     "pr_auc_physics": a.mean(), "delta": a.mean() - b.mean(), "p_value": p})
abl_df = pd.DataFrame(ablation)
print(to_markdown(abl_df, TAB / "04_feature_ablation"))

# figure: dot plot of fold scores
fig, ax = plt.subplots(figsize=(10, 4.8))
plot_models = [m for m in models if m != "dummy"]
for i, name in enumerate(plot_models):
    for j, (fs, color) in enumerate((("raw", SERIES[0]), ("physics", SERIES[1]))):
        s = fold_scores[(name, fs)]
        yy = i + (j - 0.5) * 0.32
        jitter = np.random.default_rng(i * 2 + j).uniform(-0.07, 0.07, len(s))
        ax.scatter(s, yy + jitter, s=14, color=color, alpha=0.45, lw=0)
        ax.plot([s.mean()] * 2, [yy - 0.14, yy + 0.14], color=color, lw=3,
                label=("Fitur mentah" if fs == "raw" else "Mentah + fisika") if i == 0 else None)
ax.set_yticks(range(len(plot_models)), [MODEL_LABELS[m] for m in plot_models])
ax.set_xlabel("PR-AUC per fold (garis tebal = rata-rata)")
dummy_pr = fold_scores[("dummy", "raw")].mean()
ax.axvline(dummy_pr, color=INK["muted"], ls="--", lw=1)
ax.text(dummy_pr, -0.55, f" baseline acak = {dummy_pr:.3f}", fontsize=8,
        color=INK["secondary"], va="center")
ax.set_ylim(len(plot_models) - 0.5, -0.75)
ax.legend(loc="upper left", bbox_to_anchor=(0.08, 0.9), fontsize=9)
ax.set_title(f"Perbandingan model — repeated stratified {N_SPLITS}-fold ×{N_REPEATS}")
_top = abl_df.sort_values("delta", ascending=False).iloc[0]
subtitle(ax, f"Fitur fisika menaikkan PR-AUC hingga +{_top.delta:.2f} ({_top.model}); "
             "model tree-based jauh mengungguli linear")
ax.grid(axis="y", visible=False)
save(fig, FIG / "04_model_comparison.png")

# =========================================================================== 4.2
banner("4.2 Hyperparameter tuning (RandomizedSearchCV, PR-AUC)")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
tuned: dict[str, Pipeline] = {}
best_params: dict[str, dict] = {}
for name in [m for m in models if m != "dummy"]:
    space = search_space(name, POS_WEIGHT)
    n_iter = min(N_ITER, 10) if name.startswith("logreg") else N_ITER
    search = RandomizedSearchCV(
        build_pipeline(name, cfg, engineered=True, pos_weight=POS_WEIGHT), space,
        n_iter=n_iter, scoring="average_precision", cv=skf, n_jobs=-1,
        random_state=SEED, refit=False)
    search.fit(X, y)
    best_params[name] = {k: (v.item() if hasattr(v, "item") else v)
                         for k, v in search.best_params_.items()}
    tuned[name] = build_pipeline(name, cfg, True, POS_WEIGHT).set_params(**search.best_params_)
    print(f"  {name:15s} best CV PR-AUC {search.best_score_:.4f}  {best_params[name]}")
(MODELS / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))

# =========================================================================== 4.3
banner("4.3 Out-of-fold predictions & ensemble")
oof: dict[str, np.ndarray] = {}
for name, pipe in tuned.items():
    oof[name] = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]

tree_models = [m for m in ("random_forest", "hist_gb", "xgboost") if m in tuned]
# Soft-voting = rata-rata probabilitas model tree-based (identik dengan VotingClassifier)
oof["ensemble"] = np.column_stack([oof[m] for m in tree_models]).mean(axis=1)

oof_rows = []
for name, p in oof.items():
    oof_rows.append({"model": MODEL_LABELS.get(name, "Soft-voting ensemble"), "key": name,
                     "oof_pr_auc": average_precision_score(y, p),
                     "oof_roc_auc": roc_auc_score(y, p), "oof_brier": brier_score_loss(y, p)})
oof_df = pd.DataFrame(oof_rows).sort_values("oof_pr_auc", ascending=False)
print(to_markdown(oof_df.drop(columns="key"), TAB / "04_oof_tuned"))

# Model final: ensemble hanya dipilih bila unggul jelas (> +0.005 PR-AUC) dibanding
# model tunggal terbaik — prinsip parsimoni (lebih sederhana, lebih mudah dirawat).
best_single = oof_df[oof_df["key"] != "ensemble"].iloc[0]["key"]
ens_gain = average_precision_score(y, oof["ensemble"]) - average_precision_score(y, oof[best_single])
final_key = "ensemble" if ens_gain > 0.005 else best_single
print(f"Model tunggal terbaik: {best_single}; gain ensemble = {ens_gain:+.4f} -> final: {final_key}")

if final_key == "ensemble":
    base_estimator = VotingClassifier([(m, tuned[m]) for m in tree_models], voting="soft")
else:
    base_estimator = tuned[final_key]

# =========================================================================== 4.4
banner("4.4 Kalibrasi probabilitas")
cal_rows, cal_oof = [], {"none": oof[final_key]}
for method in ("sigmoid", "isotonic"):
    cal = CalibratedClassifierCV(clone(base_estimator), method=method, cv=3)
    cal_oof[method] = cross_val_predict(cal, X, y, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
for method, p in cal_oof.items():
    cal_rows.append({"calibration": method, "oof_brier": brier_score_loss(y, p),
                     "oof_pr_auc": average_precision_score(y, p),
                     "mean_pred": p.mean(), "observed_rate": y.mean()})
cal_df = pd.DataFrame(cal_rows)
print(to_markdown(cal_df, TAB / "04_calibration"))
# pilih Brier terendah selama PR-AUC tidak turun > 0.01
ok = cal_df[cal_df["oof_pr_auc"] >= cal_df.loc[cal_df.calibration == "none", "oof_pr_auc"].item() - 0.01]
cal_method = ok.sort_values("oof_brier").iloc[0]["calibration"]
print("Metode kalibrasi terpilih:", cal_method)
final_estimator = (base_estimator if cal_method == "none" else
                   CalibratedClassifierCV(clone(base_estimator), method=cal_method, cv=5))
final_oof = cal_oof[cal_method]

# =========================================================================== 4.5
banner("4.5 Threshold berbasis biaya (pada OOF, bukan test)")
thr, curve = select_threshold(y, final_oof, C_FN, C_FP, MIN_RECALL)
curve.to_csv(TAB / "04_oof_cost_curve.csv", index=False)
r = curve.loc[(curve.threshold - thr).abs().idxmin()]
print(f"Threshold = {thr:.3f} | OOF recall {r.recall:.3f} precision {r.precision:.3f} "
      f"| biaya OOF {r.expected_cost:,.0f}")

fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(curve.threshold, curve.expected_cost / 1e6, color=SERIES[0])
ax.axvline(thr, color=SERIES[1], lw=1.2, ls="--")
ax.annotate(f"threshold terpilih = {thr:.3f}\nrecall {r.recall:.0%}, precision {r.precision:.0%}",
            xy=(thr, r.expected_cost / 1e6), xytext=(thr + 0.12, curve.expected_cost.max() / 1e6 * 0.7),
            fontsize=9, color=INK["primary"], arrowprops=dict(arrowstyle="-", color=INK["muted"]))
ax.set_xlabel("Threshold probabilitas"); ax.set_ylabel("Total biaya OOF (juta)")
ax.set_title("Kurva biaya vs threshold (data train, out-of-fold)")
subtitle(ax, f"Kegagalan lolos (FN) = {C_FN:,}; inspeksi per alarm = {C_FP:,}; syarat recall ≥ {MIN_RECALL:.0%}")
save(fig, FIG / "04_cost_curve_oof.png")

# =========================================================================== 4.6
banner("4.6 Learning curve")
sizes, tr_s, va_s = learning_curve(clone(base_estimator), X, y, cv=skf, n_jobs=1,
                                   train_sizes=np.linspace(0.15, 1.0, 6),
                                   scoring="average_precision", random_state=SEED)
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for s, lab, c in ((tr_s, "Train", SERIES[0]), (va_s, "Validasi (CV)", SERIES[1])):
    ax.plot(sizes, s.mean(1), marker="o", ms=5, color=c, label=lab)
    ax.fill_between(sizes, s.mean(1) - s.std(1), s.mean(1) + s.std(1), color=c, alpha=0.15, lw=0)
ax.set_xlabel("Jumlah sampel training"); ax.set_ylabel("PR-AUC")
ax.legend(loc="lower right")
ax.set_title("Learning curve model final")
_gain = va_s.mean(1)[-1] - va_s.mean(1)[-2]
subtitle(ax, f"Gap train-validasi akhir = {tr_s.mean(1)[-1] - va_s.mean(1)[-1]:.3f}; "
             + ("kurva validasi masih naik → data tambahan kemungkinan membantu" if _gain > 0.005
                else "kurva validasi mulai mendatar → model mendekati plateau"))
save(fig, FIG / "04_learning_curve.png")
pd.DataFrame({"n": sizes, "train": tr_s.mean(1), "valid": va_s.mean(1),
              "valid_std": va_s.std(1)}).to_csv(TAB / "04_learning_curve.csv", index=False)

# =========================================================================== 4.7
banner("4.7 Model diagnosis mode kegagalan (multi-label)")
MODE_LABELS = ["TWF", "HDF", "PWF", "OSF"]   # RNF = acak menurut desain -> tidak dimodelkan
numeric, categorical = feature_lists(cfg, True)
mode_model = Pipeline([
    ("physics", PhysicsFeatures(overstrain_limits=cfg["features"]["overstrain_limits"])),
    ("prep", build_preprocessor(numeric, categorical)),
    ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                     class_weight="balanced_subsample", n_jobs=-1,
                                     random_state=SEED)),
])
Y_modes = train[MODE_LABELS].to_numpy()
mode_oof = cross_val_predict(mode_model, X, Y_modes, cv=list(skf.split(X, y)),
                             method="predict_proba")
mode_rows = [{"mode": m, "n_train": int(Y_modes[:, i].sum()),
              "oof_pr_auc": average_precision_score(Y_modes[:, i], mode_oof[i][:, 1]),
              "oof_roc_auc": roc_auc_score(Y_modes[:, i], mode_oof[i][:, 1])}
             for i, m in enumerate(MODE_LABELS)]
print(to_markdown(pd.DataFrame(mode_rows), TAB / "04_mode_model_oof"))
mode_model.fit(X, Y_modes)

# =========================================================================== 4.8
banner("4.8 Fit model final pada seluruh data train")
final_estimator.fit(X, y)
joblib.dump(final_estimator, MODELS / "final_pipeline.joblib", compress=3)
joblib.dump(mode_model, MODELS / "mode_model.joblib", compress=3)
for name, pipe in tuned.items():   # untuk perbandingan transparan di tahap evaluasi
    joblib.dump(pipe.fit(X, y), MODELS / f"tuned_{name}.joblib", compress=3)

summary = {
    "final_model": final_key,
    "final_model_label": MODEL_LABELS.get(final_key, "Soft-voting ensemble"),
    "ensemble_members": tree_models if final_key == "ensemble" else [],
    "calibration": cal_method,
    "threshold": thr,
    "oof_pr_auc": float(average_precision_score(y, final_oof)),
    "oof_roc_auc": float(roc_auc_score(y, final_oof)),
    "oof_brier": float(brier_score_loss(y, final_oof)),
    "best_params": best_params.get(final_key, {k: best_params[k] for k in tree_models}),
    "mode_labels": MODE_LABELS,
    "cv": {"n_splits": N_SPLITS, "n_repeats": N_REPEATS, "search_iter": N_ITER},
    "cost": {"fn": C_FN, "fp": C_FP, "min_recall": MIN_RECALL},
    "runtime_min": round((time.time() - t0) / 60, 1),
}
(MODELS / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print(json.dumps(summary, indent=2, default=str))
