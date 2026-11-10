"""CRISP-DM Tahap 5 — Evaluation.

Test set (20%, dikunci sejak tahap 3) dipakai SATU KALI di sini dengan model
dan threshold yang sudah difinalkan di tahap 4.

  5.1  Metrik test + 95% bootstrap CI; threshold 0.5 vs threshold biaya
  5.2  Kurva PR/ROC semua kandidat (transparansi), kalibrasi, confusion matrix
  5.3  Dampak bisnis: biaya strategi reaktif vs inspeksi-semua vs model;
       cumulative gains (berapa % kegagalan tertangkap jika inspeksi top-k%)
  5.4  Explainability: permutation importance, partial dependence,
       surrogate decision tree (aturan yang dapat dibaca teknisi)
  5.5  Error analysis per mode kegagalan
  5.6  Evaluasi model diagnosis mode kegagalan
  5.7  Deployment gate (kriteria keberhasilan dari tahap Business Understanding)
"""
from __future__ import annotations

import json

import _bootstrap  # noqa: F401
from _bootstrap import banner, to_markdown

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.tree import DecisionTreeClassifier, export_text

from mfp.config import load_config
from mfp.data import load_split, xy
from mfp.evaluation import bootstrap_ci, full_metrics
from mfp.models import MODEL_LABELS
from mfp.viz import INK, MODEL_COLORS, SEQ_BLUE, SERIES, STATUS, save, set_style, subtitle

cfg = load_config()
set_style()
FIG, TAB, MODELS = cfg.path("figures_dir"), cfg.path("tables_dir"), cfg.path("models_dir")
C_FN, C_FP = cfg["business"]["cost_false_negative"], cfg["business"]["cost_false_positive"]
MIN_RECALL = cfg["business"]["min_recall"]
MODES_ALL = cfg["data"]["failure_modes"]

summary = json.loads((MODELS / "training_summary.json").read_text())
final = joblib.load(MODELS / "final_pipeline.joblib")
mode_model = joblib.load(MODELS / "mode_model.joblib")
THR = summary["threshold"]
FINAL_KEY = summary["final_model"]

train, test = load_split(cfg)
X_tr, y_tr = xy(train, cfg)
X_te, y_te = xy(test, cfg)
p_te = final.predict_proba(X_te)[:, 1]
print(f"Model final: {summary['final_model_label']} | kalibrasi: {summary['calibration']} "
      f"| threshold: {THR:.3f} | test n={len(y_te)}, failures={y_te.sum()}")

# =========================================================================== 5.1
banner("5.1 Metrik test set + bootstrap CI")
m_default = full_metrics(y_te, p_te, 0.5, C_FN, C_FP)
m_cost = full_metrics(y_te, p_te, THR, C_FN, C_FP)
cmp = pd.DataFrame({"threshold 0.50": m_default, f"threshold biaya {THR:.3f}": m_cost}).T
cmp = cmp[["pr_auc", "roc_auc", "brier", "recall", "precision", "f1", "f2", "mcc",
           "specificity", "tp", "fp", "fn", "tn", "expected_cost"]]
print(to_markdown(cmp, TAB / "05_test_metrics", index=True))

ci = bootstrap_ci(y_te, p_te, THR, n_boot=cfg["evaluation"]["n_bootstrap"],
                  level=cfg["evaluation"]["ci_level"], seed=cfg.seed)
print(to_markdown(ci, TAB / "05_test_metrics_ci", index=True))

# =========================================================================== 5.2
banner("5.2 Kurva PR/ROC, kalibrasi, confusion matrix")
candidates = {}
for key in ("logreg", "logreg_spline", "random_forest", "hist_gb", "xgboost"):
    f = MODELS / f"tuned_{key}.joblib"
    if f.exists():
        candidates[key] = joblib.load(f).predict_proba(X_te)[:, 1]
cand_rows = [{"model": MODEL_LABELS[k], "test_pr_auc": average_precision_score(y_te, p),
              "test_roc_auc": roc_auc_score(y_te, p)} for k, p in candidates.items()]
cand_rows.append({"model": f"FINAL — {summary['final_model_label']}"
                           f" (kalibrasi: {summary['calibration']})",
                  "test_pr_auc": m_cost["pr_auc"], "test_roc_auc": m_cost["roc_auc"]})
print(to_markdown(pd.DataFrame(cand_rows).sort_values("test_pr_auc", ascending=False),
                  TAB / "05_test_all_candidates"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for key, p in candidates.items():
    is_final = key == FINAL_KEY
    prec, rec, _ = precision_recall_curve(y_te, p)
    fpr, tpr, _ = roc_curve(y_te, p)
    kw = dict(color=MODEL_COLORS[key], lw=2.4 if is_final else 1.4, alpha=1 if is_final else 0.75)
    axes[0].plot(rec, prec, label=f"{MODEL_LABELS[key]} (AP={average_precision_score(y_te, p):.3f})", **kw)
    axes[1].plot(fpr, tpr, label=f"{MODEL_LABELS[key]} (AUC={roc_auc_score(y_te, p):.3f})", **kw)
if FINAL_KEY not in candidates:
    prec, rec, _ = precision_recall_curve(y_te, p_te)
    axes[0].plot(rec, prec, color=MODEL_COLORS["ensemble"], lw=2.4, label=f"Final (AP={m_cost['pr_auc']:.3f})")
axes[0].scatter([m_cost["recall"]], [m_cost["precision"]], s=60, color=INK["primary"], zorder=5,
                label=f"Titik operasi (t={THR:.3f})")
axes[0].axhline(y_te.mean(), color=INK["muted"], ls="--", lw=1)
axes[0].set_xlabel("Recall"); axes[0].set_ylabel("Precision")
axes[0].set_title("Precision–Recall (test set)")
subtitle(axes[0], "Kurva utama untuk data imbalanced; garis putus = baseline acak")
axes[0].legend(loc="lower left", fontsize=8)
axes[1].plot([0, 1], [0, 1], color=INK["muted"], ls="--", lw=1)
axes[1].set_xlabel("False positive rate"); axes[1].set_ylabel("True positive rate")
axes[1].set_title("ROC (test set)")
subtitle(axes[1], "ROC-AUC tinggi semua model — kurang diskriminatif saat kelas langka")
axes[1].legend(loc="lower right", fontsize=8)
fig.tight_layout()
save(fig, FIG / "05_pr_roc_curves.png")

# kalibrasi: reliability diagram + histogram (dua panel, satu sumbu y masing-masing)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [3, 1.2]})
frac, mean_pred = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
ax1.plot([0, 1], [0, 1], color=INK["muted"], ls="--", lw=1, label="Kalibrasi sempurna")
ax1.plot(mean_pred, frac, marker="o", ms=6, color=SERIES[0], label="Model final")
ax1.set_ylabel("Frekuensi kegagalan aktual")
ax1.set_title("Reliability diagram (test set)")
subtitle(ax1, f"Brier score = {m_cost['brier']:.4f}; bin kuantil")
ax1.legend(loc="upper left", fontsize=8)
ax2.hist(p_te, bins=40, color=SERIES[0], alpha=0.8)
ax2.set_yscale("log"); ax2.set_xlabel("Probabilitas prediksi"); ax2.set_ylabel("Jumlah (log)")
fig.tight_layout()
save(fig, FIG / "05_calibration.png")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, (lab, m) in zip(axes, ((f"Threshold 0.50", m_default), (f"Threshold biaya {THR:.3f}", m_cost))):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap=SEQ_BLUE, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac_ = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]:,}\n({frac_:.1%})", ha="center", va="center", fontsize=10,
                    color="#ffffff" if frac_ > 0.55 else INK["primary"])
    ax.set_xticks([0, 1], ["Pred: Normal", "Pred: Failure"])
    ax.set_yticks([0, 1], ["Aktual: Normal", "Aktual: Failure"])
    ax.grid(visible=False)
    ax.set_title(lab, fontsize=11)
    subtitle(ax, f"Recall {m['recall']:.1%} · Precision {m['precision']:.1%}")
fig.tight_layout()
save(fig, FIG / "05_confusion_matrices.png")

# =========================================================================== 5.3
banner("5.3 Dampak bisnis")
n, n_pos = len(y_te), int(y_te.sum())
strategies = pd.DataFrame([
    {"strategi": "Reaktif (run-to-failure)", "biaya": n_pos * C_FN,
     "kegagalan_lolos": n_pos, "inspeksi": 0},
    {"strategi": "Inspeksi semua unit", "biaya": n * C_FP, "kegagalan_lolos": 0, "inspeksi": n},
    {"strategi": "Model @ threshold 0.50", "biaya": m_default["expected_cost"],
     "kegagalan_lolos": m_default["fn"], "inspeksi": m_default["tp"] + m_default["fp"]},
    {"strategi": f"Model @ threshold biaya", "biaya": m_cost["expected_cost"],
     "kegagalan_lolos": m_cost["fn"], "inspeksi": m_cost["tp"] + m_cost["fp"]},
])
base_cost = strategies.loc[0, "biaya"]
strategies["penghematan_vs_reaktif"] = 1 - strategies["biaya"] / base_cost
print(to_markdown(strategies, TAB / "05_business_impact", floatfmt=",.2f"))

fig, ax = plt.subplots(figsize=(8.5, 3.8))
colors = [INK["muted"], INK["muted"], SERIES[0], SERIES[1]]
ax.barh(strategies["strategi"], strategies["biaya"] / 1e3, color=colors, height=0.55)
for i, r in strategies.iterrows():
    ax.text(r.biaya / 1e3, i, f"  {r.biaya / 1e3:,.0f}k  ({r.penghematan_vs_reaktif:+.0%})",
            va="center", fontsize=9, color=INK["primary"])
ax.invert_yaxis(); ax.set_xlabel("Total biaya pada test set (ribu satuan biaya)")
ax.set_xlim(0, strategies["biaya"].max() / 1e3 * 1.3)
ax.set_title("Dampak bisnis — 2.000 unit test")
subtitle(ax, f"Asumsi: kegagalan lolos = {C_FN:,}, inspeksi = {C_FP:,} per unit; % = hemat vs reaktif")
ax.grid(axis="y", visible=False)
save(fig, FIG / "05_business_impact.png")

order = np.argsort(-p_te)
gains = np.cumsum(y_te[order]) / n_pos
frac_inspected = np.arange(1, n + 1) / n
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.plot(frac_inspected, gains, color=SERIES[0], label="Model final")
ax.plot([0, 1], [0, 1], color=INK["muted"], ls="--", lw=1, label="Acak")
ax.plot([0, y_te.mean(), 1], [0, 1, 1], color=SERIES[2], lw=1, ls=":", label="Sempurna")
gain_rows = []
for k in (0.02, 0.05, 0.10):
    g = gains[int(k * n) - 1]
    gain_rows.append({"top_k_pct_inspeksi": f"{k:.0%}", "pct_kegagalan_tertangkap": g,
                      "lift": g / k})
    ax.scatter([k], [g], s=30, color=SERIES[1], zorder=5)
    ax.text(k + 0.015, g - 0.06, f"top {k:.0%} → {g:.0%}", fontsize=8.5, color=INK["primary"])
ax.set_xlim(0, 0.4)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Proporsi unit yang diinspeksi (urut risiko tertinggi)")
ax.set_ylabel("Proporsi kegagalan tertangkap")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Cumulative gains — prioritisasi inspeksi")
subtitle(ax, "Bila kapasitas tim terbatas, inspeksi unit berisiko tertinggi lebih dulu")
save(fig, FIG / "05_cumulative_gains.png")
print(to_markdown(pd.DataFrame(gain_rows), TAB / "05_cumulative_gains"))

# =========================================================================== 5.4
banner("5.4 Explainability")
# (a) permutation importance level input mentah (pipeline penuh, metrik PR-AUC)
pi_raw = permutation_importance(final, X_te, y_te, scoring="average_precision",
                                n_repeats=20, random_state=cfg.seed, n_jobs=1)
# (b) permutation importance level fitur rekayasa (model tree tanpa kalibrasi)
expl_key = FINAL_KEY if FINAL_KEY in candidates else "random_forest"
expl_pipe = joblib.load(MODELS / f"tuned_{expl_key}.joblib")
Xt_te = expl_pipe[:-1].transform(X_te)
feat_names = list(expl_pipe[:-1].get_feature_names_out())
Xt_te = pd.DataFrame(np.asarray(Xt_te), columns=feat_names)
pi_eng = permutation_importance(expl_pipe[-1], Xt_te.to_numpy(), y_te, scoring="average_precision",
                                n_repeats=20, random_state=cfg.seed, n_jobs=1)
imp_raw = pd.DataFrame({"feature": X_te.columns, "mean": pi_raw.importances_mean,
                        "std": pi_raw.importances_std}).sort_values("mean")
imp_eng = pd.DataFrame({"feature": feat_names, "mean": pi_eng.importances_mean,
                        "std": pi_eng.importances_std}).sort_values("mean")
to_markdown(imp_raw.iloc[::-1], TAB / "05_perm_importance_raw")
print(to_markdown(imp_eng.iloc[::-1], TAB / "05_perm_importance_engineered"))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.4]})
for ax, imp, title, sub in (
        (axes[0], imp_raw, "Sensor mentah (pipeline penuh)", "Penurunan PR-AUC saat sensor diacak"),
        (axes[1], imp_eng, f"Fitur rekayasa ({MODEL_LABELS[expl_key]})",
         "Fitur fisika mendominasi → model belajar mekanisme, bukan kebetulan")):
    ax.barh(imp["feature"], imp["mean"], xerr=imp["std"], color=SERIES[0], height=0.6,
            error_kw=dict(ecolor=INK["secondary"], lw=1, capsize=2))
    ax.set_xlabel("Δ PR-AUC (rata-rata ± std, 20 permutasi)")
    ax.set_title(title, fontsize=11); subtitle(ax, sub)
    ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, FIG / "05_permutation_importance.png")

# (c) Partial dependence: 2D torsi x putaran -> memperlihatkan "pita daya aman"
sample = X_te.sample(400, random_state=cfg.seed)
pd2 = partial_dependence(final, sample, features=["rot_speed_rpm", "torque_nm"],
                         grid_resolution=40, percentiles=(0.005, 0.995), kind="average")
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
ax = axes[0]
gx, gy = pd2["grid_values"]
cs = ax.contourf(gx, gy, pd2["average"][0].T, levels=14, cmap=SEQ_BLUE)
fig.colorbar(cs, ax=ax, label="P(failure) rata-rata")
w = np.linspace(gx.min(), gx.max(), 200)
for P in (3500, 9000):
    tq = P / (w * 2 * np.pi / 60)
    ax.plot(w, tq, color=SERIES[1], lw=1.2, ls="--")
    inside = np.flatnonzero((tq > gy.min() + 2) & (tq < gy.max() - 2))
    if len(inside):
        k = inside[len(inside) // 2]
        ax.text(w[k], tq[k], f" {P} W", fontsize=8.5, color=INK["primary"], va="bottom")
ax.set_ylim(gy.min(), gy.max())
ax.set_xlabel("Rotational speed [rpm]"); ax.set_ylabel("Torque [Nm]")
ax.grid(visible=False)
ax.set_title("Partial dependence 2D: putaran × torsi")
subtitle(ax, "Risiko tinggi di luar pita daya 3500–9000 W (garis putus = iso-daya)")
ax = axes[1]
for i, col in enumerate(["tool_wear_min", "air_temp_k"]):
    pdx = partial_dependence(final, sample, features=[col], grid_resolution=40,
                             percentiles=(0.005, 0.995), kind="average")
    gv = pdx["grid_values"][0]
    norm = (gv - gv.min()) / (gv.max() - gv.min())
    ax.plot(norm, pdx["average"][0], color=SERIES[i],
            label=f"{col} ({gv.min():.0f}–{gv.max():.0f})")
ax.set_xlabel("Posisi dalam rentang fitur (0 = min, 1 = max)")
ax.set_ylabel("P(failure) rata-rata")
ax.legend(loc="upper left", fontsize=8)
ax.set_title("Partial dependence 1D")
subtitle(ax, "Tool wear > ~200 min dan suhu udara tinggi (ΔT menyempit → HDF) menaikkan risiko")
fig.tight_layout()
save(fig, FIG / "05_partial_dependence.png")

# (d) Surrogate tree: aturan global yang meniru keputusan model, dalam SATUAN
#     FISIK asli (bukan z-score) agar dapat dibaca langsung oleh teknisi.
def physical_frame(X):
    F = expl_pipe[0].transform(X)
    return pd.concat([F.drop(columns="type"),
                      pd.get_dummies(F["type"], prefix="type", dtype=float)
                        .reindex(columns=["type_H", "type_L", "type_M"], fill_value=0.0)], axis=1)
F_tr, F_te = physical_frame(X_tr), physical_frame(X_te)
model_flag_tr = (final.predict_proba(X_tr)[:, 1] >= THR).astype(int)
model_flag_te = (p_te >= THR).astype(int)
surrogate = DecisionTreeClassifier(max_depth=4, min_samples_leaf=20, random_state=cfg.seed)
surrogate.fit(F_tr, model_flag_tr)
fidelity = (surrogate.predict(F_te) == model_flag_te).mean()
rules = export_text(surrogate, feature_names=list(F_tr.columns), decimals=1)
(TAB / "05_surrogate_rules.txt").write_text(
    f"Surrogate decision tree (depth 4) — fidelity terhadap keputusan model final "
    f"(threshold {THR:.3f}) pada test set: {fidelity:.2%}\n"
    "Satuan fisik asli: K, rpm, Nm, min, W, min·Nm; strain_ratio = strain / batas varian.\n\n"
    + rules)
print(f"Fidelity surrogate tree: {fidelity:.2%}")
print(rules)

try:  # SHAP opsional
    import shap  # noqa: F401
    explainer = shap.TreeExplainer(expl_pipe[-1])
    sv = explainer.shap_values(Xt_te)
    sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if np.ndim(sv) == 3 else sv)
    shap.summary_plot(sv, Xt_te, show=False, max_display=12)
    plt.title("SHAP summary"); save(plt.gcf(), FIG / "05_shap_summary.png")
    print("SHAP summary plot dibuat.")
except Exception as e:  # pragma: no cover
    print(f"SHAP dilewati ({type(e).__name__}): pip install shap untuk mengaktifkan.")

# =========================================================================== 5.5
banner("5.5 Error analysis per mode kegagalan")
te = test.copy()
te["proba"], te["flag"] = p_te, model_flag_te
te["mode"] = te[MODES_ALL].apply(lambda r: "+".join(m for m in MODES_ALL if r[m]) or "tanpa mode", axis=1)
fails = te[te["machine_failure"] == 1]
by_mode = []
for m in MODES_ALL + ["tanpa mode"]:
    sub = fails[fails["mode"] == "tanpa mode"] if m == "tanpa mode" else fails[fails[m] == 1]
    if len(sub):
        by_mode.append({"mode": m, "n_test": len(sub), "terdeteksi": int(sub["flag"].sum()),
                        "recall": sub["flag"].mean(), "median_proba": sub["proba"].median()})
by_mode = pd.DataFrame(by_mode)
print(to_markdown(by_mode, TAB / "05_recall_by_mode"))
fn_rows = fails[fails["flag"] == 0][["udi", "type", *cfg["features"]["numeric_raw"], "mode", "proba"]]
to_markdown(fn_rows, TAB / "05_false_negatives")
print("False negatives:\n", fn_rows.to_string(index=False))
rnf_only = te[(te["RNF"] == 1) & (te["machine_failure"] == 0)]
print(f"RNF tanpa failure di test: {len(rnf_only)} (label 0 — konsisten dengan target)")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
bm = by_mode.sort_values("recall")
cols = [STATUS["good"] if r >= MIN_RECALL else STATUS["critical"] for r in bm["recall"]]
ax.barh(bm["mode"], bm["recall"], color=cols, height=0.55)
for i, r in enumerate(bm.itertuples()):
    mark = "✓" if r.recall >= MIN_RECALL else "✗"
    ax.text(r.recall + 0.01, i, f"{mark} {r.terdeteksi}/{r.n_test}", va="center", fontsize=9)
ax.axvline(MIN_RECALL, color=INK["muted"], ls="--", lw=1)
ax.set_xlim(0, 1.18)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_xlabel("Recall pada test set")
ax.set_title("Recall per mode kegagalan")
subtitle(ax, f"Garis = target recall {MIN_RECALL:.0%}; TWF & kegagalan tanpa mode bersifat (semi-)acak")
ax.grid(axis="y", visible=False)
save(fig, FIG / "05_recall_by_mode.png")

# =========================================================================== 5.6
banner("5.6 Model diagnosis mode kegagalan (test)")
mode_labels = summary["mode_labels"]
mp = mode_model.predict_proba(X_te)
mrows = [{"mode": m, "n_test": int(test[m].sum()),
          "pr_auc": average_precision_score(test[m], mp[i][:, 1]),
          "roc_auc": roc_auc_score(test[m], mp[i][:, 1])} for i, m in enumerate(mode_labels)]
print(to_markdown(pd.DataFrame(mrows), TAB / "05_mode_model_test"))
# top-1 diagnosis untuk kegagalan yang terdeteksi & memiliki mode yang dimodelkan
P = np.column_stack([mp[i][:, 1] for i in range(len(mode_labels))])
mask = (te["flag"] == 1) & (te[mode_labels].sum(axis=1) > 0)
top1 = np.asarray(mode_labels)[P[mask.to_numpy()].argmax(axis=1)]
hit = [te.loc[idx, t] == 1 for idx, t in zip(te.index[mask], top1)]
top1_acc = float(np.mean(hit))
print(f"Top-1 akurasi diagnosis mode (kegagalan terdeteksi): {top1_acc:.1%} dari {mask.sum()} kasus")

# =========================================================================== 5.7
banner("5.7 Deployment gate")
criteria = {
    "pr_auc_test >= 0.80": m_cost["pr_auc"] >= 0.80,
    f"recall_test >= {MIN_RECALL}": m_cost["recall"] >= MIN_RECALL,
    "biaya < 50% strategi reaktif": m_cost["expected_cost"] < 0.5 * base_cost,
    "lebih baik dari inspeksi-semua": m_cost["expected_cost"] < n * C_FP,
}
for k, v in criteria.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")
report = {
    "model": summary["final_model_label"], "calibration": summary["calibration"],
    "threshold": THR,
    "test_metrics": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v))
                     for k, v in m_cost.items()},
    "test_metrics_threshold_0_5": {k: float(v) for k, v in m_default.items()},
    "ci95": ci.round(4).to_dict(orient="index"),
    "business": strategies.round(4).to_dict(orient="records"),
    "recall_by_mode": by_mode.round(4).to_dict(orient="records"),
    "mode_model_test": pd.DataFrame(mrows).round(4).to_dict(orient="records"),
    "mode_top1_accuracy": top1_acc,
    "surrogate_fidelity": float(fidelity),
    "gate": {k: bool(v) for k, v in criteria.items()},
    "approved": bool(all(criteria.values())),
}
(MODELS / "evaluation_report.json").write_text(json.dumps(report, indent=2, default=str))
print("APPROVED for deployment" if report["approved"] else "NOT approved — kembali ke tahap modeling")
