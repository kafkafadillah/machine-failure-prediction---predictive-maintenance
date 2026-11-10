"""CRISP-DM Tahap 2 — Data Understanding.

Output:
  reports/figures/02_*.png       visualisasi EDA
  reports/tables/02_*.csv|md     ringkasan statistik & uji hipotesis

Pertanyaan yang dijawab:
  1. Seberapa bersih dan konsisten data? (audit kualitas)
  2. Seberapa imbalanced target, dan bagaimana komposisi mode kegagalan?
  3. Sensor mana yang membedakan unit gagal vs normal? (uji statistik + effect size)
  4. Apakah mekanisme fisik kegagalan terlihat di data? (validasi hipotesis domain)
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
from _bootstrap import banner, to_markdown

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from mfp.config import load_config
from mfp.data import audit_quality, load_raw
from mfp.features import PhysicsFeatures
from mfp.viz import CLASS_COLORS, CLASS_NAMES, DIVERGING, INK, SERIES, save, set_style, subtitle

cfg = load_config()
set_style()
FIG, TAB = cfg.path("figures_dir"), cfg.path("tables_dir")
TARGET = cfg["data"]["target"]
MODES = cfg["data"]["failure_modes"]
SENSORS = cfg["features"]["numeric_raw"]
LABELS = {
    "air_temp_k": "Air temperature [K]", "process_temp_k": "Process temperature [K]",
    "rot_speed_rpm": "Rotational speed [rpm]", "torque_nm": "Torque [Nm]",
    "tool_wear_min": "Tool wear [min]", "temp_diff_k": "Temp. difference [K]",
    "power_w": "Power [W]", "strain_minnm": "Strain [min·Nm]",
    "strain_ratio": "Strain / limit", "torque_per_krpm": "Torque per krpm",
    "wear_fraction": "Wear fraction",
}

df = load_raw(cfg.path("raw_data"))
dfe = PhysicsFeatures(overstrain_limits=cfg["features"]["overstrain_limits"]).fit_transform(df)

# --------------------------------------------------------------------------- 1
banner("1. Audit kualitas data")
q = audit_quality(df, cfg)
print(to_markdown(q.to_frame(), TAB / "02_quality_audit"))
print("UDI failure tanpa mode:", q.failure_without_mode)
print("Mode=1 tapi failure=0 :", q.mode_without_failure)

summary = df[SENSORS].describe().T
summary["skew"] = df[SENSORS].skew()
summary["kurtosis"] = df[SENSORS].kurt()
to_markdown(summary.round(3), TAB / "02_descriptive_stats", index=True, floatfmt=".3f")

# --------------------------------------------------------------------------- 2
banner("2. Target & mode kegagalan")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
counts = df[TARGET].value_counts().sort_index()
ax = axes[0]
bars = ax.bar([CLASS_NAMES[i] for i in counts.index], counts.values,
              color=[CLASS_COLORS[i] for i in counts.index], width=0.55)
for b, v in zip(bars, counts.values):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / len(df):.1%})",
            ha="center", va="bottom", fontsize=9, color=INK["primary"])
ax.set_ylim(0, counts.max() * 1.22)
ax.set_title("Distribusi target")
subtitle(ax, "Sangat imbalanced: 1 kegagalan per ~29 unit")
ax.grid(axis="x", visible=False)

mc = df[MODES].sum().sort_values()
ax = axes[1]
ax.barh(mc.index, mc.values, color=SERIES[1], height=0.55)
for i, v in enumerate(mc.values):
    ax.text(v + 2, i, str(v), va="center", fontsize=9, color=INK["primary"])
ax.set_title("Jumlah kejadian per mode kegagalan")
subtitle(ax, "Satu kegagalan bisa memiliki >1 mode; RNF acak 0,1%")
ax.grid(axis="y", visible=False)
save(fig, FIG / "02_target_and_modes.png")

co = df.loc[df[MODES].sum(axis=1) > 0, MODES].T.dot(df.loc[df[MODES].sum(axis=1) > 0, MODES])
to_markdown(co, TAB / "02_mode_cooccurrence", index=True, floatfmt=".0f")

# --------------------------------------------------------------------------- 3
banner("3. Perbedaan distribusi sensor: gagal vs normal")
feats = SENSORS + ["temp_diff_k", "power_w", "strain_minnm"]
fig, axes = plt.subplots(2, 4, figsize=(15, 6.6))
for ax, col in zip(axes.ravel(), feats):
    for cls in (0, 1):
        vals = dfe.loc[dfe[TARGET] == cls, col]
        ax.hist(vals, bins=40, density=True, alpha=0.55 if cls == 0 else 0.65,
                color=CLASS_COLORS[cls], label=CLASS_NAMES[cls])
    ax.set_title(LABELS[col], fontsize=10)
    ax.set_yticks([])
    ax.grid(visible=False)
axes[0, 0].legend(loc="upper left", fontsize=8)
fig.suptitle("Distribusi (density) per kelas — fitur turunan memisahkan kelas lebih jelas",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
save(fig, FIG / "02_distributions_by_class.png")

rows = []
for col in feats:
    a = dfe.loc[dfe[TARGET] == 1, col]
    b = dfe.loc[dfe[TARGET] == 0, col]
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    rbc = 2 * u / (len(a) * len(b)) - 1          # rank-biserial correlation
    rows.append({"feature": col, "median_failure": a.median(), "median_normal": b.median(),
                 "mann_whitney_p": p, "rank_biserial": rbc,
                 "effect": ("besar" if abs(rbc) >= 0.43 else "sedang" if abs(rbc) >= 0.28
                            else "kecil" if abs(rbc) >= 0.11 else "dapat diabaikan")})
mw = pd.DataFrame(rows).sort_values("rank_biserial", key=np.abs, ascending=False)
print(to_markdown(mw, TAB / "02_mannwhitney", floatfmt=".4g"))

ct = pd.crosstab(df["type"], df[TARGET])
chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
cramers_v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
print(f"Chi-square Type vs failure: chi2={chi2:.2f}, p={p_chi:.2e}, Cramer's V={cramers_v:.3f}")

# failure rate per type with Wilson CI
fig, ax = plt.subplots(figsize=(6, 3.6))
order = ["L", "M", "H"]
rates, lo, hi = [], [], []
for t in order:
    k, n = int(ct.loc[t, 1]), int(ct.loc[t].sum())
    r = stats.binomtest(k, n).proportion_ci(method="wilson")
    rates.append(k / n); lo.append(k / n - r.low); hi.append(r.high - k / n)
ax.bar(order, rates, color=SERIES[1], width=0.5)
ax.errorbar(order, rates, yerr=[lo, hi], fmt="none", ecolor=INK["secondary"], capsize=4, lw=1)
for i, r in enumerate(rates):
    ax.text(i + 0.28, r, f"{r:.1%}", va="center", fontsize=9)
ax.set_ylabel("Failure rate")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.set_title("Failure rate per varian kualitas produk (95% Wilson CI)")
subtitle(ax, f"Varian L paling rentan; chi² p={p_chi:.1e}, Cramér's V={cramers_v:.2f} (lemah)")
ax.grid(axis="x", visible=False)
save(fig, FIG / "02_failure_rate_by_type.png")

# korelasi
corr = dfe[feats + [TARGET]].corr(method="spearman")
fig, ax = plt.subplots(figsize=(8.5, 7))
im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), [LABELS.get(c, "Machine failure") for c in corr], rotation=45, ha="right")
ax.set_yticks(range(len(corr)), [LABELS.get(c, "Machine failure") for c in corr])
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                color="#ffffff" if abs(v) > 0.6 else INK["primary"])
ax.grid(visible=False)
fig.colorbar(im, ax=ax, fraction=0.04, label="Spearman ρ")
ax.set_title("Korelasi Spearman")
subtitle(ax, "Torsi & putaran berkorelasi kuat negatif (ρ≈-0,9); suhu udara & proses ρ≈0,87")
save(fig, FIG / "02_correlation.png")

# --------------------------------------------------------------------------- 4
banner("4. Validasi hipotesis mekanisme fisik")
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
normal = dfe[dfe[MODES].sum(axis=1) == 0].sample(2500, random_state=cfg.seed)

def _scatter(ax, x, y, mode, title, sub):
    ax.scatter(normal[x], normal[y], s=8, color=SERIES[0], alpha=0.25, lw=0, label="Normal (sampel)")
    m = dfe[dfe[mode] == 1]
    ax.scatter(m[x], m[y], s=22, color=SERIES[1], edgecolor=INK["surface"], lw=0.8,
               label=f"{mode} (n={len(m)})")
    ax.set_xlabel(LABELS[x]); ax.set_ylabel(LABELS[y])
    ax.set_title(title, fontsize=11); subtitle(ax, sub)
    ax.legend(loc="best", fontsize=8)

_scatter(axes[0], "rot_speed_rpm", "temp_diff_k", "HDF", "Heat dissipation failure",
         "HDF terkonsentrasi di ΔT < 8,6 K DAN putaran < 1380 rpm")
axes[0].axhline(8.6, color=INK["muted"], lw=1, ls="--"); axes[0].axvline(1380, color=INK["muted"], lw=1, ls="--")
_scatter(axes[1], "rot_speed_rpm", "power_w", "PWF", "Power failure",
         "PWF muncul saat daya < 3500 W ATAU > 9000 W")
axes[1].axhline(3500, color=INK["muted"], lw=1, ls="--"); axes[1].axhline(9000, color=INK["muted"], lw=1, ls="--")
_scatter(axes[2], "tool_wear_min", "torque_nm", "OSF", "Overstrain failure",
         "OSF muncul saat wear×torque melewati batas per varian")
w = np.linspace(60, 253, 100)
for t, lim in cfg["features"]["overstrain_limits"].items():
    axes[2].plot(w, lim / w, lw=1, ls="--", color=INK["muted"])
    axes[2].text(w[-1], lim / w[-1], f" {t}", fontsize=8, color=INK["secondary"], va="center")
axes[2].set_ylim(0, 80)
fig.tight_layout()
save(fig, FIG / "02_physics_mechanisms.png")

twf = df[df["TWF"] == 1]["tool_wear_min"]
print(f"TWF: tool wear rentang {twf.min()}–{twf.max()} min (median {twf.median():.0f}) "
      f"→ terjadi di akhir umur tool, tetapi waktu tepatnya acak.")
print("\nSelesai. Figur di", FIG)
