"""Membangun & mengeksekusi notebook naratif `machine_failure_prediction.ipynb`.

Notebook membaca artefak hasil pipeline (`make all`) — tidak melatih ulang —
sehingga cepat dieksekusi dan identik dengan hasil pipeline.

    python notebooks/_build_notebook.py

Bila Jupyter terpasang, notebook juga bisa dijalankan ulang langsung.
Builder ini memiliki executor minimal (stdout, DataFrame, gambar) agar
notebook ber-output tetap dapat dibuat tanpa dependensi jupyter/nbformat.
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "machine_failure_prediction.ipynb"

CELLS: list[tuple[str, str]] = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))   # noqa: E731
code = lambda s: CELLS.append(("code", s.strip("\n")))     # noqa: E731

# --------------------------------------------------------------------------- isi
md(r"""
# Machine Failure Prediction — Predictive Maintenance (CRISP-DM)

**Tujuan:** memprediksi unit mesin yang berisiko gagal *sebelum* kegagalan terjadi,
meminimalkan total biaya maintenance, dan memberi teknisi dugaan **mode kegagalan**.

| CRISP-DM | Isi notebook | Kode produksi |
|---|---|---|
| 1. Business Understanding | biaya, kriteria sukses | `docs/01_business_understanding.md` |
| 2. Data Understanding | audit kualitas, EDA, uji statistik, validasi mekanisme fisik | `pipelines/02_data_understanding.py` |
| 3. Data Preparation | leakage, skema, split terkunci, fitur fisika | `pipelines/03_data_preparation.py`, `src/mfp/features.py` |
| 4. Modeling | repeated CV, ablation, tuning, ensemble, kalibrasi, threshold biaya | `pipelines/04_modeling.py` |
| 5. Evaluation | test + bootstrap CI, dampak bisnis, explainability, error analysis | `pipelines/05_evaluation.py` |
| 6. Deployment | model bundle, REST API, CLI, monitoring drift | `pipelines/06_deployment.py`, `app/` |

> Notebook ini adalah **laporan naratif**: membaca artefak hasil `make all`, sehingga
> angka di sini identik dengan pipeline produksi.
""")
code(r"""
import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
from IPython.display import Image, display
from mfp.config import load_config
from mfp.data import load_raw, audit_quality

pd.set_option("display.float_format", lambda v: f"{v:,.4f}")
cfg = load_config()
FIG, TAB, MODELS = ROOT / "reports/figures", ROOT / "reports/tables", ROOT / "models"
table = lambda name: pd.read_csv(TAB / f"{name}.csv")
""")

md(r"""
## 1. Business Understanding

* Kegagalan tak terdeteksi (FN) = **10.000** satuan biaya; setiap alarm memicu inspeksi = **500**.
* Rasio 20:1 → threshold Bayes-optimal teoretis ≈ 0,05, **bukan 0,5**.
* Kriteria sukses: PR-AUC ≥ 0,80, recall ≥ 80%, biaya < 50% strategi reaktif & < inspeksi-semua.
""")

md("## 2. Data Understanding\n### 2.1 Sekilas data & audit kualitas")
code(r"""
df = load_raw(ROOT / cfg["paths"]["raw_data"])
print(df.shape)
display(df.head())
display(audit_quality(df, cfg).to_frame())
""")
md(r"""
**Temuan penting**
* Tidak ada nilai kosong/duplikat; prefix `Product ID` = `Type` (redundan → dibuang).
* **Label anomali**: 9 baris `Machine failure=1` tanpa mode, 18 baris `RNF=1` dengan failure=0.
  Label tidak diubah (ground truth operator); dampaknya dianalisis di error analysis.
* Kolom TWF/HDF/PWF/OSF/RNF = **target leakage** (diketahui setelah kegagalan).
""")
code(r"""display(Image(filename=str(FIG / "02_target_and_modes.png")))""")
md("### 2.2 Sensor mana yang membedakan unit gagal? (Mann-Whitney U + rank-biserial effect size)")
code(r"""
display(table("02_mannwhitney"))
display(Image(filename=str(FIG / "02_distributions_by_class.png")))
""")
md(r"""
### 2.3 Validasi hipotesis mekanisme fisik
Setiap mode kegagalan muncul di wilayah yang dapat dijelaskan fisika — inilah dasar
*feature engineering* di tahap 3.
""")
code(r"""display(Image(filename=str(FIG / "02_physics_mechanisms.png")))""")
code(r"""display(Image(filename=str(FIG / "02_correlation.png")))""")

md(r"""
## 3. Data Preparation

* Fitur: hanya 6 kolom yang tersedia **sebelum** kegagalan (`type` + 5 sensor).
* Validasi skema eksplisit (`mfp.data.validate_features`) dipakai saat training **dan** serving.
* Split stratified 80/20 dikunci ke disk; test set tidak disentuh sampai tahap 5.
* Fitur fisika dibuat **di dalam** `sklearn.Pipeline` → di-fit ulang per fold, tanpa
  training-serving skew.
""")
code(r"""
print(json.dumps(json.loads((ROOT / "data/processed/data_manifest.json").read_text()), indent=2))

from mfp.features import PhysicsFeatures
sample = df[["type", *cfg["features"]["numeric_raw"]]].head(3)
display(PhysicsFeatures(overstrain_limits=cfg["features"]["overstrain_limits"]).fit_transform(sample))
""")

md(r"""
## 4. Modeling
### 4.1 Perbandingan model × set fitur (repeated stratified 5-fold × 3, hanya data train)
Uji signifikansi memakai **corrected resampled t-test** (Nadeau & Bengio) karena skor
fold CV tidak independen.
""")
code(r"""
display(table("04_model_comparison"))
display(Image(filename=str(FIG / "04_model_comparison.png")))
""")
md("### 4.2 Ablation: apakah fitur fisika benar-benar membantu?")
code(r"""display(table("04_feature_ablation"))""")
md(r"""
Fitur fisika menaikkan PR-AUC Random Forest **+0,15** dan gradient boosting **+0,09**
(signifikan, p < 0,001). Domain knowledge > model yang lebih kompleks.

### 4.3 Model tuned (prediksi out-of-fold) & ensemble
""")
code(r"""
display(table("04_oof_tuned"))
print(json.dumps(json.loads((MODELS / "training_summary.json").read_text()), indent=2))
""")
md(r"""
Soft-voting ensemble hanya +0,004 PR-AUC di atas Random Forest → **di bawah ambang +0,005**,
jadi model tunggal dipilih (parsimoni: lebih cepat, lebih mudah dijelaskan & dirawat).

### 4.4 Kalibrasi & 4.5 threshold berbasis biaya
""")
code(r"""
display(table("04_calibration"))
display(Image(filename=str(FIG / "04_cost_curve_oof.png")))
display(Image(filename=str(FIG / "04_learning_curve.png")))
""")

md("## 5. Evaluation (test set terkunci, dipakai sekali)\n### 5.1 Metrik + 95% bootstrap CI")
code(r"""
display(pd.read_csv(TAB / "05_test_metrics.csv"))
display(pd.read_csv(TAB / "05_test_metrics_ci.csv"))
display(Image(filename=str(FIG / "05_confusion_matrices.png")))
display(Image(filename=str(FIG / "05_pr_roc_curves.png")))
display(Image(filename=str(FIG / "05_calibration.png")))
""")
md("### 5.2 Dampak bisnis")
code(r"""
display(table("05_business_impact"))
display(Image(filename=str(FIG / "05_business_impact.png")))
display(Image(filename=str(FIG / "05_cumulative_gains.png")))
""")
md("### 5.3 Explainability — apakah model belajar mekanisme yang benar?")
code(r"""
display(Image(filename=str(FIG / "05_permutation_importance.png")))
display(Image(filename=str(FIG / "05_partial_dependence.png")))
print((TAB / "05_surrogate_rules.txt").read_text())
""")
md(r"""
Pohon surrogate (fidelity ~98%) **menemukan kembali** aturan fisika tanpa diberi tahu:
daya < ~3.550 W atau > ~8.900 W (PWF), strain > ~11.000 min·Nm (OSF), dan keausan
tool > ~80% umur (TWF). Partial dependence 2D menunjukkan pita daya aman 3.500–9.000 W.

### 5.4 Error analysis per mode kegagalan
""")
code(r"""
display(table("05_recall_by_mode"))
display(table("05_false_negatives"))
display(Image(filename=str(FIG / "05_recall_by_mode.png")))
display(table("05_mode_model_test"))
""")
md(r"""
Kegagalan yang lolos adalah **TWF** (acak di rentang keausan 200–240 min) dan satu
kegagalan **tanpa mode** — keduanya secara desain tidak dapat diprediksi dari snapshot sensor.
""")

md("## 6. Deployment\n### 6.1 Scoring skenario dengan model bundle")
code(r"""
from mfp.inference import ModelBundle
bundle = ModelBundle.load(MODELS / "model_bundle.joblib")
scenarios = pd.DataFrame([
    dict(skenario="Operasi normal",           type="M", air_temp_k=300.0, process_temp_k=310.0, rot_speed_rpm=1500, torque_nm=40.0, tool_wear_min=50),
    dict(skenario="Daya berlebih (PWF)",      type="L", air_temp_k=300.5, process_temp_k=310.5, rot_speed_rpm=1300, torque_nm=70.0, tool_wear_min=60),
    dict(skenario="Pendinginan buruk (HDF)",  type="L", air_temp_k=302.8, process_temp_k=310.9, rot_speed_rpm=1330, torque_nm=48.0, tool_wear_min=90),
    dict(skenario="Overstrain (OSF)",         type="L", air_temp_k=300.0, process_temp_k=310.0, rot_speed_rpm=1400, torque_nm=55.0, tool_wear_min=215),
    dict(skenario="Tool hampir habis (TWF?)", type="H", air_temp_k=299.0, process_temp_k=309.5, rot_speed_rpm=1600, torque_nm=35.0, tool_wear_min=225),
])
res = scenarios[["skenario"]].join(bundle.predict(scenarios.drop(columns="skenario")))
display(res[["skenario", "failure_probability", "risk_level", "likely_failure_mode", "recommended_action"]])
""")
md("### 6.2 Validasi level risiko & monitoring drift")
code(r"""
display(table("06_risk_band_validation"))
display(Image(filename=str(FIG / "06_drift_monitoring.png")))
""")
md(r"""
## 7. Kesimpulan

* **Random Forest + fitur fisika** mencapai PR-AUC test 0,906 (CI 0,84–0,96), recall 95,6%,
  dan menghemat **~80% biaya** dibanding strategi reaktif.
* Inspeksi 5% unit berisiko tertinggi saja sudah menangkap ~87% kegagalan.
* Model dapat dijelaskan: aturan yang ia pelajari sesuai fisika mesin.
* Level risiko tervalidasi: HIGH ≈ 96% benar-benar gagal, LOW ≈ 0,2%.

**Langkah berikutnya:** data deret waktu untuk *remaining useful life*, shadow deployment
di lini nyata, retraining terjadwal saat PSI > 0,25, dan biaya aktual dari CMMS perusahaan.
""")


# --------------------------------------------------------------------- executor
class _Recorder:
    def __init__(self):
        self.outputs: list[dict] = []
        self.count = 0


REC = _Recorder()


class Image:  # pengganti IPython.display.Image untuk executor
    def __init__(self, filename: str):
        self.data = Path(filename).read_bytes()


def display(obj):
    import pandas as pd
    if isinstance(obj, Image):
        REC.outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"image/png": base64.b64encode(obj.data).decode(),
                                     "text/plain": ["<IPython.core.display.Image object>"]}})
    elif isinstance(obj, pd.DataFrame):
        REC.outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"text/html": obj.to_html(max_rows=30),
                                     "text/plain": obj.to_string(max_rows=30)}})
    else:
        REC.outputs.append({"output_type": "display_data", "metadata": {},
                            "data": {"text/plain": repr(obj)}})


def _lines(text: str) -> list[str]:
    parts = text.splitlines(keepends=True)
    return parts


def build() -> None:
    fake = types.ModuleType("IPython")
    fake_display = types.ModuleType("IPython.display")
    fake_display.Image, fake_display.display = Image, display
    fake.display = fake_display
    sys.modules["IPython"], sys.modules["IPython.display"] = fake, fake_display

    os.chdir(HERE)
    ns: dict = {"__name__": "__main__"}
    nb_cells = []
    for kind, src in CELLS:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(src)})
            continue
        REC.count += 1
        REC.outputs = []
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(src, f"<cell {REC.count}>", "exec"), ns)
        outputs = []
        if buf.getvalue():
            outputs.append({"output_type": "stream", "name": "stdout",
                            "text": _lines(buf.getvalue())})
        outputs += REC.outputs
        nb_cells.append({"cell_type": "code", "execution_count": REC.count, "metadata": {},
                         "outputs": outputs, "source": _lines(src)})
    nb = {"cells": nb_cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python", "version": sys.version.split()[0]}}}
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Notebook tersimpan: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB, {REC.count} code cells)")


if __name__ == "__main__":
    build()
