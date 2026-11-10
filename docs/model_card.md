# Model Card — Machine Failure Prediction v1.0.0

## Ringkasan
| | |
|---|---|
| Tugas | Klasifikasi biner: apakah unit mesin akan gagal (`machine_failure`) |
| Model utama | Random Forest (500 pohon, max_depth 8, max_features 0,7, min_samples_leaf 7) dalam pipeline `PhysicsFeatures → ColumnTransformer → RF` |
| Model sekunder | Random Forest multi-label untuk mode kegagalan TWF / HDF / PWF / OSF |
| Threshold keputusan | 0,025 (minimisasi biaya pada prediksi out-of-fold, syarat recall ≥ 80%) |
| Artefak | `models/model_bundle.joblib` (pipeline + threshold + model mode + metadata) |

## Data
AI4I 2020 Predictive Maintenance Dataset (10.000 baris, 3,39% gagal).
Split stratified 80/20 (seed 42): train 8.000 (271 gagal), test 2.000 (68 gagal).
Input: `type` (L/M/H), suhu udara [K], suhu proses [K], putaran [rpm], torsi [Nm], keausan tool [min].

## Performa (test set, belum pernah dilihat selama pengembangan)

| Metrik | Nilai | 95% bootstrap CI |
|---|---|---|
| PR-AUC | 0,906 | 0,839 – 0,960 |
| ROC-AUC | 0,979 | 0,955 – 0,997 |
| Recall @ 0,025 | 95,6% (65/68) | 89,7% – 100% |
| Precision @ 0,025 | 31,6% | 28,2% – 35,4% |
| Brier score | 0,0065 | — |

Recall per mode: HDF 29/29, PWF 13/13, OSF 16/16, TWF 8/10, tanpa mode 1/2.
Biaya total pada test: 133k vs 680k (reaktif) → **hemat 80%**.

## Penggunaan yang dimaksud
Alat bantu prioritisasi inspeksi bagi tim maintenance. Output: probabilitas,
level risiko (LOW / MEDIUM / HIGH), tindakan yang disarankan, dan dugaan mode kegagalan.

## Di luar cakupan
* Bukan pengganti interlock/prosedur keselamatan.
* Tidak memprediksi *kapan* kegagalan terjadi (data snapshot, bukan deret waktu).
* Tidak valid untuk mesin/varian di luar rentang data training (lihat kontrak skema).

## Keterbatasan yang diketahui
* **TWF** (tool wear failure) terjadi acak di rentang keausan ~200–240 min; model hanya
  dapat menandai "tool mendekati akhir umur", sehingga sebagian besar false positive
  berasal dari tool aus yang belum gagal. Ini konsisten dengan kebijakan ganti-tool.
* **RNF** acak 0,1% secara desain — tidak dapat diprediksi.
* 9 baris berlabel gagal tanpa mode dan 18 baris RNF berlabel tidak gagal; label tidak
  diubah (ground truth operator) dan dampaknya dilaporkan di error analysis.
* Dataset sintetis; performa di pabrik nyata harus divalidasi ulang (shadow deployment).

## Monitoring
PSI per fitur terhadap `models/reference_data.csv` (stabil < 0,10; investigasi 0,10–0,25;
retrain > 0,25), plus pemantauan rata-rata probabilitas prediksi dan recall aktual
per bulan setelah label kegagalan tersedia.
