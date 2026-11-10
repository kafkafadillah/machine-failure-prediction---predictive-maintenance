# CRISP-DM Tahap 1 — Business Understanding

## 1. Latar belakang

Sebuah lini produksi milling menjalankan ribuan siklus mesin. Kegagalan mesin yang
tidak terduga menyebabkan *unplanned downtime*, kerusakan tool/benda kerja, dan
perbaikan darurat yang mahal. Strategi saat ini ada dua ekstrem:

| Strategi | Masalah |
|---|---|
| **Reaktif** (perbaiki setelah rusak) | Setiap kegagalan = downtime + biaya darurat |
| **Preventif terjadwal** (inspeksi semua unit) | Biaya inspeksi besar; >96% inspeksi ternyata tidak perlu |

**Predictive maintenance** menempatkan inspeksi hanya pada unit yang datanya
menunjukkan risiko tinggi.

## 2. Tujuan bisnis

1. Mendeteksi unit berisiko gagal **sebelum** kegagalan terjadi.
2. Meminimalkan **total biaya** = biaya kegagalan yang lolos + biaya inspeksi.
3. Memberi **alasan yang dapat ditindaklanjuti** (mode kegagalan yang paling mungkin)
   agar teknisi tahu apa yang harus diperiksa.

## 3. Tujuan data mining

| Tujuan bisnis | Formulasi ML |
|---|---|
| Deteksi dini | Klasifikasi biner `machine_failure` dari 6 pembacaan sensor/proses |
| Minimalkan biaya | Pemilihan threshold berbasis matriks biaya (bukan 0,5 default) |
| Prioritisasi kapasitas tim | Probabilitas terkalibrasi → level risiko LOW/MEDIUM/HIGH |
| Alasan yang dapat ditindaklanjuti | Klasifikasi multi-label mode kegagalan (TWF/HDF/PWF/OSF) + explainability |

## 4. Asumsi biaya (dapat dikonfigurasi di `configs/config.yaml`)

| Kejadian | Biaya (satuan) | Keterangan |
|---|---|---|
| Kegagalan tidak terdeteksi (FN) | 10.000 | downtime + perbaikan darurat + scrap |
| Setiap alarm → inspeksi (TP & FP) | 500 | jam teknisi + jeda produksi singkat |

Rasio 20 : 1 berarti, untuk probabilitas yang terkalibrasi, threshold Bayes-optimal
secara teori ≈ 500 / 10.000 = **0,05** — jauh di bawah 0,5. Threshold akhir dipilih
secara empiris pada prediksi out-of-fold.

## 5. Kriteria keberhasilan (deployment gate)

Model hanya dipaketkan jika **semua** kriteria terpenuhi pada test set yang dikunci:

1. PR-AUC ≥ 0,80 (metrik utama — data sangat imbalanced, accuracy menyesatkan)
2. Recall ≥ 80% (permintaan tim maintenance)
3. Total biaya < 50% strategi reaktif
4. Total biaya < strategi inspeksi-semua

## 6. Sumber daya, batasan, dan risiko

* **Data**: AI4I 2020 Predictive Maintenance Dataset (UCI / Kaggle), 10.000 baris,
  sintetis tetapi dimodelkan dari mesin nyata. Satu baris = satu snapshot, bukan
  deret waktu → tidak dapat memodelkan *remaining useful life*.
* **Leakage**: kolom TWF/HDF/PWF/OSF/RNF hanya diketahui **setelah** kegagalan —
  dilarang menjadi fitur.
* **Kegagalan acak (RNF)** menurut desain tidak dapat diprediksi; recall 100% mustahil.
* **Risiko etis/operasional**: false negative bersifat keselamatan; model adalah
  alat bantu keputusan, bukan pengganti prosedur keselamatan.

## 7. Rencana proyek

| Tahap | Artefak |
|---|---|
| 2. Data Understanding | `pipelines/02_data_understanding.py` → audit kualitas, EDA, uji statistik |
| 3. Data Preparation | `pipelines/03_data_preparation.py` → validasi skema, split terkunci, manifest |
| 4. Modeling | `pipelines/04_modeling.py` → CV berulang, ablation, tuning, kalibrasi, threshold |
| 5. Evaluation | `pipelines/05_evaluation.py` → test + CI, biaya, explainability, error analysis, gate |
| 6. Deployment | `pipelines/06_deployment.py`, `app/` → bundle, REST API, CLI, monitoring drift |
