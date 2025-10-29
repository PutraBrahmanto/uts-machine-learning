# Prediksi Harga Properti dengan Regresi Polinomial

Proyek ini mengimplementasikan model Regresi Polinomial untuk memprediksi harga properti berdasarkan berbagai fitur. Implementasi meliputi eksplorasi data, pra-pemrosesan, pelatihan model dengan berbagai derajat polinomial dan teknik regularisasi, serta evaluasi model yang komprehensif.

## Prasyarat

Sebelum memulai, pastikan Anda telah menginstal:
- Python 3.7 atau lebih baru
- pip (manajer paket Python)

## Instalasi

1. **Clone repositori ini**
   ```bash
   git clone [URL_REPOSITORY]
   cd MachineLearning
   ```

  ## Daftar Dependensi

Berikut adalah daftar paket Python yang diperlukan untuk menjalankan proyek ini:

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
joblib>=1.0.0
jupyter>=1.0.0  # Opsional, untuk menjalankan notebook
```

### Cara Menginstal

1. **Menggunakan pip** (cara termudah):
   ```bash
   pip install -r requirements.txt
   ```

2. **Menginstal secara manual**:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn joblib jupyter
   ```

### Versi Python

Proyek ini dikembangkan dan diuji menggunakan Python 3.8. Disarankan untuk menggunakan Python versi 3.7 atau lebih baru.

### Catatan Instalasi

- Pastikan pip sudah diperbarui ke versi terbaru:
  ```bash
  python -m pip install --upgrade pip
  ```

- Jika terjadi masalah kompatibilitas, Anda dapat mencoba membuat virtual environment baru:
  ```bash
  python -m venv myenv
  source myenv/bin/activate  # Linux/MacOS
  .\myenv\Scripts\activate  # Windows
  pip install -r requirements.txt
  ```

- Untuk pengguna Anaconda, Anda dapat membuat environment baru dengan:
  ```bash
  conda create -n properti python=3.8
  conda activate properti
  pip install -r requirements.txt
  ```

## Menjalankan Kode

1. **Menjalankan analisis EDA (Exploratory Data Analysis)**
   ```bash
   python generate_visualizations.py
   ```
   Akan menghasilkan file visualisasi di folder `plots/`

2. **Menjalankan pelatihan dan evaluasi model**
   ```bash
   python model_visualizations.py
   ```
   Akan menghasilkan model terbaik dan visualisasi performa model

3. **Membuat prediksi**
   ```bash
   python final_predictions.py
   ```
   Akan membuat prediksi menggunakan model terlatih

## Struktur Proyek

```
MachineLearning/

|── dataset_uts.csv     # Dataset utama
├── models/                 # Model yang sudah dilatih
│   ├── best_model.pkl      # Model terbaik
│   ├── poly_transformer.pkl # Transformer untuk fitur polinomial
│   └── scaler.pkl          # Scaler untuk normalisasi data
├── plots/                  # File visualisasi
│   ├── 1.1_eda_stat_summary.png
│   ├── 1.2_histograms_features.png
│   ├── 1.3_scatter_each_vs_price.png
│   ├── 1.4_correlation_heatmap.png
│   └── ...
├── additional_visualizations.py  # Visualisasi tambahan
├── final_predictions.py    # Script untuk prediksi
├── generate_visualizations.py  # Script untuk EDA
├── model_visualizations.py # Script untuk visualisasi model
├── property_price_prediction.py # Script utama
└── requirements.txt        # Daftar dependensi
|__ Report.md               # Laporan
```

## File yang Dihasilkan

### 1. Analisis Data Eksplorasi (EDA)
- `1.1_eda_stat_summary.png`: Ringkasan statistik deskriptif
- `1.2_histograms_features.png`: Distribusi setiap fitur
- `1.3_scatter_each_vs_price.png`: Hubungan setiap fitur dengan harga properti
- `1.4_correlation_heatmap.png`: Peta panas korelasi antar variabel
- `1.5_outliers_boxplots.png`: Analisis outlier menggunakan boxplot
- `correlation_matrix.png`: Matriks korelasi antar fitur
- `feature_distributions.png`: Visualisasi distribusi seluruh fitur

### 2. Analisis Fitur Polinomial
- `2.1_poly_feature_counts.png`: Jumlah fitur polinomial untuk setiap derajat

### 3. Evaluasi Model
- `3.1_metrics_table_train_test.png`: Tabel metrik evaluasi model
- `3.3_r2_vs_degree.png`: Perbandingan R² untuk setiap derajat polinomial
- `3.3_r2_vs_degree_comparison.png`: Perbandingan R² antar model
- `3.4_predicted_vs_actual.png`: Perbandingan nilai prediksi vs aktual
- `3.5_residual_plot.png`: Plot residual untuk analisis error
- `3.6_polynomial_curves.png`: Kurva polinomial yang dihasilkan
- `actual_vs_predicted.png`: Visualisasi prediksi vs nilai sebenarnya

### 4. Analisis Regularisasi
- `4.1_r2_vs_alpha_ridge_lasso.png`: Pengaruh alpha terhadap performa Ridge dan Lasso
- `4.3_feature_importance_bar.png`: Kepentingan setiap fitur dalam prediksi

### 5. Prediksi dan Analisis Lanjutan
- `5.1_predictions_with_intervals.png`: Prediksi dengan interval kepercayaan
- `5.2_feature_importance_ci.png`: Kepentingan fitur dengan interval kepercayaan
- `5.3_new_samples_predictions.png`: Prediksi untuk sampel baru

## Kustomisasi

Anda dapat menyesuaikan parameter berikut dalam skrip:
- **Derajat polinomial maksimum** (default: 5)
- **Nilai alpha untuk regularisasi** (default: [0.1, 1, 10])
- **Rasio pembagian data latih-uji** (default: 70-30)
- **Random seed** untuk reproduktibilitas (default: 42)
- **Parameter visualisasi** (ukuran gambar, warna, dll)

## Pemilihan Model

Model terbaik dipilih berdasarkan nilai R² tertinggi pada data uji. Skrip akan menampilkan tabel perbandingan semua model dan metrik kinerjanya.

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).

This project is open source and available under the MIT License.
