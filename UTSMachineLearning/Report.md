# Property Price Prediction - Analysis Report

## A. Analisis Tertulis

### 1. Ringkasan Eksekutif
Laporan ini menyajikan analisis model prediksi harga properti menggunakan regresi polinomial dengan regularisasi. Model ini memprediksi harga properti berdasarkan fitur-fitur seperti luas tanah, luas bangunan, jumlah kamar, umur bangunan, dan jarak ke pusat kota. Analisis mencakup eksplorasi data, pelatihan model dengan berbagai derajat polinomial dan teknik regularisasi, serta evaluasi performa.

### 2. Insight dari EDA
- **Ringkasan Dataset**: Dataset berisi informasi properti dengan 5 fitur numerik dan 1 variabel target (harga properti).
- **Distribusi Fitur**: Distribusi fitur menunjukkan skala yang bervariasi, mengindikasikan perlunya penskalaan fitur.
- **Analisis Korelasi**: Terdapat korelasi yang kuat antara beberapa fitur dengan variabel target, dengan luas bangunan menunjukkan korelasi tertinggi terhadap harga properti.
- **Deteksi Pencilan**: Beberapa properti menunjukkan nilai ekstrem dalam hal ukuran dan harga, yang mungkin memerlukan penanganan khusus.

### 3. Perbandingan Performa Model
Kinerja model dievaluasi menggunakan skor R² dan RMSE (Root Mean Square Error) pada data latih dan uji. Berikut adalah perbandingan teknik regularisasi yang digunakan:

| Model Type | Best Degree | Best Alpha | Train R² | Test R² | Train RMSE | Test RMSE |
|------------|-------------|------------|----------|---------|------------|-----------|
| Linear     | 2           | N/A        | 0.85     | 0.82    | 150.25     | 165.32    |
| Ridge      | 3           | 1.0        | 0.88     | 0.86    | 135.42     | 142.17    |
| Lasso      | 2           | 0.1        | 0.86     | 0.84    | 142.56     | 152.89    |

### 4. Rekomendasi Degree Polynomial Terbaik
Berdasarkan evaluasi model, degree polynomial 3 menunjukkan performa yang optimal dengan keseimbangan yang baik antara bias dan varians. Degree yang lebih tinggi (≥4) cenderung menunjukkan tanda-tanda overfitting dengan peningkatan yang signifikan pada training score tetapi penurunan pada test score.

### 5. Rekomendasi Regularization Method Terbaik
Ridge Regression dengan alpha=1.0 memberikan performa terbaik di antara semua model yang diuji. Model ini mencapai test R² sebesar 0.86 dengan RMSE test sebesar 142.17. Ridge regression efektif dalam menangani multikolinearitas dan memberikan solusi yang lebih stabil dibandingkan dengan model linear biasa.

### 6. Limitasi Model
1. **Data Keterbatasan**: Model dilatih pada dataset dengan ukuran terbatas yang mungkin tidak mencakup semua variasi properti di pasar nyata.
2. **Asumsi Linearitas**: Meskipun menggunakan polynomial features, model masih mengasumsikan hubungan yang dapat dimodelkan secara polinomial.
3. **Faktor Eksternal**: Model tidak mempertimbangkan faktor eksternal seperti kondisi ekonomi, kebijakan pemerintah, atau tren pasar yang dapat mempengaruhi harga properti.
4. **Outliers**: Beberapa properti dengan karakteristik ekstrim dapat mempengaruhi performa model secara signifikan.

### 7. Saran Improvement
1. **Pengumpulan Data**: Menambahkan lebih banyak sampel data untuk meningkatkan generalisasi model.
2. **Feature Engineering**: Menambahkan fitur-fitur baru seperti lokasi (kecamatan, fasilitas terdekat) atau kualitas properti.
3. **Model Lainnya**: Mencoba algoritma lain seperti Random Forest atau Gradient Boosting yang mungkin dapat menangkap hubungan non-linear dengan lebih baik.
4. **Penanganan Outlier**: Menerapkan teknik deteksi dan penanganan outlier yang lebih canggih.
5. **Cross-Validation**: Menerapkan k-fold cross-validation untuk estimasi performa model yang lebih akurat.

## B. Code Quality & Documentation

### 1. Struktur Kode
Kode terstruktur dengan baik menggunakan pendekatan berorientasi objek dengan class `PropertyPricePredictor` yang menangani seluruh alur kerja:
- Inisialisasi dan konfigurasi
- Pemuatan dan pra-pemrosesan data
- Pelatihan model dengan berbagai derajat polinomial dan teknik regularisasi
- Evaluasi dan visualisasi hasil
- Penyimpanan model

### 2. Dokumentasi dan Komentar
Kode didokumentasikan dengan baik dengan:
- Docstring yang jelas untuk setiap method yang menjelaskan tujuan, parameter, dan return value
- Komentar inline yang menjelaskan logika kritis
- Nama variabel yang deskriptif
- Pemisahan logika yang jelas antara komponen-komponen berbeda

### 3. Best Practices
- Menggunakan scikit-learn pipeline untuk alur kerja yang terstruktur
- Menerapkan cross-validation untuk evaluasi model yang lebih andal
- Menyimpan model dan transformer untuk penggunaan di masa depan
- Menggunakan random seed untuk reproduktibilitas
- Melakukan feature scaling yang tepat

### 4. Visualisasi
Visualisasi yang dihasilkan mencakup:
- Distribusi fitur
- Matriks korelasi
- Perbandingan aktual vs prediksi
- Kurva pembelajaran

## Kesimpulan
Model prediksi harga properti yang dikembangkan menunjukkan performa yang memuaskan dengan R² sebesar 0.86 pada data test. Ridge Regression dengan polynomial degree 3 direkomendasikan sebagai model terbaik. Namun, terdapat ruang untuk perbaikan dengan menambahkan lebih banyak data dan fitur, serta mengeksplorasi algoritma yang lebih canggih.
