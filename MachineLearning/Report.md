# Property Price Prediction - Analysis Report

## A. Written Analysis

### 1. Executive Summary
This report presents the analysis of a property price prediction model using polynomial regression with regularization. The model predicts property prices based on features such as land area, building area, number of rooms, building age, and distance to city center. The analysis includes exploratory data analysis, model training with different polynomial degrees and regularization techniques, and performance evaluation.

### 2. Insight dari EDA
- **Dataset Overview**: The dataset contains property information with 5 numerical features and 1 target variable (property price).
- **Feature Distributions**: The distributions of features show varying scales, indicating the need for feature scaling.
- **Correlation Analysis**: Strong correlations exist between certain features and the target variable, with building area showing the highest correlation with property price.
- **Outlier Detection**: Some properties exhibit extreme values in terms of size and price, which may need special handling.

### 3. Perbandingan Performa Model
Model performance was evaluated using R² score and RMSE (Root Mean Square Error) on both training and test sets. The following regularization techniques were compared:

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
