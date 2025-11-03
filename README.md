# Prediksi Harga Properti dengan Regresi Polinomial

Proyek ini mengimplementasikan model prediksi harga properti menggunakan berbagai teknik regresi polinomial dan regularisasi. Dataset yang digunakan adalah data sintetis properti dengan fitur-fitur seperti luas tanah, luas bangunan, jumlah kamar, umur bangunan, dan jarak ke pusat kota.

## Daftar Isi
1. [Instalasi Dependensi](#instalasi-dependensi)
2. [Menjalankan Kode](#menjalankan-kode)
3. [Struktur Project](#struktur-project)
4. [Model yang Tersedia](#model-yang-tersedia)
5. [Hasil](#hasil)

## Instalasi Dependensi

Pastikan Anda memiliki Python 3.8+ terinstal. Kemudian, instal dependensi yang diperlukan dengan perintah berikut:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## Menjalankan Kode

1. Clone repository ini:
```bash
git clone [URL_REPOSITORY]
cd UTS_MachineLearning_PutraBrahmanto
```

2. Buka Jupyter Notebook:
```bash
jupyter notebook UTS_MachineLearning_PutraBrahmanto.ipynb
```

3. Jalankan notebook secara berurutan dari sel pertama hingga terakhir.

## Struktur Project

```
UTS_MachineLearning_PutraBrahmanto/
├── UTS_MachineLearning_PutraBrahmanto.ipynb  # Notebook utama
├── standard_scaler.pkl                      # Model scaler (StandardScaler)
├── minmax_scaler.pkl                       # Model scaler (MinMaxScaler)
└── README.md                               # File ini
```

## Model yang Tersedia

Proyek ini mengimplementasikan beberapa model regresi dengan berbagai tingkat kompleksitas:

1. **Linear Regression**
   - Regresi linier standar tanpa regularisasi
   - Cocok untuk baseline model

2. **Ridge Regression**
   - Regresi dengan regularisasi L2
   - Mencegah overfitting dengan menambahkan penalty pada koefisien besar
   - Nilai alpha yang diuji: 0.1, 1, 10

3. **Lasso Regression**
   - Regresi dengan regularisasi L1
   - Dapat melakukan feature selection dengan membuat beberapa koefisien menjadi nol
   - Nilai alpha yang diuji: 0.1, 1, 10

Setiap model diuji dengan derajat polinomial 1 sampai 5 untuk menangkap hubungan non-linear dalam data.

## Hasil

Model dievaluasi menggunakan metrik berikut:
- **MSE (Mean Squared Error)**: Mengukur rata-rata kuadrat kesalahan prediksi
- **R² Score**: Mengukur seberapa baik model menjelaskan variasi dalam data (0-1, semakin mendekati 1 semakin baik)

### Ringkasan Performa Model

| Model | Degree | Alpha | Test MSE | Test R² |
|-------|--------|-------|----------|---------|
| Linear Regression | 2 | - | 1,234,567 | 0.85 |
| Ridge Regression | 2 | 1 | 1,234,567 | 0.85 |
| Lasso Regression | 2 | 0.1 | 1,234,567 | 0.85 |

*Catatan: Nilai di atas adalah contoh. Silakan lihat output notebook untuk hasil aktual.*

### Visualisasi

Notebook ini menyertakan berbagai visualisasi untuk menganalisis data dan hasil model, termasuk:
- Distribusi fitur
- Matriks korelasi
- Scatter plot fitur vs target
- Perbandingan performa model

## Kontribusi

1. Fork repository
2. Buat branch fitur baru (`git checkout -b fitur/namafitur`)
3. Commit perubahan (`git commit -m 'Menambahkan fitur'`)
4. Push ke branch (`git push origin fitur/namafitur`)
5. Buat Pull Request

## Lisensi

Proyek ini dilisensikan di bawah [MIT License](LICENSE).
