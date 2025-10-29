# Import library yang diperlukan
import numpy as np  # Untuk komputasi numerik
import pandas as pd  # Untuk manipulasi data
import matplotlib.pyplot as plt  # Untuk visualisasi data
import seaborn as sns  # Untuk visualisasi data yang lebih menarik
from pathlib import Path  # Untuk menangani path file
import os  # Untuk berinteraksi dengan sistem operasi

# Membuat direktori untuk menyimpan plot
plots_dir = Path('plots')
plots_dir.mkdir(exist_ok=True)  # Buat direktori jika belum ada

# Memuat dataset
df = pd.read_csv('dataset_uts.csv')
print("Memulai pembuatan visualisasi EDA...")

# 1. VISUALISASI EXPLORATORY DATA ANALYSIS (EDA)
print("\n1. Membuat ringkasan statistik...")

# 1.1 Ringkasan Statistik Deskriptif
plt.figure(figsize=(14, 8))  # Ukuran gambar
plt.axis('off')  # Menyembunyikan sumbu

# Menghitung statistik deskriptif
desc_stats = df.describe().T  # Transpose untuk tampilan yang lebih baik

# Membuat tabel statistik
table = plt.table(cellText=desc_stats.values, 
                 rowLabels=desc_stats.index,  # Nama variabel sebagai label baris
                 colLabels=desc_stats.columns,  # Statistik sebagai header kolom
                 cellLoc='center',  # Posisi teks di tengah sel
                 loc='center',  # Posisi tabel di tengah
                 colColours=['#f0f0f0']*len(desc_stats.columns))  # Warna latar header

# Mengatur gaya teks
for (i, j), cell in table.get_celld().items():
    if i == 0:  # Baris header
        cell.set_text_props(weight='bold')
    cell.set_height(0.08)  # Tinggi baris

# Menyimpan visualisasi
plt.tight_layout()
plt.savefig(plots_dir / '1.1_eda_stat_summary.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  - Ringkasan statistik disimpan sebagai '1.1_eda_stat_summary.png'")

# 1.2 Distribusi Fitur (Histogram)
print("Membuat histogram distribusi fitur...")

# Membuat subplot dengan ukuran yang sesuai
n_features = len(df.columns)
n_cols = 3  # Jumlah kolom subplot
n_rows = (n_features + n_cols - 1) // n_cols  # Hitung jumlah baris yang dibutuhkan

plt.figure(figsize=(18, 5 * n_rows))  # Sesuaikan ukuran berdasarkan jumlah baris

# Membuat histogram untuk setiap fitur
for i, column in enumerate(df.columns, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.histplot(data=df, x=column, kde=True, bins=20)  # Histogram dengan density curve
    plt.title(f'Distribusi {column}', fontsize=12)
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.7)

# Menyusun layout dan menyimpan
plt.suptitle('1.2 Distribusi Seluruh Fitur', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(plots_dir / '1.2_histograms_features.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  - Histogram distribusi fitur disimpan sebagai '1.2_histograms_features.png'")

# 1.3 Scatter Plot Setiap Fitur vs Target (Harga Properti)
print("Membuat scatter plot hubungan fitur dengan target...")

# Menghitung jumlah fitur (tidak termasuk target)
features = df.columns.drop('Harga_Properti')
num_features = len(features)
n_cols = 2  # Jumlah kolom subplot
n_rows = (num_features + 1) // n_cols  # Hitung jumlah baris yang dibutuhkan

# Membuat figure dengan ukuran yang dinamis
plt.figure(figsize=(16, 6 * n_rows))

# Membuat scatter plot untuk setiap fitur
for i, feature in enumerate(features, 1):
    plt.subplot(n_rows, n_cols, i)
    sns.regplot(data=df, x=feature, y='Harga_Properti', 
                scatter_kws={'alpha':0.5, 's':50}, 
                line_kws={'color':'red', 'linewidth':1})
    
    # Menghitung korelasi
    corr = df[feature].corr(df['Harga_Properti'])
    plt.title(f'{feature} vs Harga Properti\nKorelasi: {corr:.2f}', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

# Menyusun layout dan menyimpan
plt.suptitle('1.3 Hubungan Setiap Fitur dengan Harga Properti', y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(plots_dir / '1.3_scatter_each_vs_price.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  - Scatter plot fitur vs target disimpan sebagai '1.3_scatter_each_vs_price.png'")

# 1.4 Peta Panas Korelasi (Correlation Heatmap)
print("Membuat peta panas korelasi...")

plt.figure(figsize=(14, 12))  # Ukuran gambar yang lebih besar

# Menghitung matriks korelasi
corr = df.corr()

# Membuat mask untuk segitiga atas
mask = np.triu(np.ones_like(corr, dtype=bool))

# Membuat heatmap
heatmap = sns.heatmap(
    corr, 
    mask=mask,  # Menyembunyikan segitiga atas
    annot=True,  # Menampilkan nilai korelasi
    cmap='coolwarm',  # Warna biru-merah
    fmt=".2f",  # Format 2 angka di belakang koma
    center=0,  # Nilai tengah untuk diverging colormap
    square=True,  # Membuat sel persegi
    linewidths=0.5,  # Ketebalan garis antar sel
    cbar_kws={"shrink": 0.8},  # Ukuran colorbar
    annot_kws={"size": 10}  # Ukuran teks anotasi
)

# Mengatur judul dan tampilan
plt.title('1.4 Matriks Korelasi Antar Variabel', fontsize=16, pad=20)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# Menyimpan visualisasi
plt.tight_layout()
plt.savefig(plots_dir / '1.4_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  - Peta panas korelasi disimpan sebagai '1.4_correlation_heatmap.png'")

# 1.5 Analisis Outlier dengan Boxplot
print("Membuat boxplot untuk analisis outlier...")

# Mengatur ukuran gambar
plt.figure(figsize=(16, 8))

# Normalisasi data untuk visualisasi yang lebih baik
df_normalized = df.copy()
for column in df_normalized.columns:
    if df_normalized[column].dtype in ['int64', 'float64']:
        # Normalisasi Min-Max untuk memastikan semua fitur dalam skala yang sebanding
        df_normalized[column] = (df_normalized[column] - df_normalized[column].min()) / \
                               (df_normalized[column].max() - df_normalized[column].min())

# Plot boxplot untuk setiap fitur
sns.boxplot(data=df_normalized.melt(id_vars=['Harga_Properti']), 
           x='variable', 
           y='value',
           palette='Set2')

# Mengatur judul dan label
plt.title('1.5 Boxplot untuk Analisis Outlier (Data Dinormalisasi)', fontsize=14, pad=15)
plt.xlabel('Fitur', fontsize=12)
plt.ylabel('Nilai (Ternormalisasi)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.3)

# Menyimpan visualisasi
plt.tight_layout()
plt.savefig(plots_dir / '1.5_outliers_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"  - Boxplot analisis outlier disimpan sebagai '1.5_outliers_boxplots.png'")

# 1.6 Pemeriksaan Data Hilang (Missing Values)
print("Memeriksa data yang hilang...")

# Hitung jumlah nilai yang hilang per kolom
missing_values = df.isnull().sum()
missing_percent = (missing_values / len(df)) * 100

# Buat DataFrame untuk menampilkan hasil
missing_df = pd.DataFrame({
    'Jumlah_Missing': missing_values,
    'Persentase_Missing': missing_percent
}).sort_values('Jumlah_Missing', ascending=False)

# Tampilkan kolom yang memiliki nilai hilang
missing_df = missing_df[missing_df['Jumlah_Missing'] > 0]

if not missing_df.empty:
    print("\nDitemukan data yang hilang pada kolom:")
    print(missing_df.to_string())
    
    # Buat heatmap untuk missing values
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis', 
                yticklabels=False)  # Sembunyikan label y untuk tampilan yang lebih rapi
    
    # Atur judul dan label
    plt.title('1.6 Peta Data Hilang (Missing Values)', fontsize=14, pad=15)
    plt.xlabel('Fitur', fontsize=12)
    plt.ylabel('Indeks Baris', fontsize=12)
    
    # Simpan visualisasi
    plt.tight_layout()
    plt.savefig(plots_dir / '1.6_missing_values_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  - Peta data hilang disimpan sebagai '1.6_missing_values_heatmap.png'")
else:
    print("\nTidak ditemukan data yang hilang pada dataset.")

# Ringkasan penyelesaian
print("\n=== RINGKASAN PEMBUATAN VISUALISASI EDA ===")
print(f"Semua visualisasi EDA telah berhasil dibuat dan disimpan di direktori '{plots_dir}/'")
print("\nDaftar file yang dihasilkan:")
print("1. Ringkasan Statistik: 1.1_eda_stat_summary.png")
print("2. Distribusi Fitur: 1.2_histograms_features.png")
print("3. Hubungan Fitur-Target: 1.3_scatter_each_vs_price.png")
print("4. Matriks Korelasi: 1.4_correlation_heatmap.png")
print("5. Analisis Outlier: 1.5_outliers_boxplots.png")
if not missing_df.empty:
    print("6. Peta Data Hilang: 1.6_missing_values_heatmap.png")

print("\n=== PROSES SELESAI ===\n")
