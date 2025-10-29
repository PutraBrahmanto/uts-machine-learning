# Import library yang diperlukan
import numpy as np  # Untuk komputasi numerik
import pandas as pd  # Untuk manipulasi data
import matplotlib.pyplot as plt  # Untuk visualisasi data
import seaborn as sns  # Untuk visualisasi data yang lebih menarik
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Untuk pra-pemrosesan data
from sklearn.linear_model import LinearRegression  # Model regresi linear
from sklearn.pipeline import make_pipeline  # Untuk membuat alur kerja
from sklearn.utils import resample  # Untuk teknik bootstrap
from pathlib import Path  # Untuk menangani path file

# Membuat direktori untuk menyimpan plot
plots_dir = Path('plots')
plots_dir.mkdir(exist_ok=True)  # Buat direktori jika belum ada

# Memuat dataset
df = pd.read_csv('dataset_uts.csv')

# Mempersiapkan data
X = df.drop('Harga_Properti', axis=1)  # Fitur-fitur prediktor
y = df['Harga_Properti']  # Target yang akan diprediksi

# Membagi data menjadi data latih dan uji (70% latih, 30% uji)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 30% data untuk testing
    random_state=42  # Untuk memastikan hasil yang konsisten
)

# Melakukan penskalaan fitur untuk meningkatkan performa model
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit dan transform data latih
X_test_scaled = scaler.transform(X_test)  # Hanya transform data uji (tanpa fit)

# Melatih model akhir dengan derajat polinomial terbaik (degree=1)
degree = 1  # Menggunakan derajat 1 karena memberikan performa terbaik

# Membuat pipeline untuk model dengan polynomial features dan regresi linear
model = make_pipeline(
    PolynomialFeatures(degree=degree, include_bias=False),  # Fitur polinomial
    LinearRegression()  # Model regresi linear
)

# Melatih model dengan data latih yang telah diskalakan
model.fit(X_train_scaled, y_train)
print("Model akhir telah dilatih dengan derajat polinomial =", degree)

def get_prediction_intervals(X, y, model, n_bootstrap=1000, confidence=0.95):
    """Menghasilkan interval prediksi menggunakan teknik bootstrap.
    
    Args:
        X: Data fitur
        y: Target
        model: Model yang akan digunakan
        n_bootstrap: Jumlah iterasi bootstrap (default: 1000)
        confidence: Tingkat kepercayaan interval (default: 0.95)
        
    Returns:
        Dictionary yang berisi prediksi dan interval kepercayaannya
    """
    preds = []
    
    # Melakukan bootstrap untuk mendapatkan distribusi prediksi
    for _ in range(n_bootstrap):
        # Membuat sampel bootstrap dengan penggantian
        X_sample, y_sample = resample(X, y, random_state=_)
        
        # Membuat model baru untuk setiap iterasi bootstrap
        model_clone = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),  # Fitur polinomial
            LinearRegression()  # Model regresi linear
        )
        model_clone.fit(X_sample, y_sample)
        
        # Melakukan prediksi dengan model bootstrap
        pred = model_clone.predict(X)  # Memprediksi dengan model bootstrap
        preds.append(pred)  # Menyimpan hasil prediksi
    
    # Menghitung interval kepercayaan dari distribusi prediksi bootstrap
    alpha = (1 - confidence) / 2  # Menghitung alpha untuk interval kepercayaan dua sisi
    lower = np.percentile(preds, alpha * 100, axis=0)  # Batas bawah interval
    upper = np.percentile(preds, (1 - alpha) * 100, axis=0)  # Batas atas interval
    
    return lower, upper  # Mengembalikan batas bawah dan atas interval kepercayaan

def plot_final_predictions():
    """Membuat visualisasi prediksi akhir dengan interval kepercayaan.
    
    Fungsi ini akan:
    1. Melakukan prediksi pada data uji
    2. Menghitung interval kepercayaan menggunakan bootstrap
    3. Menyajikan visualisasi perbandingan nilai aktual dan prediksi
    """
    print("Membuat visualisasi prediksi akhir...")
    
    # Mendapatkan prediksi untuk data uji
    y_pred = model.predict(X_test_scaled)
    
    # Inisialisasi array untuk menyimpan prediksi bootstrap
    n_bootstrap = 1000  # Jumlah iterasi bootstrap
    bootstrap_preds = np.zeros((n_bootstrap, len(X_test_scaled)))
    
    # Melakukan bootstrap untuk mendapatkan distribusi prediksi
    for i in range(n_bootstrap):
        # Membuat sampel bootstrap dari data latih
        X_sample, y_sample = resample(X_train_scaled, y_train, random_state=i)
        
        # Membuat dan melatih model baru untuk setiap iterasi bootstrap
        model_clone = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),  # Fitur polinomial
            LinearRegression()  # Model regresi linear
        )
        model_clone.fit(X_sample, y_sample)
        
        # Melakukan prediksi pada data uji
        bootstrap_preds[i] = model_clone.predict(X_test_scaled)
    
    # Menghitung interval prediksi (persentil ke-2.5 dan ke-97.5)
    lower = np.percentile(bootstrap_preds, 2.5, axis=0)  # Batas bawah 95% CI
    upper = np.percentile(bootstrap_preds, 97.5, axis=0)  # Batas atas 95% CI
    
    # Memastikan semua array memiliki panjang yang sama
    min_len = min(len(y_test), len(y_pred), len(lower), len(upper))
    
    # Membuat DataFrame untuk menyimpan hasil
    results = pd.DataFrame({
        'Actual': y_test.values[:min_len],  # Nilai aktual
        'Predicted': y_pred[:min_len],  # Nilai prediksi
        'Lower_Bound': lower[:min_len],  # Batas bawah interval
        'Upper_Bound': upper[:min_len]   # Batas atas interval
    }).sort_values('Actual').reset_index(drop=True)  # Mengurutkan berdasarkan nilai aktual
    
    # Membuat plot prediksi dengan interval kepercayaan
    plt.figure(figsize=(14, 7))
    
    # Plot nilai aktual vs prediksi
    plt.scatter(range(len(results)), results['Actual'], 
               color='blue', alpha=0.5, label='Nilai Aktual')  # Titik biru = nilai aktual
    
    # Plot predicted values
    plt.plot(range(len(results)), results['Predicted'], 
            'r-', label='Predicted Values')
    
    # Plot confidence interval
    plt.fill_between(range(len(results)), 
                    results['Lower_Bound'], 
                    results['Upper_Bound'],
                    color='gray', alpha=0.2, 
                    label=f'95% Confidence Interval')
    
    plt.xlabel('Test Samples')
    plt.ylabel('Harga Properti')
    plt.title('5.1 Predicted vs Actual Values with Confidence Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / '5.1_predictions_with_intervals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

def plot_feature_importance_with_ci():
    """Menganalisis dan memvisualisasikan tingkat kepentingan fitur dengan interval kepercayaan.
    
    Fungsi ini akan:
    1. Menghitung koefisien model menggunakan bootstrap
    2. Menghitung rata-rata dan standar deviasi koefisien
    3. Memvisualisasikan tingkat kepentingan fitur dengan interval kepercayaan 95%
    """
    print("Menganalisis tingkat kepentingan fitur...")
    
    # Inisialisasi parameter bootstrap
    n_bootstrap = 1000  # Jumlah iterasi bootstrap
    coefs = []  # Untuk menyimpan koefisien dari setiap iterasi
    
    # Melakukan bootstrap untuk mendapatkan distribusi koefisien
    for _ in range(n_bootstrap):
        # Membuat sampel bootstrap dari data latih
        X_sample, y_sample = resample(X_train_scaled, y_train, random_state=_)
        
        # Membuat dan melatih model dengan sampel bootstrap
        model_clone = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),  # Fitur polinomial
            LinearRegression()  # Model regresi linear
        )
        model_clone.fit(X_sample, y_sample)
        
        # Menyimpan koefisien model
        coefs.append(model_clone.named_steps['linearregression'].coef_)
    
    # Menghitung statistik dari distribusi bootstrap
    coefs = np.array(coefs)  # Mengubah ke array numpy untuk perhitungan yang lebih mudah
    mean_coefs = np.mean(coefs, axis=0)  # Rata-rata koefisien
    std_coefs = np.std(coefs, axis=0)    # Standar deviasi koefisien
    
    # Mendapatkan nama-nama fitur
    feature_names = X.columns
    
    # Membuat DataFrame untuk visualisasi
    importance_df = pd.DataFrame({
        'Feature': feature_names,  # Nama fitur
        'Importance': mean_coefs,  # Rata-rata nilai koefisien
        'Std_Error': std_coefs    # Standar error koefisien
    }).sort_values('Importance', key=abs, ascending=False)  # Mengurutkan berdasarkan nilai absolut koefisien
    
    # Membuat visualisasi tingkat kepentingan fitur
    plt.figure(figsize=(12, 6))
    
    # Membuat plot batang horizontal dengan interval kepercayaan
    plt.errorbar(importance_df['Importance'],  # Nilai koefisien
                range(len(importance_df)),  # Posisi y untuk setiap fitur
                xerr=1.96 * importance_df['Std_Error'],  # Interval kepercayaan 95%
                fmt='o',  # Format marker
                capsize=5,  # Ukuran cap pada ujung error bar
                color='#1f77b4')  # Warna biru
    
    # Mengatur label sumbu y dengan nama fitur
    plt.yticks(range(len(importance_df)), importance_df['Feature'])
    plt.xlabel('Nilai Koefisien')  # Label sumbu x
    plt.title('5.2 Tingkat Kepentingan Fitur dengan Interval Kepercayaan 95%')
    plt.axvline(0, color='r', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / '5.2_feature_importance_ci.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualisasi tingkat kepentingan fitur disimpan sebagai '{plots_dir}/5.2_feature_importance_ci.png'")
    return importance_df

def plot_predictions_on_new_samples():
    """Membuat prediksi untuk sampel baru dan memvisualisasikan hasilnya.
    
    Fungsi ini akan:
    1. Mengambil sampel acak dari data uji
    2. Memprediksi nilai properti untuk sampel tersebut
    3. Menghitung interval kepercayaan untuk prediksi
    4. Menampilkan perbandingan nilai aktual dan prediksi
    """
    print("Membuat prediksi untuk sampel baru...")
    
    # Mengambil sampel acak dari data uji (untuk simulasi data baru)
    n_samples = 20  # Jumlah sampel yang akan diambil
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)  # Indeks acak tanpa pengembalian
    X_new = X_test.iloc[sample_indices]  # Mengambil fitur untuk sampel baru
    y_new = y_test.iloc[sample_indices]  # Nilai aktual untuk sampel baru
    
    # Melakukan penskalaan pada sampel baru
    X_new_scaled = scaler.transform(X_new)  # Menggunakan scaler yang sudah di-fit sebelumnya
    
    # Melakukan prediksi untuk sampel baru
    y_pred = model.predict(X_new_scaled)  # Memprediksi dengan model yang sudah dilatih
    
    # Menghitung interval prediksi menggunakan bootstrap
    n_bootstrap = 1000  # Jumlah iterasi bootstrap
    bootstrap_preds = np.zeros((n_bootstrap, len(X_new_scaled)))  # Menyimpan prediksi bootstrap
    
    for i in range(n_bootstrap):
        # Membuat sampel bootstrap dari data latih
        X_sample, y_sample = resample(X_train_scaled, y_train, random_state=i)
        
        # Membuat dan melatih model baru untuk setiap iterasi bootstrap
        model_clone = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),  # Fitur polinomial
            LinearRegression()  # Model regresi linear
        )
        model_clone.fit(X_sample, y_sample)  # Melatih model dengan sampel bootstrap
        
        # Menyimpan prediksi untuk sampel baru
        bootstrap_preds[i] = model_clone.predict(X_new_scaled)
    
    # Menghitung interval kepercayaan 95% dari distribusi bootstrap
    lower = np.percentile(bootstrap_preds, 2.5, axis=0)  # Batas bawah (percentil ke-2.5)
    upper = np.percentile(bootstrap_preds, 97.5, axis=0)  # Batas atas (percentil ke-97.5)
    
    # Membuat DataFrame untuk menyimpan hasil prediksi
    results = pd.DataFrame({
        'Actual': y_new.values,  # Nilai aktual
        'Predicted': y_pred,     # Nilai prediksi
        'Lower_Bound': lower,    # Batas bawah interval kepercayaan
        'Upper_Bound': upper     # Batas atas interval kepercayaan
    }).sort_values('Actual').reset_index(drop=True)  # Mengurutkan berdasarkan nilai aktual
    
    # Menambahkan nomor sampel untuk keperluan plotting
    results['Sample'] = range(1, len(results) + 1)
    
    # Membuat visualisasi perbandingan nilai aktual dan prediksi
    plt.figure(figsize=(14, 7))
    
    # Plot nilai aktual (batang biru)
    plt.bar(results['Sample'] - 0.2, results['Actual'], 
           width=0.4, label='Nilai Aktual', color='blue', alpha=0.7)
    
    # Plot nilai prediksi (batang oranye)
    plt.bar(results['Sample'] + 0.2, results['Predicted'], 
           width=0.4, label='Nilai Prediksi', color='orange', alpha=0.7)
    
    # Menambahkan error bar untuk interval kepercayaan
    plt.errorbar(results['Sample'], results['Predicted'],
                yerr=[results['Predicted'] - results['Lower_Bound'], 
                      results['Upper_Bound'] - results['Predicted']],
                fmt='none', color='black', capsize=5, label='95% CI')
    
    # Mengatur label dan judul plot
    plt.xlabel('Sampel')
    plt.ylabel('Harga Properti (juta Rupiah)')
    plt.title('5.3 Perbandingan Nilai Aktual dan Prediksi untuk Sampel Baru')
    plt.xticks(results['Sample'])  # Menampilkan nomor sampel pada sumbu x
    plt.legend()  # Menampilkan legenda
    plt.grid(True, alpha=0.3)  # Menambahkan grid dengan transparansi 30%
    
    # Menyimpan plot
    plt.tight_layout()  # Menyesuaikan layout
    plt.savefig(plots_dir / '5.3_predictions_on_new_samples.png', 
               dpi=300,  # Resolusi tinggi
               bbox_inches='tight')  # Memastikan tidak ada bagian yang terpotong
    plt.close()  # Menutup plot untuk menghemat memori
    
    print(f"Visualisasi prediksi untuk sampel baru disimpan sebagai '{plots_dir}/5.3_predictions_on_new_samples.png'")
    return results  # Mengembalikan DataFrame berisi hasil prediksi

if __name__ == "__main__":
    """Fungsi utama untuk menjalankan semua visualisasi prediksi akhir.
    
    Visualisasi yang akan dihasilkan:
    1. Plot prediksi akhir dengan interval kepercayaan
    2. Analisis tingkat kepentingan fitur
    3. Prediksi untuk sampel baru
    """
    print("=== MEMULAI PEMBUATAN VISUALISASI PREDIKSI AKHIR ===\n")
    
    # Menjalankan semua fungsi visualisasi
    print("1. Membuat visualisasi prediksi akhir...")
    pred_results = plot_final_predictions()  # 5.1
    
    print("\n2. Menganalisis tingkat kepentingan fitur...")
    importance_df = plot_feature_importance_with_ci()  # 5.2
    
    print("\n3. Membuat prediksi untuk sampel baru...")
    sample_results = plot_predictions_on_new_samples()  # 5.3
    
    # Menyimpan hasil prediksi ke file CSV
    print("\nMenyimpan hasil prediksi ke file CSV...")
    pred_results.to_csv(plots_dir / 'final_predictions.csv', index=False)
    importance_df.to_csv(plots_dir / 'feature_importance.csv', index=False)
    sample_results.to_csv(plots_dir / 'new_samples_predictions.csv', index=False)
    
    # Menampilkan ringkasan
    print("\n=== SEMUA VISUALISASI TELAH SELESAI DIBUAT ===")
    print(f"Semua file visualisasi telah disimpan di direktori '{plots_dir}/'")
    
    print("\nDaftar file yang dihasilkan:")
    print("\nVisualisasi:")
    print(f"- {plots_dir}/5.1_predictions_with_intervals.png")
    print(f"- {plots_dir}/5.2_feature_importance_with_ci.png")
    print(f"- {plots_dir}/5.3_predictions_on_new_samples.png")
    
    print("\nFile Data:")
    print(f"- {plots_dir}/final_predictions.csv")
    print(f"- {plots_dir}/feature_importance.csv")
    print(f"- {plots_dir}/new_samples_predictions.csv")
