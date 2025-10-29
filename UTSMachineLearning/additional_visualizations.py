# Import library yang diperlukan
import numpy as np  # Untuk komputasi numerik
import pandas as pd  # Untuk manipulasi data
import matplotlib.pyplot as plt  # Untuk visualisasi data
import seaborn as sns  # Untuk visualisasi data yang lebih menarik
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Untuk pra-pemrosesan data
from sklearn.linear_model import LinearRegression  # Model regresi linear
from sklearn.metrics import r2_score, mean_squared_error  # Metrik evaluasi
from sklearn.pipeline import make_pipeline  # Untuk membuat alur kerja
from pathlib import Path  # Untuk menangani path file
import os  # Untuk berinteraksi dengan sistem operasi

# Membuat direktori untuk menyimpan plot
plots_dir = Path('plots')
plots_dir.mkdir(exist_ok=True)  # Buat direktori jika belum ada

# Memuat dataset
print("Memuat dataset...")
df = pd.read_csv('dataset_uts.csv')

# Mempersiapkan data
# X berisi fitur-fitur prediktor
y = df['Harga_Properti']  # Target yang akan diprediksi
X = df.drop('Harga_Properti', axis=1)  # Semua kolom kecuali Harga_Properti

# Membagi data menjadi data latih dan data uji (70% latih, 30% uji)
print("Membagi data menjadi data latih dan uji...")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,  # 30% data untuk testing
    random_state=42  # Untuk memastikan hasil yang konsisten
)

# Melakukan penskalaan fitur untuk meningkatkan performa model
print("Melakukan penskalaan fitur...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit dan transform data latih
X_test_scaled = scaler.transform(X_test)  # Hanya transform data uji (tanpa fit)

# 3.4 Plot Nilai Prediksi vs Aktual
def plot_predicted_vs_actual():
    """Membuat plot perbandingan antara nilai aktual dan prediksi.
    
    Plot ini membantu memvisualisasikan seberapa baik prediksi model
    mendekati nilai aktual. Garis merah menunjukkan prediksi sempurna.
    """
    print("Membuat plot nilai prediksi vs aktual...")
    
    # Membuat model regresi linear dengan fitur polinomial derajat 1
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),  # Fitur polinomial
        LinearRegression()  # Model regresi linear
    )
    
    # Melatih model dengan data latih
    model.fit(X_train_scaled, y_train)
    
    # Melakukan prediksi pada data uji
    y_pred = model.predict(X_test_scaled)
    
    # Membuat plot
    plt.figure(figsize=(10, 8))
    
    # Plot titik-titik prediksi vs aktual
    plt.scatter(y_test, y_pred, alpha=0.6, label='Data Uji')
    
    # Garis diagonal merah untuk prediksi sempurna
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2, label='Prediksi Sempurna')
    
    # Menambahkan label dan judul
    plt.xlabel('Nilai Aktual (juta Rupiah)')
    plt.ylabel('Nilai Prediksi (juta Rupiah)')
    plt.title('3.4 Perbandingan Nilai Aktual vs Prediksi (Data Uji)')
    plt.legend()
    plt.grid(True)
    
    # Menghitung dan menambahkan metrik evaluasi ke plot
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.2f}', 
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(plots_dir / '3.4_predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()

# 3.5 Plot Residual
def plot_residuals():
    """Membuat plot residual untuk mengevaluasi model.
    
    Residual adalah selisih antara nilai aktual dan prediksi.
    Plot ini membantu mendeteksi:
    - Heteroskedastisitas (pola tertentu dalam residual)
    - Outlier
    - Asumsi normalitas residual
    """
    print("Membuat plot residual...")
    
    # Membuat model yang sama seperti sebelumnya
    model = make_pipeline(
        PolynomialFeatures(degree=1, include_bias=False),  # Fitur polinomial derajat 1
        LinearRegression()  # Model regresi linear
    )
    
    # Melatih model
    model.fit(X_train_scaled, y_train)
    
    # Memprediksi dan menghitung residual
    y_pred = model.predict(X_test_scaled)
    residuals = y_test - y_pred  # Menghitung residual (aktual - prediksi)
    
    # Membuat plot
    plt.figure(figsize=(10, 6))
    
    # Plot residual vs prediksi
    plt.scatter(y_pred, residuals, alpha=0.6, label='Residual')
    
    # Garis horizontal di y=0 (residual sempurna)
    plt.axhline(y=0, color='r', linestyle='--', label='Residual = 0')
    
    # Menambahkan label dan judul
    plt.xlabel('Nilai Prediksi (juta Rupiah)')
    plt.ylabel('Residual (Aktual - Prediksi)')
    plt.title('3.5 Plot Residual (Data Uji)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Menambahkan garis standar deviasi (2σ)
    std_residuals = residuals.std()  # Standar deviasi residual
    plt.axhline(y=2*std_residuals, color='g', linestyle=':', alpha=0.6)
    plt.axhline(y=-2*std_residuals, color='g', linestyle=':', alpha=0.6, 
                label='±2σ')
    
    plt.legend()
    plt.tight_layout()
    
    # Menyimpan plot
    plt.savefig(plots_dir / '3.5_residual_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot residual disimpan sebagai '{plots_dir}/3.5_residual_plot.png'")

# 3.6 Polynomial Curve for Important Features
def plot_polynomial_curves():
    """Membuat visualisasi kurva polinomial untuk fitur-fitur penting.
    
    Fungsi ini:
    1. Memilih 2 fitur dengan korelasi tertinggi terhadap target
    2. Memvisualisasikan hubungan non-linear antara fitur dan target
    3. Menampilkan kurva polinomial dengan berbagai derajat
    """
    print("Membuat visualisasi kurva polinomial untuk fitur penting...")
    
    # Memilih 2 fitur dengan korelasi absolut tertinggi terhadap target
    corr_with_target = df.corr()['Harga_Properti'].abs().sort_values(ascending=False)
    top_features = corr_with_target.index[1:3]  # Pilih 2 fitur teratas (indeks 0 adalah target itu sendiri)
    
    # Membuat figure dengan 2 subplot bersebelahan
    plt.figure(figsize=(15, 6))
    
    # Loop melalui setiap fitur penting
    for i, feature in enumerate(top_features, 1):
        # Menyiapkan data untuk fitur saat ini
        X_feature = X[feature].values.reshape(-1, 1)  # Mengubah ke bentuk 2D untuk kompatibilitas
        y_target = y.values
        
        # Daftar derajat polinomial yang akan dicoba
        degrees = [1, 2, 3]  # Linear, kuadratik, dan kubik
        
        # Membuat subplot untuk setiap fitur
        plt.subplot(1, 2, i)
        
        # Plot data asli (titik-titik biru)
        plt.scatter(X_feature, y_target, color='navy', s=30, marker='o', 
                   label="Data Asli", alpha=0.5)
        
        # Membuat range nilai untuk plot kurva
        X_test_plot = np.linspace(X_feature.min(), X_feature.max(), 300).reshape(-1, 1)
        
        # Mencoba setiap derajat polinomial
        for degree in degrees:
            # Membuat fitur polinomial
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_poly = poly.fit_transform(X_feature)
            
            # Melatih model regresi linear dengan fitur polinomial
            model = LinearRegression()
            model.fit(X_poly, y_target)
            
            # Memprediksi untuk plot kurva
            X_test_poly = poly.transform(X_test_plot)
            y_pred = model.predict(X_test_poly)
            
            # Plot kurva polinomial
            plt.plot(X_test_plot, y_pred, linewidth=2, 
                    label=f'Derajat {degree}')
        
        # Menambahkan label dan judul
        plt.xlabel(feature)
        plt.ylabel('Harga Properti (juta Rupiah)')
        plt.title(f'3.6 Kurva Polinomial untuk {feature}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Menyimpan plot
    plt.tight_layout()
    plt.savefig(plots_dir / '3.6_polynomial_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualisasi kurva polinomial disimpan sebagai '{plots_dir}/3.6_polynomial_curves.png'")

# 3.3 Perbandingan Skor R² untuk Berbagai Derajat
def plot_r2_comparison():
    """Membandingkan performa model dengan berbagai derajat polinomial.
    
    Fungsi ini akan:
    1. Mencoba model dengan derajat polinomial 1 sampai 5
    2. Menghitung skor R² untuk data latih dan uji
    3. Memvisualisasikan perbandingan performa
    """
    print("Membuat perbandingan skor R² untuk berbagai derajat polinomial...")
    
    # Daftar derajat polinomial yang akan diuji
    degrees = range(1, 6)  # Derajat 1 sampai 5
    train_scores = []  # Untuk menyimpan skor R² data latih
    test_scores = []   # Untuk menyimpan skor R² data uji
    
    # Mencoba setiap derajat polinomial
    for degree in degrees:
        # Membuat pipeline dengan polynomial features dan regresi linear
        model = make_pipeline(
            PolynomialFeatures(degree=degree, include_bias=False),  # Fitur polinomial
            LinearRegression()  # Model regresi linear
        )
        
        # Menghitung skor R² dengan cross-validation pada data latih
        train_score = cross_val_score(
            model, 
            X_train_scaled, 
            y_train, 
            cv=5,  # 5-fold cross-validation
            scoring='r2'  # Metrik evaluasi: R²
        ).mean()  # Rata-rata skor cross-validation
        
        # Melatih model pada seluruh data latih dan evaluasi pada data uji
        model.fit(X_train_scaled, y_train)
        test_score = r2_score(y_test, model.predict(X_test_scaled))
        
        # Menyimpan skor
        train_scores.append(train_score)
        test_scores.append(test_score)
    
    # Membuat plot perbandingan skor R²
    plt.figure(figsize=(10, 6))
    
    # Plot skor data latih dan uji
    plt.plot(degrees, train_scores, 'o-', label='Skor Data Latih', linewidth=2)
    plt.plot(degrees, test_scores, 'o-', label='Skor Data Uji', linewidth=2)
    
    # Menambahkan label dan judul
    plt.xlabel('Derajat Polinomial')
    plt.ylabel('Nilai R²')
    plt.title('3.3 Perbandingan Skor R² untuk Berbagai Derajat Polinomial')
    plt.xticks(degrees)  # Menampilkan semua derajat pada sumbu x
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Menambahkan nilai skor di atas titik-titik
    for i, (train_val, test_val) in enumerate(zip(train_scores, test_scores)):
        plt.text(degrees[i], train_val, f'{train_val:.3f}', 
                ha='center', va='bottom', fontweight='bold')
        plt.text(degrees[i], test_val, f'{test_val:.3f}', 
                ha='center', va='top', fontweight='bold')
    
    # Menyimpan plot
    plt.tight_layout()
    plt.savefig(plots_dir / '3.3_r2_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot perbandingan skor R² disimpan sebagai '{plots_dir}/3.3_r2_comparison.png'")

if __name__ == "__main__":
    """Fungsi utama untuk menjalankan semua visualisasi.
    
    Visualisasi yang akan dihasilkan:
    1. Plot nilai prediksi vs aktual
    2. Plot residual
    3. Kurva polinomial untuk fitur penting
    4. Perbandingan skor R² untuk berbagai derajat polinomial
    """
    print("=== MEMULAI PEMBUATAN VISUALISASI TAMBAHAN ===\n")
    
    # Membuat semua visualisasi
    plot_predicted_vs_actual()  # 3.4
    plot_residuals()           # 3.5
    plot_polynomial_curves()   # 3.6
    plot_r2_comparison()       # 3.3
    
    # Pesan penyelesaian
    print("\n=== SEMUA VISUALISASI TELAH SELESAI DIBUAT ===")
    print(f"Semua file visualisasi telah disimpan di direktori '{plots_dir}/'")
    print("\nDaftar file yang dihasilkan:")
    print(f"- {plots_dir}/3.3_r2_comparison.png")
    print(f"- {plots_dir}/3.4_predicted_vs_actual.png")
    print(f"- {plots_dir}/3.5_residual_plot.png")
    print(f"- {plots_dir}/3.6_polynomial_curves.png")
