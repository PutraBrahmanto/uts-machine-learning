# Import library yang diperlukan
import numpy as np  # Untuk komputasi numerik
import pandas as pd  # Untuk manipulasi data
import matplotlib.pyplot as plt  # Untuk visualisasi data
import seaborn as sns  # Untuk visualisasi data yang lebih menarik
from sklearn.model_selection import learning_curve, cross_val_score  # Untuk validasi model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Untuk pra-pemrosesan data
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Model regresi
from sklearn.metrics import r2_score, mean_squared_error  # Metrik evaluasi
from sklearn.pipeline import Pipeline  # Untuk membuat alur kerja
from pathlib import Path  # Untuk menangani path file
import joblib  # Untuk menyimpan dan memuat model
import os  # Untuk berinteraksi dengan sistem operasi

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

# 2. Visualisasi Pelatihan Model
# 2.1 Jumlah Fitur Polinomial
print("Membuat visualisasi jumlah fitur polinomial...")

degrees = range(1, 6)  # Derajat polinomial yang akan diuji (1 sampai 5)
feature_counts = []  # Untuk menyimpan jumlah fitur setiap derajat

# Menghitung jumlah fitur untuk setiap derajat polinomial
for degree in degrees:
    poly = PolynomialFeatures(degree=degree, include_bias=False)  # Membuat objek polynomial features
    X_poly = poly.fit_transform(X_train_scaled)  # Mentransformasi data latih
    feature_counts.append(X_poly.shape[1])  # Menyimpan jumlah fitur yang dihasilkan

# Membuat plot hubungan antara derajat polinomial dan jumlah fitur
plt.figure(figsize=(10, 6))
plt.plot(degrees, feature_counts, 'bo-', linewidth=2, markersize=8)  # Garis biru dengan titik
plt.xlabel('Derajat Polinomial', fontsize=12)
plt.ylabel('Jumlah Fitur', fontsize=12)
plt.title('2.1 Jumlah Fitur vs Derajat Polinomial', fontsize=14, pad=15)
plt.xticks(degrees)  # Menampilkan semua derajat pada sumbu x
plt.grid(True, linestyle='--', alpha=0.7)  # Grid dengan garis putus-putus

# Menyimpan plot
plt.tight_layout()
plt.savefig(plots_dir / '2.1_poly_feature_counts.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Visualisasi jumlah fitur polinomial disimpan sebagai '{plots_dir}/2.1_poly_feature_counts.png'")

# 3. Visualisasi Evaluasi Model
print("\nMemulai evaluasi model dengan berbagai konfigurasi...")

# Daftar derajat polinomial dan nilai alpha yang akan diuji
degrees = [1, 2, 3, 4, 5]  # Derajat polinomial
alphas = [0.1, 1, 10]  # Nilai alpha untuk regularisasi
results = []  # Untuk menyimpan hasil evaluasi

# Melakukan pelatihan dan evaluasi untuk setiap derajat polinomial
for degree in degrees:
    print(f"\nMemproses derajat polinomial {degree}...")
    
    # Membuat fitur polinomial
    poly = PolynomialFeatures(degree=degree, include_bias=False)  # Tanpa bias karena sudah ada intercept
    X_train_poly = poly.fit_transform(X_train_scaled)  # Transformasi data latih
    X_test_poly = poly.transform(X_test_scaled)  # Transformasi data uji
    
    # Daftar model yang akan dievaluasi
    models = {
        'Linear': LinearRegression(),  # Regresi linear tanpa regularisasi
        'Ridge (α=0.1)': Ridge(alpha=0.1, random_state=42),  # Ridge regression dengan alpha kecil
        'Ridge (α=1)': Ridge(alpha=1, random_state=42),      # Ridge regression dengan alpha sedang
        'Ridge (α=10)': Ridge(alpha=10, random_state=42),    # Ridge regression dengan alpha besar
        'Lasso (α=0.1)': Lasso(alpha=0.1, random_state=42, max_iter=10000),  # Lasso dengan alpha kecil
        'Lasso (α=1)': Lasso(alpha=1, random_state=42, max_iter=10000),      # Lasso dengan alpha sedang
        'Lasso (α=10)': Lasso(alpha=10, random_state=42, max_iter=10000)     # Lasso dengan alpha besar
    }
    
    # Melatih dan mengevaluasi setiap model
    for name, model in models.items():
        print(f"  Melatih model {name}...")
        
        # Melatih model dengan data latih
        model.fit(X_train_poly, y_train)
        
        # Memprediksi data latih dan uji
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Menghitung metrik evaluasi
        train_r2 = r2_score(y_train, y_train_pred)  # R² score data latih
        test_r2 = r2_score(y_test, y_test_pred)      # R² score data uji
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))  # RMSE data latih
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))     # RMSE data uji
        
        # Menyimpan hasil evaluasi
        results.append({
            'Degree': degree,         # Derajat polinomial
            'Model': name,            # Nama model
            'Train_R2': train_r2,     # R² score data latih
            'Test_R2': test_r2,       # R² score data uji
            'Train_RMSE': train_rmse, # RMSE data latih
            'Test_RMSE': test_rmse    # RMSE data uji
        })
        
        print(f"    Selesai - R² (train/test): {train_r2:.4f}/{test_r2:.4f}")

# Mengubah hasil evaluasi menjadi DataFrame untuk analisis lebih lanjut
results_df = pd.DataFrame(results)
print("\nEvaluasi model selesai. Menyimpan hasil...")

# 3.1 Tabel Metrik Evaluasi
print("Membuat tabel metrik evaluasi...")

# Membuat pivot table untuk menampilkan metrik evaluasi
metrics_table = results_df.pivot(
    index=['Degree', 'Model'],  # Baris berdasarkan derajat dan model
    columns=[],  # Tidak ada kolom tambahan
    values=['Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE']  # Nilai yang akan ditampilkan
)

# Membuat visualisasi tabel
plt.figure(figsize=(14, len(results_df) * 0.4))  # Ukuran dinamis berdasarkan jumlah model
plt.axis('off')  # Menonaktifkan sumbu

# Membuat tabel
plt.table(
    cellText=metrics_table.round(4).values,  # Nilai sel, dibulatkan 4 desimal
    rowLabels=metrics_table.index,  # Label baris (Degree, Model)
    colLabels=metrics_table.columns.get_level_values(0),  # Label kolom (Train_R2, dll)
    cellLoc='center',  # Posisi teks di tengah sel
         loc='center',  # Posisi tabel di tengah
         colWidths=[0.15, 0.15, 0.15, 0.15])  # Lebar kolom

# Menyimpan tabel metrik evaluasi
plt.tight_layout()
plt.savefig(plots_dir / '3.1_metrics_table_train_test.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Tabel metrik evaluasi disimpan sebagai '{plots_dir}/3.1_metrics_table_train_test.png'")

# 3.3 Visualisasi R² vs Derajat Polinomial
print("Membuat visualisasi R² vs Derajat Polinomial...")
plt.figure(figsize=(14, 7))

# Mengelompokkan data berdasarkan model dan memplot R² untuk setiap derajat
for model_type in results_df['Model'].unique():
    # Filter data untuk model saat ini
    model_data = results_df[results_df['Model'] == model_type]
    
    # Plot R² data uji untuk setiap derajat polinomial
    plt.plot(model_data['Degree'], model_data['Test_R2'], 'o-', 
             linewidth=2, markersize=8, label=model_type)
# Mengatur label dan judul plot
plt.xlabel('Derajat Polinomial', fontsize=12)
plt.ylabel('Nilai R² (Data Uji)', fontsize=12)
plt.title('3.3 Perbandingan R² Score Berdasarkan Derajat Polinomial', fontsize=14, pad=15)

# Menambahkan legenda di luar plot
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

# Menambahkan grid dan mengatur tampilan
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(degrees)  # Menampilkan semua derajat pada sumbu x

# Menyimpan plot
plt.tight_layout()
plt.savefig(plots_dir / '3.3_r2_vs_degree.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Visualisasi R² vs Derajat Polinomial disimpan sebagai '{plots_dir}/3.3_r2_vs_degree.png'")

# 4. Analisis Regularisasi
print("\nMemulai analisis regularisasi...")

# 4.1 Pengaruh Kekuatan Regularisasi (Alpha) terhadap Kinerja Model
alphas = [0.001, 0.01, 0.1, 1, 10, 100]  # Rentang nilai alpha yang akan diuji
ridge_scores = []  # Untuk menyimpan skor R² Ridge
lasso_scores = []  # Untuk menyimpan skor R² Lasso

# Menggunakan derajat polinomial 2 sebagai contoh
degree = 2  # Derajat polinomial yang digunakan
print(f"Menggunakan derajat polinomial {degree} untuk analisis regularisasi...")

# Membuat fitur polinomial
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)  # Transformasi data latih
X_test_poly = poly.transform(X_test_scaled)  # Transformasi data uji

# Melatih model dengan berbagai nilai alpha
for alpha in alphas:
    print(f"  Memproses alpha = {alpha}...")
    
    # Melatih model Ridge dengan alpha saat ini
    ridge = Ridge(alpha=alpha, random_state=42)
    ridge.fit(X_train_poly, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test_poly)))  # Menyimpan R² score
    
    # Melatih model Lasso dengan alpha saat ini
    lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_poly, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test_poly)))  # Menyimpan R² score

# Membuat plot perbandingan performa Ridge dan Lasso
plt.figure(figsize=(14, 7))
plt.semilogx(alphas, ridge_scores, 'o-', linewidth=2, markersize=8, label='Ridge')
plt.semilogx(alphas, lasso_scores, 's-', linewidth=2, markersize=8, label='Lasso')

# Mengatur label dan judul plot
plt.xlabel('Nilai Alpha (skala logaritmik)', fontsize=12)
plt.ylabel('Nilai R² (Data Uji)', fontsize=12)
plt.title('4.1 Pengaruh Kekuatan Regularisasi terhadap Kinerja Model', fontsize=14, pad=15)

# Menambahkan legenda dan grid
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)

# Menyimpan plot
plt.tight_layout()
plt.savefig(plots_dir / '4.1_r2_vs_alpha.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Visualisasi pengaruh regularisasi disimpan sebagai '{plots_dir}/4.1_r2_vs_alpha.png'")

# 4.3 Analisis Kepentingan Fitur (Feature Importance)
print("\nMenganalisis kepentingan fitur...")

# Menggunakan model Linear Regression sederhana (degree=1) sebagai contoh
# Karena koefisien pada model linear lebih mudah diinterpretasikan
best_model = LinearRegression()
best_model.fit(X_train_scaled, y_train)

# Membuat DataFrame untuk menyimpan kepentingan fitur
feature_importance = pd.DataFrame({
    'Feature': X.columns,  # Nama-nama fitur
    'Importance': np.abs(best_model.coef_)  # Nilai absolut koefisien
}).sort_values('Importance', ascending=False)  # Urutkan dari yang terpenting

print("\nUrutan kepentingan fitur:")
print(feature_importance.to_string(index=False))

# Membuat visualisasi kepentingan fitur
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance, palette='viridis')

# Mengatur judul dan label
plt.title('4.3 Analisis Kepentingan Fitur (Linear Regression)', fontsize=14, pad=15)
plt.xlabel('Tingkat Kepentingan', fontsize=12)
plt.ylabel('Fitur', fontsize=12)

# Menyimpan visualisasi
plt.tight_layout()
plt.savefig(plots_dir / '4.3_feature_importance_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"\nVisualisasi kepentingan fitur disimpan sebagai '{plots_dir}/4.3_feature_importance_bar.png'")

# 5. Ringkasan dan Penyimpanan Hasil
print("\n=== RINGKASAN HASIL ===")
print(f"Semua visualisasi model telah berhasil dibuat dan disimpan di direktori '{plots_dir}/'")
print("\nDaftar file yang dihasilkan:")
print("\n1. Analisis Fitur Polinomial:")
print(f"- {plots_dir}/2.1_poly_feature_counts.png")

print("\n2. Tabel dan Grafik Evaluasi Model:")
print(f"- {plots_dir}/3.1_metrics_table_train_test.png")
print(f"- {plots_dir}/3.3_r2_vs_degree.png")
print(f"- {plots_dir}/3.4_rmse_vs_degree.png")

print("\n3. Analisis Regularisasi:")
print(f"- {plots_dir}/4.1_r2_vs_alpha.png")

print("\n4. Analisis Kepentingan Fitur:")
print(f"- {plots_dir}/4.3_feature_importance_bar.png")

print("\n5. Data Hasil Evaluasi:")
print(f"- {plots_dir}/model_evaluation_results.csv")

print("\n=== PROSES SELESAI ===\n")
