# Import library yang diperlukan
import numpy as np  # Untuk komputasi numerik
import pandas as pd  # Untuk manipulasi data
import matplotlib.pyplot as plt  # Untuk visualisasi data
import seaborn as sns  # Untuk visualisasi data yang lebih menarik
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve  # Untuk membagi data dan validasi model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # Untuk pra-pemrosesan data
from sklearn.linear_model import LinearRegression, Ridge, Lasso  # Model regresi
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error  # Metrik evaluasi
from sklearn.pipeline import Pipeline  # Untuk membuat alur kerja
import joblib  # Untuk menyimpan model
import os  # Untuk berinteraksi dengan sistem operasi
from pathlib import Path  # Untuk menangani path file

# Set random seed for reproducibility
np.random.seed(42)

class PropertyPricePredictor:
    """Kelas untuk memprediksi harga properti menggunakan regresi polinomial."""
    
    def __init__(self, data_path):
        """Inisialisasi objek predictor dengan path data.
        
        Args:
            data_path (str): Path menuju file dataset
        """
        self.data_path = data_path  # Path menuju file dataset
        self.df = None  # Untuk menyimpan dataframe
        self.X_train = None  # Fitur data latih
        self.X_test = None   # Fitur data uji
        self.y_train = None  # Target data latih
        self.y_test = None   # Target data uji
        self.scaler = StandardScaler()  # Untuk standardisasi data
        self.best_model = None  # Model terbaik hasil pelatihan
        self.best_degree = None  # Derajat polinomial terbaik
        self.models = {}  # Untuk menyimpan semua model yang dilatih
        
    def load_data(self):
        """Memuat dan mempersiapkan dataset untuk analisis.
        
        Fungsi ini akan:
        1. Membaca file CSV ke dalam DataFrame
        2. Menampilkan informasi dasar tentang dataset
        """
        # Membaca file CSV ke dalam DataFrame
        self.df = pd.read_csv(self.data_path)
        
        # Menampilkan informasi dasar dataset
        print("\n=== Ringkasan Dataset ===")
        print(f"Dimensi dataset: {self.df.shape} (baris, kolom)")
        print("\n5 baris pertama data:")
        print(self.df.head())
        
    def explore_data(self):
        """Melakukan analisis eksplorasi data (EDA).
        
        Fungsi ini akan:
        1. Menampilkan statistik deskriptif
        2. Membuat visualisasi distribusi fitur
        3. Membuat matriks korelasi
        """
        # Menampilkan statistik deskriptif
        print("\n=== Ringkasan Statistik ===")
        print(self.df.describe())
        
        # Membuat plot distribusi untuk setiap fitur
        print("\nMembuat visualisasi distribusi fitur...")
        self.df.hist(figsize=(12, 10))
        plt.suptitle('Distribusi Fitur', y=1.02)
        plt.tight_layout()
        plt.savefig('feature_distributions.png')
        print("Gambar disimpan sebagai 'feature_distributions.png'")
        
        # Membuat matriks korelasi
        print("\nMembuat matriks korelasi...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', center=0)
        plt.title('Matriks Korelasi Antar Fitur')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        print("Matriks korelasi disimpan sebagai 'correlation_matrix.png'")
        
        # Correlation matrix
        print("\nGenerating correlation heatmap...")
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        
    def preprocess_data(self):
        """Mempersiapkan data untuk pemodelan.
        
        Langkah-langkah:
        1. Memisahkan fitur dan target
        2. Membagi data menjadi data latih dan uji
        3. Melakukan penskalaan fitur
        """
        # Memisahkan fitur (X) dan target (y)
        X = self.df.drop('Harga_Properti', axis=1)  # Semua kolom kecuali Harga_Properti
        y = self.df['Harga_Properti']  # Target yang akan diprediksi
        
        # Membagi data menjadi data latih (80%) dan data uji (20%)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Melakukan penskalaan fitur menggunakan StandardScaler
        # Fit pada data latih, lalu transform data latih dan uji
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("\n=== Pra-pemrosesan Data ===")
        print(f"Jumlah data latih: {len(self.X_train)} sampel")
        print(f"Jumlah data uji: {len(self.X_test)} sampel")
        
    def train_models(self, max_degree=5):
        """Melatih model dengan berbagai derajat polinomial dan regularisasi.
        
        Args:
            max_degree (int): Derajat polinomial maksimum yang akan dicoba
            
        Model yang dilatih:
        - Linear Regression (tanpa regularisasi)
        - Ridge Regression (L2 regularization)
        - Lasso Regression (L1 regularization)
        """
        # Daftar nilai alpha (kekuatan regularisasi) yang akan diuji
        alphas = [0.01, 0.1, 1.0, 10.0]
        
        # Melatih model untuk setiap derajat polinomial
        for degree in range(1, max_degree + 1):
            print(f"\nMelatih model dengan degree={degree}...")
            
            # Membuat fitur polinomial
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_train_poly = poly.fit_transform(self.X_train)
            
            # 1. Melatih Linear Regression (tanpa regularisasi)
            print(f"  - Melatih Linear Regression...")
            lr = LinearRegression()
            lr.fit(X_train_poly, self.y_train)
            self.models[f'degree_{degree}_linear'] = {
                'model': lr,
                'poly': poly,
                'type': 'Linear',
                'degree': degree
            }
            
            # 2. Melatih Ridge Regression (L2 regularization)
            for alpha in alphas:
                print(f"  - Melatih Ridge (alpha={alpha})...")
                ridge = Ridge(alpha=alpha, random_state=42)
                ridge.fit(X_train_poly, self.y_train)
                self.models[f'degree_{degree}_ridge_{alpha}'] = {
                    'model': ridge,
                    'poly': poly,
                    'type': 'Ridge',
                    'alpha': alpha,
                    'degree': degree
                }
                
            # 3. Melatih Lasso Regression (L1 regularization)
            for alpha in alphas:
                print(f"  - Melatih Lasso (alpha={alpha})...")
                lasso = Lasso(alpha=alpha, random_state=42, max_iter=10000)
                lasso.fit(X_train_poly, self.y_train)
                self.models[f'degree_{degree}_lasso_{alpha}'] = {
                    'model': lasso,
                    'poly': poly,
                    'type': 'Lasso',
                    'alpha': alpha,
                    'degree': degree
                }
                
    def evaluate_models(self):
        """Mengevaluasi semua model yang telah dilatih dan memilih yang terbaik.
        
        Returns:
            DataFrame: Hasil evaluasi semua model dalam bentuk DataFrame
            
        Metrik yang dihitung:
        - R² Score (Train & Test): Mendekati 1 berarti model lebih baik
        - RMSE (Train & Test): Semakin kecil semakin baik
        """
        results = []
        
        # Evaluasi setiap model yang telah dilatih
        for name, model_info in self.models.items():
            # Mendapatkan model dan transformer polinomial
            model = model_info['model']
            poly = model_info['poly']
            
            # Transformasi fitur menggunakan polinomial
            X_train_poly = poly.transform(self.X_train)
            X_test_poly = poly.transform(self.X_test)
            
            # Melakukan prediksi
            y_train_pred = model.predict(X_train_poly)
            y_test_pred = model.predict(X_test_poly)
            
            # Menghitung metrik evaluasi
            train_r2 = r2_score(self.y_train, y_train_pred)  # R² untuk data latih
            test_r2 = r2_score(self.y_test, y_test_pred)      # R² untuk data uji
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))  # RMSE data latih
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))     # RMSE data uji
            
            # Menyimpan hasil evaluasi
            result = {
                'Model': name,                       # Nama model
                'Type': model_info['type'],          # Jenis model (Linear/Ridge/Lasso)
                'Degree': model_info['degree'],      # Derajat polinomial
                'Alpha': model_info.get('alpha', 'N/A'),  # Parameter regularisasi
                'Train_R2': train_r2,                # R² data latih
                'Test_R2': test_r2,                  # R² data uji
                'Train_RMSE': train_rmse,            # RMSE data latih
                'Test_RMSE': test_rmse,              # RMSE data uji
                'Model_Object': model,               # Objek model
                'Poly_Transformer': poly             # Transformer polinomial
            }
            results.append(result)
            
        # Konversi ke DataFrame untuk analisis lebih lanjut
        results_df = pd.DataFrame(results)
        
        # Memilih model terbaik berdasarkan R² score pada data uji
        best_model_idx = results_df['Test_R2'].idxmax()
        self.best_model = results_df.loc[best_model_idx, 'Model_Object']
        self.best_poly = results_df.loc[best_model_idx, 'Poly_Transformer']
        self.best_degree = results_df.loc[best_model_idx, 'Degree']
        
        # Menampilkan hasil evaluasi
        print("\n=== Hasil Evaluasi Model ===")
        print("Model terbaik dipilih berdasarkan R² score tertinggi pada data uji.")
        print("\nDetail Model Terbaik:")
        print(f"- Jenis: {results_df.loc[best_model_idx, 'Type']}")
        print(f"- Derajat: {self.best_degree}")
        print(f"- Alpha: {results_df.loc[best_model_idx, 'Alpha']}")
        print(f"- R² Score (Test): {results_df.loc[best_model_idx, 'Test_R2']:.4f}")
        print(f"- RMSE (Test): {results_df.loc[best_model_idx, 'Test_RMSE']:.4f}")
        
        # Menampilkan ringkasan semua model
        print("\nRingkasan Semua Model:")
        print(results_df[['Model', 'Type', 'Degree', 'Alpha', 'Train_R2', 'Test_R2', 'Train_RMSE', 'Test_RMSE']])
        
        return results_df
    
    def plot_results(self):
        """Membuat visualisasi untuk evaluasi model.
        
        Visualisasi yang dihasilkan:
        1. Plot aktual vs prediksi untuk model terbaik
        2. Garis diagonal merah menunjukkan prediksi sempurna
        """
        # Transformasi data uji menggunakan polinomial terbaik
        X_test_poly = self.best_poly.transform(self.X_test)
        
        # Melakukan prediksi menggunakan model terbaik
        y_test_pred = self.best_model.predict(X_test_poly)
        
        # Membuat plot
        plt.figure(figsize=(10, 6))
        
        # Plot titik-titik prediksi vs aktual
        plt.scatter(self.y_test, y_test_pred, alpha=0.5, 
                   label='Data Uji')
        
        # Menambahkan garis diagonal (prediksi sempurna)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                 [self.y_test.min(), self.y_test.max()], 
                 'r--', lw=2, label='Prediksi Sempurna')
        
        # Menambahkan label dan judul
        plt.xlabel('Harga Aktual (juta Rupiah)')
        plt.ylabel('Harga Prediksi (juta Rupiah)')
        plt.title('Perbandingan Harga Aktual vs Prediksi')
        plt.legend()
        
        # Menambahkan grid untuk memudahkan pembacaan
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Menyimpan gambar
        plt.tight_layout()
        plt.savefig('actual_vs_predicted.png')
        print("\nVisualisasi perbandingan harga aktual vs prediksi disimpan sebagai 'actual_vs_predicted.png'")
        
        # Menampilkan R² score dan RMSE pada plot
        r2 = r2_score(self.y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        
        plt.figtext(0.15, 0.8, f"R² Score: {r2:.4f}\nRMSE: {rmse:.2f}", 
                   bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 5})
        
        # Menyimpan gambar dengan metrik
        plt.savefig('actual_vs_predicted_with_metrics.png')
        plt.close()
        
    def save_models(self):
        """Menyimpan model terbaik dan objek terkait ke dalam file.
        
        File yang disimpan:
        - best_model.pkl: Model prediksi terbaik
        - poly_transformer.pkl: Transformer untuk fitur polinomial
        - scaler.pkl: Scaler untuk normalisasi data
        """
        # Membuat direktori models jika belum ada
        if not os.path.exists('models'):
            os.makedirs('models')
            
        # Menyimpan model dan transformer ke dalam file
        joblib.dump(self.best_model, 'models/best_model.pkl')  # Model terbaik
        joblib.dump(self.best_poly, 'models/poly_transformer.pkl')  # Transformer polinomial
        joblib.dump(self.scaler, 'models/scaler.pkl')  # Scaler untuk normalisasi
        
        # Menampilkan informasi penyimpanan
        print("\n=== Model dan Transformer Disimpan ===")
        print("File yang disimpan di direktori 'models/':")
        print("- best_model.pkl: Model prediksi terbaik")
        print("- poly_transformer.pkl: Transformer untuk fitur polinomial")
        print("- scaler.pkl: Scaler untuk normalisasi data")

def main():
    """Fungsi utama untuk menjalankan alur prediksi harga properti.
    
    Langkah-langkah yang dilakukan:
    1. Inisialisasi predictor
    2. Memuat dan mengeksplorasi data
    3. Melakukan pra-pemrosesan data
    4. Melatih model dengan berbagai konfigurasi
    5. Mengevaluasi dan memilih model terbaik
    6. Membuat visualisasi hasil
    7. Menyimpan model terbaik
    """
    print("=== MEMULAI PROSES PREDIKSI HARGA PROPERTI ===\n")
    
    # 1. Inisialisasi predictor dengan path dataset
    print("1. Menginisialisasi predictor...")
    predictor = PropertyPricePredictor('dataset_uts.csv')
    
    # 2. Memuat dan mengeksplorasi data
    print("\n2. Memuat dan menganalisis data...")
    predictor.load_data()
    predictor.explore_data()
    
    # 3. Pra-pemrosesan data
    print("\n3. Melakukan pra-pemrosesan data...")
    predictor.preprocess_data()
    
    # 4. Melatih model dengan berbagai derajat polinomial
    print("\n4. Melatih model dengan berbagai konfigurasi...")
    predictor.train_models(max_degree=5)  # Mencoba hingga polinomial derajat 5
    
    # 5. Evaluasi model dan pilih yang terbaik
    print("\n5. Mengevaluasi model...")
    results = predictor.evaluate_models()
    
    # 6. Buat visualisasi hasil
    print("\n6. Membuat visualisasi hasil...")
    predictor.plot_results()
    
    # 7. Simpan model terbaik
    print("\n7. Menyimpan model terbaik...")
    predictor.save_models()
    
    # Tampilkan pesan selesai
    print("\n=== PROSES SELESAI ===")
    print(f"\nRingkasan Hasil:")
    print(f"- Model terbaik menggunakan polinomial derajat: {predictor.best_degree}")
    print("- Visualisasi hasil dapat dilihat pada file gambar yang dihasilkan")
    print("- Model dan transformer disimpan di direktori 'models/'")

if __name__ == "__main__":
    main()
