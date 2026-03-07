import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_log_error

# Langkah 1: Baca Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"Ukuran data training: {train_data.shape[0]} baris, {train_data.shape[1]} kolom")
print(f"Ukuran data testing: {test_data.shape[0]} baris, {test_data.shape[1]} kolom")

print("\nData Training:")
print(train_data.head())

# Langkah 2: Pisahkan fitur dan target
target = train_data['SalePrice']
print(f"Target (SalePrice):")
print(target.head())

# Buang kolom SalePrice dari data training
fitur_train = train_data.drop('SalePrice', axis=1)
fitur_test = test_data.copy()
test_ids = fitur_test['Id'] # Simpan ID untuk nanti

# Buang kolom Id
fitur_train = fitur_train.drop('Id', axis=1)
fitur_test = fitur_test.drop('Id', axis=1)
print(f"\nJumlah fitur: {fitur_train.shape[1]}")


# Langkah 3: Cek Tipe Data
# Cari kolom numerik
kolom_numerik = fitur_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Jumlah kolom numerik: {len(kolom_numerik)}")
print(f"Contoh: {kolom_numerik[:5]}")

# Cari kolom kategorikal
kolom_kategorikal = fitur_train.select_dtypes(include=['object']).columns.tolist()
print(f"\nJumlah kolom kategorikal: {len(kolom_kategorikal)}")
print(f"Contoh: {kolom_kategorikal[:5]}")


# Langkah 4: Isi Missing Value
total_kosong_train = fitur_train.isnull().sum().sum()
total_kosong_test = fitur_test.isnull().sum().sum()
print(f"Total data kosong di training: {total_kosong_train}")
print(f"Total data kosong di testing: {total_kosong_test}")

for kolom in kolom_numerik:
    # Hitung nilai median (nilai tengah)
    nilai_tengah = fitur_train[kolom].median()
    # Isi data kosong dengan nilai median
    fitur_train[kolom] = fitur_train[kolom].fillna(nilai_tengah)
    fitur_test[kolom] = fitur_test[kolom].fillna(nilai_tengah)

for kolom in kolom_kategorikal:
    # Isi data kosong dengan teks 'None'
    fitur_train[kolom] = fitur_train[kolom].fillna('None')
    fitur_test[kolom] = fitur_test[kolom].fillna('None')

print("\nSetelah isi missing value:")
print(f"Data kosong di training: {fitur_train.isnull().sum().sum()}")
print(f"Data kosong di testing: {fitur_test.isnull().sum().sum()}")


# Langkah 5: Enkode Fitur Kategorikal
fitur_train = pd.get_dummies(fitur_train, columns=kolom_kategorikal, drop_first=True)
fitur_test = pd.get_dummies(fitur_test, columns=kolom_kategorikal, drop_first=True)

fitur_test = fitur_test.reindex(columns=fitur_train.columns, fill_value=0)
print(f"\nJumlah fitur setelah encoding: {fitur_train.shape[1]}")


# Langkah 6: Standarisasi Fitur
scaler = StandardScaler()
fitur_train_scaled = scaler.fit_transform(fitur_train)
fitur_test_scaled = scaler.transform(fitur_test)
print(f"\nBentuk data sesudah standarisasi: {fitur_train_scaled.shape}")


# Langkah 7: Gunakan PCA
pca = PCA(n_components=0.95, random_state=42) # 95% variansi

# Fit dan transform
fitur_train_pca = pca.fit_transform(fitur_train_scaled)
fitur_test_pca = pca.transform(fitur_test_scaled)
print(f"\nFitur sebelum PCA: {fitur_train_scaled.shape[1]}")
print(f"Fitur setelah PCA: {fitur_train_pca.shape[1]}")


# Langkah 8: Training Model
model = LinearRegression()
model.fit(fitur_train_pca, target)


# Langkah 9: Evaluasi Model
# Prediksi data training
prediksi_train = model.predict(fitur_train_pca)
prediksi_train = np.maximum(prediksi_train, 0) # Pastikan tidak ada negatif

# Hitung RMSLE
rmsle = np.sqrt(mean_squared_log_error(target, prediksi_train))
print(f"\nRMSLE: {rmsle:.6f}")


# Langkah 10: Prediksi Data Testing
prediksi_test = model.predict(fitur_test_pca)
prediksi_test = np.maximum(prediksi_test, 0) # Pastikan tidak ada negatif

print(f"\nJumlah prediksi: {len(prediksi_test)}")
print(f"Harga rata-rata: ${prediksi_test.mean():,.0f}")
print(f"Harga terendah: ${prediksi_test.min():,.0f}")
print(f"Harga tertinggi: ${prediksi_test.max():,.0f}")


# Langkah 11: Buat File CSV
# Buat DataFrame
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': prediksi_test
})

# Simpan ke file CSV
nama_file = 'submission-pca.csv'
submission.to_csv(nama_file, index=False)