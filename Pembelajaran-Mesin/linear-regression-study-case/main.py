import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_log_error


# Langkah 1: Baca Data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print(f"Training: {train_data.shape[0]} baris, {train_data.shape[1]} kolom")
print(f"Testing: {test_data.shape[0]} baris, {test_data.shape[1]} kolom")

print("\nData Training:")
print(train_data.head())


# Langkah 2: Pisahkan fitur dan target
target = train_data['SalePrice']
print(f"\nTarget (SalePrice):")
print(target.head())

# Buang kolom SalePrice dari data training (karena ini yang mau kita prediksi)
fitur_train = train_data.drop('SalePrice', axis=1)
fitur_test = test_data.copy()
test_ids = fitur_test['Id'] # Simpan ID untuk submission nanti

# Buang kolom Id (karena bukan fitur yang berguna untuk prediksi)
fitur_train = fitur_train.drop('Id', axis=1)
fitur_test = fitur_test.drop('Id', axis=1)

print(f"\nJumlah fitur: {fitur_train.shape[1]}")


# Langkah 3: Cek Tipe Data
# Cari kolom numerik
kolom_numerik = fitur_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nJumlah kolom numerik: {len(kolom_numerik)}")
print(f"Contoh: {kolom_numerik[:5]}")

# Cari kolom kategorikal
kolom_kategorikal = fitur_train.select_dtypes(include=['object']).columns.tolist()
print(f"\nJumlah kolom kategorikal: {len(kolom_kategorikal)}")
print(f"Contoh: {kolom_kategorikal[:5]}")


# Langkah 4: Isi Missing Value
total_kosong_train = fitur_train.isnull().sum().sum()
total_kosong_test = fitur_test.isnull().sum().sum()
print(f"\nTotal data kosong di training: {total_kosong_train}")
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


# Langkah 5: Enkode Fitur Kategorikal menggunakan Label Encoding
# Untuk menghindari terlalu banyak fitur dummy
print("\nMengenkode fitur kategorikal dengan Label Encoding...")
label_encoders = {}
for kolom in kolom_kategorikal:
    le = LabelEncoder()
    # Gabungkan train dan test untuk fit
    combined = pd.concat([fitur_train[kolom], fitur_test[kolom]])
    le.fit(combined)
    
    fitur_train[kolom] = le.transform(fitur_train[kolom])
    fitur_test[kolom] = le.transform(fitur_test[kolom])
    label_encoders[kolom] = le

print(f"Jumlah fitur setelah encoding: {fitur_train.shape[1]}")


# Langkah 6: Standarisasi Fitur
scaler = StandardScaler()
fitur_train_scaled = scaler.fit_transform(fitur_train)
fitur_test_scaled = scaler.transform(fitur_test)
print(f"Bentuk data sesudah standarisasi: {fitur_train_scaled.shape}")


# Langkah 7: Training Model (Tanpa PCA)
model = LinearRegression()
model.fit(fitur_train_scaled, target)


# Langkah 8: Evaluasi Model
# Prediksi data training
prediksi_train = model.predict(fitur_train_scaled)
prediksi_train = np.maximum(prediksi_train, 0) # Pastikan tidak ada prediksi negatif

# Hitung RMSLE
rmsle = np.sqrt(mean_squared_log_error(target, prediksi_train))
print(f"\nRMSLE (tanpa PCA): {rmsle:.6f}")


# Langkah 9: Prediksi Data Testing
prediksi_test = model.predict(fitur_test_scaled)
prediksi_test = np.maximum(prediksi_test, 0) # Pastikan tidak ada prediksi negatif

print(f"\nJumlah prediksi: {len(prediksi_test)}")
print(f"Harga rata-rata: ${prediksi_test.mean():,.0f}")
print(f"Harga terendah: ${prediksi_test.min():,.0f}")
print(f"Harga tertinggi: ${prediksi_test.max():,.0f}")


# Langkah 10: Buat File CSV
# Buat DataFrame
submission = pd.DataFrame({'Id': test_ids, 'SalePrice': prediksi_test})

# Simpan ke file CSV
nama_file = 'submission-no-pca.csv'
submission.to_csv(nama_file, index=False)