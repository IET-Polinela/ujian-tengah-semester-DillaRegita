
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Membaca CSV dengan delimiter titik koma dan koma desimal
df = pd.read_csv("/content/ujian-tengah-semester-DillaRegita/healthcare-dataset-stroke-data.csv", delimiter=';', decimal=',')

# 1. Tangani data hilang
# Tampilkan data hilang
print("🔹 Data hilang sebelum ditangani:")
print(df.isnull().sum())

# Membersihkan kolom bmi dari karakter yang tidak diinginkan dan mengubahnya menjadi numerik
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')  # Mengubah menjadi numerik dan mengganti yang tidak bisa menjadi NaN

# Imputasi: Mengisi missing value di 'bmi' dengan median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# 2. Encoding data kategorikal
# Kolom kategorikal
categorical_cols = df.select_dtypes(include='object').columns.tolist()
print("\n🔹 Kolom kategorikal:", categorical_cols)

# Drop kolom 'id' karena tidak berguna sebagai fitur
df.drop('id', axis=1, inplace=True)

# Label encoding untuk semua kolom kategorikal
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 3. Normalisasi fitur numerik
scaler = MinMaxScaler()

# Kolom numerik (setelah encoding)
numeric_cols = df.select_dtypes(include='number').drop('stroke', axis=1).columns  # kecuali target
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 4. Pisahkan data menjadi fitur dan target
X = df.drop('stroke', axis=1)  # Fitur
y = df['stroke']  # Target

# 5. Split data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Jumlah data latih: {len(X_train)}")
print(f"Jumlah data uji: {len(X_test)}")


# 6. Membuat dan melatih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Prediksi dan evaluasi model
y_pred = model.predict(X_test)
print("\n🔹 Evaluasi Model:")
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Classification Report:\n{classification_report(y_test, y_pred)}")

# Tampilkan data setelah preprocessing
print("\n🔹 Data setelah preprocessing:")
print(df.head())

# Simpan data setelah preprocessing
df.to_csv("/content/ujian-tengah-semester-DillaRegita/data_setelah_dibersihkan.csv", index=False)
print("✅ Data setelah dibersihkan berhasil disimpan.")

