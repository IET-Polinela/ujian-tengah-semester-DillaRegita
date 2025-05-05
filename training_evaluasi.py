
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Baca data
df = pd.read_csv("/content/ujian-tengah-semester-DillaRegita/healthcare-dataset-stroke-data.csv", delimiter=';', decimal=',')

# 2. Tangani data hilang
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')  # ubah ke numerik
df['bmi'] = df['bmi'].fillna(df['bmi'].median())       # isi NaN dengan median

# 3. Hapus kolom yang tidak relevan
df.drop('id', axis=1, inplace=True)

# 4. Label encoding untuk kolom kategorikal
categorical_cols = df.select_dtypes(include='object').columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# 5. Normalisasi fitur numerik
scaler = MinMaxScaler()
X_numerik = df.drop('stroke', axis=1)
df[X_numerik.columns] = scaler.fit_transform(X_numerik)

# 6. Pisahkan fitur dan target
X = df.drop('stroke', axis=1)
y = df['stroke']

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Jumlah data latih: {len(X_train)}")
print(f"Jumlah data uji: {len(X_test)}")

# 8. Latih model Decision Tree
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# 9. Prediksi dan evaluasi
y_pred = model.predict(X_test)

print("\n🔹 Evaluasi Model:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 10. Visualisasi Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Stroke', 'Stroke'], yticklabels=['No Stroke', 'Stroke'])
plt.title('Confusion Matrix')
plt.xlabel('Prediksi')
plt.ylabel('Aktual')
plt.tight_layout()
plt.show()

# 11. Visualisasi Decision Tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=X.columns, class_names=['No Stroke', 'Stroke'], filled=True, fontsize=8)
plt.title("Visualisasi Struktur Pohon Keputusan")
plt.tight_layout()
plt.show()

# 12. Simpan data setelah preprocessing
df.to_csv("/content/ujian-tengah-semester-DillaRegita/data_setelah_dibersihkan.csv", index=False)
print("\n✅ Data setelah dibersihkan berhasil disimpan.")
