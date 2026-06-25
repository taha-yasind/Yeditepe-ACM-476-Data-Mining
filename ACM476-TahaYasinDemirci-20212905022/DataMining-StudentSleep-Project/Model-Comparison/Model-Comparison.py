# GEREKLİ KÜTÜPHANELERİ İÇE AKTARIYORUM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. VERİYİ OKUMA VE ÖN İŞLEME
# ============================================================

# excel dosyasını okuyorum
df = pd.read_excel('../Students.xlsx')

# person_id kimlik sütununu çıkarıyorum
df = df.drop(columns=['person_id'])

# kategorik sütunları sayısal değerlere çeviriyorum
le = LabelEncoder()
cat_cols = [col for col in df.columns
            if df[col].dtype == object or str(df[col].dtype) == 'str']

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# bağımsız ve bağımlı değişkenleri ayırıyorum
X = df.drop(columns=['sleep_disorder_risk'])
Y = df['sleep_disorder_risk']

# veriyi %80 eğitim %20 test olarak bölüyorum
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print("Veri hazırlandı.")

# ============================================================
# 2. MODELLERİ EĞİTİYORUM
# ============================================================

# random forest modelini eğitiyorum
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest eğitildi.")

# decision tree modelini eğitiyorum
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
print("Decision Tree eğitildi.")

# ============================================================
# 3. METRİKLERİ HESAPLIYORUM
# ============================================================

# her model için 5 metriği hesaplıyorum
results = {
    'Model': ['Random Forest', 'Decision Tree'],

    # doğru tahmin oranı
    'Accuracy': [
        accuracy_score(y_test, rf_pred),
        accuracy_score(y_test, dt_pred)
    ],

    # kesinlik: tahmin ettiklerimin ne kadarı gerçekten doğru
    'Precision': [
        precision_score(y_test, rf_pred, average='weighted'),
        precision_score(y_test, dt_pred, average='weighted')
    ],

    # duyarlılık: gerçek pozitifleri ne kadar yakaladım
    'Recall': [
        recall_score(y_test, rf_pred, average='weighted'),
        recall_score(y_test, dt_pred, average='weighted')
    ],

    # precision ve recall'un harmonik ortalaması
    'F1 Score': [
        f1_score(y_test, rf_pred, average='weighted'),
        f1_score(y_test, dt_pred, average='weighted')
    ],
}

# sonuçları bir tabloya dönüştürüyorum
results_df = pd.DataFrame(results)
results_df = results_df.set_index('Model')

print("\n=== MODEL KARŞILAŞTIRMA TABLOSU ===")
print(results_df.round(4))

# ============================================================
# 4. METRİK KARŞILAŞTIRMA GRAFİĞİ
# ============================================================

# iki modelin metriklerini yan yana bar grafik ile karşılaştırıyorum
results_df.T.plot(kind='bar', figsize=(10, 6),
                  color=['steelblue', 'darkorange'])
plt.title('Model Karşılaştırma - 4 Metrik')
plt.xlabel('Metrik')
plt.ylabel('Skor')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(title='Model')
plt.tight_layout()
plt.savefig('model_karsilastirma.png')
plt.close()
print("\nmodel_karsilastirma.png kaydedildi.")

# ============================================================
# 5. CONFUSION MATRIX KARŞILAŞTIRMASI
# ============================================================

# iki modelin confusion matrix'ini yan yana çiziyorum
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sinif_isimleri = ['Healthy', 'Mild', 'Moderate', 'Severe']

# random forest confusion matrix
cm_rf = confusion_matrix(y_test, rf_pred)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=sinif_isimleri,
            yticklabels=sinif_isimleri,
            ax=axes[0])
axes[0].set_title('Random Forest - Confusion Matrix')
axes[0].set_xlabel('Tahmin Edilen')
axes[0].set_ylabel('Gerçek')

# decision tree confusion matrix
cm_dt = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Oranges',
            xticklabels=sinif_isimleri,
            yticklabels=sinif_isimleri,
            ax=axes[1])
axes[1].set_title('Decision Tree - Confusion Matrix')
axes[1].set_xlabel('Tahmin Edilen')
axes[1].set_ylabel('Gerçek')

plt.tight_layout()
plt.savefig('confusion_matrix_karsilastirma.png')
plt.close()
print("confusion_matrix_karsilastirma.png kaydedildi.")

# ============================================================
# 6. FİNAL YORUM
# ============================================================

# en iyi modeli accuracy'e göre belirliyorum
en_iyi = results_df['Accuracy'].idxmax()
en_iyi_acc = results_df['Accuracy'].max()

print("\n=== FİNAL YORUM ===")
print(f"En iyi model : {en_iyi}")
print(f"Accuracy     : {en_iyi_acc:.4f}")
print()
print("Özet:")
print(f"  Random Forest Accuracy : {results_df.loc['Random Forest', 'Accuracy']:.4f}")
print(f"  Decision Tree Accuracy : {results_df.loc['Decision Tree', 'Accuracy']:.4f}")
print()
print("Random Forest daha fazla ağaçla tahmin yaptığı için")
print("Decision Tree'ye kıyasla genellikle daha yüksek doğruluk verir.")
print("Ancak Decision Tree görsel olarak yorumlanması daha kolay bir modeldir.")

print("\nModel Comparison tamamlandı.")