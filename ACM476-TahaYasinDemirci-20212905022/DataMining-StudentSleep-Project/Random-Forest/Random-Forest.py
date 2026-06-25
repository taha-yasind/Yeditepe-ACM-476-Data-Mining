# GEREKLİ KÜTÜPHANELERİ İÇE AKTARIYORUM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder

# ============================================================
# 1. VERİYİ OKUMA
# ============================================================

# excel dosyasını okuyorum
df = pd.read_excel('../Students.xlsx')

# kaç satır sütun var kontrol ediyorum
print(f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# ============================================================
# 2. VERİ ÖN İŞLEME
# ============================================================

# person_id kimlik sütunu, modele katkısı yok, çıkarıyorum
df = df.drop(columns=['person_id'])

# kategorik sütunları sayısal değerlere çeviriyorum
le = LabelEncoder()
cat_cols = [col for col in df.columns
            if df[col].dtype == object or str(df[col].dtype) == 'str']

for col in cat_cols:
    # her kategorik sütunu sayıya dönüştürüyorum
    df[col] = le.fit_transform(df[col].astype(str))

print("Encoding tamamlandı.")

# ============================================================
# 3. X VE Y AYIRMA
# ============================================================

# bağımsız değişkenler (özellikler)
X = df.drop(columns=['sleep_disorder_risk'])

# bağımlı değişken (tahmin edilecek sütun)
Y = df['sleep_disorder_risk']

print(f"X boyutu: {X.shape}")
print(f"Y dağılımı:\n{Y.value_counts()}")

# ============================================================
# 4. EĞİTİM VE TEST SETİNE BÖLME
# ============================================================

# veriyi %80 eğitim %20 test olarak bölüyorum
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)

print(f"\nEğitim seti: {X_train.shape[0]} satır")
print(f"Test seti  : {X_test.shape[0]} satır")

# ============================================================
# 5. RANDOM FOREST MODELİ
# ============================================================

# random forest modelini tanımlıyorum
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# modeli eğitim verisiyle eğitiyorum
rf_model.fit(X_train, y_train)

print("\nModel eğitimi tamamlandı.")

# ============================================================
# 6. TAHMİN VE DEĞERLENDİRME
# ============================================================

# test seti üzerinde tahmin yapıyorum
y_pred = rf_model.predict(X_test)

# accuracy hesaplıyorum
accuracy = accuracy_score(y_test, y_pred)

# f1 score hesaplıyorum (çok sınıflı için weighted)
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nAccuracy : {accuracy:.4f}")
print(f"F1 Score : {f1:.4f}")

# sınıf bazında detaylı rapor yazdırıyorum
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['Healthy', 'Mild', 'Moderate', 'Severe']))

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

# confusion matrix hesaplıyorum
cm = confusion_matrix(y_test, y_pred)

# confusion matrix'i görselleştirip kaydediyorum
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy', 'Mild', 'Moderate', 'Severe'],
            yticklabels=['Healthy', 'Mild', 'Moderate', 'Severe'])
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.tight_layout()
plt.savefig('rf_confusion_matrix.png')
plt.close()
print("rf_confusion_matrix.png kaydedildi.")

# ============================================================
# 8. FEATURE IMPORTANCE
# ============================================================

# her özelliğin modele katkısını önem skoruyla alıyorum
feature_importance = pd.Series(
    rf_model.feature_importances_, index=X.columns)

# en önemli 15 özelliği sıralıyorum
feature_importance = feature_importance.sort_values(ascending=False).head(15)

# feature importance grafiğini kaydediyorum
plt.figure(figsize=(10, 6))
feature_importance.plot(kind='bar', color='steelblue')
plt.title('Random Forest - En Önemli 15 Özellik')
plt.xlabel('Özellik')
plt.ylabel('Önem Skoru')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('rf_feature_importance.png')
plt.close()
print("rf_feature_importance.png kaydedildi.")

print("\nRandom Forest tamamlandı.")