# GEREKLİ KÜTÜPHANELERİ İÇE AKTARIYORUM
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. VERİYİ OKUMA
# ============================================================

# excel dosyasını okuyorum
df = pd.read_excel('../Students.xlsx')

# ilk 5 satırı kontrol ediyorum
print("İlk 5 satır:")
print(df.head())

# ============================================================
# 2. GENEL BİLGİLER
# ============================================================

# kaç satır kaç sütun olduğunu yazdırıyorum
print(f"\nVeri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")

# sütun isimlerini ve veri tiplerini görüyorum
print("\nSütunlar ve veri tipleri:")
print(df.dtypes)

# ============================================================
# 3. EKSİK VERİ KONTROLÜ
# ============================================================

# her sütunda kaç eksik değer olduğunu kontrol ediyorum
print("\nEksik veri sayısı (sütun bazında):")
print(df.isnull().sum())

# toplam eksik veri var mı yok mu kontrol ediyorum
print(f"\nEksik veri var mı: {df.isnull().values.any()}")

# ============================================================
# 4. TANIMLAYICI İSTATİSTİKLER
# ============================================================

# sayısal sütunların özet istatistiklerini çıkarıyorum
print("\nTanımlayıcı istatistikler:")
print(df.describe().T)

# ============================================================
# 5. TARGET DEĞİŞKENİ İNCELEME
# ============================================================

# sleep_disorder_risk sütunundaki sınıfların dağılımını görüyorum
print("\nTarget dağılımı (sleep_disorder_risk):")
print(df['sleep_disorder_risk'].value_counts())

# target dağılımını bar grafik olarak kaydediyorum
plt.figure(figsize=(8, 5))
df['sleep_disorder_risk'].value_counts().plot(kind='bar', color=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
plt.title('Sleep Disorder Risk - Sınıf Dağılımı')
plt.xlabel('Risk Seviyesi')
plt.ylabel('Kişi Sayısı')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('target_dagilimi.png')
plt.close()
print("target_dagilimi.png kaydedildi.")

# ============================================================
# 6. SAYISAL DEĞİŞKENLERİ İNCELEME
# ============================================================

# sayısal sütunları listeliyorum (person_id hariç)
num_cols = [col for col in df.columns
            if df[col].dtype in ['int64', 'float64'] and col != 'person_id']

# her sayısal değişken için özet istatistik fonksiyonu tanımlıyorum
def num_summary(data, numerical_col, plot=False):
    # değişkenin yüzdelik dilimlerini yazdırıyorum
    quantiles = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50,
                 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(data[numerical_col].describe(quantiles).T)

    if plot:
        # histogramı çiziyorum
        data[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.tight_layout()
        plt.show(block=True)

# tüm sayısal değişkenleri özetliyorum
for col in num_cols:
    num_summary(df, col, plot=False)

# ============================================================
# 7. TARGET'A GÖRE SAYISAL DEĞİŞKEN ORTALAMALARI
# ============================================================

# her risk grubunda sayısal değişkenlerin ortalamasını karşılaştırıyorum
def target_summary_with_num(dataframe, target, num_col):
    print(dataframe.groupby(target).agg({num_col: 'mean'}), end='\n\n')

# en önemli değişkenler için target'a göre ortalama yazdırıyorum
important_cols = ['sleep_duration_hrs', 'stress_score', 'sleep_quality_score',
                  'caffeine_mg_before_bed', 'rem_percentage', 'deep_sleep_percentage']

for col in important_cols:
    target_summary_with_num(df, 'sleep_disorder_risk', col)

# ============================================================
# 8. KATEGORİK DEĞİŞKENLER
# ============================================================

# kategorik sütunları listeliyorum (target hariç)
cat_cols = [col for col in df.columns
            if (df[col].dtype == object or str(df[col].dtype) == 'str')
            and col != 'sleep_disorder_risk']

# her kategorik değişkenin benzersiz değerlerini yazdırıyorum
print("\nKategorik değişkenler:")
for col in cat_cols:
    print(f"{col}: {df[col].nunique()} unique → {df[col].unique()}")

# ============================================================
# 9. KORELASYON HARİTASI
# ============================================================

# sayısal değişkenler arasındaki korelasyonu hesaplıyorum
corr = df[num_cols].corr()

# korelasyon ısı haritasını kaydediyorum
plt.figure(figsize=(16, 12))
sns.heatmap(corr, cmap='RdBu', center=0, annot=False, linewidths=0.5)
plt.title('Korelasyon Haritası')
plt.tight_layout()
plt.savefig('korelasyon_haritasi.png')
plt.close()
print("korelasyon_haritasi.png kaydedildi.")

# ============================================================
# 10. BOXPLOT - STRES SKORU
# ============================================================

# risk grubuna göre stres skorunu boxplot ile görselleştiriyorum
plt.figure(figsize=(8, 5))
sns.boxplot(x='sleep_disorder_risk', y='stress_score', data=df,
            order=['Healthy', 'Mild', 'Moderate', 'Severe'],
            hue='sleep_disorder_risk', legend=False,
            palette=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
plt.title('Risk Grubuna Göre Stres Skoru')
plt.xlabel('Sleep Disorder Risk')
plt.ylabel('Stress Score')
plt.tight_layout()
plt.savefig('stres_boxplot.png')
plt.close()
print("stres_boxplot.png kaydedildi.")

# ============================================================
# 11. BOXPLOT - UYKU SÜRESİ
# ============================================================

# risk grubuna göre uyku süresini boxplot ile görselleştiriyorum
plt.figure(figsize=(8, 5))
sns.boxplot(x='sleep_disorder_risk', y='sleep_duration_hrs', data=df,
            order=['Healthy', 'Mild', 'Moderate', 'Severe'],
            hue='sleep_disorder_risk', legend=False,
            palette=['#2ecc71', '#f39c12', '#e67e22', '#e74c3c'])
plt.title('Risk Grubuna Göre Uyku Süresi')
plt.xlabel('Sleep Disorder Risk')
plt.ylabel('Sleep Duration (hrs)')
plt.tight_layout()
plt.savefig('uyku_suresi_boxplot.png')
plt.close()
print("uyku_suresi_boxplot.png kaydedildi.")

print("\nEDA tamamlandı.")