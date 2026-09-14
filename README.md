# Halka Arz Tavan Tahmin Sistemi

BIST'te yeni halka arz olan bir şirketin, borsada işlem görmeye başladıktan sonra
**kaç gün üst üste tavan gideceğini** tahmin eden uçtan uca bir makine öğrenmesi projesi.

> *An end-to-end ML pipeline that predicts how many consecutive daily limit-up ("tavan")
> days a newly listed Borsa Istanbul IPO will have. Scrapes IPO data, computes the target
> from raw price series, trains a calibrated probability model, and outputs decision
> support for upcoming offerings. Turkish language project.*

> ⚠️ **Bu bir yatırım tavsiyesi değildir.** Sınırlı bir veri setiyle (N=136) eğitilmiş
> deneysel bir projedir.

---

## Ne yapıyor?

Henüz borsada işlem görmeye başlamamış halka arzları otomatik bulur ve her biri için
olasılık üretir:

```
### Henüz işlem görmeye başlamamış halka arzlar (1 adet) ###

======================================================================
  ÖRNEK — ÖRNEK A.Ş.
======================================================================
  Talep tarihi     : 9-10-11 Eylül 2026
  Halka arz fiyatı : 25.52 TL
  Sermaye artışı   : %71.43  |  Ortak satışı: %28.57
  Bireysel katılım : 652.248 kişi  →  kişi başına 53.66 lot

  Halka arz büyüklüğü: 2.23 milyar TL  (modelin en güçlü sinyali)

  TAHMİN — bu şirketin tavan gitme olasılığı
    ≥1 gün : % 81  ████████████████
    ≥3 gün : % 60  ███████████
    ≥5 gün : % 40  ████████
    ≥7 gün : % 23  ████

  MODEL KALİTESİ — modelin o eşikte şirketleri ayırt etme gücü
  (bu şirkete özgü değil, 136 şirketlik geçmişe dayalı genel performans)
    Eşik     AUC      %90 aralık  Poz.  Ayrım gücü
    ≥1 gün  0.77     0.69 - 0.85   114  orta düzey ayrım gücü
    ≥3 gün  0.76     0.69 - 0.83    96  orta düzey ayrım gücü
    ≥5 gün  0.63     0.55 - 0.71    66  sınırlı ayrım gücü
    ≥7 gün  0.64     0.55 - 0.72    43  sınırlı ayrım gücü
```

Çıktı bilinçli olarak **iki bloğa ayrılmış**: üstteki blok bu şirket için üretilen
olasılık, alttaki blok modelin o eşikteki genel isabeti. İkisi farklı şeyler —
yüksek olasılık, modelin o eşikte iyi olduğu anlamına gelmiyor.

---

## Kurulum ve kullanım

```bash
git clone <repo-url>
cd halka-arz-tahmin
pip install pandas scikit-learn joblib requests beautifulsoup4

python run.py          # bekleyen halka arzları bul + tahmin üret (~10 sn)
python run.py tam      # her şeyi sıfırdan üret: veri + hedef + model (birkaç dakika)
```

Elle veri girişi yok — sistem gerekli her şeyi kendisi çeker.

---

## Nasıl çalışıyor?

```
halkarz.com (web scraping) ──> X: arz büyüklüğü, sermaye yapısı ─┐
                                                                  ├──> model eğitimi
İş Yatırım API (fiyat verisi) ──> y: kaç gün tavan gitti ────────┘         │
                                                                            ▼
                    yeni bekleyen halka arz ───────────────────────> olasılık tahmini
```

| Aşama | Script | Ne yapar |
|---|---|---|
| 1. Veri toplama | `fetch_halkarz.py` | halkarz.com'dan arz bilgilerini çeker |
| 2. Hedef üretimi | `compute_target.py` | Günlük fiyat serisinden tavan günlerini hesaplar |
| 3. Feature | `prepare_features_v2.py` | Modelleme tablosunu oluşturur |
| 4. Eğitim | `train_v2.py` | Model karşılaştırmaları + nihai model |
| 5. Tahmin | `predict.py` | Bekleyen arzlar için olasılık üretir |

### Hedef değişken
BIST'te günlük fiyat limiti ±%10. Bir gün "tavan" sayılır eğer kapanış bir önceki
kapanışa göre ≥%9.5 arttıysa. İlk işlem gününden itibaren kesintisiz kaç gün tavan
olduğu sayılır (takip penceresi: ilk 30 işlem günü). Hesaplama, bağımsız bir kaynakla
karşılaştırılarak doğrulandı (51 şirketin 41'i tam, 46'sı ±1 gün içinde eşleşti).

### Model
- **Algoritma:** Logistic Regression (`SimpleImputer` → `StandardScaler` → `LogReg`)
- **Feature'lar:** `log(halka arz büyüklüğü)`, `sermaye artışı %`, `ortak satışı %`
  — hepsi halka arz **öncesi** bilinen değerler (veri sızıntısı yok)
- **Eğitim verisi:** 136 şirket (2023-2026, hedefi güvenilir hesaplanabilenler)
- **Değerlendirme:** Stratified 5-fold CV, ROC AUC + bootstrap güven aralığı

| Eşik | Pozitif örnek | ROC AUC | %90 güven aralığı | Ayrım gücü |
|---|---|---|---|---|
| **≥1 gün** | 114 | **0.774** | 0.69 – 0.85 | orta düzey |
| **≥3 gün** | 96 | **0.762** | 0.69 – 0.83 | orta düzey |
| ≥5 gün | 66 | 0.632 | 0.55 – 0.71 | sınırlı |
| ≥7 gün | 43 | 0.637 | 0.55 – 0.72 | sınırlı |

Değerlendirme, AUC'nin kendisine değil **güven aralığının alt sınırına** göre yapılıyor:
az örnekte tek bir AUC sayısı yanıltıcı olabiliyor. Kademeler bilinçli olarak yumuşak
tutuldu, çünkü aralıklar birbiriyle çakışıyor. Pratik sonuç: **≥1 ve ≥3 gün eşikleri
kullanılabilir; ≥5 ve ≥7 gün tahminleri zayıf, temkinli okunmalı.**

**En güçlü sinyal:** halka arz büyüklüğü. Standardize katsayısı −0.97 ile diğer iki
feature'ı (±0.10) ezici şekilde geride bırakıyor. Küçük arz = piyasada az hisse =
fiyatı yukarı itmek kolay.

**Neden Logistic Regression?** Gradient Boosting ve Random Forest ile aynı veri ve
feature setinde karşılaştırıldı:

| Model | Test AUC (CV) | Eğitim AUC | Ezber farkı | Brier |
|---|---|---|---|---|
| **Logistic Regression** | **0.762** | 0.779 | **0.017** | **0.172** |
| Random Forest | 0.753 | 0.922 | 0.170 | 0.174 |
| Gradient Boosting | 0.699 | 0.924 | 0.225 | 0.196 |

Logistic Regression üç metrikte de önde. Ayrıca eğitim–test farkı çok küçük (0.017),
yani öğrendiği örüntü bu veri setine özgü değil; ağaç modelleri eğitim verisini
neredeyse ezberliyor. Katsayılarının yorumlanabilir olması da ek avantaj.

---

## Proje yapısı

```
halka-arz-tahmin/
├── run.py                      Tek giriş noktası
├── src/
│   ├── paths.py                Dosya yolları (göreli)
│   ├── fetch_halkarz.py        Web scraping
│   ├── compute_target.py       Hedef değişken hesaplama
│   ├── prepare_features_v2.py  Feature engineering
│   ├── train_v2.py             Model eğitimi + karşılaştırmalar
│   └── predict.py              Tahmin
└── data/
    ├── raw/                    Çekilen ham veri
    └── processed/              İşlenmiş veri + eğitilmiş model
```

## Veri kaynakları
- **halkarz.com** — halka arz bilgileri (web scraping)
- **İş Yatırım** — günlük fiyat verisi (public JSON endpoint)
- **arzanaliz.com** — hedef değişkenin bağımsız doğrulaması

Veriler kişisel/eğitim amaçlı, düşük hızda (istekler arası bekleme ile) çekilmektedir.

## Lisans
MIT
