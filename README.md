# Halka Arz Tavan Tahmin Sistemi

BIST'te yeni halka arz olan bir şirketin, borsada işlem görmeye başladıktan sonra
**kaç gün üst üste tavan gideceğini** tahmin eden uçtan uca bir makine öğrenmesi projesi.

> *An end-to-end ML pipeline that predicts how many consecutive daily limit-up ("tavan")
> days a newly listed Borsa Istanbul IPO will have. Scrapes IPO data, computes the target
> from raw price series, trains a calibrated probability model, and outputs decision
> support for upcoming offerings. Turkish language project.*

> ⚠️ **Bu bir yatırım tavsiyesi değildir.** Küçük bir veri setiyle (N=66) eğitilmiş
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
    ≥1 gün : % 61  ████████████
    ≥3 gün : % 46  █████████
    ≥5 gün : % 24  ████
    ≥7 gün : % 12  ██

  MODEL KALİTESİ — modelin o eşikte şirketleri ayırt etme gücü
  (bu şirkete özgü değil, 66 şirketlik geçmişe dayalı genel performans)
    Eşik     AUC      %90 aralık  Poz.  Ayrım gücü
    ≥1 gün  0.66     0.54 - 0.77    42  sınırlı ayrım gücü
    ≥3 gün  0.75     0.65 - 0.85    35  orta düzey ayrım gücü
    ≥5 gün  0.62     0.49 - 0.74    20  ayırt edemiyor (aralık 0.50'yi içeriyor)
    ≥7 gün  0.70     0.57 - 0.82    13  sınırlı ayrım gücü
```

Çıktı bilinçli olarak **iki bloğa ayrılmış**: üstteki blok bu şirket için üretilen
olasılık, alttaki blok modelin o eşikteki genel isabeti. İkisi farklı şeyler —
yüksek olasılık, modelin o eşikte iyi olduğu anlamına gelmiyor.

---

## Kurulum ve kullanım

```bash
git clone <repo-url>
cd halka-arz-tahmin
pip install pandas scikit-learn joblib requests

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
karşılaştırılarak doğrulandı (51 şirketin 45'i ±1 gün içinde eşleşti).

### Model
- **Algoritma:** Logistic Regression (`SimpleImputer` → `StandardScaler` → `LogReg`)
- **Feature'lar:** `log(halka arz büyüklüğü)`, `sermaye artışı %`, `ortak satışı %`
  — hepsi halka arz **öncesi** bilinen değerler (veri sızıntısı yok)
- **Eğitim verisi:** 66 şirket (2024-2026)
- **Değerlendirme:** Stratified 5-fold CV, ROC AUC + bootstrap güven aralığı

| Eşik | Pozitif örnek | ROC AUC | %90 güven aralığı | Ayrım gücü |
|---|---|---|---|---|
| ≥1 gün | 42 | 0.657 | 0.54 – 0.77 | sınırlı |
| **≥3 gün** | 35 | **0.752** | **0.65 – 0.85** | **orta düzey** |
| ≥5 gün | 20 | 0.616 | 0.49 – 0.74 | ayırt edemiyor (aralık 0.50'yi içeriyor) |
| ≥7 gün | 13 | 0.705 | 0.57 – 0.82 | sınırlı |

Değerlendirme, AUC'nin kendisine değil **güven aralığının alt sınırına** göre yapılıyor:
az örnekte tek bir AUC sayısı yanıltıcı olabiliyor. Kademeler bilinçli olarak yumuşak
tutuldu, çünkü aralıklar birbiriyle çakışıyor — örneğin ≥1 gün (0.54–0.77) ile ≥3 gün
(0.65–0.85) arasındaki fark istatistiksel olarak anlamlı olmayabilir. Pratik sonuç:
**≥3 gün en güvenilir eşik, ≥5 gün'e hiç güvenilmemeli.**

**En güçlü sinyal:** halka arz büyüklüğü (negatif korelasyon). Küçük arz = piyasada az
hisse = fiyatı yukarı itmek kolay.

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
