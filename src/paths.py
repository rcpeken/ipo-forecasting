"""
Proje içi dosya yolları - tek yerden yönetilir.

Yollar, bu dosyanın konumundan türetilir (mutlak yol YAZILMAZ). Böylece proje
herhangi bir klasöre/bilgisayara taşındığında ya da repo klonlandığında
değişiklik yapmadan çalışır.
"""
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent      # .../halka-arz-tahmin
RAW = KOK / "data" / "raw"
PROCESSED = KOK / "data" / "processed"

# Ham veri (çekilen)
HALKARZ_GUNCEL = RAW / "halkarz_guncel.csv"       # bekleyen arzlar -> tahmin girdisi
HALKARZ_GECMIS = RAW / "halkarz_gecmis.csv"       # geçmiş arşiv -> eğitim kaynağı

# İşlenmiş veri
HEDEF_HESAPLANAN = PROCESSED / "hedef_hesaplanan.csv"
FEATURES_V2 = PROCESSED / "features_v2.csv"
MODELS_V2 = PROCESSED / "models_v2.joblib"

# Doğrulama referansı: bağımsız bir kaynaktan (arzanaliz.com) alınmış tavan serileri.
# Modelde kullanılmıyor ama compute_target.py çıktısını karşılaştırmak için değerli.
ARZANALIZ_REFERANS = RAW / "arzanaliz_tavan_serileri.csv"

# Klasörler yoksa oluştur
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
