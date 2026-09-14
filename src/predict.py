"""
Yeni halka arzlar için "girmeye değer mi" tahmini — OTOMATİK.

Artık şirket bilgilerini elle girmene gerek yok. Script, halkarz.com'dan çekilen
güncel veriyi (data/raw/halkarz_guncel.csv) okuyup, henüz borsada işlem görmeye
BAŞLAMAMIŞ halka arzlar için tahmin üretir — yani "girsem mi girmesem mi" kararını
vereceğin, tam da o anda aktüel olan şirketler için.

KULLANIM:
    python src/fetch_halkarz.py     # önce güncel veriyi çek (haftada bir yeterli)
    python src/predict.py           # sonra tahminleri gör

    python src/predict.py TUMU      # sadece bekleyenleri değil, çekilen tüm
                                    # şirketleri göster (geçmişi test etmek için)

UYARI: Küçük bir veri setiyle eğitilmiş DENEYSEL bir model (eğitim örnek sayısı
model dosyasında tutulur ve çıktıda gösterilir). Yatırım tavsiyesi değildir.
"""
import sys

import joblib
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import MODELS_V2 as MODEL_PATH, HALKARZ_GUNCEL as DATA_PATH


def feature_uret(row):
    """halkarz.com satırından modelin beklediği feature'ları üretir."""
    import numpy as np
    buyukluk = row.get("toplam_lot", float("nan")) * row.get("halka_arz_fiyati", float("nan"))
    return {
        "log_buyukluk": np.log(buyukluk) if buyukluk == buyukluk and buyukluk > 0 else float("nan"),
        "sermaye_artisi_yuzde": row.get("sermaye_artisi_yuzde", float("nan")),
        "ortak_satisi_yuzde": row.get("ortak_satisi_yuzde", float("nan")),
    }


def tahmin_yazdir(row, models, features, egitim_n="?"):
    print(f"\n{'='*70}")
    print(f"  {row['kod']} — {row['sirket_adi']}")
    print(f"{'='*70}")
    print(f"  Talep tarihi     : {row.get('talep_tarihi','')}")
    print(f"  İlk işlem tarihi : {row.get('ilk_islem_tarihi','')}")
    print(f"  Halka arz fiyatı : {row.get('halka_arz_fiyati','')} TL")
    print(f"  Sermaye artışı   : %{row.get('sermaye_artisi_yuzde','')}"
          f"  |  Ortak satışı: %{row.get('ortak_satisi_yuzde','')}")
    lb = row.get("bireysel_lot_basina_kisi")
    if lb == lb:  # NaN değilse
        print(f"  Bireysel katılım : {int(row['bireysel_kisi']):,} kişi"
              f"  →  kişi başına {lb} lot".replace(",", "."))
    print()

    uretilen = feature_uret(row)
    X_new = pd.DataFrame([{f: uretilen.get(f, float("nan")) for f in features}])

    buyukluk_tl = row.get("toplam_lot", 0) * row.get("halka_arz_fiyati", 0)
    if buyukluk_tl:
        print(f"  Halka arz büyüklüğü: {buyukluk_tl/1e9:.2f} milyar TL  "
              f"(modelin en güçlü sinyali)")
    print()

    # --- BLOK 1: bu şirket için tahmin ---
    print("  TAHMİN — bu şirketin tavan gitme olasılığı")
    for esik, bilgi in sorted(models.items()):
        model = bilgi["model"] if isinstance(bilgi, dict) else bilgi
        proba = model.predict_proba(X_new)[0, 1]
        bar = "█" * int(proba * 20)
        print(f"    ≥{esik} gün : %{proba*100:>3.0f}  {bar}")

    # --- BLOK 2: modelin genel kalitesi (bu şirkete özgü DEĞİL) ---
    # Ayrı blokta gösteriliyor çünkü yan yana konduğunda karışıyordu: yüksek olasılık
    # ile yüksek model kalitesi farklı şeyler, biri diğerini ima etmiyor.
    print("\n  MODEL KALİTESİ — modelin o eşikte şirketleri ayırt etme gücü")
    print(f"  (bu şirkete özgü değil, {egitim_n} şirketlik geçmişe dayalı genel performans)")
    print(f"    {'Eşik':<6} {'AUC':>5} {'%90 aralık':>15} {'Poz.':>5}  Ayrım gücü")
    print(f"    {'-'*6} {'-'*5} {'-'*15} {'-'*5}  {'-'*36}")
    for esik, bilgi in sorted(models.items()):
        if not isinstance(bilgi, dict):
            continue
        auc, alt, ust = bilgi.get("auc"), bilgi.get("auc_alt"), bilgi.get("auc_ust")
        n_poz, yorum = bilgi.get("pozitif_sayisi", "?"), bilgi.get("yorum", "")
        aralik = f"{alt:.2f} - {ust:.2f}" if alt is not None else "—"
        auc_s = f"{auc:.2f}" if auc is not None else "—"
        print(f"    ≥{esik} gün {auc_s:>5} {aralik:>15} {n_poz:>5}  {yorum}")
    print("\n  Not: Aralıklar birbiriyle çakışıyorsa eşikler arası kalite farkı")
    print(f"  anlamlı olmayabilir — N={egitim_n} ölçeğinde belirsizlik yüksek.")


def main():
    hepsi = len(sys.argv) > 1 and sys.argv[1].upper() == "TUMU"

    saved = joblib.load(MODEL_PATH)
    models, features = saved["models"], saved["features"]

    try:
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    except FileNotFoundError:
        print("Güncel veri bulunamadı. Önce şunu çalıştır:")
        print("    python src/fetch_halkarz.py")
        return

    if hepsi:
        secilen = df
        baslik = f"Çekilen tüm halka arzlar ({len(df)} adet)"
    else:
        # Henüz borsada işlem görmeye başlamamış olanlar = karar verilecek olanlar
        mask = df["ilk_islem_tarihi"].astype(str).str.contains("Hazırlanıyor", na=False)
        secilen = df[mask]
        baslik = f"Henüz işlem görmeye başlamamış halka arzlar ({len(secilen)} adet)"

    print(f"\n### {baslik} ###")

    if len(secilen) == 0:
        print("\nŞu an bekleyen (henüz işlem görmemiş) bir halka arz yok.")
        print("Geçmiş şirketleri görmek için: python src/predict.py TUMU")
        return

    for _, row in secilen.iterrows():
        tahmin_yazdir(row, models, features, saved.get("egitim_n", "?"))

    n = saved.get("egitim_n", "?")
    print(f"\n{'='*70}")
    print(f"  NOT: Bu bir yatırım tavsiyesi değildir. Model {n} şirketlik bir veri")
    print("  setiyle eğitildi (2023-2026, hedefi güvenilir hesaplanabilenler).")
    print("  Tahminleri kesinlik değil, kaba bir olasılık tahmini olarak oku;")
    print("  bu ölçekte AUC farkları da gürültü içerir.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
