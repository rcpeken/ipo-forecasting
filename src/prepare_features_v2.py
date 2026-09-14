"""
v2 feature hazırlama - genişletilmiş veri seti (2023-2026, ~145 şirket).

v1'den farkları:
  1. Veri kaynağı tamamen halkarz.com (tutarlı, hepsi halka arz ÖNCESİ bilinen değerler).
     v1'de bazı satırlar "gerçekleşen" değerlerle doluydu -> train/serve tutarsızlığı vardı.
  2. Yeni feature: bireysel_lot_basina_kisi (talep yoğunluğunun dolaylı ölçüsü).
     Bu, "kaç kat talep" yayınlamayan şirketlerde bile mevcut.
  3. Hedef değişken fiyat verisinden kendimiz hesaplıyoruz (compute_target.py),
     böylece arzanaliz'in kapsamadığı 2023-2024 şirketleri de kullanılabiliyor.

Girdi:  data/raw/halkarz_gecmis.csv + data/processed/hedef_hesaplanan.csv
Çıktı:  data/processed/features_v2.csv
"""
import sys

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import (HALKARZ_GECMIS as OZELLIK_PATH,
                   HEDEF_HESAPLANAN as HEDEF_PATH,
                   FEATURES_V2 as OUT_PATH)


def main():
    ozellik = pd.read_csv(OZELLIK_PATH, encoding="utf-8-sig")
    hedef = pd.read_csv(HEDEF_PATH, encoding="utf-8-sig")

    df = ozellik.merge(
        hedef[["kod", "acilis_tavan_serisi", "en_uzun_seri_hesaplanan",
               "toplam_tavan_gunu", "ilk_gun_getiri_yuzde", "ilk_gun_fiyat_guvenilir"]],
        on="kod", how="inner",
    )
    print(f"Özellik tablosu: {len(ozellik)} satır, hedef tablosu: {len(hedef)} satır")
    print(f"Eşleşen (modellenebilir): {len(df)} şirket\n")

    # --- feature'lar ---
    # Talep yoğunluğu: kişi başına düşen lot. Çarpık dağıldığı için log alıyoruz
    # (1.4 ile 221 arasında değişiyor, log bunu lineer modeller için ehlileştirir).
    df["log_lot_basina_kisi"] = np.log(df["bireysel_lot_basina_kisi"].replace(0, np.nan))

    # Halka arz büyüklüğü (TL) - nominal olduğu için enflasyondan etkilenir,
    # bu yüzden hem log'unu alıyoruz hem de yıl feature'ı ile birlikte kullanıyoruz.
    df["halka_arz_buyuklugu_tl"] = df["toplam_lot"] * df["halka_arz_fiyati"]
    df["log_buyukluk"] = np.log(df["halka_arz_buyuklugu_tl"].replace(0, np.nan))

    df["esit_dagitim"] = (
        df["dagitim_yontemi"].astype(str).str.contains("Eşit", na=False).astype(int)
    )

    # --- hedefler ---
    df["hedef_acilis_serisi"] = df["acilis_tavan_serisi"]
    df["hedef_en_az_3_gun"] = (df["acilis_tavan_serisi"] >= 3).astype(int)
    df["hedef_en_az_5_gun"] = (df["acilis_tavan_serisi"] >= 5).astype(int)

    kolonlar = [
        "kod", "sirket_adi", "yil", "ilk_islem_tarihi",
        "sermaye_artisi_yuzde", "ortak_satisi_yuzde",
        "bireysel_lot_basina_kisi", "log_lot_basina_kisi", "bireysel_kisi",
        "halka_arz_fiyati", "toplam_lot", "halka_arz_buyuklugu_tl", "log_buyukluk",
        "esit_dagitim",
        "hedef_acilis_serisi", "hedef_en_az_3_gun", "hedef_en_az_5_gun",
        "en_uzun_seri_hesaplanan", "toplam_tavan_gunu", "ilk_gun_getiri_yuzde",
        # Hedefin ne kadar güvenilir olduğunu gösteren bayrak. 0 ise ilk gün için
        # halka arz fiyatı referansı kullanılamamış (düzeltilmiş fiyat sorunu),
        # "tavanda kilitlenme" imzasına düşülmüş -> hedef eksik saymış olabilir.
        "ilk_gun_fiyat_guvenilir",
    ]
    out = df[[k for k in kolonlar if k in df.columns]]
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("Yıl dağılımı:")
    print(out["yil"].value_counts().sort_index())
    print("\nAçılış tavan serisi dağılımı:")
    print(out["hedef_acilis_serisi"].value_counts().sort_index())
    print(f"\n'En az 3 gün' pozitif oranı: {out['hedef_en_az_3_gun'].mean():.1%}")
    print(f"Kişi başına lot verisi eksik olan: {out['bireysel_lot_basina_kisi'].isna().sum()}")
    print(f"\nYazıldı: {OUT_PATH}  ({len(out)} satır)")


if __name__ == "__main__":
    main()
