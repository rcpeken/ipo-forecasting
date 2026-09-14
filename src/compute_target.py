"""
Hedef değişkeni (tavan gün sayısı) fiyat verisinden KENDİMİZ hesaplar.

Neden gerekli: arzanaliz.com sadece 2025 Ocak'tan itibaren tavan serisi veriyor.
Veri setini 2023'e kadar genişletmek için eski şirketlerin tavan serilerini de
bilmemiz lazım - bunu isyatirimhisse'den çektiğimiz günlük fiyatlardan hesaplıyoruz.

Yöntem: BIST'te günlük fiyat limiti ±%10. Bir gün "tavan" sayılır eğer kapanış,
bir önceki kapanışa göre >= %9.5 arttıysa (yuvarlama payı bırakıyoruz). İlk gün için
karşılaştırma halka arz fiyatına göre yapılır.

Doğrulama: KPEKS için bu yöntem 5 gün veriyor; arzanaliz de 5 gün diyor (eşleşiyor).

Girdi:  data/raw/halkarz_gecmis.csv  (fetch_halkarz.py yillar ... çıktısı)
Çıktı:  data/processed/hedef_hesaplanan.csv

Kullanım:
    python src/compute_target.py
    python src/compute_target.py 20      # sadece ilk 20 şirketi işle (test için)
"""
import sys
import time

import pandas as pd
import requests

# NOT: isyatirimhisse kütüphanesi yerine İş Yatırım API'si DOĞRUDAN çağrılıyor.
# Sebep: kütüphanenin timeout'u 10 sn'ye sabitlenmiş ve sunucu yoğun olduğunda
# sürekli zaman aşımına düşüyordu. Doğrudan çağrıda timeout'u uzatıp retry ekliyoruz.
API_URL = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
TIMEOUT = 60
DENEME_SAYISI = 3


def fiyat_verisi_cek(kod: str, bas: str, bit: str):
    """İş Yatırım API'sinden günlük fiyat verisi çeker. Hata olursa tekrar dener."""
    son_hata = None
    for deneme in range(DENEME_SAYISI):
        try:
            r = requests.get(
                API_URL,
                params={"hisse": kod, "startdate": bas, "enddate": bit},
                timeout=TIMEOUT,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            r.raise_for_status()
            d = r.json()
            if not d.get("ok"):
                return []
            return d.get("value") or []
        except Exception as e:  # ağ hatası / json hatası
            son_hata = e
            time.sleep(2 * (deneme + 1))
    raise RuntimeError(f"{DENEME_SAYISI} denemede çekilemedi: {son_hata}")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import HALKARZ_GECMIS as IN_PATH, HEDEF_HESAPLANAN as OUT_PATH

TAVAN_ESIGI = 0.095   # %9.5 ve üzeri artış = tavan
TAKIP_GUNU = 30       # ilk işlem gününden sonra kaç işlem günü incelensin

TR_AYLAR = {
    "ocak": 1, "şubat": 2, "subat": 2, "mart": 3, "nisan": 4, "mayıs": 5, "mayis": 5,
    "haziran": 6, "temmuz": 7, "ağustos": 8, "agustos": 8, "eylül": 9, "eylul": 9,
    "ekim": 10, "kasım": 11, "kasim": 11, "aralık": 12, "aralik": 12,
}


def parse_tr_tarih(s: str):
    """'6 Şubat 2025' -> pd.Timestamp"""
    if not isinstance(s, str):
        return None
    parts = s.strip().split()
    if len(parts) != 3:
        return None
    try:
        gun = int(parts[0])
        ay = TR_AYLAR.get(parts[1].lower())
        yil = int(parts[2])
        if ay is None:
            return None
        return pd.Timestamp(year=yil, month=ay, day=gun)
    except (ValueError, TypeError):
        return None


def tavan_serisi_hesapla(gunler, halka_arz_fiyati):
    """
    Günlük verilerden tavan günlerini bulur.
    `gunler`: [(kapanis, min, max), ...] sırayla ilk işlem gününden itibaren.

    1. GÜN özel durum: İş Yatırım DÜZELTİLMİŞ fiyat veriyor (bedelsiz sermaye
    artırımı sonrası geçmiş fiyatlar geriye dönük bölünür) ve halkarz.com'daki
    halka arz fiyatı bazen aralık olabiliyor. Bu yüzden 1. günü halka arz
    fiyatıyla kıyaslamak güvenilmez sonuç veriyordu (ör. AKFIS'te -%85 gibi
    imkansız bir "ilk gün getirisi" çıkıyordu).
    Çözüm: 1. gün için önce halka arz fiyatı ile hesaplanan değişim mantıklı
    aralıkta mı diye bakıyoruz (±%15); değilse "tavanda kilitlenme" imzasını
    kullanıyoruz (gün boyu hiç işlem görmemiş = min == max == kapanış).

    Döner: (acilis_serisi, en_uzun_seri, toplam_tavan_gunu, ilk_gun_guvenilir)
    """
    tavan_bayraklari = []
    ilk_gun_guvenilir = True

    for i, (kapanis, dusuk, yuksek) in enumerate(gunler):
        if i == 0:
            degisim = None
            if halka_arz_fiyati and halka_arz_fiyati > 0:
                d = (kapanis - halka_arz_fiyati) / halka_arz_fiyati
                if -0.15 <= d <= 0.15:      # mantıklı aralıkta -> güvenilir
                    degisim = d
            if degisim is not None:
                tavan_bayraklari.append(degisim >= TAVAN_ESIGI)
            else:
                # fiyat referansı güvenilmez -> kilitlenme imzasına bak
                ilk_gun_guvenilir = False
                kilitli = (
                    dusuk == dusuk and yuksek == yuksek
                    and dusuk == yuksek == kapanis
                )
                tavan_bayraklari.append(bool(kilitli))
        else:
            onceki = gunler[i - 1][0]
            if onceki and onceki > 0:
                tavan_bayraklari.append((kapanis - onceki) / onceki >= TAVAN_ESIGI)
            else:
                tavan_bayraklari.append(False)

    # açılış serisi
    acilis = 0
    for b in tavan_bayraklari:
        if b:
            acilis += 1
        else:
            break

    # en uzun seri
    en_uzun = mevcut = 0
    for b in tavan_bayraklari:
        mevcut = mevcut + 1 if b else 0
        en_uzun = max(en_uzun, mevcut)

    return acilis, en_uzun, sum(tavan_bayraklari), ilk_gun_guvenilir


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None

    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    if limit:
        df = df.head(limit)

    sonuclar = []
    for i, row in df.iterrows():
        kod = row["kod"]
        tarih = parse_tr_tarih(row.get("ilk_islem_tarihi"))
        fiyat = row.get("halka_arz_fiyati")

        if tarih is None or not (fiyat == fiyat) or fiyat <= 0:
            print(f"  {kod:8s} ATLANDI (tarih/fiyat yok: '{row.get('ilk_islem_tarihi')}')")
            continue

        bas = tarih.strftime("%d-%m-%Y")
        bit = (tarih + pd.Timedelta(days=TAKIP_GUNU * 2)).strftime("%d-%m-%Y")
        try:
            kayitlar = fiyat_verisi_cek(kod, bas, bit)
        except Exception as e:
            print(f"  {kod:8s} HATA: {e}")
            continue

        if not kayitlar:
            print(f"  {kod:8s} VERİ YOK")
            continue

        veri = pd.DataFrame(kayitlar)
        veri["_t"] = pd.to_datetime(veri["HGDG_TARIH"], format="%d-%m-%Y", errors="coerce")
        veri = veri.sort_values("_t")
        veri = veri.dropna(subset=["HGDG_KAPANIS"]).head(TAKIP_GUNU)
        if len(veri) == 0:
            print(f"  {kod:8s} KAPANIŞ VERİSİ YOK")
            continue

        gunler = list(zip(veri["HGDG_KAPANIS"], veri.get("HGDG_MIN"), veri.get("HGDG_MAX")))
        acilis, en_uzun, toplam, ilk_guv = tavan_serisi_hesapla(gunler, float(fiyat))
        ilk_gun_getiri = (gunler[0][0] - float(fiyat)) / float(fiyat) * 100
        sonuclar.append({
            "kod": kod,
            "ilk_islem_tarihi": row.get("ilk_islem_tarihi"),
            "halka_arz_fiyati": fiyat,
            "acilis_tavan_serisi": acilis,
            "en_uzun_seri_hesaplanan": en_uzun,
            "toplam_tavan_gunu": toplam,
            "ilk_gun_getiri_yuzde": round(ilk_gun_getiri, 2),
            "ilk_gun_fiyat_guvenilir": int(ilk_guv),
            "islem_gunu_sayisi": len(gunler),
        })
        bayrak = "" if ilk_guv else "  [1.gün fiyat referansı güvenilmez, kilitlenme imzası kullanıldı]"
        print(f"  {kod:8s} açılış: {acilis:2d} | en uzun: {en_uzun:2d} | toplam: {toplam:2d}{bayrak}")
        time.sleep(0.3)

    if not sonuclar:
        print("Hiç sonuç hesaplanamadı.")
        return

    out = pd.DataFrame(sonuclar)
    out.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n{len(out)} şirket için hedef hesaplandı: {OUT_PATH}")
    print("\nAçılış tavan serisi dağılımı:")
    print(out["acilis_tavan_serisi"].value_counts().sort_index())


if __name__ == "__main__":
    main()
