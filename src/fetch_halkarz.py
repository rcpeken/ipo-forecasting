"""
halkarz.com'dan güncel halka arz verilerini otomatik çeker.

Amaç: predict.py'daki şirket bilgilerini elle girmek zorunda kalmamak.

Çıktı: data/raw/halkarz_guncel.csv
Kolonlar: kod, sirket_adi, url, talep_tarihi, sonuclar_aciklandi, halka_arz_fiyati,
          dagitim_yontemi, toplam_lot, sermaye_artisi_lot, ortak_satisi_lot,
          sermaye_artisi_yuzde, ortak_satisi_yuzde, bireysel_kisi, bireysel_lot,
          bireysel_lot_basina_kisi, ilk_islem_tarihi

NOT - "bireysel_lot_basina_kisi" hakkında: Eşit dağıtım yapan şirketler genelde
"X kat talep toplandı" rakamı yayınlamıyor. Ama bireysel gruba kaç kişinin katıldığı
ve o gruba kaç lot ayrıldığı yayınlanıyor. Kişi başına düşen lot = tahsis / katılımcı
sayısı; bu, talep yoğunluğunun dolaylı ama kullanışlı bir ölçüsü (çok katılım = kişi
başına az lot = yüksek talep). Talep toplama oranının bulunamadığı şirketlerde bunu
alternatif sinyal olarak kullanabiliriz.

Kullanım:
    python src/fetch_halkarz.py            # VARSAYILAN: sadece henüz borsada işlem
                                           # görmeye BAŞLAMAMIŞ halka arzları çeker
                                           # (karar verilecek olanlar bunlar)
    python src/fetch_halkarz.py 25         # en yeni 25 tanesini çeker (bekleyen/işlem
                                           # gören ayrımı yapmadan)
    python src/fetch_halkarz.py yillar 2023 2026
                                           # 2023-2026 arası TÜM halka arzları çeker
                                           # (veri setini büyütmek için) ->
                                           # data/raw/halkarz_gecmis.csv
"""
import csv
import re
import sys
import time
import urllib.request

from bs4 import BeautifulSoup

BASE = "https://halkarz.com/"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
from paths import HALKARZ_GUNCEL as OUT_PATH, HALKARZ_GECMIS as GECMIS_OUT_PATH

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def parse_listing(html: str):
    """
    Halka arz listesini ayrıştırır.

    ÖNEMLİ: Ana sayfada iki sekme var - "İlk Halka Arzlar" (gerçek, fiyatı belli
    olmuş arzlar) ve "Taslak Arzlar" (201 adet; sadece başvuru yapmış, henüz
    fiyatlanmamış şirketler). Taslaklar da "henüz işlem görmüyor" durumunda
    olduğu için bunları ayıklamazsak bekleyen-arz taraması 200+ şirketi boşuna
    geziyor. Bu yüzden sekme yapısı varsa SADECE İLK sekmeyi alıyoruz.
    (Yıl arşivi sayfalarında sekme yok, orada tüm liste geçerli.)
    """
    soup = BeautifulSoup(html, "html.parser")
    kapsam = soup.select_one("div.tab_item") or soup

    items = []
    for art in kapsam.select("article.index-list"):
        kod_el = art.select_one(".il-bist-kod")
        baglanti = art.select_one(".il-halka-arz-sirket a")
        if not (kod_el and baglanti and baglanti.get("href")):
            continue
        zaman = art.select_one("time")
        items.append({
            "kod": kod_el.get_text(strip=True),
            "sirket_adi": baglanti.get_text(strip=True),
            "url": baglanti["href"],
            "talep_tarihi": zaman.get("datetime", "") if zaman else "",
            # Rozet: sonuçlar açıklandıysa özel bir ikon konuyor
            "sonuclar_aciklandi": int(bool(art.select_one(".snc-badge"))),
        })
    return items


def _sayi(metin: str) -> float:
    """'62.500.000' -> 62500000.0   ('.' binlik ayırıcı)"""
    m = re.search(r"[\d.]+", metin or "")
    if not m:
        return float("nan")
    try:
        return float(m.group(0).replace(".", ""))
    except ValueError:
        return float("nan")


def parse_detail(html: str) -> dict:
    """
    Şirket detay sayfasından halka arz yapısını ve sonuç tablosunu ayrıştırır.

    Sayfada iki farklı yapı var, ikisi de BeautifulSoup ile geziliyor:
      1. Künye tablosu: <tr><td><em>Etiket :</em></td><td>Değer</td></tr>
      2. "Özet Bilgiler" listesi: <ul class="aex-in"><li><h5>Başlık</h5><p>İçerik</p>
      3. Sonuç tablosu: <tr>Yurt İçi Bireysel | kişi | lot | %oran</tr>
    """
    soup = BeautifulSoup(html, "html.parser")
    out = {}

    # --- 1) Künye tablosu: <em> etiketli satırlar ---
    kunye = {}
    for tr in soup.select("tr"):
        em = tr.find("em")
        if not em:
            continue
        hucreler = tr.find_all("td")
        if len(hucreler) >= 2:
            etiket = em.get_text(strip=True).rstrip(":").strip()
            kunye[etiket] = hucreler[1].get_text(" ", strip=True)

    fiyat = kunye.get("Halka Arz Fiyatı/Aralığı", "")
    fm = re.search(r"([\d.,]+)\s*TL", fiyat)
    out["halka_arz_fiyati"] = fm.group(1).replace(".", "").replace(",", ".") if fm else ""

    # "Eşit Dağıtım **" -> yıldızları ve fazla boşluğu at
    out["dagitim_yontemi"] = kunye.get("Dağıtım Yöntemi", "").replace("*", "").strip()
    out["toplam_lot"] = _sayi(kunye.get("Pay", ""))
    out["ilk_islem_tarihi"] = kunye.get("Bist İlk İşlem Tarihi", "").strip()

    # --- 2) Özet Bilgiler: "Halka Arz Şekli" ve "Fonun Kullanım Yeri" ---
    ozet = {}
    for li in soup.select("ul.aex-in li"):
        h5, p = li.find("h5"), li.find("p")
        if h5 and p:
            ozet[h5.get_text(strip=True)] = p.get_text(" ", strip=True)

    sekil = ozet.get("Halka Arz Şekli", "")
    sa_m = re.search(r"Sermaye Artırımı\s*:\s*([\d.]+)\s*Lot", sekil)
    os_m = re.search(r"Ortak Satışı\s*:\s*([\d.]+)\s*Lot", sekil)
    sa = _sayi(sa_m.group(1)) if sa_m else float("nan")
    os_ = _sayi(os_m.group(1)) if os_m else 0.0
    out["sermaye_artisi_lot"] = sa
    out["ortak_satisi_lot"] = os_
    toplam = (sa if sa == sa else 0) + (os_ if os_ == os_ else 0)
    if toplam > 0 and sa == sa:
        out["sermaye_artisi_yuzde"] = round(sa / toplam * 100, 2)
        out["ortak_satisi_yuzde"] = round(os_ / toplam * 100, 2)
    else:
        out["sermaye_artisi_yuzde"] = float("nan")
        out["ortak_satisi_yuzde"] = float("nan")

    # Fon kullanım yeri metni (ham). Yüzdelere ayrıştırma ileride yapılacak -
    # bkz. context.md "Test Edilmiş Hipotezler" ve açık işler #4.
    out["fon_kullanim_yeri"] = ozet.get("Fonun Kullanım Yeri", "")

    # --- 3) Sonuç tablosu: "Yurt İçi Bireysel | 652.248 | 35.000.000 | %40" ---
    out["bireysel_kisi"] = float("nan")
    out["bireysel_lot"] = float("nan")
    out["bireysel_lot_basina_kisi"] = float("nan")
    for tr in soup.select("tr"):
        hucreler = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(hucreler) >= 3 and hucreler[0].strip() == "Yurt İçi Bireysel":
            kisi, lot = _sayi(hucreler[1]), _sayi(hucreler[2])
            out["bireysel_kisi"] = kisi
            out["bireysel_lot"] = lot
            if kisi == kisi and lot == lot and kisi:
                out["bireysel_lot_basina_kisi"] = round(lot / kisi, 2)
            break

    return out


def yil_arsivi_tara(bas_yil: int, son_yil: int):
    """Yıl arşivi sayfalarını (sayfalama dahil) gezip tüm halka arzları toplar."""
    hepsi = []
    gorulen_kodlar = set()
    for yil in range(bas_yil, son_yil + 1):
        sayfa = 1
        yil_sayisi = 0
        while True:
            url = f"{BASE}k/halka-arz/{yil}/" if sayfa == 1 else f"{BASE}k/halka-arz/{yil}/page/{sayfa}/"
            try:
                html = get(url)
            except Exception:
                break  # 404 = bu yıl için sayfa bitti
            items = parse_listing(html)
            if not items:
                break
            for it in items:
                if it["kod"] in gorulen_kodlar:
                    continue
                gorulen_kodlar.add(it["kod"])
                it["yil"] = yil
                hepsi.append(it)
                yil_sayisi += 1
            sayfa += 1
            time.sleep(0.5)
        print(f"  {yil}: {yil_sayisi} halka arz")
    return hepsi


def detaylari_cek(items, out_path):
    rows = []
    for i, it in enumerate(items, 1):
        print(f"  [{i}/{len(items)}] {it['kod']:8s} {it['sirket_adi'][:42]:42s} ", end="", flush=True)
        try:
            it.update(parse_detail(get(it["url"])))
            rows.append(it)
            print(f"OK (sermaye artışı %{it.get('sermaye_artisi_yuzde')}, "
                  f"kişi başı {it.get('bireysel_lot_basina_kisi')} lot)")
        except Exception as e:
            print(f"HATA: {e}")
        time.sleep(0.8)

    if not rows:
        print("Hiç veri çekilemedi.")
        return
    yaz(rows, out_path)


ALANLAR = [
    "kod", "sirket_adi", "url", "talep_tarihi", "sonuclar_aciklandi",
    "halka_arz_fiyati", "dagitim_yontemi", "toplam_lot",
    "sermaye_artisi_lot", "ortak_satisi_lot",
    "sermaye_artisi_yuzde", "ortak_satisi_yuzde",
    "bireysel_kisi", "bireysel_lot", "bireysel_lot_basina_kisi",
    "ilk_islem_tarihi", "fon_kullanim_yeri",
]


def yaz(rows, out_path):
    """CSV'ye yazar. rows boş olsa bile başlık satırını yazar (predict.py
    'bekleyen yok' durumunu düzgün gösterebilsin diye)."""
    alanlar = list(ALANLAR)
    if any("yil" in r for r in rows):
        alanlar.append("yil")
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} satır yazıldı: {out_path}")


def islem_gormemis_mi(detay: dict) -> bool:
    """Borsada işlem görmeye henüz başlamamış mı? ('Hazırlanıyor...' = başlamamış)"""
    t = str(detay.get("ilk_islem_tarihi", "")).strip()
    return (t == "") or ("Hazırlanıyor" in t)


def bekleyenleri_cek():
    """
    Sadece henüz işlem görmeye başlamamış halka arzları çeker.

    Ana sayfadaki liste en yeniden eskiye sıralı; bekleyenler en üstte olur.
    Bu yüzden baştan başlayıp, üst üste birkaç tane 'zaten işlem görüyor' ile
    karşılaşınca duruyoruz - böylece 100 şirketin detayını boşuna çekmiyoruz.
    """
    DUR_ESIGI = 3  # üst üste bu kadar 'işlem görüyor' görünce dur

    print("halkarz.com kontrol ediliyor (sadece bekleyen halka arzlar)...\n")
    items = parse_listing(get(BASE))

    bekleyenler = []
    ardisik_islem_goren = 0
    for it in items:
        try:
            detay = parse_detail(get(it["url"]))
        except Exception as e:
            print(f"  {it['kod']:8s} HATA: {e}")
            continue
        time.sleep(0.8)

        if islem_gormemis_mi(detay):
            it.update(detay)
            bekleyenler.append(it)
            ardisik_islem_goren = 0
            print(f"  {it['kod']:8s} {it['sirket_adi'][:42]:42s} ← BEKLİYOR (henüz işlem görmüyor)")
        else:
            ardisik_islem_goren += 1
            print(f"  {it['kod']:8s} {it['sirket_adi'][:42]:42s}   zaten işlem görüyor "
                  f"({detay.get('ilk_islem_tarihi')})")
            if ardisik_islem_goren >= DUR_ESIGI:
                print(f"\n  → {DUR_ESIGI} ardışık 'işlem görüyor' bulundu, tarama durduruldu.")
                break

    return bekleyenler


def main():
    if len(sys.argv) > 1 and sys.argv[1].lower() == "yillar":
        bas = int(sys.argv[2]) if len(sys.argv) > 2 else 2023
        son = int(sys.argv[3]) if len(sys.argv) > 3 else 2026
        print(f"{bas}-{son} yıl arşivleri taranıyor...")
        items = yil_arsivi_tara(bas, son)
        print(f"\nToplam {len(items)} benzersiz halka arz bulundu. Detaylar çekiliyor...\n")
        detaylari_cek(items, GECMIS_OUT_PATH)
        return

    # Argüman verilmemişse: SADECE bekleyen (henüz işlem görmemiş) halka arzlar
    if len(sys.argv) == 1:
        bekleyenler = bekleyenleri_cek()
        if bekleyenler:
            print(f"\n{len(bekleyenler)} bekleyen halka arz bulundu.")
        else:
            print("\nŞu an bekleyen (henüz işlem görmemiş) halka arz YOK.")
        yaz(bekleyenler, OUT_PATH)   # boş olsa bile başlıklı dosya yaz
        return

    limit = int(sys.argv[1])

    print("halkarz.com ana sayfası çekiliyor...")
    items = parse_listing(get(BASE))
    print(f"{len(items)} halka arz bulundu, en yeni {min(limit, len(items))} tanesi işlenecek.\n")

    rows = []
    for it in items[:limit]:
        print(f"  {it['kod']:8s} {it['sirket_adi'][:45]:45s} ", end="", flush=True)
        try:
            detail = parse_detail(get(it["url"]))
            it.update(detail)
            rows.append(it)
            sa = it.get("sermaye_artisi_yuzde")
            lb = it.get("bireysel_lot_basina_kisi")
            print(f"OK (sermaye artışı %{sa}, kişi başı {lb} lot)")
        except Exception as e:
            print(f"HATA: {e}")
        time.sleep(1)  # siteye nazik davran

    if not rows:
        print("Hiç veri çekilemedi.")
        return
    yaz(rows, OUT_PATH)


if __name__ == "__main__":
    main()
