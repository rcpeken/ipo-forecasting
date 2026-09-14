"""
HALKA ARZ TAHMİN SİSTEMİ — TEK KOMUTLUK ÇALIŞTIRICI

Hiçbir bilgiyi elle girmene gerek yok. Her şeyi kendisi çeker.

KULLANIM:
    python run.py           # GÜNLÜK KULLANIM (hızlı, ~10 saniye)
                            #   - halkarz.com'da SADECE henüz borsada işlem görmeye
                            #     başlamamış halka arzları bulur (karar verilecek olanlar)
                            #   - onlar için tahmin basar
                            #   - bekleyen arz yoksa bunu söyler ve çıkar

    python run.py tam       # TAM YENİLEME (yavaş, birkaç dakika)
                            #   - 2023'ten bugüne tüm halka arzları çeker
                            #   - tavan serilerini fiyat verisinden hesaplar
                            #   - modeli yeniden eğitir
                            #   - sonra tahminleri basar
                            # Ayda bir / yeni veri biriktiğinde çalıştır.
"""
import subprocess
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

KOK = Path(__file__).parent
PY = sys.executable


def kosu(baslik, script, *args):
    # flush şart: alt süreçler stdout'u doğrudan kullanıyor, biz tamponlarsak
    # başlıklar çıktının sonunda toplanıp sıra karışıyor.
    print(f"\n{'─'*70}\n▶ {baslik}\n{'─'*70}", flush=True)
    sonuc = subprocess.run([PY, str(KOK / "src" / script), *args], cwd=str(KOK))
    if sonuc.returncode != 0:
        print(f"\n✗ '{script}' hata verdi (kod {sonuc.returncode}). Devam edilmiyor.")
        sys.exit(sonuc.returncode)


def main():
    tam = len(sys.argv) > 1 and sys.argv[1].lower() in ("tam", "full", "yenile")

    print("=" * 70)
    print("  HALKA ARZ TAVAN TAHMİN SİSTEMİ")
    print("  " + ("TAM YENİLEME modu (veri + model yeniden üretilecek)" if tam
                  else "Hızlı mod (mevcut model kullanılacak)"))
    print("=" * 70, flush=True)

    if tam:
        kosu("1/5  Geçmiş halka arzlar çekiliyor (2023-bugün)", "fetch_halkarz.py", "yillar", "2023", "2026")
        kosu("2/5  Tavan serileri fiyat verisinden hesaplanıyor", "compute_target.py")
        kosu("3/5  Feature'lar hazırlanıyor", "prepare_features_v2.py")
        kosu("4/5  Model eğitiliyor ve test ediliyor", "train_v2.py")
        kosu("5/5  Bekleyen (henüz işlem görmemiş) halka arzlar çekiliyor", "fetch_halkarz.py")
    else:
        kosu("1/2  Bekleyen (henüz işlem görmemiş) halka arzlar çekiliyor", "fetch_halkarz.py")

    kosu("Tahminler", "predict.py")


if __name__ == "__main__":
    main()
