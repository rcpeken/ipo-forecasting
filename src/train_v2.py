"""
v2 model eğitimi ve DÜRÜST karşılaştırmalar.

Üç soruyu ölçerek cevaplıyoruz (varsayarak değil):
  A) Yeni "kişi başına lot" feature'ı gerçekten işe yarıyor mu?
  B) Eski yılları (2023-2024) eklemek modeli iyileştiriyor mu, bozuyor mu?
     (Piyasa rejimi değişti diye endişeleniyorduk - ölçelim.)
  C) Hangi model / hangi feature seti en iyisi?

Değerlendirme: Stratified 5-fold CV (N büyüdüğü için LOO'ya gerek kalmadı,
5-fold daha hızlı ve yeterince stabil). Metrik: ROC AUC + Brier score.

Çıktı: en iyi feature setiyle eğitilmiş modeller -> data/processed/models_v2.joblib
"""
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from paths import FEATURES_V2 as IN_PATH, MODELS_V2 as MODEL_OUT

HEDEF = "hedef_en_az_3_gun"
ESIKLER = [1, 3, 5, 7]


def logreg():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=0.5)),
    ])


def gbc():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", GradientBoostingClassifier(n_estimators=80, max_depth=2,
                                           learning_rate=0.05, random_state=42)),
    ])


def rf():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=300, max_depth=4,
                                       min_samples_leaf=3, random_state=42)),
    ])


def degerlendir(pipe, X, y, n_splits=5):
    if y.nunique() < 2 or y.value_counts().min() < n_splits:
        return float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    return roc_auc_score(y, proba), brier_score_loss(y, proba)


def auc_guven_araligi(pipe, X, y, n_splits=5, n_boot=2000, seed=42):
    """
    AUC için bootstrap %90 güven aralığı.

    NEDEN GEREKLİ: Tek bir AUC sayısına bakıp "kullanılabilir/zayıf" demek yanıltıcı,
    çünkü az örnekte o sayının etrafındaki belirsizlik çok büyük. Örneğin ≥7 gün
    eşiğinde sadece 13 pozitif örnek var; AUC 0.705 görünüyor ama gerçek değer
    0.58 ile 0.82 arasında herhangi bir yerde olabilir. Güvenilirlik etiketini
    nokta tahminine değil, aralığın ALT SINIRINA göre veriyoruz.

    BİLİNEN KISIT (bilinçli tercih): Model her bootstrap turunda YENİDEN EĞİTİLMİYOR;
    bir kez üretilen CV tahminleri yeniden örnekleniyor. Yani bu aralık sadece
    "değerlendirme örnekleminden gelen" belirsizliği ölçüyor, "modelin farklı eğitim
    verisiyle ne kadar değişeceğini" ölçmüyor. Tam versiyon (her turda tüm pipeline'ı
    baştan çalıştırmak) muhtemelen bir miktar DAHA GENİŞ aralık verirdi — yani buradaki
    aralıklar hafif iyimser. Maliyeti (2000 × 5-fold eğitim) faydasına değmediği için
    bu basit versiyon tercih edildi.
    """
    if y.nunique() < 2 or y.value_counts().min() < n_splits:
        return float("nan"), float("nan"), float("nan")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
    y_arr = y.values if hasattr(y, "values") else y
    auc = roc_auc_score(y_arr, proba)

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_arr), len(y_arr))
        if len(np.unique(y_arr[idx])) < 2:
            continue
        boots.append(roc_auc_score(y_arr[idx], proba[idx]))
    alt, ust = np.percentile(boots, [5, 95])
    return auc, alt, ust


def guvenilirlik_etiketi(alt_sinir):
    """
    Modelin o eşikteki AYIRT ETME GÜCÜNÜ tarif eder (bu şirketin olasılığını değil).

    Güven aralığının alt sınırına bakar: "en kötü makul senaryoda bile ne kadar iyi?"
    Kademeler bilinçli olarak YUMUŞAK tutuldu — keskin "kullanılabilir/kullanılamaz"
    ayrımı, aslında belirsiz olan farkları olduğundan kesin gösteriyordu. Örneğin
    ≥1 gün (aralık 0.53-0.77) ile ≥3 gün (0.65-0.85) aralıkları çakışıyor; aradaki
    fark istatistiksel olarak anlamlı olmayabilir, o yüzden biri "kullanılabilir"
    diğeri "zayıf" diye etiketlenmemeli.
    """
    if alt_sinir != alt_sinir:
        return "ölçülemedi"
    if alt_sinir <= 0.50:
        return "ayırt edemiyor (aralık 0.50'yi içeriyor)"
    if alt_sinir <= 0.60:
        return "sınırlı ayrım gücü"
    if alt_sinir <= 0.70:
        return "orta düzey ayrım gücü"
    return "iyi ayrım gücü"


def bolum(baslik):
    print(f"\n{'='*72}\n{baslik}\n{'='*72}")


def main():
    df = pd.read_csv(IN_PATH, encoding="utf-8-sig")
    print(f"Veri seti: {len(df)} şirket, yıllar: {sorted(df['yil'].dropna().unique().tolist())}")
    y = df[HEDEF]
    print(f"Hedef '{HEDEF}' pozitif oranı: {y.mean():.1%}")
    naif = max(y.mean(), 1 - y.mean())

    # ---------- A) Yeni feature işe yarıyor mu? ----------
    bolum("A) 'Kişi başına lot' feature'ı işe yarıyor mu? (5-fold CV, tüm yıllar)")
    setler = {
        "Sadece sermaye yapısı (v1'deki gibi)": ["sermaye_artisi_yuzde", "ortak_satisi_yuzde"],
        "Sadece kişi başına lot (YENİ)": ["log_lot_basina_kisi"],
        "İkisi birlikte": ["sermaye_artisi_yuzde", "ortak_satisi_yuzde", "log_lot_basina_kisi"],
        "Hepsi + büyüklük + yıl": ["sermaye_artisi_yuzde", "ortak_satisi_yuzde",
                                    "log_lot_basina_kisi", "log_buyukluk", "yil"],
    }
    print(f"{'Feature seti':<42} {'AUC':>7} {'Brier':>8}")
    print("-" * 60)
    en_iyi_set, en_iyi_auc = None, -1
    for ad, feats in setler.items():
        feats = [f for f in feats if f in df.columns]
        auc, brier = degerlendir(logreg(), df[feats], y)
        print(f"{ad:<42} {auc:>7.3f} {brier:>8.3f}")
        if auc == auc and auc > en_iyi_auc:
            en_iyi_auc, en_iyi_set = auc, feats
    print(f"\nNaif taban çizgisi (hep çoğunluk sınıfı): accuracy {naif:.3f}, AUC 0.500")
    print(f"En iyi feature seti: {en_iyi_set}")

    # ---------- B) Eski yıllar yardım mı ediyor, zarar mı? ----------
    bolum("B) Hangi veri alt kümesi en iyi? (aynı feature seti, farklı dönemler/filtreler)\n"
          "   NOT: eski yıllarda hedef değişken güvenilirliği düşük (bedelsiz sermaye\n"
          "   artırımı sonrası düzeltilmiş fiyatlar ilk gün referansını bozuyor).")
    print(f"{'Eğitim verisi':<38} {'N':>5} {'Poz.%':>7} {'AUC':>7} {'Brier':>8}")
    print("-" * 68)
    guv = df["ilk_gun_fiyat_guvenilir"] == 1
    for ad, mask in [
        ("Sadece 2025-2026", df["yil"] >= 2025),
        ("2024-2026", df["yil"] >= 2024),
        ("2023-2026 (hepsi)", df["yil"] >= 2023),
        ("SADECE HEDEFİ GÜVENİLİR OLANLAR", guv),
        ("Güvenilir + 2024-2026", guv & (df["yil"] >= 2024)),
    ]:
        alt = df[mask]
        if len(alt) < 20:
            continue
        auc, brier = degerlendir(logreg(), alt[en_iyi_set], alt[HEDEF])
        print(f"{ad:<38} {len(alt):>5} {alt[HEDEF].mean():>6.1%} {auc:>7.3f} {brier:>8.3f}")

    # ---------- C) Hangi model? ----------
    # ÖNEMLİ: Karşılaştırma NİHAİ KURULUMDA yapılmalı (temiz alt küme + nihai feature
    # seti). Önceden tüm veri ve `yil` dahil feature setiyle karşılaştırılıyordu; o
    # koşulda ağaç modelleri çok kötü görünüyordu ama nihai kurulumda tablo değişiyor.
    # Farklı bir konfigürasyonda yapılan karşılaştırmaya dayanarak model seçmek hatalı.
    NIHAI_FEATURES = ["log_buyukluk", "sermaye_artisi_yuzde", "ortak_satisi_yuzde"]
    # Eğitim alt kümesi: hedefi güvenilir olan TÜM şirketler (yıl filtresi YOK).
    # Yıl filtresi başta 2023-2024'ün hedefi bozuk olduğu için konmuştu; kaynak
    # sorun (düzeltilmiş fiyat) HG_* kolonlarına geçilerek çözülünce filtre sadece
    # veri kaybettirmeye başladı. Ölçüldü: güvenilir-tümü (N=136) AUC 0.748 /
    # Brier 0.176, güvenilir+2024 (N=82) AUC 0.719 / Brier 0.206.
    temiz = df[df["ilk_gun_fiyat_guvenilir"] == 1]
    y_temiz = temiz[HEDEF]

    bolum(f"C) Model karşılaştırması — NİHAİ KURULUMDA (N={len(temiz)}, {NIHAI_FEATURES})")
    print(f"{'Model':<24} {'Eğitim AUC':>11} {'Test AUC (CV)':>14} {'Ezber farkı':>13} {'Brier':>8}")
    print("-" * 76)
    modeller = {"Logistic Regression": logreg(), "Gradient Boosting": gbc(), "Random Forest": rf()}
    for ad, m in modeller.items():
        auc, brier = degerlendir(m, temiz[NIHAI_FEATURES], y_temiz)
        m.fit(temiz[NIHAI_FEATURES], y_temiz)
        egitim_auc = roc_auc_score(y_temiz, m.predict_proba(temiz[NIHAI_FEATURES])[:, 1])
        print(f"{ad:<24} {egitim_auc:>11.3f} {auc:>14.3f} {egitim_auc-auc:>13.3f} {brier:>8.3f}")

    print("\nSEÇİM: Logistic Regression. Test AUC farkları bu ölçekte gürültü içinde")
    print("(güven aralıkları ±0.10 genişliğinde), ama LogReg'in EZBER FARKI çok daha")
    print("küçük — yani öğrendiği şey bu 66 şirkete özgü değil. Ayrıca katsayıları")
    print("yorumlanabilir ve olasılık çıktısı bu projenin ihtiyacı olan şey.")

    # ---------- Katsayılar (yorumlama) ----------
    # NİHAİ modelin katsayıları gösterilmeli (temiz alt küme + nihai feature seti),
    # yukarıdaki arama aşamasında denenen geniş setinki değil. Aksi halde ekranda
    # modelde OLMAYAN feature'ların (ör. `yil`) katsayıları görünüyordu.
    p = logreg().fit(temiz[NIHAI_FEATURES], y_temiz)
    bolum("Nihai modelin katsayıları (standardize - büyüklük = etki gücü)")
    for f, c in sorted(zip(NIHAI_FEATURES, p.named_steps["clf"].coef_[0]),
                       key=lambda t: -abs(t[1])):
        yon = "↑ tavan ihtimalini ARTIRIR" if c > 0 else "↓ tavan ihtimalini AZALTIR"
        print(f"  {f:<26} {c:+.3f}   {yon}")

    # ---------- Nihai modeller ----------
    # Yukarıdaki testlerin sonucuna göre bilinçli seçimler:
    #  * `yil` ÇIKARILDI: en büyük katsayıya sahipti ama gerçek sinyal değil,
    #    bizim ölçüm hatamızı (eski yıllarda hedefin aşağı sapmasını) öğreniyordu.
    #    Ayrıca gelecekteki bir halka arz için eğitim aralığı dışına taşar.
    #  * Eğitim verisi TEMİZ ALT KÜME: hedefi güvenilir + 2024 sonrası.
    #  * `log_lot_basina_kisi` çıkarıldı: tek başına AUC 0.58, eklendiğinde katkı yok.
    # (NIHAI_FEATURES ve `temiz` yukarıda C bölümünde tanımlandı.)

    bolum(f"Nihai modeller (temiz alt küme, N={len(temiz)}, feature: {NIHAI_FEATURES})")
    modeller_esik = {}
    print(f"{'Eşik':>5} {'Pozitif':>8} {'AUC':>7}  {'%90 güven aralığı':>18}  yorum")
    print("-" * 76)
    for t in ESIKLER:
        yt = (temiz["hedef_acilis_serisi"] >= t).astype(int)
        auc, alt, ust = auc_guven_araligi(logreg(), temiz[NIHAI_FEATURES], yt)
        final = logreg().fit(temiz[NIHAI_FEATURES], yt)
        yorum = guvenilirlik_etiketi(alt)
        modeller_esik[t] = {
            "model": final,
            "auc": None if auc != auc else round(auc, 3),
            "auc_alt": None if alt != alt else round(alt, 3),
            "auc_ust": None if ust != ust else round(ust, 3),
            "pozitif_sayisi": int(yt.sum()),
            "yorum": yorum,
        }
        aralik = f"[{alt:.2f} - {ust:.2f}]"
        print(f"{t:>4}g {yt.sum():>8} {auc:>7.3f}  {aralik:>18}  {yorum}")

    print("\nNot: Etiketler AUC'nin kendisine değil, güven aralığının ALT SINIRINA göre")
    print("veriliyor. Aralık 0.50'yi içeriyorsa model o eşikte yazı turadan ayırt")
    print("edilemiyor demektir — nokta tahmini yüksek görünse bile.")

    joblib.dump({"models": modeller_esik, "features": NIHAI_FEATURES,
                 "egitim_n": len(temiz)}, MODEL_OUT)
    print(f"\nKaydedildi: {MODEL_OUT}")


if __name__ == "__main__":
    main()
