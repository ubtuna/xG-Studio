"""
add_extra_features.py — build_features.py çıktısı olan temiz şut tablosunu okur,
şut bazlı xG modeli için ek yorumlanabilir feature'lar üretir.

Girdi:
    data/shots_features.parquet

Çıktı:
    data/shots_features_extra.parquet

Ne yapar:
- Mevcut feature engineering çıktısına dokunmaz
- x, y, distance ve angle kolonlarını kullanarak yeni futbol mantıklı değişkenler üretir
- Ceza sahası içi / merkez şut / yakın mesafe / uzak şut gibi açıklanabilir feature'lar ekler
- Hedef değişken olan is_goal'a dokunmaz
- StatsBomb xG referans kolonunu değiştirmez

Çalıştırma:
    python src/add_extra_features.py

Not:
Bu script mevcut pipeline'ın üçüncü adımıdır:

    fetch_shots.py
        ↓
    build_features.py
        ↓
    add_extra_features.py

Bu sayede ilk iki dosya bozulmadan ek feature denemeleri yapılabilir.
"""

import pathlib
import pandas as pd


# build_features.py tarafından üretilen temiz tablo
IN = pathlib.Path("data/shots_features.parquet")

# Ek feature'lar eklenmiş yeni çıktı
OUT = pathlib.Path("data/shots_features_extra.parquet")


def main():
    """
    Temiz şut verisini okur, ek feature'ları üretir ve yeni parquet dosyasına yazar.
    """

    # 1) Temiz feature tablosunu oku
    df = pd.read_parquet(IN)
    print(f"Temiz tablo okundu: {len(df)} satır, {df.shape[1]} kolon")

    # 2) Ceza sahası içi mi?
    #
    # StatsBomb koordinat sistemi:
    # - Saha uzunluğu yaklaşık 120 birim
    # - Saha genişliği yaklaşık 80 birim
    # - Rakip kale x=120, y=40 merkezindedir
    #
    # Ceza sahası yaklaşık:
    # - x >= 102
    # - y 18 ile 62 arası
    #
    # Bu feature şutun ceza sahası içinden gelip gelmediğini gösterir.
    df["inside_box"] = (
        (df["x"] >= 102) &
        (df["y"].between(18, 62))
    ).astype(int)

    # 3) Merkezden şut mu?
    #
    # y koordinatı 40'a ne kadar yakınsa şut o kadar merkezden gelir.
    # Burada 30-50 aralığını merkezi koridor olarak kabul ediyoruz.
    #
    # Merkezden çekilen şutlar genellikle daha geniş açıya sahiptir.
    df["central_shot"] = (
        df["y"].between(30, 50)
    ).astype(int)

    # 4) Yakın mesafe şut mu?
    #
    # build_features.py zaten distance kolonunu üretmişti.
    # Burada 12 birim ve altını yakın mesafe kabul ediyoruz.
    #
    # Yakın mesafe şutların gol olma ihtimali genellikle daha yüksektir.
    df["close_range"] = (
        df["distance"] <= 12
    ).astype(int)

    # 5) Uzak şut mu?
    #
    # 25 birim ve üzerindeki şutları uzak şut kabul ediyoruz.
    #
    # Uzak şutlar genellikle düşük xG değerine sahiptir.
    df["long_shot"] = (
        df["distance"] >= 25
    ).astype(int)

    # 6) Ek feature'lı tabloyu kaydet
    df.to_parquet(OUT, index=False)

    print(f"BİTTİ. Ek feature'lı tablo: {len(df)} satır, {df.shape[1]} kolon -> {OUT}")
    print("Eklenen kolonlar:")
    print([
        "inside_box",
        "central_shot",
        "close_range",
        "long_shot",
    ])

    print("\nGol oranı kontrolleri:")
    for col in ["inside_box", "central_shot", "close_range", "long_shot"]:
        rates = df.groupby(col)["is_goal"].mean()
        print(f"\n{col}")
        print(rates)


if __name__ == "__main__":
    main()
