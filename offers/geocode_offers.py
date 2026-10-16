"""
Place the Warsaw flats on offer (parse_price_lists.py) on the map and keep one current price per flat.

  1. one row per flat: developer + investment street + number + unit, latest price date ("Data od której
     obowiązuje cena m²"; mixed date formats, parsed day-first)
  2. street name normalised (no "ul./al./aleja/ulica", lower case, no diacritics) and matched to the city's address
     points (data/address_points.gpkg) by its last word ("Włady Majewskiej" -> "majewskiej"; "Posag 7 Panien /
     Gierdziejewskiego" -> the first part)
  3. house number: the first number token ("63, 63A, 63B" -> 63; "6G/1" -> 6G, then 6); matched to that street's
     points, else the median position of the street's points (geo = "street")

    python offers/geocode_offers.py   -> offers/data/offers_warszawa.gpkg (layer offers)
"""
import os
import re
import unicodedata

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "data", "offers_warszawa.gpkg")
PREFIX = re.compile(r"^(ulica|ul\.?|aleja|al\.?|aleje|plac|pl\.?|osiedle|os\.?|rondo)\s+", re.I)


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).replace("ł", "l").replace("Ł", "L")).encode("ascii", "ignore").decode()
    s = PREFIX.sub("", s.strip().lower())
    return re.sub(r"[^a-z0-9 ]", " ", s).split()


def street_key(s):
    """Last word of the street name, numbers / postcodes / prefixes stripped ("ul.Popularna", "UlicaBukowiecka",
    "Silnikowa 25A", "E. Barszczewskiej", "Marywilska 73, 03-042 Warszawa" all reduce to the surname word)."""
    s = re.sub(r"^(ulica|ul\.?|aleja|al\.?)(?=[A-ZŁŚŻŹĆŃÓĘĄ])", "", str(s).split("/")[0].split(",")[0].strip())
    w = [x for x in norm(s) if not x.isdigit() and not re.fullmatch(r"\d+[a-z]?", x)]
    return w[-1] if w else ""


def house(s):
    m = re.search(r"\d+[a-zA-Z]?", str(s))
    return m.group(0).upper() if m else None


def main():
    f = pd.read_csv(os.path.join(HERE, "data", "flats_warszawa.csv"), low_memory=False)
    f["date"] = pd.to_datetime(f["price_m2_from"], errors="coerce", dayfirst=True)
    f = f.sort_values("date").drop_duplicates(["developer", "inv_street", "inv_no", "unit"], keep="last")
    f = f[f["area"].between(15, 300)]
    print(f"{len(f):,} distinct flats on offer")

    a = gpd.read_file(os.path.join(ROOT, "data", "address_points.gpkg"))
    a["skey"] = a["street"].map(street_key)
    a["hno"] = a["number"].astype(str).str.upper().str.replace(" ", "")
    exact = a.drop_duplicates(["skey", "hno"]).set_index(["skey", "hno"]).geometry
    a["hnum"] = a["number"].map(house).str.extract(r"(\d+)")[0]
    num = a.drop_duplicates(["skey", "hnum"]).set_index(["skey", "hnum"]).geometry
    street = a.groupby("skey").apply(lambda g: gpd.points_from_xy([g.geometry.x.median()], [g.geometry.y.median()])[0])

    f["skey"] = f["inv_street"].map(street_key)
    f["hno"] = f["inv_no"].map(house)
    in_street = f["inv_street"].astype(str).str.split(",").str[0].map(house)       # "Silnikowa 25A"
    f["hno"] = f["hno"].fillna(in_street)
    f["hnum"] = f["hno"].str.extract(r"(\d+)")[0]
    geo, how = [], []
    for sk, hn, hnum in zip(f["skey"], f["hno"], f["hnum"]):
        if (sk, hn) in exact.index:
            geo.append(exact[(sk, hn)]); how.append("number")
        elif (sk, hnum) in num.index:
            geo.append(num[(sk, hnum)]); how.append("number")
        elif sk in street.index:
            geo.append(street[sk]); how.append("street")
        else:
            geo.append(None); how.append("none")
    f["geo"] = how
    g = gpd.GeoDataFrame(f, geometry=geo, crs=a.crs).to_crs(2180)
    print(g["geo"].value_counts().to_string())
    g = g[g["geo"] != "none"].drop(columns=["skey", "hno", "hnum"])
    g["date"] = g["date"].astype(str)
    g.to_file(OUT, layer="offers", driver="GPKG")
    print(f"Saved {OUT}: {len(g):,} flats, {g[['developer', 'inv_street']].drop_duplicates().shape[0]:,} investments; "
          f"median {g['price_m2'].median():,.0f} PLN/m²")


if __name__ == "__main__":
    main()
