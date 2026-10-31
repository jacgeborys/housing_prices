"""
New flats on offer in Lublin (developers' price lists, parse_offers.py -> data/offers_lublin.csv) against what flats in
nearby recent buildings are worth. Run from lublin/:  python offers.py [--lang pl]

Geocoding: an investment's street (and number where given) is matched to the addresses of flat sales in the register
(data/rcn_lokale_lublin.gpkg, LOK_ADRES 'UL:<street>;NR_PORZ:<no>'): the number's point if it sold flats, else the median
point of the street (street-level accuracy, fine for a city map). Investments on streets with no recorded sale are left off.
Comparison: per investment the median asking price per m² vs the median typical-flat value (output/price_model.gpkg) of
buildings from 2011-20 (output/building_age.gpkg) within 1.5 km.

-> output/report/src[_pl]/r15_offers.png, output/offers_lublin.gpkg
"""
import re
import sys
import unicodedata

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from common import T, basemap, districts, fmt, headline, page, pct, save, set_lang, source, style

NORM = Normalize(7000, 15000)


def key(s):
    s = unicodedata.normalize("NFKD", str(s).lower()).encode("ascii", "ignore").decode()
    s = re.sub(r"\b(ul|al|ulica|aleja|pl|plac|os|osiedle)\b\.?", " ", s)
    words = re.findall(r"[a-z0-9]+", s)
    return words[-1] if words else ""


def number(s):
    m = re.match(r"^\s*(\d+[a-z]?)", str(s).lower())
    return m.group(1) if m else None


def geocode(f):
    r = gpd.read_file("data/rcn_lokale_lublin.gpkg", columns=["LOK_ADRES"]).to_crs(2180)
    a = r["LOK_ADRES"].fillna("")
    r["street"] = a.str.extract(r"UL:([^;]+)")[0].map(key)
    r["no"] = a.str.extract(r"NR_PORZ:([^;]+)")[0].str.lower().str.strip()
    r = r[r["street"] != ""]
    xy = pd.DataFrame({"street": r["street"], "no": r["no"], "x": r.geometry.x, "y": r.geometry.y})
    by_no = xy.groupby(["street", "no"])[["x", "y"]].median()
    by_st = xy.groupby("street")[["x", "y"]].median()
    inv = f.groupby(["developer", "inv_street", "inv_no"], dropna=False).agg(
        flats=("price_m2", "size"), ask=("price_m2", "median"), area=("area", "median")).reset_index()
    pts, how = [], []
    for st, no in zip(inv["inv_street"], inv["inv_no"]):
        s, n = key(st), number(no)
        if (s, n) in by_no.index:
            pts.append(by_no.loc[(s, n)].to_numpy()); how.append("number")
        elif s in by_st.index:
            pts.append(by_st.loc[s].to_numpy()); how.append("street")
        else:
            pts.append([np.nan, np.nan]); how.append("none")
    p = np.array(pts, dtype=float)
    inv["how"] = how
    inv = gpd.GeoDataFrame(inv, geometry=gpd.points_from_xy(p[:, 0], p[:, 1]), crs=2180)
    print("geocoded:", inv["how"].value_counts().to_dict(), f"({inv.loc[inv['how'] != 'none', 'flats'].sum():,} of "
          f"{inv['flats'].sum():,} flats)")
    return inv[inv["how"] != "none"].copy()


def main(lang="en"):
    set_lang(lang)
    f = pd.read_csv("data/offers_lublin.csv", low_memory=False)
    f = f[f["area"].between(15, 200) & f["price_m2"].between(5000, 30000)]
    f = f.drop_duplicates(["developer", "inv_street", "unit", "area", "price_m2"])
    inv = geocode(f)
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value")
    age = gpd.read_file("output/building_age.gpkg", ignore_geometry=True)[["osm_id", "decade_end"]]
    bv = bv.merge(age, on="osm_id", how="left")
    recent = bv[bv["decade_end"] == 2020]
    rp = recent.geometry.representative_point()
    near = []
    for g in inv.geometry:
        v = recent.loc[rp.distance(g) < 1500, "value_2025"]
        near.append(v.median() if len(v) >= 3 else np.nan)
    inv["near"] = near
    inv["gap"] = 100 * (inv["ask"] / inv["near"] - 1)
    inv.to_file("output/offers_lublin.gpkg", layer="investments", driver="GPKG")
    ok = inv.dropna(subset=["gap"])
    w_gap = np.average(ok["gap"], weights=ok["flats"])
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True)
    res25 = t.loc[(t["year"] == 2025) & (t["market"] == "secondary"), "price_m2"].median()
    new25 = t.loc[(t["year"] == 2025) & (t["market"] == "primary"), "price_m2"].median()
    d = districts()
    fig = page(T(f"New flats ask {pct(w_gap)} over 5-15-year-old flats nearby",
                 f"Nowe mieszkania w ofercie: {pct(w_gap)} wobec 5-15-letnich w pobliżu"),
               T("Flats in developers' latest price lists, by investment, against nearby buildings from 2011-20",
                 "Mieszkania z najnowszych cenników deweloperów wg inwestycji, wobec pobliskich budynków z lat 2011-20"),
               T(f"Price lists published under the 2025 law (dane.gov.pl), {fmt(len(f))} flats in {len(inv)} located investments; asking, "
                 "not sale prices. Nearby = median typical-flat value of 2011-20 buildings within 1.5 km. Located by street. ",
                 f"Cenniki deweloperów (dane.gov.pl), {fmt(len(f))} mieszkań w {len(inv)} inwestycjach; ceny ofertowe. W pobliżu = "
                 "mediana typowego mieszkania w budynkach z lat 2011-20 do 1,5 km. Lokalizacja wg ulicy. ") + source(", dane.gov.pl"))
    headline(fig, T(f"Developers ask a median {fmt(f['price_m2'].median())} PLN/m² for {fmt(len(f))} listed flats (median "
                    f"{f['area'].median():.0f} m²); weighted by flats that is {pct(w_gap)} against nearby 2011-20 buildings. In 2025 "
                    f"new flats sold for {fmt(new25)} and resales for {fmt(res25)} (deeds).",
                    f"Deweloperzy chcą w medianie {fmt(f['price_m2'].median())} zł/m² za {fmt(len(f))} mieszkań w cennikach "
                    f"(mediana {f['area'].median():.0f} m²); ważone liczbą mieszkań to {pct(w_gap)} wobec pobliskich budynków z lat "
                    f"2011-20. W 2025 nowe sprzedawano po {fmt(new25)}, wtórne po {fmt(res25)} zł/m² (akty)."))
    ax = basemap(fig, d, rect=(0.03, 0.36, 0.94, 0.47))
    o = inv.sort_values("flats", ascending=False)
    ax.scatter(o.geometry.x, o.geometry.y, s=8 + 9 * np.sqrt(o["flats"]), c=o["ask"], cmap=style.PRICE_CMAP, norm=NORM,
               lw=0, alpha=0.95, zorder=8)
    cax = fig.add_axes([0.70, 0.37, 0.24, 0.011])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=style.PRICE_CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([7000, 9000, 11000, 13000, 15000])
    cb.set_ticklabels(T(["7K", "9K", "11K", "13K", "15K+"], ["7", "9", "11", "13", "15+ tys."]))
    cax.tick_params(labelsize=8.5, length=0)
    cax.set_title(T("asking PLN per m²\n(dot size = flats)", "cena ofertowa zł/m²\n(wielkość = liczba mieszkań)"),
                  fontsize=8.5, loc="left")
    a = fig.add_axes([0.12, 0.09, 0.84, 0.21])
    bins = np.arange(-40, 61, 10)
    a.hist(ok["gap"].clip(-40, 59.9), bins=bins, weights=ok["flats"], color=style.ORANGE, edgecolor=style.SURFACE, lw=1)
    a.axvline(0, color=style.AXIS, lw=1)
    a.axvline(w_gap, color=style.INK, lw=1.4, ls=(0, (4, 3)))
    a.text(w_gap + 1.5, a.get_ylim()[1] * 0.95, f"{T('weighted mean', 'średnia ważona')} {pct(w_gap)}", fontsize=9, va="top")
    a.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    a.set_ylabel(T("flats on offer", "mieszkań w ofercie"))
    style.hgrid_only(a)
    a.set_title(T("Asking price per m² against nearby 2011-20 buildings (flats)",
                  "Cena ofertowa za m² wobec pobliskich budynków z lat 2011-20 (mieszkania)"), fontsize=10.5, loc="left")
    save(fig, "r15_offers.png")


if __name__ == "__main__":
    main("pl" if "pl" in sys.argv[1:] else "en")
