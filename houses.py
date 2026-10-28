"""
House or flat? Single-family house sales from the RCN "budynki" layer (fetch_rcn.py --layer budynki) next to flat
resales in the same district.

House sale = Warsaw (TERYT 1465), built-up land in full ownership (nieruchomoscGruntowaZabudowana, share 1/1),
free market (an empty share field counts as 1/1: most 2021-22 deeds), one residential building per deed (deeds with
several buildings repeat the total price on every one), any size. Year from the notarial deed number ("4660/2023"): the layer has no date. Price per m²
is per m² of the house INCLUDING its plot, and only where the deed records the usable area (50-400 m²): about one
deed in three (2023-26); where recorded it is ~1.6x the footprint, so plausible. Totals use every deed. RCN records
houses from 2017.

Outputs: output/houses.gpkg (clean house sales, for QGIS), output/summary/35_house_or_flat.png (two bar charts)
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap
from summary_maps import load

style.apply()
RECENT = (2023, 2026)
MIN_HOUSES = 40
MIN_WITH_AREA = 25
USABLE = (50, 400)
FLOOR_RATIO = {1: 1.2, 2: 1.52, 3: 1.97, 4: 2.4}


def estimate_area(h):
    """Usable area for every house: the deed's own figure where it records one (about 1 deed in 6), else the BDOT10k
    outline x a ratio by number of floors. Ratios = median recorded usable area / outline on the 1,115 deeds that have
    both (1 floor 1.2, 2 floors 1.52, 3 floors 1.97, 4+ guessed 2.4); typical miss 14 % per house (18 % with one ratio
    for all), so fine for medians over hundreds of sales, not for a single house."""
    bd = gpd.read_file("data/bdot_buildings.gpkg", columns=["n_floors"]).to_crs(2180)
    bd["outline_m2"] = bd.geometry.area
    j = gpd.sjoin(h, bd[["n_floors", "outline_m2", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()].drop(columns="index_right")
    ratio = j["n_floors"].fillna(2).clip(1, 4).map(FLOOR_RATIO)
    est = (j["outline_m2"] * ratio).where(lambda s: s.between(*USABLE))
    j["usable_est"] = est
    j["usable_all"] = j["usable_m2"].fillna(est)
    j["area_source"] = np.where(j["usable_m2"].notna(), "deed", np.where(est.notna(), "estimated", "none"))
    j["price_m2"] = j["price"] / j["usable_all"]
    print("area source:", j["area_source"].value_counts().to_dict())
    return j


def houses(d):
    b = gpd.read_file("output/rcn_budynki_warszawa.gpkg",
                      where="TERYT='1465' AND NIER_RODZAJ='nieruchomoscGruntowaZabudowana'")
    n_bud = b.groupby("TRAN_LOKALNY_ID_IIP")["gid"].transform("size")
    b = b[n_bud == 1].copy()
    b["year"] = pd.to_numeric(b["DOK_OZNACZENIE"].str.extract(r"/(\d{4})")[0], errors="coerce")
    b["usable_m2"] = pd.to_numeric(b["BUD_POW_UZYT"], errors="coerce")
    b["price"] = pd.to_numeric(b["NIER_CENA_BRUTTO"], errors="coerce")
    b["plot_m2"] = pd.to_numeric(b["NIER_POW_GRUNTU"], errors="coerce")
    # 2021-22 deeds mostly leave the ownership share empty (1,380 of them look like 1/1 sales: same price and plot
    # as 1/1 deeds in 2020 and 2023), so an empty share counts as full ownership
    full = (b["NIER_UDZIAL"] == "1/1") | b["NIER_UDZIAL"].isna()
    h = b[(b["BUD_RODZAJ"] == "mieszkalny") & (b["TRAN_RODZAJ_TRANS"] == "wolnyRynek") & full
          & (b["price"] > 100_000)].copy()
    h["usable_m2"] = h["usable_m2"].where(h["usable_m2"].between(*USABLE))
    h["market"] = np.where(h["TRAN_RODZAJ_RYNKU"] == "rynekPierwotny", "primary", "secondary")
    h = gpd.GeoDataFrame(h[["year", "price", "usable_m2", "plot_m2", "market", "BUD_ADRES"]],
                         geometry=h.geometry.to_crs(2180).representative_point().values, crs=2180)
    h = gpd.sjoin(h, d[["district", "geometry"]], predicate="within").drop(columns="index_right")
    h = estimate_area(h)
    print(f"clean house sales: {len(h):,}; by year:\n" + h["year"].value_counts().sort_index().to_string())
    h.to_file("output/houses.gpkg", layer="houses", driver="GPKG")
    return h


def curves(ax, house, flat, ticks, labels, xlabel):
    """Two overlapping density curves on a log axis, medians marked."""
    from scipy.stats import gaussian_kde
    lo, hi = np.log10(ticks[0] * 0.8), np.log10(ticks[-1] * 1.25)
    x = np.linspace(lo, hi, 400)
    for vals, colr, lab in ((flat, "#9a968b", "Resold flat"), (house, "#2a6b35", "House (with plot)")):
        v = np.log10(vals.dropna())
        y = gaussian_kde(v, bw_method=0.18)(x)
        ax.fill_between(10 ** x, y, color=colr, alpha=0.45, lw=0)
        ax.plot(10 ** x, y, color=colr, lw=1.6, label=lab)
        m = np.median(vals.dropna())
        ax.axvline(m, color=colr, lw=1.2, ls=(0, (4, 3)))
        ax.text(m, y.max() * 1.07, labels(m), color=colr, ha="center", va="bottom", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.85))
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(10 ** lo, 10 ** hi)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_xlabel(xlabel, fontsize=9)


def main():
    v, d, w = load()
    h = houses(d)
    r = h[h["year"].between(*RECENT)]
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True)
    f = t[t["year"].between(*RECENT) & (t["market"] == "secondary")]
    n_by = r.groupby("district").size()
    keep = n_by[n_by >= MIN_HOUSES].index                       # districts with enough house sales; flats from the same ones
    r, f = r[r["district"].isin(keep)], f[f["district"].isin(keep)]
    hp, fp = r["price"], f["price"]
    hm2, fm2 = r["price_m2"].where(r["price_m2"].between(2000, 40000)), f["price_m2"].where(f["price_m2"].between(2000, 40000))
    tot_x, m2_x = hp.median() / fp.median(), hm2.median() / fm2.median()
    print(f"{len(r):,} houses, {len(f):,} flats in {len(keep)} districts; median house {hp.median():,.0f} vs flat "
          f"{fp.median():,.0f} ({tot_x:.2f}x); per m2 {hm2.median():,.0f} vs {fm2.median():,.0f} ({m2_x:.2f}x)")

    fig = page(f"A house costs {tot_x:.1f}x a flat, a square metre {1 - m2_x:.0%} less",
               f"Houses and resold flats sold {RECENT[0]}–{RECENT[1]} in the {len(keep)} districts with at least {MIN_HOUSES} "
               "house sales",
               f"Houses: {len(r):,} sales with plot (RCN, from 2017); flats: {len(f):,} resales, same districts and years. House area: "
               f"the deed's ({(r['area_source'] == 'deed').mean():.0%}) or estimated from the building outline and floors. " + style.SOURCE)
    mil = lambda m: f"{m / 1e6:.2g} M" if m >= 1e6 else f"{m / 1e6:.2f} M"
    k = lambda m: f"{m / 1000:.0f}K"
    fig.text(0.03, 0.885, wrap(f"In total a median house costs {mil(hp.median())} and a flat {mil(fp.median())}. Per m² it flips: "
                               f"{k(hm2.median())} for a house, plot included, {k(fm2.median())} for a flat. A house buys "
                               "much more space, and the land comes with it.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    a1 = fig.add_axes([0.06, 0.52, 0.90, 0.29])
    curves(a1, hp, fp, [0.3e6, 0.5e6, 1e6, 2e6, 4e6], mil, "price paid, PLN")
    a1.legend(loc="upper right", frameon=False, fontsize=9.5)
    a1.set_title("What you pay in total", fontsize=11.5, loc="left")
    a2 = fig.add_axes([0.06, 0.135, 0.90, 0.29])
    curves(a2, hm2, fm2, [4000, 6000, 10000, 15000, 25000], k, "price per m² of living space, PLN")
    a2.set_title("What a square metre costs", fontsize=11.5, loc="left")
    save(fig, "35_house_or_flat.png")


if __name__ == "__main__":
    main()
