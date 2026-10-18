"""
House or flat? Single-family house sales from the RCN "budynki" layer (fetch_rcn.py --layer budynki) next to flat
resales in the same district.

House sale = Warsaw (TERYT 1465), built-up land in full ownership (nieruchomoscGruntowaZabudowana, share 1/1),
free market, one residential building per deed (deeds with several buildings repeat the total price on every one),
any size. Year from the notarial deed number ("4660/2023"): the layer has no date. Price per m²
is per m² of the house INCLUDING its plot, and only where the deed records the usable area (50-400 m²): about one
deed in three (2023-26); where recorded it is ~1.6x the footprint, so plausible. Totals use every deed. RCN records
houses from 2017; 2021-22 are nearly empty (register gap).

Outputs: output/houses.gpkg (clean house sales), output/summary/35_house_or_flat.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, legend_patches, load, outlines

style.apply()
RECENT = (2023, 2026)
MIN_HOUSES = 40
MIN_WITH_AREA = 25
USABLE = (50, 400)


def houses(d):
    b = gpd.read_file("output/rcn_budynki_warszawa.gpkg",
                      where="TERYT='1465' AND NIER_RODZAJ='nieruchomoscGruntowaZabudowana'")
    n_bud = b.groupby("TRAN_LOKALNY_ID_IIP")["gid"].transform("size")
    b = b[n_bud == 1].copy()
    b["year"] = pd.to_numeric(b["DOK_OZNACZENIE"].str.extract(r"/(\d{4})")[0], errors="coerce")
    b["usable_m2"] = pd.to_numeric(b["BUD_POW_UZYT"], errors="coerce")
    b["price"] = pd.to_numeric(b["NIER_CENA_BRUTTO"], errors="coerce")
    b["plot_m2"] = pd.to_numeric(b["NIER_POW_GRUNTU"], errors="coerce")
    h = b[(b["BUD_RODZAJ"] == "mieszkalny") & (b["TRAN_RODZAJ_TRANS"] == "wolnyRynek") & (b["NIER_UDZIAL"] == "1/1")
          & (b["price"] > 100_000)].copy()
    h["usable_m2"] = h["usable_m2"].where(h["usable_m2"].between(*USABLE))
    h["price_m2"] = h["price"] / h["usable_m2"]
    h["market"] = np.where(h["TRAN_RODZAJ_RYNKU"] == "rynekPierwotny", "primary", "secondary")
    h = gpd.GeoDataFrame(h[["year", "price", "usable_m2", "plot_m2", "price_m2", "market", "BUD_ADRES"]],
                         geometry=h.geometry.to_crs(2180).representative_point().values, crs=2180)
    h = gpd.sjoin(h, d[["district", "geometry"]], predicate="within").drop(columns="index_right")
    print(f"clean house sales: {len(h):,}; by year:\n" + h["year"].value_counts().sort_index().to_string())
    h.to_file("output/houses.gpkg", layer="houses", driver="GPKG")
    return h


def main():
    v, d, w = load()
    h = houses(d)
    r = h[h["year"].between(*RECENT)]
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True)
    f = t[t["year"].between(*RECENT) & (t["market"] == "secondary")]
    g = r.groupby("district").agg(n=("price", "size"), house=("price", "median"), house_m2=("price_m2", "median"),
                                  n_m2=("price_m2", "count"), usable=("usable_m2", "median"), plot=("plot_m2", "median"))
    g = g.join(f.groupby("district").agg(flat=("price", "median"), flat_m2=("price_m2", "median")))
    g = g[(g["n"] >= MIN_HOUSES) & (g["n_m2"] >= MIN_WITH_AREA)].sort_values("house_m2")
    print(g.round(0).to_string())
    k = g["n"].idxmax()                                   # the district with the most house sales
    cheaper = g.index[g["house_m2"] < g["flat_m2"]]

    fig = page("House or flat?",
               f"Houses and resold flats sold {RECENT[0]}–{RECENT[1]}, price per m² of living space",
               f"Houses: {len(r):,} sales of a single-family house with its plot (RCN, full ownership, one house per "
               f"deed). Price per m² of the house INCLUDING the plot, from the {r['price_m2'].notna().sum():,} deeds that "
               f"record the usable area (about one in three); totals from all. Flats: RCN resales in the same years. "
               f"Medians; districts with ≥ {MIN_HOUSES} house sales. Terraced and "
               "semi-detached houses on their own plot count as houses. The register records houses from 2017. "
               + style.SOURCE)
    where = ("In every district with enough house sales" if len(cheaper) == len(g)
             else "In " + ", ".join(cheaper[:-1]) + " and " + cheaper[-1])
    fig.text(0.03, 0.885, wrap(f"{where}, a house costs less per m² than a flat, plot included: "
                               f"in {k} {g.loc[k, 'house_m2'] / 1000:.1f}K vs {g.loc[k, 'flat_m2'] / 1000:.1f}K, with a "
                               f"median {g.loc[k, 'plot']:.0f} m² of land. The catch is the total: a median house there "
                               f"has {g.loc[k, 'usable']:.0f} m² and costs {g.loc[k, 'house'] / 1e6:.1f} million, a flat "
                               f"{g.loc[k, 'flat'] / 1e6:.2f} million.", 100), fontsize=10.5, fontweight="semibold", va="top")

    ax = basemap(fig, d, w)
    ax.set_position([0.02, 0.38, 0.55, 0.42])
    cmap = plt.cm.colors.ListedColormap(["#c9dfc0", "#8fc27f", "#4f9a4a", "#2a6b35", "#123d1f"])
    bounds = [0, 12000, 15000, 18000, 24000, 1e9]
    norm = plt.cm.colors.BoundaryNorm(bounds, 5)
    rs = r[r["price_m2"].notna()].sort_values("price_m2")
    ax.scatter(rs.geometry.x, rs.geometry.y, s=5, c=rs["price_m2"], cmap=cmap, norm=norm, lw=0, zorder=4)
    outlines(ax, d, labels=False)
    legend_patches(ax, [cmap(i) for i in range(5)], ["< 12K", "12–15K", "15–18K", "18–24K", "24K+"],
                   "House, PLN/m²")
    ax.set_title(f"Houses sold {RECENT[0]}–{RECENT[1]} (deeds with a recorded size)", fontsize=10.5)

    a2 = fig.add_axes([0.66, 0.44, 0.31, 0.36])
    y = np.arange(len(g))
    a2.barh(y + 0.19, g["house_m2"], height=0.36, color="#2a6b35", label="House (with plot)")
    a2.barh(y - 0.19, g["flat_m2"], height=0.36, color=style.CONTEXT, label="Resold flat")
    for yi, (hm, fm) in enumerate(zip(g["house_m2"], g["flat_m2"])):
        a2.text(hm, yi + 0.19, f" {hm / 1000:.1f}K", va="center", fontsize=7.5)
        a2.text(fm, yi - 0.19, f" {fm / 1000:.1f}K", va="center", fontsize=7.5, color=style.INK_2)
    a2.set_yticks(y)
    a2.set_yticklabels(g.index, fontsize=8.5)
    a2.set_xlim(0, g["house_m2"].max() * 1.25)
    a2.set_xticks([])
    for sp in ("top", "right", "bottom"):
        a2.spines[sp].set_visible(False)
    a2.legend(loc="upper left", bbox_to_anchor=(0, -0.01), fontsize=8, frameon=False, ncol=2)
    a2.set_title("Price per m², median", fontsize=10.5, loc="left")

    a3 = fig.add_axes([0.10, 0.115, 0.86, 0.20])
    x = np.arange(len(g))
    a3.bar(x - 0.2, g["house"] / 1e6, width=0.4, color="#2a6b35", label="House")
    a3.bar(x + 0.2, g["flat"] / 1e6, width=0.4, color=style.CONTEXT, label="Resold flat")
    for xi, (hv, fv, u) in enumerate(zip(g["house"], g["flat"], g["usable"])):
        a3.text(xi - 0.2, hv / 1e6, f"{hv / 1e6:.1f}\n{u:.0f} m²", ha="center", va="bottom", fontsize=7)
    a3.set_xticks(x)
    a3.set_xticklabels(g.index, fontsize=9)
    a3.set_ylabel("million PLN")
    a3.set_ylim(0, g["house"].max() / 1e6 * 1.3)
    style.hgrid_only(a3)
    a3.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
    a3.set_title("What you pay in total (median; house size under the bar label)", fontsize=10.5, loc="left")
    save(fig, "35_house_or_flat.png")


if __name__ == "__main__":
    main()
