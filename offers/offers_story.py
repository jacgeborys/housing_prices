"""
New flats on offer in Warsaw (developer price lists, dane.gov.pl, 2025 transparency law): where, for how much,
and how much more than a nearly new flat next door.

Inputs: offers/data/offers_warszawa.gpkg (geocode_offers.py), output/price_drivers.gpkg (typical-flat value
2025 per building), output/building_age.gpkg (construction decade), RCN resales 2024-26 for the size mix.

Output: output/summary/26_new_flats_on_offer.png, offers/data/investments.gpkg
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import style  # noqa: E402
from summary_charts import page, save, wrap  # noqa: E402
from summary_maps import basemap, legend_patches, load, outlines  # noqa: E402

style.apply()
COMP_M = 1000          # comparables: priced 2011-2020 buildings within this distance
MIN_COMPS = 3


def investments(o):
    """One point per investment (developer + street + building number), median asking."""
    o = o.copy()
    o["inv"] = o["developer"].astype(str) + " | " + o["inv_street"].astype(str) + " " + o["inv_no"].astype(str)
    g = o.groupby("inv").agg(developer=("developer", "first"), street=("inv_street", "first"),
                             flats=("price_m2", "size"), ask=("price_m2", "median"), area=("area", "median"),
                             x=("geometry", lambda s: s.x.median()), y=("geometry", lambda s: s.y.median()))
    return gpd.GeoDataFrame(g.reset_index(), geometry=gpd.points_from_xy(g["x"], g["y"]), crs=o.crs)


def main():
    v, d, w = load()
    o = gpd.read_file(os.path.join(HERE, "data", "offers_warszawa.gpkg")).to_crs(2180)
    o = o[o["price_m2"].between(8000, 60000)]
    inv = investments(o)
    j = gpd.sjoin(inv, d[["district", "geometry"]], predicate="within", how="left")
    inv["district"] = j[~j.index.duplicated()]["district"].to_numpy()
    o["district"] = gpd.sjoin(o, d[["district", "geometry"]], predicate="within", how="left") \
        .groupby(level=0)["district"].first().reindex(o.index).to_numpy()

    vv = v if "decade_end" in v else v.merge(gpd.read_file(os.path.join(ROOT, "output", "building_age.gpkg"),
                                                           ignore_geometry=True)[["osm_id", "decade_end"]], on="osm_id")
    rec = vv[vv["decade_end"].between(2011, 2020)]
    rp = rec.geometry.representative_point()
    tree = cKDTree(np.c_[rp.x, rp.y])
    vals = rec["value_2025"].to_numpy()
    comp = []
    for x, y in zip(inv.geometry.x, inv.geometry.y):
        nb = tree.query_ball_point([x, y], COMP_M)
        comp.append(np.median(vals[nb]) if len(nb) >= MIN_COMPS else np.nan)
    inv["comp"] = comp
    inv["premium"] = 100 * (inv["ask"] / inv["comp"] - 1)
    inv.drop(columns=["x", "y"]).to_file(os.path.join(HERE, "data", "investments.gpkg"), layer="investments",
                                         driver="GPKG")
    print(f"{len(inv)} investments, {int(inv['flats'].sum()):,} flats; premium over nearby 2011-20 buildings: "
          f"median {inv['premium'].median():+.1f}% ({inv['premium'].notna().sum()} with comparables)")

    fig = page("New flats on offer",
               "Every flat a developer is selling in Warsaw today, from the developers' own price lists",
               "Asking prices: developers' price lists on dane.gov.pl (2025 price-transparency law), latest price per flat, "
               f"{len(o):,} flats in {len(inv)} investments placed via the city's address points (~85 % of Warsaw files "
               "readable). Premium: median asking per m² vs the typical-flat value (hedonic model, resale level, 2025) of "
               f"priced buildings built 2011–2020 within {COMP_M / 1000:g} km. Resales: RCN 2024–26. " + style.SOURCE)
    # 1 supply map
    ax1 = basemap(fig, d, w)
    ax1.set_position([0.01, 0.46, 0.48, 0.31])
    cm1 = ListedColormap(["#fde4cf", "#f9b98a", "#f08a4b", "#d4602a", "#a33f14", "#6e2508"])
    b1 = [0, 13000, 15000, 17000, 20000, 25000, 1e6]
    ax1.scatter(inv.geometry.x, inv.geometry.y, s=inv["flats"].clip(upper=600) * 0.9 + 6, c=inv["ask"], cmap=cm1,
                norm=BoundaryNorm(b1, cm1.N), edgecolor=style.SURFACE, lw=0.6, alpha=0.9, zorder=4)
    outlines(ax1, d)
    legend_patches(ax1, [cm1(i) for i in range(6)], ["< 13k", "13–15k", "15–17k", "17–20k", "20–25k", "25k+"],
                   "Asking, PLN/m²")
    ax1.set_title("Where they are (circle size = flats on offer)", fontsize=10.5)

    # 2 premium map
    ax2 = basemap(fig, d, w)
    ax2.set_position([0.51, 0.46, 0.48, 0.31])
    p = inv.dropna(subset=["premium"])
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=40)
    ax2.scatter(p.geometry.x, p.geometry.y, s=p["flats"].clip(upper=600) * 0.9 + 6, c=p["premium"].clip(-20, 40),
                cmap=style.DIV_CMAP, norm=norm, edgecolor=style.INK_2, lw=0.4, zorder=4)
    outlines(ax2, d, labels=False)
    cax = fig.add_axes([0.94, 0.61, 0.013, 0.13])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=style.DIV_CMAP), cax=cax, ticks=[-20, 0, 20, 40])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, length=0)
    cb.ax.set_yticklabels(["−20%", "0", "+20%", "+40%"])
    ax2.set_title("Asking vs a nearly new flat nearby", fontsize=10.5)

    # 3 by district
    a3 = fig.add_axes([0.20, 0.12, 0.34, 0.27])
    dist = o.groupby("district").agg(flats=("price_m2", "size"), ask=("price_m2", "median"))
    dist["resale"] = v.groupby("district")["value_2025"].median()
    dist = dist[dist["flats"] >= 30].sort_values("flats")
    yy = np.arange(len(dist))
    a3.barh(yy, dist["flats"], color="#e6e3da", height=0.7)
    for yi, n in zip(yy, dist["flats"]):
        a3.text(8, yi, style.fmt_int(n), va="center", fontsize=7, color=style.MUTED)
    a3.set_yticks(yy)
    a3.set_yticklabels(dist.index, fontsize=8.5)
    a3.set_xlabel("flats on offer", fontsize=8.5, color=style.MUTED)
    a3.tick_params(axis="x", labelsize=8, colors=style.MUTED)
    style.vgrid_only(a3)
    a3b = a3.twiny()
    a3b.scatter(dist["ask"], yy, color="#c4521f", s=28, zorder=5, label="asking (new)")
    a3b.scatter(dist["resale"], yy, color=style.BLUE, s=28, zorder=5, label="typical resale")
    for yi, a_, r_ in zip(yy, dist["ask"], dist["resale"]):
        a3b.plot([r_, a_], [yi, yi], color=style.INK_2, lw=0.8, zorder=4)
    a3b.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    a3b.tick_params(labelsize=8)
    a3b.legend(loc="lower right", fontsize=8)
    a3b.grid(False)
    a3.set_title("By district: supply (bars) and price per m² (dots)", fontsize=10, pad=24)

    # 4 size mix
    a4 = fig.add_axes([0.66, 0.12, 0.31, 0.23])
    t = gpd.read_file(os.path.join(ROOT, "output", "price_model.gpkg"), layer="transactions_adj",
                      ignore_geometry=True, columns=["year", "market", "area"])
    t = t[(t["year"] >= 2024) & (t["market"] == "secondary")]
    bins = np.arange(15, 131, 7.5)
    a4.hist(t["area"].clip(15, 130), bins=bins, density=True, color=style.BLUE, alpha=0.35, label="sold (resales 2024–26)")
    a4.hist(o["area"].clip(15, 130), bins=bins, density=True, histtype="step", color="#c4521f", lw=1.8,
            label="on offer now (new)")
    a4.set_yticks([])
    a4.set_xlabel("flat size, m²", fontsize=9)
    a4.legend(fontsize=8, loc="upper right")
    style.hgrid_only(a4)
    a4.set_title("What's being built", fontsize=10.5)
    small = (o["area"] < 40).mean()
    small_s = (t["area"] < 40).mean()
    prem = inv["premium"].median()
    fig.text(0.03, 0.885, wrap(f"{style.fmt_int(len(o))} new flats on offer, median {style.fmt_int(o['price_m2'].median())} "
                               f"PLN/m² and {o['area'].median():.0f} m². The median investment asks {abs(prem):.0f}% "
                               f"{'less' if prem < 0 else 'more'} per m² than a 5–15-year-old flat nearby; towers around "
                               "Rondo Daszyńskiego ask 30–50k and lift Wola's median.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "26_new_flats_on_offer.png")
    print(dist.round(0).to_string())


if __name__ == "__main__":
    main()
