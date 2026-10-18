"""
How big are Warsaw flats, and what does size cost? (RCN flat sales 2006-2026, hedonic model coefficients)

  1  median size of flats sold each year: new (developer) vs resale; RCN gap 2017-20 shaded
  2  rooms in new flats sold each year (share 1 / 2 / 3 / 4+ rooms)
  3  price per m² by size, same building, same date (hedonic coefficients vs 45-60 m²)
  4  map: median size of flats resold 2021-26 per 600 m hexagon

Output: output/summary/29_flat_sizes.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, legend_patches, load, outlines

style.apply()
ROOMS = ["1 room", "2 rooms", "3 rooms", "4+ rooms"]
ROOM_COL = ["#d6e4f5", "#83b2ea", "#2a6fc9", "#0d3670"]
BANDS = ["<25", "25-30", "30-35", "35-40", "40-45", "45-60", "60-75", "75-90", "90-120", "120+"]


def main():
    v, d, w = load()
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t = t[t["year"].between(2006, 2026)]
    sz = t.groupby(["year", "market"])["area"].agg(["median", "size"]).unstack()
    coef = pd.read_csv("output/analysis/model_coefficients.csv").set_index("term")["beta"]
    prem = pd.Series({b: 100 * (np.exp(coef.get(f"area:{b}", 0.0)) - 1) for b in BANDS})

    fig = page("Flats are getting smaller",
               "Size of flats sold in Warsaw, 2006–2026, and what size does to the price per m²",
               "RCN flat sales (full ownership). New = sold by the developer (deed at completion, so a year or two after "
               "the flat was bought). Years with < 300 sales hidden (2017–20 RCN gap). Price by size: hedonic model, same "
               "building and quarter, vs a 45–60 m² flat. Map: median size of resold flats 2021–26, hexagons with ≥ 30 "
               "sales. Rooms = RCN 'izby': a separate kitchen counts, a kitchen annex does not, so part of the fall in "
               "room counts is kitchens opening into living rooms. " + style.SOURCE)
    a1 = fig.add_axes([0.09, 0.55, 0.38, 0.23])
    for mk, lab, colr in (("primary", "new (developer)", "#c4521f"), ("secondary", "resale", style.BLUE)):
        s = sz[("median", mk)].where(sz[("size", mk)] >= 300).reindex(range(2006, 2027))
        a1.plot(s.index, s.values, "-o", color=colr, lw=2.2, ms=4, mec=style.SURFACE, label=lab)
        a1.annotate(f"{s.dropna().iloc[-1]:.0f} m²", (s.dropna().index[-1], s.dropna().iloc[-1]),
                    xytext=(6, -7 if mk == "primary" else 7),
                    textcoords="offset points", va="center", fontsize=9, color=colr, fontweight="semibold")
    style.shade_gap(a1)
    a1.set_ylim(40, 70)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f} m²"))
    style.hgrid_only(a1)
    a1.legend(loc="lower left", fontsize=8.5)
    a1.set_title("Median size of flats sold", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.59, 0.55, 0.37, 0.23])
    p = t[(t["market"] == "primary") & t["rooms"].between(1, 20)]
    mix = pd.crosstab(p["year"], p["rooms"].clip(upper=4), normalize="index")
    mix = mix[p.groupby("year").size().reindex(mix.index) >= 300]
    bottom = np.zeros(len(mix))
    for k, (lab, colr) in enumerate(zip(ROOMS, ROOM_COL), 1):
        vals = 100 * mix.get(float(k), mix.get(k, pd.Series(0, index=mix.index))).to_numpy()
        a2.bar(mix.index, vals, bottom=bottom, color=colr, width=0.8, label=lab, edgecolor=style.SURFACE, lw=0.5)
        bottom += vals
    a2.set_ylim(0, 100)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    a2.legend(loc="lower left", fontsize=7.5, ncol=4, bbox_to_anchor=(0, 1.0), frameon=False, handlelength=1)
    style.hgrid_only(a2)
    a2.set_title("Rooms in new flats sold (register count: a separate kitchen is a room)", fontsize=10.5,
                  loc="left", pad=20)

    a3 = fig.add_axes([0.09, 0.12, 0.38, 0.30])
    x = np.arange(len(prem))
    a3.bar(x, prem.values, color=[style.BLUE if val > 0 else "#c4521f" for val in prem.values], width=0.7)
    for xi, val in zip(x, prem.values):
        a3.text(xi, val + (0.6 if val >= 0 else -0.6), f"{val:+.0f}%", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8.5)
    a3.axhline(0, color=style.AXIS, lw=1)
    a3.set_xticks(x)
    a3.set_xticklabels([b.replace("-", "–") for b in BANDS], fontsize=8, rotation=40, ha="right")
    a3.set_xlabel("flat size, m²", fontsize=9)
    a3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:+.0f}%"))
    style.hgrid_only(a3)
    a3.set_title("Price per m² by size, same building", fontsize=10.5, loc="left")

    ax = basemap(fig, d, w)
    ax.set_position([0.52, 0.08, 0.46, 0.38])
    r = t[(t["market"] == "secondary") & (t["year"] >= 2021)]
    cmap = plt.cm.colors.ListedColormap(["#fde4cf", "#f9b98a", "#f08a4b", "#c4521f", "#7a2a08"])
    norm = plt.cm.colors.BoundaryNorm([0, 40, 45, 50, 57, 200], 5)
    ax.hexbin(r.geometry.x, r.geometry.y, C=r["area"], gridsize=(48, 42), reduce_C_function=np.median, mincnt=30,
              cmap=cmap, norm=norm, linewidths=0.3, edgecolors=style.SURFACE, zorder=3,
              extent=(*ax.get_xlim(), *ax.get_ylim()))
    outlines(ax, d, labels=False)
    legend_patches(ax, [cmap(i) for i in range(5)], ["< 40 m²", "40–45", "45–50", "50–57", "57+"], "Median size")
    ax.set_title("Median size of resold flats, 2021–26", fontsize=10.5)

    n_old = sz.loc[2006:2013, ("median", "primary")].median()
    n_new = sz.loc[2025:2026, ("median", "primary")].median()
    fig.text(0.03, 0.885, wrap(f"A new flat sold today is {n_new:.0f} m², down from {n_old:.0f} m² in 2006–13 "
                               f"({100 * (n_new / n_old - 1):+.0f}%): new flats are now no bigger than old ones. Small "
                               f"flats cost more per m²: under 25 m² {prem['<25']:+.0f}% vs a 45–60 m² flat in the same "
                               "building.", 100), fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "29_flat_sizes.png")


if __name__ == "__main__":
    main()
