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
               "New = sold by the developer (deed at completion, a year or two after purchase). Years with < 300 sales hidden (2017–20 register "
               "gap). Price by size: same building and quarter, vs a 45–60 m² flat. " + style.SOURCE)
    a1 = fig.add_axes([0.10, 0.50, 0.84, 0.30])
    for mk, lab, colr in (("primary", "new (developer)", "#c4521f"), ("secondary", "resale", style.BLUE)):
        s = sz[("median", mk)].where(sz[("size", mk)] >= 300).reindex(range(2006, 2027))
        a1.plot(s.index, s.values, "-o", color=colr, lw=2.2, ms=4, mec=style.SURFACE, label=lab)
        a1.annotate(f"{s.dropna().iloc[-1]:.0f} m²", (s.dropna().index[-1], s.dropna().iloc[-1]),
                    xytext=(6, -7 if mk == "primary" else 7),
                    textcoords="offset points", va="center", fontsize=9, color=colr, fontweight="semibold")
    style.shade_gap(a1)
    a1.set_ylim(40, 70)
    a1.set_xticks(range(2006, 2027, 2))
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f} m²"))
    style.hgrid_only(a1)
    a1.legend(loc="lower left", fontsize=8.5)
    a1.set_title("Median size of flats sold", fontsize=10.5, loc="left")

    a3 = fig.add_axes([0.10, 0.12, 0.84, 0.27])
    x = np.arange(len(prem))
    a3.bar(x, prem.values, color=[style.BLUE if val > 0 else "#c4521f" for val in prem.values], width=0.7)
    for xi, val in zip(x, prem.values):
        a3.text(xi, val + (0.6 if val >= 0 else -0.6), f"{val:+.0f}%", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=8.5)
    a3.axhline(0, color=style.AXIS, lw=1)
    a3.set_ylim(prem.min() - 3, prem.max() + 3)
    a3.set_xticks(x)
    a3.set_xticklabels([b.replace("-", "–") for b in BANDS], fontsize=9)
    a3.set_xlabel("flat size, m²", fontsize=9)
    a3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: f"{v_:+.0f}%"))
    style.hgrid_only(a3)
    a3.set_title("Price per m² by size, same building", fontsize=10.5, loc="left")

    n_old = sz.loc[2006:2013, ("median", "primary")].median()
    n_new = sz.loc[2025:2026, ("median", "primary")].median()
    fig.text(0.03, 0.885, wrap(f"A new flat sold today is {n_new:.0f} m², down from {n_old:.0f} m² in 2006–13 "
                               f"({100 * (n_new / n_old - 1):+.0f}%): new flats are now no bigger than old ones. Small "
                               f"flats cost more per m²: under 25 m² {prem['<25']:+.0f}% vs a 45–60 m² flat in the same "
                               "building.", 100), fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "29_flat_sizes.png")


if __name__ == "__main__":
    main()
