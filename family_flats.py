"""
Two buyer's-eye figures from actual sales (no model, no benchmark flat):

  33  What 1 million złoty buys: size of flats resold for 0.9-1.1 million PLN in 2024-26, by hexagon and district
  34  Room for a family: where flats of 70 m² or more are resold (share of resales 2021-26) and what a 65-80 m²
      flat costs (median total price, resales 2024-26)

Resales only: developer deeds are recorded at completion, at a price agreed 1-2 years earlier, so they would make
recent money look like it buys more. Rooms in RCN are "izby" (a separate kitchen counts as a room, a kitchen annex
does not), so size in m² is the measure here.

Output: output/summary/33_what_1m_buys.png, 34_family_flats.png
"""
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, legend_patches, load, outlines

style.apply()
BUDGET = (900_000, 1_100_000)
RECENT = (2024, 2026)
FAMILY_M2 = 70
FAMILY_BAND = (65, 80)
MIN_DISTRICT = 25


def resales():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    return t[(t["market"] == "secondary") & t["district"].notna() & t["area"].between(12, 300)]


def district_bars(ax, s, fmt, title, color):
    s = s.sort_values()
    y = np.arange(len(s))
    ax.barh(y, s.values, color=color, height=0.68)
    for yi, val in zip(y, s.values):
        ax.text(val, yi, " " + fmt(val), va="center", fontsize=8.5)
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=8.5)
    ax.set_xlim(0, s.max() * 1.22)
    ax.set_xticks([])
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.set_title(title, fontsize=10.5, loc="left")


def what_1m_buys(t, d, w):
    r = t[t["year"].between(*RECENT) & t["price"].between(*BUDGET)]
    g = r.groupby("district")["area"].agg(["median", "size"])
    g = g[g["size"] >= MIN_DISTRICT]["median"]
    lo, hi = g.idxmin(), g.idxmax()
    fig = page("What 1 million złoty buys",
               f"Size of flats resold for {BUDGET[0] / 1e6:.1f}–{BUDGET[1] / 1e6:.1f} million PLN, {RECENT[0]}–{RECENT[1]}",
               f"{len(r):,} resales (full ownership) recorded in RCN at {BUDGET[0]:,}–{BUDGET[1]:,} PLN, "
               f"{RECENT[0]}–{RECENT[1]}. Price as paid, without fees or renovation. Map: median size per hexagon "
               f"(≥ 8 sales); bars: districts with ≥ {MIN_DISTRICT} such sales. Developer sales left out: their "
               "deeds carry prices agreed 1–2 years earlier. " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"The same million buys a {g[lo]:.0f} m² flat in {lo} and a {g[hi]:.0f} m² one in {hi}: "
                               f"{g[hi] / g[lo]:.1f}× the space. Across the river or out past the ring roads, the "
                               "budget of a city-centre two-room flat becomes a family flat.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    ax = basemap(fig, d, w)
    ax.set_position([0.02, 0.10, 0.60, 0.72])
    cmap = plt.cm.colors.ListedColormap(["#f6d6c2", "#eea583", "#d8694c", "#a83a3e", "#6b1d3a"])   # darker = more space
    bounds = [0, 50, 55, 62, 70, 300]
    norm = plt.cm.colors.BoundaryNorm(bounds, 5)
    ax.hexbin(r.geometry.x, r.geometry.y, C=r["area"], gridsize=(34, 30), reduce_C_function=np.median, mincnt=8,
              cmap=cmap, norm=norm, linewidths=0.4, edgecolors=style.SURFACE, zorder=3,
              extent=(*ax.get_xlim(), *ax.get_ylim()))
    outlines(ax, d, labels=False)
    legend_patches(ax, [cmap(i) for i in range(5)], ["< 50 m²", "50–55", "55–62", "62–70", "70+ m²"],
                   "1 million buys")
    a2 = fig.add_axes([0.72, 0.12, 0.24, 0.66])
    district_bars(a2, g, lambda v_: f"{v_:.0f} m²", "Median size, by district", "#a83a3e")
    save(fig, "33_what_1m_buys.png")
    print(g.sort_values().round(0).to_string())


def family_flats(t, d, w):
    r = t[t["year"] >= 2021]
    share = r.groupby("district")["area"].apply(lambda s: (s >= FAMILY_M2).mean())
    f = t[t["year"].between(*RECENT) & t["area"].between(*FAMILY_BAND)]
    cost = f.groupby("district")["price"].agg(["median", "size"])
    cost = cost[cost["size"] >= MIN_DISTRICT]["median"]
    lo, hi = cost.idxmin(), cost.idxmax()
    fig = page("Room for a family",
               f"Where flats of {FAMILY_M2} m² or more are resold, and what a {FAMILY_BAND[0]}–{FAMILY_BAND[1]} m² flat costs",
               f"RCN resales (full ownership). Map: share of resales {r['year'].min()}–{r['year'].max()} that are "
               f"{FAMILY_M2} m² or more, hexagons with ≥ 30 resales. Bars: median price as paid for a "
               f"{FAMILY_BAND[0]}–{FAMILY_BAND[1]} m² resale, {RECENT[0]}–{RECENT[1]}, districts with ≥ {MIN_DISTRICT} "
               "sales. Size, not rooms: the register counts a separate kitchen as a room. " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"Only {100 * (r['area'] >= FAMILY_M2).mean():.0f} % of flats resold since 2021 have "
                               f"{FAMILY_M2} m² or more. A {FAMILY_BAND[0]}–{FAMILY_BAND[1]} m² flat costs "
                               f"{cost[lo] / 1e6:.2f} million in {lo} and {cost[hi] / 1e6:.2f} million in {hi}.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    ax = basemap(fig, d, w)
    ax.set_position([0.02, 0.10, 0.60, 0.72])
    cmap = plt.cm.colors.ListedColormap(["#dbe8f6", "#a9c8ec", "#6fa2dd", "#2f6fc0", "#123f80"])
    norm = plt.cm.colors.BoundaryNorm([0, 0.08, 0.14, 0.20, 0.30, 1], 5)
    big = (r["area"] >= FAMILY_M2).astype(float)
    ax.hexbin(r.geometry.x, r.geometry.y, C=big, gridsize=(48, 42), reduce_C_function=np.mean, mincnt=30,
              cmap=cmap, norm=norm, linewidths=0.3, edgecolors=style.SURFACE, zorder=3,
              extent=(*ax.get_xlim(), *ax.get_ylim()))
    outlines(ax, d, labels=False)
    legend_patches(ax, [cmap(i) for i in range(5)], ["< 8 %", "8–14 %", "14–20 %", "20–30 %", "30 %+"],
                   f"Resales ≥ {FAMILY_M2} m²")
    a2 = fig.add_axes([0.72, 0.12, 0.24, 0.66])
    district_bars(a2, cost, lambda v_: f"{v_ / 1e6:.2f} mln", f"A {FAMILY_BAND[0]}–{FAMILY_BAND[1]} m² flat costs",
                  style.BLUE)
    save(fig, "34_family_flats.png")
    print(cost.sort_values().round(-3).to_string())


def main():
    v, d, w = load()
    t = resales()
    what_1m_buys(t, d, w)
    family_flats(t, d, w)


if __name__ == "__main__":
    main()
