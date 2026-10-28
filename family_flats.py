"""
Two buyer's-eye figures from actual sales (no model, no benchmark flat):

  33  What 1 million złoty buys: size of flats resold for 0.9-1.1 million PLN in 2024-26, by hexagon and district
  34  Room for a family: where flats of 70 m² or more are resold (share of resales 2021-26) and what a 65-80 m²
      flat costs (median total price, resales 2024-26)
  36  Where the big flats are: median size and share of 4+ room flats among all flat sales since 2016 (developer
      and resale: where such flats exist), per hexagon, with the top city neighbourhoods (MSI) named

Resales only: developer deeds are recorded at completion, at a price agreed 1-2 years earlier, so they would make
recent money look like it buys more. Rooms in RCN are "izby" (a separate kitchen counts as a room, a kitchen annex
does not), so size in m² is the measure here.

  40  Where flats are big (2026-10-28, replaces 33 + 34 in the report): one map, one "bigness" score per hexagon
      (half median m², half mean rooms, both as ranks across hexagons), all sales since 2021, plus districts below

Output: output/summary/33_what_1m_buys.png, 34_family_flats.png, 36_big_flats.png, 40_where_flats_are_big.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
               f"{len(r):,} resales at {BUDGET[0]:,}–{BUDGET[1]:,} PLN, {RECENT[0]}–{RECENT[1]}; price as paid. Map: hexagons with ≥ 8 sales; "
               f"bars: districts with ≥ {MIN_DISTRICT} such sales. Developer sales left out (prices agreed 1–2 years earlier). " + style.SOURCE)
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
               f"Map: share of resales {r['year'].min()}–{r['year'].max()} that are {FAMILY_M2} m² or more, hexagons with ≥ 30 resales. "
               f"Bars: median price as paid for a {FAMILY_BAND[0]}–{FAMILY_BAND[1]} m² resale, {RECENT[0]}–{RECENT[1]}, districts with ≥ {MIN_DISTRICT} sales. " + style.SOURCE)
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


def top_list(fig, x, y, title, s, fmt):
    fig.text(x, y, title, fontsize=10, fontweight="semibold", va="top")
    for k, (name, val) in enumerate(s.items()):
        fig.text(x, y - 0.028 * (k + 1.3), f"{fmt(val)}", fontsize=9, va="top", ha="right", color=style.INK)
        fig.text(x + 0.012, y - 0.028 * (k + 1.3), name, fontsize=9, va="top", color=style.INK_2)


def big_flats(d, w):
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t = t[(t["year"] >= 2016) & t["district"].notna() & t["area"].between(12, 300)]
    t["many"] = (t["rooms"] >= 4).astype(float)
    msi = gpd.read_file("data/msi.gpkg").to_crs(2180)
    j = gpd.sjoin(t[["area", "many", "geometry"]], msi[["name", "geometry"]], predicate="within")
    nb = j.groupby("name").agg(n=("area", "size"), med=("area", "median"), many=("many", "mean"))
    nb = nb[nb["n"] >= 300]
    many_small = nb[(nb["many"] >= 0.35) & (nb["med"] < 55)].sort_values("med")      # many rooms, little space
    fig = page("Where the big flats are",
               f"All flat sales since 2016 (developer and resale): size and number of rooms",
               f"{len(t):,} flat sales 2016–{int(t['year'].max())}. Hexagons with ≥ 40 sales; lists: MSI neighbourhoods with ≥ 300 sales. "
               "Rooms = the register's 'izby' (a separate kitchen counts as a room). " + style.SOURCE)
    ex = many_small.index[0] if len(many_small) else None
    fig.text(0.03, 0.885, wrap("Space and rooms are not the same thing. The biggest flats sell in Wawer, on the Saska "
                               "Kępa riverside and around Stegny and Wyględów"
                               + (f". Many rooms also come in prefab estates: in {ex}, {100 * many_small.loc[ex, 'many']:.0f} % "
                                  f"of flats have 4+ rooms in a median {many_small.loc[ex, 'med']:.0f} m²." if ex else "."),
                               100), fontsize=10.5, fontweight="semibold", va="top")
    panels = (("area", np.median, ["#f3e4d0", "#e6bb8a", "#d08a4f", "#a65a2a", "#6b3214"], [0, 45, 52, 60, 70, 500],
               ["< 45 m²", "45–52", "52–60", "60–70", "70+ m²"], "Median size", [0.02, 0.44, 0.62, 0.37]),
              ("many", np.mean, ["#e2e6f3", "#b3bde3", "#7f8fcc", "#4d5fae", "#26337a"], [0, 0.15, 0.22, 0.30, 0.40, 1],
               ["< 15 %", "15–22 %", "22–30 %", "30–40 %", "40 %+"], "4+ rooms", [0.02, 0.06, 0.62, 0.38]))
    for col, red, colors, bounds, labels, title, box in panels:
        ax = basemap(fig, d, w)
        ax.set_position(box)
        cmap = plt.cm.colors.ListedColormap(colors)
        ax.hexbin(t.geometry.x, t.geometry.y, C=t[col], gridsize=(48, 42), reduce_C_function=red, mincnt=40,
                  cmap=cmap, norm=plt.cm.colors.BoundaryNorm(bounds, 5), linewidths=0.3, edgecolors=style.SURFACE,
                  zorder=3, extent=(*ax.get_xlim(), *ax.get_ylim()))
        outlines(ax, d, labels=False)
        legend_patches(ax, colors, labels, title)
    top_list(fig, 0.70, 0.80, "Biggest flats (median)", nb["med"].nlargest(9), lambda v_: f"{v_:.0f} m²")
    top_list(fig, 0.70, 0.41, "Most 4+ room flats", nb["many"].nlargest(9), lambda v_: f"{100 * v_:.0f} %")
    save(fig, "36_big_flats.png")


def where_flats_are_big(d, w):
    from matplotlib.colors import LinearSegmentedColormap
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t = t[(t["year"] >= 2021) & t["district"].notna() & t["area"].between(12, 300) & t["rooms"].between(1, 10)]
    dist = t.groupby("district").agg(area=("area", "median"), rooms=("rooms", "mean"), n=("area", "size"))
    dist = dist[dist["n"] >= 300].sort_values("area")
    lo, hi = dist.index[0], dist.index[-1]
    fig = page("Where Warsaw's flats are big",
               "Size and number of rooms of every flat sold since 2021, new and resale",
               f"{len(t):,} flat sales 2021–{int(t['year'].max())}; hexagons with ≥ 15 sales. Score = average of a hexagon's rank "
               "on median m² and on mean rooms (the register's 'izby'). " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"A typical flat sold in {lo} is {dist.loc[lo, 'area']:.0f} m² with "
                               f"{dist.loc[lo, 'rooms']:.1f} rooms on average; in {hi}, {dist.loc[hi, 'area']:.0f} m² and "
                               f"{dist.loc[hi, 'rooms']:.1f}. The darker the hexagon, the bigger the flats and the more rooms.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    ax = basemap(fig, d, w, box=[0.02, 0.30, 0.96, 0.55])
    kw = dict(gridsize=(52, 46), mincnt=15, extent=(*ax.get_xlim(), *ax.get_ylim()))
    ha = ax.hexbin(t.geometry.x, t.geometry.y, C=t["area"], reduce_C_function=np.median, **kw)
    hr = ax.hexbin(t.geometry.x, t.geometry.y, C=t["rooms"], reduce_C_function=np.mean, **kw)
    assert np.allclose(ha.get_offsets(), hr.get_offsets())
    score = 0.5 * pd.Series(ha.get_array()).rank(pct=True) + 0.5 * pd.Series(hr.get_array()).rank(pct=True)
    cmap = LinearSegmentedColormap.from_list("big", ["#f4eedd", "#f2c97d", "#e8914d", "#c9533f", "#8e2f55", "#4b1d5e"])
    ha.set_cmap(cmap)
    ha.set_clim(0.1, 0.95)
    ha.set_array(score.to_numpy())
    ha.set_edgecolor(style.SURFACE)
    ha.set_linewidth(0.25)
    ha.set_zorder(3)
    hr.remove()
    outlines(ax, d)
    cax = fig.add_axes([0.62, 0.755, 0.32, 0.012])
    cax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=cmap)
    cax.set_axis_off()
    fig.text(0.62, 0.772, "smaller flats", fontsize=8.5, color=style.INK_2, va="bottom")
    fig.text(0.94, 0.772, "bigger flats, more rooms", fontsize=8.5, color=style.INK_2, va="bottom", ha="right")

    ax2 = fig.add_axes([0.06, 0.115, 0.90, 0.14])
    x = np.arange(len(dist))
    rk = (dist["area"].rank(pct=True) * 0.5 + dist["rooms"].rank(pct=True, method="average") * 0.5).to_numpy()
    ax2.bar(x, dist["area"], width=0.72, color=[cmap(0.1 + 0.85 * v_) for v_ in rk])
    for xi, (a_, r_) in zip(x, zip(dist["area"], dist["rooms"])):
        ax2.text(xi, a_ + 1, f"{a_:.0f}", ha="center", va="bottom", fontsize=8.5, color=style.INK)
        ax2.text(xi, 2.5, f"{r_:.1f}", ha="center", va="bottom", fontsize=8, color="white", fontweight="semibold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(dist.index, rotation=55, ha="right", fontsize=8.5)
    ax2.set_yticks([])
    ax2.set_ylim(0, dist["area"].max() * 1.15)
    style.hgrid_only(ax2)
    ax2.grid(False)
    for sp in ("left",):
        ax2.spines[sp].set_visible(False)
    ax2.set_title("Typical flat by district: median m² on top, average rooms in the bar", fontsize=10.5)
    save(fig, "40_where_flats_are_big.png")
    print(dist.round(1).to_string())


def main():
    v, d, w = load()
    t = resales()
    what_1m_buys(t, d, w)
    family_flats(t, d, w)
    big_flats(d, w)
    where_flats_are_big(d, w)


if __name__ == "__main__":
    main()
