"""
What a building's age does to price (construction decade from fetch_building_age.py, source GIS Expert).

  top     typical-flat value by construction decade, relative to the same district (median of
          log value_2025 minus the district's median), buildings with >= 3 sales
  bottom  new-build premium by place: 2011-2020 buildings vs 1961-1990 (prefab era) buildings in the same
          1.5 km hexagon (>= 3 of each), map + by district

Output: output/summary/18_building_age.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load, outlines

style.apply()

ORDER = [(1900, "before\n1901"), (1910, "1901–\n10"), (1920, "1911–\n20"), (1930, "1921–\n30"), (1940, "1931–\n40"),
         (1950, "1941–\n50"), (1960, "1951–\n60"), (1970, "1961–\n70"), (1980, "1971–\n80"), (1990, "1981–\n90"),
         (2000, "1991–\n2000"), (2010, "2001–\n10"), (2020, "2011–\n20"), (2030, "2021+")]
ERA_COL = {1900: "#8c6d46", 1910: "#8c6d46", 1920: "#8c6d46", 1930: "#8c6d46", 1940: "#8c6d46",
           1950: "#b5651d", 1960: "#b5651d", 1970: "#7d7d7d", 1980: "#7d7d7d", 1990: "#7d7d7d",
           2000: "#3a7fc1", 2010: "#2a78d6", 2020: "#2a78d6", 2030: "#2a78d6"}


def main():
    v, d, w = load()
    age = gpd.read_file("output/age_check.gpkg", ignore_geometry=True)[["osm_id", "decade_end"]]
    v = v.drop(columns=[c for c in ("decade_end",) if c in v]).merge(age, on="osm_id", how="left")
    v = v[v["decade_end"] > 0].copy()
    v["lv"] = np.log(v["value_2025"])
    v["rel"] = v["lv"] - v.groupby("district")["lv"].transform("median")
    g = v.groupby("decade_end")["rel"].agg(["median", "size"])
    g = g[g["size"] >= 30]
    pct = 100 * (np.exp(g["median"]) - 1)
    print(pd.DataFrame({"pct": pct.round(1), "n": g["size"]}).to_string())

    fig = page("Old, older, newest",
               "What a building's age does to the price of its flats",
               "Construction decade: GIS Expert (mapy.gis-expert.pl) and others. Price: typical-flat value 2025 (≥ 3 sales) vs the district's "
               "median building. Map: 2011–20 vs 1961–90 buildings per 1.5 km hexagon. " + style.SOURCE)
    a1 = fig.add_axes([0.10, 0.59, 0.86, 0.21])
    labs = dict(ORDER)
    x = np.arange(len(pct))
    a1.bar(x, pct.values, color=[ERA_COL[k] for k in pct.index], width=0.7)
    for xi, (k, val) in zip(x, pct.items()):
        a1.text(xi, val + (0.6 if val >= 0 else -0.6), f"{val:+.0f}%", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=8.5, color=style.INK)
    a1.axhline(0, color=style.AXIS, lw=1)
    a1.set_xticks(x)
    a1.set_xticklabels([labs[k] for k in pct.index], fontsize=8)
    lo, hi = pct.min(), pct.max()
    a1.set_ylim(min(lo, 0) - 4, hi + 5)
    style.pct(a1)
    style.hgrid_only(a1)
    a1.set_title("Typical flat vs the median building of its district, by construction decade", fontsize=11)
    from matplotlib.patches import Patch
    a1.legend(handles=[Patch(color="#8c6d46", label="pre-war"), Patch(color="#b5651d", label="post-war rebuild"),
                       Patch(color="#7d7d7d", label="prefab era"), Patch(color="#2a78d6", label="after 1990")],
              loc="upper left", fontsize=8.5, ncol=4)

    # new vs prefab premium per 1.5 km hexagon
    new = v["decade_end"].between(2011, 2020)
    old = v["decade_end"].between(1961, 1990)
    pts = v.geometry.representative_point()
    ax = basemap(fig, d, w)
    ax.set_position([0.03, 0.08, 0.60, 0.44])
    kw = dict(gridsize=(19, 17), extent=(*ax.get_xlim(), *ax.get_ylim()), reduce_C_function=np.median)
    hn = ax.hexbin(pts.x[new], pts.y[new], C=v.loc[new, "lv"], mincnt=3, **kw)
    ho = ax.hexbin(pts.x[old], pts.y[old], C=v.loc[old, "lv"], mincnt=3, **kw)
    on = {tuple(np.round(o, 3)): val for o, val in zip(hn.get_offsets(), hn.get_array())}
    oo = {tuple(np.round(o, 3)): val for o, val in zip(ho.get_offsets(), ho.get_array())}
    both = [k for k in on if k in oo]
    prem = np.array([100 * (np.exp(on[k] - oo[k]) - 1) for k in both])
    hn.remove()
    ho.remove()
    # breaks around the hexagons' own spread (median ~ +34 %); one hue, clearly separated lightness steps
    cmap = ListedColormap(["#e6effa", "#b9d3f3", "#83b2ea", "#4c8bd9", "#2461b0", "#0d3670"])
    bounds = [-100, 20, 27, 34, 42, 60, 1000]
    norm = BoundaryNorm(bounds, cmap.N)
    xy = np.array(both)
    ax.scatter(xy[:, 0], xy[:, 1], c=prem, cmap=cmap, norm=norm, marker="h", s=150, edgecolor=style.SURFACE,
               lw=0.6, zorder=3)
    outlines(ax, d)
    ax.legend(handles=[Patch(color=cmap(i), label=l) for i, l in enumerate(
        ["< 20%", "20–27%", "27–34%", "34–42%", "42–60%", "60%+"])], title="New vs prefab", loc="upper right",
        fontsize=8, title_fontsize=8.5, frameon=True, framealpha=0.9, edgecolor="none")
    ax.set_title("Premium of a 2011–20 building over a 1961–90 one nearby", fontsize=10.5)

    dist = pd.DataFrame({"new": v[new].groupby("district")["lv"].median(),
                         "old": v[old].groupby("district")["lv"].median(),
                         "nn": v[new].groupby("district").size(), "no": v[old].groupby("district").size()})
    dist = dist[(dist["nn"] >= 10) & (dist["no"] >= 10)]
    dist["prem"] = 100 * (np.exp(dist["new"] - dist["old"]) - 1)
    dist = dist.sort_values("prem")
    a3 = fig.add_axes([0.80, 0.10, 0.17, 0.40])
    a3.barh(np.arange(len(dist)), dist["prem"], color=style.BLUE, height=0.65)
    a3.set_yticks(np.arange(len(dist)))
    a3.set_yticklabels(dist.index, fontsize=8.5)
    for yi, val in enumerate(dist["prem"]):
        a3.text(val + 1, yi, f"{val:+.0f}%", va="center", fontsize=8, color=style.INK_2)
    a3.axvline(0, color=style.AXIS, lw=1)
    a3.set_xlim(min(0, dist["prem"].min()) - 3, dist["prem"].max() * 1.35)
    style.pct(a3, "x")
    style.vgrid_only(a3)
    a3.tick_params(axis="x", labelsize=8)
    a3.set_title("By district", fontsize=10)
    prewar = pct.loc[[k for k in pct.index if k <= 1930]].median()
    prefab = pct.loc[[k for k in (1970, 1980, 1990) if k in pct.index]]
    fig.text(0.03, 0.885, wrap(f"A U-shape: pre-war tenements sell {prewar:+.0f}% vs their district's median building, "
                               f"prefab estates of 1961–90 {prefab.min():+.0f} to {prefab.max():+.0f}%, buildings of "
                               f"2011–20 {pct.get(2020, np.nan):+.0f}%. A new building's lead over a prefab one is "
                               f"biggest in the centre ({dist.index[-1]} {dist['prem'].iloc[-1]:+.0f}%) and smallest on "
                               f"the edges ({dist.index[0]} {dist['prem'].iloc[0]:+.0f}%).", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "18_building_age.png")
    print(dist.round(1).to_string())


if __name__ == "__main__":
    main()
