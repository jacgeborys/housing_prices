"""
Which neighbourhoods sell furthest above / below what their features predict?

Residuals from price_drivers.py (out-of-sample gradient boosting, no district variable) averaged per MSI
neighbourhood (the city's 143 official areas, data/msi.gpkg) in log points, then shrunk toward zero with
empirical Bayes: weight = n / (n + k), k = within-area residual variance / between-area variance. An area
with a handful of buildings cannot top the list by luck.

Outputs:
  output/analysis/estates_ranking.csv
  output/summary/15_estates.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load

style.apply()

MIN_BUILDINGS = 10
TOP = 10


def ranking(v, msi):
    pts = gpd.GeoDataFrame(v[["resid_pct"]], geometry=v.geometry.representative_point(), crs=2180)
    j = gpd.sjoin(pts, msi[["name", "district", "geometry"]], predicate="within", how="inner")
    j = j[~j.index.duplicated()]
    j["r"] = np.log1p(j["resid_pct"] / 100)
    g = j.groupby(["name", "district"])["r"].agg(["mean", "var", "size"]).reset_index()
    g = g[g["size"] >= MIN_BUILDINGS]
    within = np.average(g["var"], weights=g["size"])
    between = max(np.average((g["mean"] - np.average(g["mean"], weights=g["size"])) ** 2, weights=g["size"])
                  - within / g["size"].mean(), 1e-6)
    k = within / between
    g["w"] = g["size"] / (g["size"] + k)
    g["shrunk"] = g["w"] * g["mean"]
    g["se"] = np.sqrt(g["w"] * within / g["size"])
    for c in ("mean", "shrunk"):
        g[f"{c}_pct"] = 100 * (np.exp(g[c]) - 1)
    g["lo_pct"] = 100 * (np.exp(g["shrunk"] - 1.96 * g["se"]) - 1)
    g["hi_pct"] = 100 * (np.exp(g["shrunk"] + 1.96 * g["se"]) - 1)
    print(f"{len(g)} neighbourhoods with >= {MIN_BUILDINGS} buildings; shrinkage k = {k:.1f} buildings")
    return g.sort_values("shrunk")


def main():
    v, d, w = load()
    msi = gpd.read_file("data/msi.gpkg")
    g = ranking(v, msi)
    g.drop(columns=["var"]).round(3).to_csv("output/analysis/estates_ranking.csv", index=False)

    fig = page("Warsaw's over- and under-achievers",
               "Neighbourhoods that sell furthest above or below what their location, amenities and buildings predict",
               f"MSI neighbourhoods with ≥ {MIN_BUILDINGS} priced buildings. Value = average gap between a building's typical-flat price and "
               "the out-of-sample prediction, shrunk toward 0 for small areas; bars = 95 % interval. " + style.SOURCE)
    ax = basemap(fig, d, w)
    ax.set_position([0.10, 0.52, 0.66, 0.36])
    norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    m = msi.merge(g[["name", "shrunk_pct"]], on="name", how="left")
    m[m["shrunk_pct"].isna()].plot(ax=ax, color=style.CONTEXT, edgecolor=style.SURFACE, linewidth=0.4, zorder=3)
    m.dropna(subset=["shrunk_pct"]).plot(ax=ax, column="shrunk_pct", cmap=style.DIV_CMAP, norm=norm,
                                          edgecolor=style.SURFACE, linewidth=0.4, zorder=3)
    d.boundary.plot(ax=ax, color=style.INK_2, linewidth=0.6, zorder=4)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=style.DIV_CMAP)
    cax = fig.add_axes([0.80, 0.58, 0.02, 0.22])
    cb = fig.colorbar(sm, cax=cax, ticks=[-15, -10, -5, 0, 5, 10, 15])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0)
    cb.ax.set_yticklabels(["−15%", "−10%", "−5%", "as\npredicted", "+5%", "+10%", "+15%"])

    sel = pd.concat([g.head(TOP), g.tail(TOP)])
    ax2 = fig.add_axes([0.36, 0.12, 0.58, 0.37])
    y = np.arange(len(sel))
    colr = [style.DIV_CMAP(norm(x)) for x in sel["shrunk_pct"]]
    ax2.hlines(y, 0, sel["shrunk_pct"], color=colr, lw=3)
    ax2.errorbar(sel["shrunk_pct"], y, xerr=[sel["shrunk_pct"] - sel["lo_pct"], sel["hi_pct"] - sel["shrunk_pct"]],
                 fmt="none", ecolor=style.INK_2, elinewidth=0.8, capsize=0)
    ax2.scatter(sel["shrunk_pct"], y, s=50, color=colr, edgecolor=style.INK_2, lw=0.6, zorder=3)
    for yi, (_, r) in zip(y, sel.iterrows()):
        ax2.text(r["hi_pct"] + 0.8 if r["shrunk_pct"] > 0 else r["lo_pct"] - 0.8, yi, f"{r['shrunk_pct']:+.0f}%",
                 va="center", ha="left" if r["shrunk_pct"] > 0 else "right", fontsize=9, color=style.INK)
    ax2.axvline(0, color=style.AXIS, lw=1)
    ax2.axhline(TOP - 0.5, color=style.GRID, lw=1, ls=(0, (3, 3)))
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{n}  ·  {dd}  ({s})" for n, dd, s in zip(sel["name"], sel["district"], sel["size"])],
                        fontsize=9)
    lim = max(abs(sel["lo_pct"]).max(), abs(sel["hi_pct"]).max()) + 6
    ax2.set_xlim(-lim, lim)
    style.pct(ax2, "x")
    style.vgrid_only(ax2)
    ax2.set_xlabel("Price vs prediction (buildings in brackets)", fontsize=10)
    save(fig, "15_estates.png")
    print(sel[["name", "district", "size", "mean_pct", "shrunk_pct"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
