"""
What a house costs per m², by 300 m cell. Houses (houses.py -> output/houses.gpkg, 2017-2026) are grouped into 300 m
squares like flats are grouped into buildings: a cell with >= 3 sales gets the median price per m² of living space
(house INCLUDING its plot; area = the deed's own where recorded, else BDOT10k outline x floors, see houses.py).
Sales are brought to 2025 prices with a house price index from the same sales (log price per m² on district and year).
Plain description, no model of features.

-> output/summary/39_house_cells.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load

style.apply()

CELL = 300
MIN_IN_CELL = 3
MIN_DISTRICT = 40
M2 = (2000, 40000)
CMAP = style.PRICE_CMAP


def year_index(h):
    """House price level by year (relative to 2025), net of district: OLS of log price per m² on district + year."""
    X = pd.concat([pd.get_dummies(h["district"]), pd.get_dummies(h["year"].astype(int), prefix="y")], axis=1)
    X = X.drop(columns="y_2025").astype(float)
    beta = np.linalg.lstsq(X.to_numpy(), np.log(h["price_m2"].to_numpy()), rcond=None)[0]
    g = pd.Series(beta, index=X.columns)
    gy = {int(c[2:]): g[c] for c in X.columns if c.startswith("y_")}
    gy[2025] = 0.0
    return gy


def main():
    v, d, w = load()
    h = gpd.read_file("output/houses.gpkg")
    h = h[h["price_m2"].between(*M2) & (h["year"] >= 2017)].copy()
    gy = year_index(h)
    print("house price level by year (2025 = 100):", {y: round(100 * np.exp(-gy[y] * -1), 0) for y in sorted(gy)})
    h["adj"] = h["price_m2"] * np.exp(-h["year"].astype(int).map(gy))        # to 2025 prices
    h["cx"] = (h.geometry.x // CELL).astype(int)
    h["cy"] = (h.geometry.y // CELL).astype(int)
    cells = h.groupby(["cx", "cy"]).agg(n=("adj", "size"), val=("adj", "median"), district=("district", lambda s: s.mode()[0]))
    good = cells[cells["n"] >= MIN_IN_CELL]
    covered = h.merge(good.reset_index()[["cx", "cy"]], on=["cx", "cy"]).shape[0] / len(h)
    print(f"{len(h):,} house sales, {len(good):,} cells with >= {MIN_IN_CELL} sales covering {covered:.0%}")

    dist = h.groupby("district").agg(n=("adj", "size"), val=("adj", "median"))
    dist = dist[dist["n"] >= MIN_DISTRICT].sort_values("val")
    lo, hi = dist.iloc[0], dist.iloc[-1]
    norm = Normalize(6000, 18000)
    fig = page(f"A house m² costs {lo['val'] / 1000:.0f}K in {dist.index[0]} and {hi['val'] / 1000:.0f}K in {dist.index[-1]}",
               f"Median price per m² of living space (plot included) of houses sold 2017–2026, in 2025 prices, per {CELL} m cell",
               f"{len(h):,} house sales (RCN, from 2017) in 2025 prices (house price index net of district). Cells with ≥ {MIN_IN_CELL} sales "
               f"coloured ({len(good):,} cells, {covered:.0%} of sales). Area: deed or estimated from outline and floors. " + style.SOURCE)
    ax = basemap(fig, d, w)
    ax.set_position([0.03, 0.40, 0.94, 0.45])
    rest = h.merge(good.reset_index()[["cx", "cy"]], on=["cx", "cy"], how="left", indicator=True)
    rest = rest[rest["_merge"] == "left_only"]
    ax.scatter(rest.geometry.x, rest.geometry.y, s=1.6, color="#bdbbb2", lw=0, zorder=3)
    for (cx, cy), r in good.iterrows():
        ax.add_patch(Rectangle((cx * CELL, cy * CELL), CELL, CELL, fc=CMAP(norm(r["val"])), ec="none", zorder=4))
    cax = fig.add_axes([0.70, 0.405, 0.24, 0.011])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([6000, 10000, 14000, 18000])
    cb.set_ticklabels(["6K", "10K", "14K", "18K+"])
    cax.tick_params(labelsize=8.5, length=0)
    cax.set_title("PLN per m²", fontsize=8.5, loc="left")

    a = fig.add_axes([0.16, 0.085, 0.78, 0.25])
    yy = np.arange(len(dist))
    a.barh(yy, dist["val"], color=[CMAP(norm(x)) for x in dist["val"]], height=0.7)
    for i, (x, n) in enumerate(zip(dist["val"], dist["n"])):
        a.text(x, i, f" {x / 1000:.1f}K  ({n:,} sales)", va="center", fontsize=8.5)
    a.set_yticks(yy)
    a.set_yticklabels(dist.index, fontsize=8.5)
    a.set_xlim(0, dist["val"].max() * 1.35)
    a.set_xticks([])
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title("Median by district", fontsize=10.5, loc="left")
    save(fig, "39_house_cells.png")


if __name__ == "__main__":
    main()
