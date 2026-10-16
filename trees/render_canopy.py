"""
Tree map of Warsaw: denser tree cover in deeper, more saturated greens.

Reads trees/data/tree_density_warsaw_10m.tif (fetch_tree_density.py, Copernicus HRL 2018). Writes
  trees/output/canopy_warsaw.png   whole city, 300 dpi
  trees/output/canopy_centre.png   5 x 5 km around the centre
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.features import geometry_mask

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import style  # noqa: E402

style.apply()

CHM = os.path.join(HERE, "data", "tree_density_warsaw_10m.tif")
OUT = os.path.join(HERE, "output")
BOUNDS = [10, 25, 40, 55, 70, 85, 101]                      # % of the 10 m pixel under tree crowns
GREENS = ["#dcedc8", "#aed581", "#7cb342", "#4f8f2f", "#2e6b1f", "#16461a"]   # light -> deep, lush
LABELS = ["10–25 %", "25–40", "40–55", "55–70", "70–85", "85–100 %"]
PAPER = "#f7f6f2"
WATER = "#c9d8e6"


def layers():
    d = gpd.read_file(os.path.join(ROOT, "data", "districts.gpkg")).to_crs(2180)
    city = d.unary_union
    w = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/water.gpkg",
                      bbox=tuple(d.to_crs(4326).total_bounds)).to_crs(2180)
    w = w[w.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    w = gpd.GeoDataFrame(geometry=w.geometry.buffer(0).intersection(city), crs=2180)
    w = w[~w.is_empty & (w.area > 1e4)]
    b = gpd.read_file(os.path.join(ROOT, "output", "warsaw_prices_viz.gpkg"), layer="buildings_all",
                      columns=["osm_id"], mask=city)
    return d, city, w, b


def draw(ax, a, extent, d, w, b, lw=0.4):
    cmap = ListedColormap(GREENS)
    norm = BoundaryNorm(BOUNDS, cmap.N)
    ax.set_facecolor(PAPER)
    d.plot(ax=ax, color="white", edgecolor="none", zorder=0)
    w.plot(ax=ax, color=WATER, linewidth=0, zorder=1)
    b.plot(ax=ax, color="#d6d4cc", linewidth=0, zorder=2)
    ax.imshow(np.ma.masked_less(a, BOUNDS[0]), extent=extent, cmap=cmap, norm=norm, interpolation="nearest",
              zorder=3)
    d.boundary.plot(ax=ax, color=style.INK_2, linewidth=lw, zorder=4)
    ax.set_axis_off()
    ax.set_aspect("equal")


def main():
    os.makedirs(OUT, exist_ok=True)
    d, city, w, b = layers()
    with rasterio.open(CHM) as src:
        a = src.read(1)
        tr = src.transform
        outside = geometry_mask([city], out_shape=a.shape, transform=tr)
        a[outside | (a > 100)] = 0
        extent = (tr.c, tr.c + tr.a * a.shape[1], tr.f + tr.e * a.shape[0], tr.f)
    share = a[~outside].mean() / 100
    tall = (a[~outside] >= 70).mean()

    fig = plt.figure(figsize=(10, 12.5), facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.09, 0.96, 0.80])
    draw(ax, a, extent, d, w, b)
    x0, y0, x1, y1 = d.total_bounds
    ax.set_xlim(x0 - 200, x1 + 200)
    ax.set_ylim(y0 - 200, y1 + 200)
    for _, r in d.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, r["district"], fontsize=8, ha="center", va="center", color=style.INK, zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(GREENS, LABELS)], title="Tree cover",
              loc="upper right", fontsize=9, title_fontsize=10, frameon=True, facecolor=PAPER, edgecolor="none")
    fig.text(0.03, 0.975, "Warsaw's trees", fontsize=22, fontweight="bold", color=style.INK, va="top")
    fig.text(0.03, 0.945, f"Tree crowns cover {share:.0%} of Warsaw; {tall:.0%} of the city lies under dense (70 %+) tree cover. "
                          "Every patch of trees is mapped, not just the parks.", fontsize=12, color=style.INK_2, va="top")
    fig.text(0.03, 0.02, "Tree Cover Density 2018: Copernicus Land Monitoring Service (EEA), 10 m, from summer "
                         "Sentinel-2 imagery; shown from 10 %. Buildings, water: OpenStreetMap. Districts: BDOT10k.",
             fontsize=8.5, color=style.MUTED)
    p = os.path.join(OUT, "canopy_warsaw.png")
    fig.savefig(p, dpi=300, facecolor=PAPER)
    plt.close(fig)
    print(f"Saved {p} (canopy {share:.1%}, taller than 15 m {tall:.1%})")

    # centre zoom: 5 x 5 km around the Palace of Culture
    from pyproj import Transformer
    cx, cy = Transformer.from_crs(4326, 2180, always_xy=True).transform(21.0060, 52.2318)
    fig = plt.figure(figsize=(10, 10.6), facecolor=PAPER)
    ax = fig.add_axes([0.02, 0.06, 0.96, 0.86])
    draw(ax, a, extent, d, w, b, lw=0.8)
    ax.set_xlim(cx - 2500, cx + 2500)
    ax.set_ylim(cy - 2500, cy + 2500)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(GREENS, LABELS)], title="Tree cover",
              loc="upper right", fontsize=9, title_fontsize=10, frameon=True, facecolor=PAPER, edgecolor="none")
    fig.text(0.03, 0.975, "Central Warsaw's trees", fontsize=20, fontweight="bold", color=style.INK, va="top")
    fig.text(0.03, 0.945, "5 × 5 km around the Palace of Culture, 10 m detail", fontsize=12, color=style.INK_2,
             va="top")
    fig.text(0.03, 0.02, "Tree Cover Density 2018: Copernicus Land Monitoring Service (EEA), 10 m. Buildings, water: "
                         "OpenStreetMap.",
             fontsize=8.5, color=style.MUTED)
    p = os.path.join(OUT, "canopy_centre.png")
    fig.savefig(p, dpi=300, facecolor=PAPER)
    plt.close(fig)
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
