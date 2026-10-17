"""
Companion to the opening map (price_ridgelines.py): the same frame and colours, but every building with recorded
flat sales as a dot (colour = 2025 value of a typical flat, size = number of sales), over all other buildings in
dark grey. Shows where the transactions behind the ridgelines actually are.

Output: output/summary/32_price_points.png (2400 x 3000, dark: zoomable)
"""
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from price_ridgelines import BG, CMAP_HI, CMAP_LO, INK, LABELS, MIN_SALES, MUTED, RAMP
from summary_charts import FIG, OUT

DPI = 300
OTHER = "#3a3d55"


def main():
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    city = d.unary_union
    x0, y0, x1, y1 = city.bounds
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value")
    bv = bv[bv["n"] >= MIN_SALES].sort_values("value_2025")         # dearest drawn last, on top
    pts = bv.geometry.representative_point()
    allb = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all", columns=["osm_id"])
    ab = gpd.GeoDataFrame(geometry=allb.geometry.representative_point(), crs=allb.crs)
    ab = gpd.sjoin(ab, d[["geometry"]], predicate="within").geometry       # city only
    w = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/water.gpkg").to_crs(2180)
    river = w[(w["water"] == "river") & (w.area > 2e5)].buffer(0)

    fig = plt.figure(figsize=FIG, facecolor=BG)
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.80])                       # same frame as the ridgeline map
    ax.set_facecolor(BG)
    river.plot(ax=ax, color="#1a2033", lw=0, zorder=0)
    d.boundary.plot(ax=ax, color="#3a3c50", lw=0.5, zorder=1)
    ax.scatter(ab.x, ab.y, s=0.15, color=OTHER, lw=0, zorder=2, rasterized=True)
    norm = Normalize(CMAP_LO, CMAP_HI)
    ax.scatter(pts.x, pts.y, s=0.25 + 0.55 * np.sqrt(np.minimum(bv["n"], 200)), c=bv["value_2025"], cmap=RAMP,
               norm=norm, lw=0, zorder=3, rasterized=True)
    ax.set_xlim(x0 - 500, x1 + 500)
    ax.set_ylim(y0 - 300, y1 + 2600)
    ax.set_aspect("equal")
    ax.axis("off")
    reps = d.set_index("district").geometry.representative_point()
    for name in LABELS:                                              # no ridge offsets: at the district itself
        pt = reps[name]
        ax.text(pt.x, pt.y, name, color=INK, fontsize=8.5, ha="center", zorder=10, fontweight="semibold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    fig.text(0.05, 0.955, "Where the flats sold", color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, f"{len(bv):,} buildings with {int(bv['n'].sum()):,} recorded flat sales, 2006–2026. "
                          "Colour: what a\ntypical flat there costs per m² in 2025; bigger dot = more sales. "
                          "Dark grey: other buildings.", color=MUTED, fontsize=12.5, va="top", linespacing=1.4)
    cax = fig.add_axes([0.66, 0.075, 0.28, 0.012])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([12000, 16000, 20000, 24000])
    cb.set_ticklabels(["12K", "16K", "20K", "24K+"])
    cax.tick_params(colors=MUTED, labelsize=9, length=0)
    cax.set_title("PLN per m²", color=MUTED, fontsize=9, loc="left")
    fig.text(0.05, 0.022, f"Hedonic value of a typical 2-room resale flat, 2025, per building (≥ {MIN_SALES} sales; "
                          "date, floor and size removed). Cooperative-right flats are not in the register (fig. 23).\n"
                          "Source: GUGiK Rejestr Cen Nieruchomości (RCN) 2006–2026, OSM, BDOT10k.",
             color=MUTED, fontsize=7.5, va="bottom")
    path = f"{OUT}/32_price_points.png"
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
