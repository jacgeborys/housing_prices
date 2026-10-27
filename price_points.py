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
from matplotlib.colors import LinearSegmentedColormap, Normalize

from house_cells import M2 as HOUSE_M2, year_index
from price_ridgelines import BG, CMAP_HI, CMAP_LO, INK, LABELS, MUTED, priced_buildings
from summary_charts import FIG, OUT

DPI = 300
RAMP_LO = 8000             # colour scale starts below the ridgelines' 11K so house prices show (2026-10-27)
OTHER = "#33354b"         # other buildings, a step darker (2026-10-27; was #3a3d55)
# blue - coral - gold (chosen 2026-10-27 from five variants): brightness rises steadily with price (L* 35 -> 91), so it
# reads for red-green colour-blind viewers too; plasma felt too aggressive
RAMP = LinearSegmentedColormap.from_list("calm", ["#34508f", "#5a6fb0", "#a9789d", "#e3876f", "#f2b45a", "#ffe37a"])


def main():
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    city = d.unary_union
    x0, y0, x1, y1 = city.bounds
    bv = priced_buildings().sort_values("value")                    # dearest drawn last, on top
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
    norm = Normalize(RAMP_LO, CMAP_HI)
    ax.scatter(pts.x, pts.y, s=0.25 + 0.55 * np.sqrt(np.minimum(bv["n"], 200)), c=bv["value"], cmap=RAMP,
               norm=norm, lw=0, zorder=3, rasterized=True)
    # houses (2026-10-27): one diamond per sale, price per m2 of living space (plot included) in 2025 prices; the ramp
    # starts at 8K (not the ridgelines' 11K) so house prices (median ~10K) are told apart
    hs = gpd.read_file("output/houses.gpkg")
    hs = hs[hs["price_m2"].between(*HOUSE_M2) & (hs["year"] >= 2017)].copy()
    hs["adj"] = hs["price_m2"] * np.exp(-hs["year"].astype(int).map(year_index(hs)))
    hs = hs.sort_values("adj")
    ax.scatter(hs.geometry.x, hs.geometry.y, s=3.4, c=hs["adj"], cmap=RAMP, norm=norm, marker="D", lw=0,
               zorder=3.5, rasterized=True)
    ax.set_xlim(x0 - 500, x1 + 500)
    ax.set_ylim(y0 - 300, y1 + 2600)
    ax.set_aspect("equal")
    ax.axis("off")
    reps = d.set_index("district").geometry.representative_point()
    for name in LABELS:                                              # no ridge offsets: at the district itself
        pt = reps[name]
        ax.text(pt.x, pt.y, name, color=INK, fontsize=8.5, ha="center", zorder=10, fontweight="semibold",
                path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    fig.text(0.05, 0.955, "Where flats and houses sold", color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, f"Dots: {len(bv):,} buildings with {int(bv['n'].sum()):,} flat sales 2006–2026 (bigger = more sales).\n"
                          f"Diamonds: {len(hs):,} house sales 2017–2026. Colour: PLN per m² in 2025. Dark grey: other buildings.",
             color=MUTED, fontsize=12.5, va="top", linespacing=1.4)
    cax = fig.add_axes([0.66, 0.075, 0.28, 0.012])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([10000, 14000, 18000, 22000, 26000])
    cb.set_ticklabels(["10K", "14K", "18K", "22K", "26K+"])
    cax.tick_params(colors=MUTED, labelsize=9, length=0)
    cax.set_title("PLN per m²", color=MUTED, fontsize=9, loc="left")
    fig.text(0.05, 0.022, "Flats: a typical 2-room resale flat in 2025 per building. Houses: price per m² of living space, plot "
                          "included, in 2025 prices.\nCooperative-right flats are not in the register. "
                          "Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k.",
             color=MUTED, fontsize=7.5, va="bottom")
    path = f"{OUT}/32_price_points.png"
    fig.savefig(path, dpi=DPI, facecolor=BG)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
