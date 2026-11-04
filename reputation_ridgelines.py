"""
Cover of the second part: reputation as ridgelines, the sibling of the opening map (price_ridgelines.py). Same grid,
smoothing, taper and dark style; the height is a building's reputation instead of its price: resid_pct from
price_drivers.py (typical-flat price vs the out-of-fold prediction from location, transit, amenities, noise, green and
building; buildings with >= 3 sales). Lines rise where flats sell above prediction and sink below their baseline where
they sell under it. Southern peaks hide the lines behind them; dips are drawn over the lines to the south.

Output: output/summary/42_reputation_ridgelines.png (dark, like 31)
"""
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.ndimage import distance_transform_edt, gaussian_filter
from shapely import contains_xy

import style
from price_ridgelines import BG, CELL, FULL_BUILDINGS, INK, LABELS, MUTED, ROW_M, SIGMA_M, TAPER_M
from summary_charts import FIG, save

PER_PCT = 45                  # metres of line per 1 % above / below prediction
CLIP_PCT = 40                 # single buildings beyond +-40 % are capped before smoothing
LIM = 12                      # colour ramp ends, %
RAMP = LinearSegmentedColormap.from_list("rep", ["#3b6fd6", "#6f9be0", "#a9b8cc", "#d9d2c0", "#e9a27a",
                                                 "#e8604c", "#d8325e"])


def surface(b, x0, y0, nx, ny):
    """Kernel-weighted mean of log(1 + reputation) per cell, and a 0-1 confidence from the number of buildings."""
    pts = b.geometry.representative_point()
    ix = ((pts.x - x0) // CELL).astype(int).clip(0, nx - 1)
    iy = ((pts.y - y0) // CELL).astype(int).clip(0, ny - 1)
    w = np.minimum(b["n"].to_numpy(float), 20) ** 0.5
    r = np.log1p(np.clip(b["resid_pct"].to_numpy(float), -CLIP_PCT, CLIP_PCT) / 100)
    s = np.zeros((ny, nx)); c = np.zeros((ny, nx)); nb = np.zeros((ny, nx))
    np.add.at(s, (iy, ix), w * r)
    np.add.at(c, (iy, ix), w)
    np.add.at(nb, (iy, ix), 1.0)
    sig = SIGMA_M / CELL
    s, c, nb = gaussian_filter(s, sig), gaussian_filter(c, sig), gaussian_filter(nb, sig)
    k = 1 / (2 * np.pi * sig ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(c > 1e-9, 100 * np.expm1(s / c), 0.0)
    t = np.clip(nb / k / FULL_BUILDINGS, 0, 1)
    return p, t * t * (3 - 2 * t)


def main():
    b = gpd.read_file("output/price_drivers.gpkg", layer="building_drivers").dropna(subset=["resid_pct"])
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    city = d.unary_union
    w = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/water.gpkg").to_crs(2180)
    river = w[(w["water"] == "river") & (w.area > 2e5)].buffer(0).unary_union.buffer(40)

    x0, y0, x1, y1 = city.bounds
    nx, ny = int((x1 - x0) // CELL) + 1, int((y1 - y0) // CELL) + 1
    P, conf = surface(b, x0, y0, nx, ny)
    xs = x0 + (np.arange(nx) + 0.5) * CELL
    ys = y0 + (np.arange(ny) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    inside_grid = contains_xy(city, gx, gy) & ~contains_xy(river, gx, gy)
    taper = np.clip(distance_transform_edt(inside_grid) * CELL / TAPER_M, 0, 1)
    taper = taper * taper * (3 - 2 * taper)
    gray = np.array([0.33, 0.33, 0.42, 1.0])

    fig = plt.figure(figsize=FIG, facecolor=BG)
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.80])
    ax.set_facecolor(BG)
    norm = Normalize(-LIM, LIM)
    rows = np.arange(y1 - ROW_M / 2, y0, -ROW_M)
    for z, yr in enumerate(rows):
        iy = int((yr - y0) // CELL)
        inside = inside_grid[iy]
        if not inside.any():
            continue
        p, f = P[iy], conf[iy]
        y = yr + p * PER_PCT * f * taper[iy]
        edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
        for a, e in zip(edges[::2], edges[1::2]):
            if e - a < 3:
                continue
            seg_x, seg_y, seg_p, seg_f = xs[a:e], y[a:e], p[a:e], f[a:e]
            ax.fill_between(seg_x, np.minimum(seg_y, yr) - 1, seg_y, color=BG, lw=0, zorder=2 * z)
            pts = np.c_[seg_x, seg_y].reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), zorder=2 * z + 1,
                                colors=[ff * np.array(RAMP(norm(v))) + (1 - ff) * gray
                                        for v, ff in zip(seg_p[:-1], seg_f[:-1])],
                                linewidths=0.55 + 0.5 * seg_f[:-1], capstyle="round")
            ax.add_collection(lc)
    ax.set_xlim(x0 - 500, x1 + 500)
    ax.set_ylim(y0 - 300, y1 + 2600)
    ax.set_aspect("equal")
    ax.axis("off")

    reps = d.set_index("district").geometry.representative_point()
    for name, (dx, dy) in LABELS.items():
        pt = reps[name]
        ax.text(pt.x + dx, pt.y + dy + 2000, name, color=INK, fontsize=8.5, ha="center", zorder=10_000,
                fontweight="semibold", path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])

    style.byline(fig, y=0.975, color=MUTED, fontsize=10, x=0.95)
    fig.text(0.05, 0.955, "Warsaw, beyond price", color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, "Where flats sell for more than their location and building predict, the lines rise;\n"
                          "where they sell for less, the lines sink. The gap is the address.",
             color=MUTED, fontsize=12.5, va="top", linespacing=1.4)
    cax = fig.add_axes([0.66, 0.075, 0.28, 0.012])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-10, 0, 10])
    cb.set_ticklabels(["−10%", "as predicted", "+10%"])
    cax.tick_params(colors=MUTED, labelsize=9, length=0)
    cax.set_title("vs prediction", color=MUTED, fontsize=9, loc="left")
    fig.text(0.05, 0.022, f"Typical-flat price vs an out-of-sample prediction from location, transit, amenities, noise, green "
                          f"and building, {len(b):,} buildings with ≥ 3 sales,\nsmoothed over ~{SIGMA_M} m; lines fade to "
                          f"gray where few flats sell. Line spacing {ROW_M} m; {PER_PCT} m of height per 1%. "
                          "Source: GUGiK RCN, OSM, BDOT10k.", color=MUTED, fontsize=7.5, va="bottom")
    mpl.rcParams["savefig.facecolor"] = BG
    save(fig, "42_reputation_ridgelines.png")


if __name__ == "__main__":
    main()
