"""
Gdańsk versions of report slides 01 (price ridgelines) and 02 (every priced building as a dot), same design as
Warsaw's price_ridgelines.py / price_points.py (imported for constants and the smoothing, not edited).
Run from the gdansk/ folder after build_prices.py:  python maps.py

Output: gdansk/output/summary/01_price_ridgelines.png, 02_price_points.png (2400 x 3000, dark)
Flats only (the house sales of Warsaw's slide 02 are not built for Gdańsk yet). Colour scale and line heights use
the Gdańsk range (9-20K PLN/m²), not Warsaw's (12-26K): the footnote says so.
"""
import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geopandas as gpd  # noqa: E402
import matplotlib as mpl  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from scipy.ndimage import distance_transform_edt  # noqa: E402
from shapely import contains_xy  # noqa: E402

import price_points as pp  # noqa: E402
import price_ridgelines as pr  # noqa: E402
from summary_charts import DPI, FIG  # noqa: E402

BUILDINGS = "data/buildings.gpkg"
pr.BASE, pr.FULL_BUILDINGS, pr.SIGMA_M = 9000, 3, 300
LO, HI = 9000, 20000
ROW_M, PER_1K = 170, 200
BG, INK, MUTED = pr.BG, pr.INK, pr.MUTED
OUT = "output/summary"
LABELS = {   # district -> (label, dx, dy) metres
    "Śródmieście": ("Śródmieście", 0, 0), "Wrzeszcz Górny": ("Wrzeszcz", 0, 0), "Oliwa": ("Oliwa", -800, 0),
    "Przymorze Wielkie": ("Przymorze", 0, 0), "Orunia - Święty Wojciech - Lipce": ("Orunia", 0, -600),
    "Chełm": ("Chełm", 0, 0), "Osowa": ("Osowa", 0, 0), "Kokoszki": ("Kokoszki", 0, 0), "Jasień": ("Jasień", 0, 0),
    "Brzeźno": ("Brzeźno", 0, 0), "Stogi": ("Stogi", 0, 0), "Ujeścisko - Łostowice": ("Łostowice", 0, 0),
    "Matarnia": ("Matarnia", 0, 0),
    "Żabianka-Wejhera-Jelitkowo-Tysiąclecia": ("Jelitkowo", 0, 0),
}
TICKS = [(10000, "10K"), (13000, "13K"), (16000, "16K"), (19000, "19K+")]


def frame(d):
    city = d.unary_union
    fig = plt.figure(figsize=FIG, facecolor=BG)
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.80])
    ax.set_facecolor(BG)
    return city, city.bounds, fig, ax


def finish(fig, ax, bounds, title, sub, foot, norm, cmap, name, shift):
    x0, y0, x1, y1 = bounds
    ax.set_xlim(x0 - 500, x1 + 500)
    mid = (y0 + y1) / 2
    half = (x1 - x0 + 1000) / 2 * (0.80 * FIG[1]) / (0.94 * FIG[0])        # half the frame height in metres
    ax.set_ylim(mid - half + shift, mid + half + shift)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.text(0.05, 0.955, title, color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, sub, color=MUTED, fontsize=12.5, va="top", linespacing=1.4)
    cax = fig.add_axes([0.66, 0.075, 0.28, 0.012])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([t for t, _ in TICKS])
    cb.set_ticklabels([lab for _, lab in TICKS])
    cax.tick_params(colors=MUTED, labelsize=9, length=0)
    cax.set_title("PLN per m²", color=MUTED, fontsize=9, loc="left")
    fig.text(0.05, 0.022, foot, color=MUTED, fontsize=7.5, va="bottom")
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}", dpi=DPI, facecolor=BG)
    plt.close(fig)
    print("saved", f"{OUT}/{name}")


def label(ax, d, offset_y=0):
    reps = d.set_index("district").geometry.representative_point()
    for key, (txt, dx, dy) in LABELS.items():
        if key in reps.index:
            pt = reps[key]
            ax.text(pt.x + dx, pt.y + dy + offset_y, txt, color=INK, fontsize=8.5, ha="center", zorder=10_000,
                    fontweight="semibold", path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])


def ridgelines(bv, d):
    city, (x0, y0, x1, y1), fig, ax = frame(d)
    cell, taper_m = pr.CELL, pr.TAPER_M
    nx, ny = int((x1 - x0) // cell) + 1, int((y1 - y0) // cell) + 1
    P, conf = pr.surface(bv, x0, y0, nx, ny)
    xs = x0 + (np.arange(nx) + 0.5) * cell
    ys = y0 + (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    inside_grid = contains_xy(city, gx, gy)
    taper = np.clip(distance_transform_edt(inside_grid) * cell / taper_m, 0, 1)
    taper = taper * taper * (3 - 2 * taper)
    grey = np.array([0.33, 0.33, 0.42, 1.0])
    norm = Normalize(LO, HI)
    for z, yr in enumerate(np.arange(y1 - ROW_M / 2, y0, -ROW_M)):
        iy = int((yr - y0) // cell)
        inside = inside_grid[iy]
        if not inside.any():
            continue
        p, f = P[iy], conf[iy]
        y = yr + np.maximum(p - pr.BASE, 0) / 1000 * PER_1K * f * taper[iy]
        edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < 3:
                continue
            sx, sy, sp, sf = xs[a:b], y[a:b], p[a:b], f[a:b]
            ax.fill_between(sx, yr - 1, sy, color=BG, lw=0, zorder=2 * z)
            pts = np.c_[sx, sy].reshape(-1, 1, 2)
            ax.add_collection(LineCollection(
                np.concatenate([pts[:-1], pts[1:]], axis=1), zorder=2 * z + 1,
                colors=[ff * np.array(pr.RAMP(norm(v))) + (1 - ff) * grey for v, ff in zip(sp[:-1], sf[:-1])],
                linewidths=0.55 + 0.5 * sf[:-1], capstyle="round"))
    label(ax, d, offset_y=1500)
    mpl.rcParams["savefig.facecolor"] = BG
    finish(fig, ax, (x0, y0, x1, y1), "Gdańsk, in price",
           "What a typical flat costs per m², 2025. Each line is a slice of the city;\nit rises where flats cost more.",
           "Hedonic value of a typical 2-room resale flat, 2025, per building, smoothed over ~300 m; lines fade to grey where few flats sell.\n"
           "Scale is the Gdańsk range (Warsaw: 12-26K); Wyspa Sobieszewska (no sales) left off. Source: GUGiK RCN 2010–2026, BDOT10k.",
           norm, pr.RAMP, "01_price_ridgelines.png", shift=1300)


def points(bv, d, bld):
    city, (x0, y0, x1, y1), fig, ax = frame(d)
    bv = bv.sort_values("value")
    pts = bv.geometry.representative_point()
    ab = gpd.GeoDataFrame(geometry=bld.geometry.representative_point(), crs=bld.crs)
    ab = gpd.sjoin(ab, d[["geometry"]], predicate="within").geometry
    w = gpd.read_file("D:/QGIS/osm_basemap/gdansk/water.gpkg").to_crs(2180)
    w = w[w["water"].isin(["river", "canal", "lake"]) & (w.area > 5e4)]
    w.buffer(0).clip(city.buffer(1000)).plot(ax=ax, color="#1a2033", lw=0, zorder=0)
    d.boundary.plot(ax=ax, color="#3a3c50", lw=0.5, zorder=1)
    ax.scatter(ab.x, ab.y, s=0.15, color=pp.OTHER, lw=0, zorder=2, rasterized=True)
    norm = Normalize(LO, HI)
    ax.scatter(pts.x, pts.y, s=0.25 + 0.55 * np.sqrt(np.minimum(bv["n"], 200)), c=bv["value"], cmap=pp.RAMP, norm=norm,
               lw=0, zorder=3, rasterized=True)
    label(ax, d)
    mpl.rcParams["savefig.facecolor"] = BG
    finish(fig, ax, (x0, y0, x1, y1), "Where flats sold",
           f"Flats: {len(bv):,} buildings with {int(bv['n'].sum()):,} flat sales 2010–2026 (bigger = more sales).\n"
           "Colour: PLN per m², 2025. Dark grey: other buildings.",
           "A typical 2-room resale flat in 2025 per building. Cooperative-right flats are not in the register. Scale is the Gdańsk range (Warsaw: 12-26K).\n"
           "Wyspa Sobieszewska (no sales) left off. Source: GUGiK Rejestr Cen Nieruchomości (RCN), BDOT10k.",
           norm, pp.RAMP, "02_price_points.png", shift=0)


def main():
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    d = d[d["district"] != "Wyspa Sobieszewska"]        # the island: no flat sales, and it halves the scale of the map
    bv = pr.priced_buildings()
    bld = gpd.read_file(BUILDINGS)
    ridgelines(bv, d)
    points(bv, d, bld)


if __name__ == "__main__":
    main()
