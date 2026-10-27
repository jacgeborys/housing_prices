"""
Opening map: Warsaw's flat prices as ridgelines ("Unknown Pleasures"). Every line is an east-west slice of the city;
it rises with the price of a typical flat in 2025 (hedonic value_2025; buildings with 1-2 sales use their own unshrunk value, obvious outliers dropped), smoothed
with a Gaussian kernel. Lines are drawn north to south, each filled with the background so southern peaks hide the
lines behind them. Outside the city the line stops; the Vistula breaks it; where no flats sell it stays flat.

Output: output/summary/31_price_ridgelines.png (1200 x 1500, dark)
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

from summary_charts import DPI, FIG, save

MIN_SALES = 3                 # buildings with at least this many sales keep their model value (shrunk toward the district)
OUTLIER_LOG = 0.5             # a 1-2 sale building is dropped if its value is > 65 % above / 40 % below the median of its
                              # 8 nearest well-sold buildings within 1 km (ln 1.65 = 0.5): obvious outliers only
CELL = 50                     # m, sampling grid
SIGMA_M = 350                 # Gaussian smoothing of the price surface
FULL_BUILDINGS = 4            # priced buildings (kernel-weighted) for a peak to reach full height; fewer: it fades,
                              # so one building (e.g. Planetowa 33, Wawer: 6 sales, value 25K) cannot make a mountain
TAPER_M = 500                 # heights ease to zero over this distance from the city edge and the Vistula
ROW_M = 250                   # spacing of the lines
BASE, PER_1K = 12000, 230     # height: metres of line per 1,000 PLN/m² above BASE
CMAP_LO, CMAP_HI = 11000, 26000
BG, INK, MUTED = "#0d0e17", "#f1ede4", "#8c8a99"
RAMP = LinearSegmentedColormap.from_list("ridge", ["#3d4a7a", "#5b7fb8", "#8fb7c9", "#e8d9a8", "#f2a65a",
                                                   "#e8604c", "#c2306b"])
# kept on purpose (2026-10-27, after 20 alternatives incl. plasma and blue-red-gold): reads like a physical map; the
# top end (red-pink) is less distinct for red-green colour-blind viewers
LABELS = {"Śródmieście": (0, 1500), "Żoliborz": (-600, 900), "Mokotów": (300, 500), "Wilanów": (1600, -300),
          "Praga-Północ": (900, -200), "Białołęka": (0, 0), "Ursynów": (-500, 0), "Bemowo": (-900, 0),
          "Wawer": (0, 0), "Ursus": (0, 0), "Targówek": (500, 0)}


def priced_buildings():
    """Every building with a recorded flat sale. >= MIN_SALES sales: model value (value_2025). 1-2 sales: the unshrunk
    value (the standardised price of those one or two flats; shrinking would pull a lone sale to its district's
    average), minus obvious outliers against well-sold buildings nearby. Column `value` is what to draw, `few` marks
    the 1-2 sale buildings (added 2026-10-27; before, only >= 3 sales were drawn)."""
    from scipy.spatial import cKDTree
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value").drop_duplicates("osm_id").reset_index(drop=True)
    bv["few"] = bv["n"] < MIN_SALES
    bv["value"] = np.where(bv["few"], bv["value_2025_raw"], bv["value_2025"])
    pts = bv.geometry.representative_point()
    xy = np.c_[pts.x, pts.y]
    core = ~bv["few"].to_numpy()
    tree = cKDTree(xy[core])
    lv = np.log(bv.loc[core, "value_2025"].to_numpy())
    _, idx = tree.query(xy[~core], k=8, distance_upper_bound=1000)
    near = np.where(idx < core.sum(), lv[np.minimum(idx, core.sum() - 1)], np.nan)
    with np.errstate(all="ignore"):
        ref = np.nanmedian(near, axis=1)                                # NaN: nothing well-sold within 1 km, keep
    dev = np.log(bv.loc[~core, "value"].to_numpy()) - ref
    drop = np.zeros(len(bv), bool)
    drop[np.flatnonzero(~core)] = np.abs(dev) > OUTLIER_LOG
    drop |= bv["value"].isna().to_numpy()
    print(f"priced buildings: {core.sum():,} with >= {MIN_SALES} sales, {int((~core).sum() - drop[~core].sum()):,} with "
          f"1-2 sales kept, {int(drop.sum()):,} outliers dropped")
    return bv[~drop].reset_index(drop=True)


def surface(bv, x0, y0, nx, ny):
    pts = bv.geometry.representative_point()
    ix = ((pts.x - x0) // CELL).astype(int).clip(0, nx - 1)
    iy = ((pts.y - y0) // CELL).astype(int).clip(0, ny - 1)
    w = np.minimum(bv["n"].to_numpy(float), 20) ** 0.5      # busy buildings count more, but not overwhelmingly
    s = np.zeros((ny, nx)); c = np.zeros((ny, nx)); nb = np.zeros((ny, nx))
    np.add.at(s, (iy, ix), w * np.log(bv["value"].to_numpy()))
    np.add.at(c, (iy, ix), w)
    np.add.at(nb, (iy, ix), 1.0)
    sig = SIGMA_M / CELL
    s, c, nb = gaussian_filter(s, sig), gaussian_filter(c, sig), gaussian_filter(nb, sig)
    k = 1 / (2 * np.pi * sig ** 2)                            # peak of the kernel: one building's weight
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.exp(s / c)
    t = np.clip(nb / k / FULL_BUILDINGS, 0, 1)
    return np.where(c > 1e-9, p, BASE), t * t * (3 - 2 * t)  # smoothstep: no cliffs where the data thins out


def main():
    bv = priced_buildings()
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    city = d.unary_union
    w = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/water.gpkg").to_crs(2180)
    river = w[(w["water"] == "river") & (w.area > 2e5)].buffer(0).unary_union.buffer(40)

    x0, y0, x1, y1 = city.bounds
    nx, ny = int((x1 - x0) // CELL) + 1, int((y1 - y0) // CELL) + 1
    P, conf = surface(bv, x0, y0, nx, ny)
    xs = x0 + (np.arange(nx) + 0.5) * CELL
    ys = y0 + (np.arange(ny) + 0.5) * CELL
    gx, gy = np.meshgrid(xs, ys)
    inside_grid = contains_xy(city, gx, gy) & ~contains_xy(river, gx, gy)
    taper = np.clip(distance_transform_edt(inside_grid) * CELL / TAPER_M, 0, 1)
    taper = taper * taper * (3 - 2 * taper)
    grey = np.array([0.33, 0.33, 0.42, 1.0])

    fig = plt.figure(figsize=FIG, facecolor=BG)
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.80])
    ax.set_facecolor(BG)
    norm = Normalize(CMAP_LO, CMAP_HI)
    rows = np.arange(y1 - ROW_M / 2, y0, -ROW_M)               # north first: later (southern) lines cover them
    for z, yr in enumerate(rows):
        iy = int((yr - y0) // CELL)
        inside = inside_grid[iy]
        if not inside.any():
            continue
        p, f = P[iy], conf[iy]
        h = np.maximum(p - BASE, 0) / 1000 * PER_1K * f * taper[iy]
        y = yr + h
        # split into runs inside the city (the river and the boundary break the line)
        edges = np.flatnonzero(np.diff(np.r_[0, inside.astype(int), 0]))
        for a, b in zip(edges[::2], edges[1::2]):
            if b - a < 3:
                continue
            seg_x, seg_y, seg_p, seg_f = xs[a:b], y[a:b], p[a:b], f[a:b]
            ax.fill_between(seg_x, yr - 1, seg_y, color=BG, lw=0, zorder=2 * z)
            pts = np.c_[seg_x, seg_y].reshape(-1, 1, 2)
            lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), zorder=2 * z + 1,
                                colors=[ff * np.array(RAMP(norm(v))) + (1 - ff) * grey
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

    fig.text(0.05, 0.955, "Warsaw, in price", color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, "What a typical flat costs per m², 2025. Each line is a slice of the city;\n"
                          "it rises where flats cost more.", color=MUTED, fontsize=12.5, va="top", linespacing=1.4)
    # legend: colour ramp + height of 10,000 PLN/m²
    cax = fig.add_axes([0.66, 0.075, 0.28, 0.012])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=RAMP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([12000, 16000, 20000, 24000])
    cb.set_ticklabels(["12K", "16K", "20K", "24K+"])
    cax.tick_params(colors=MUTED, labelsize=9, length=0)
    cax.set_title("PLN per m²", color=MUTED, fontsize=9, loc="left")
    fig.text(0.05, 0.022, "Hedonic value of a typical 2-room resale flat, 2025, per building (1-2 sale buildings "
                          f"included, obvious outliers removed),\n"
                          f"smoothed over ~{SIGMA_M} m; lines fade to grey where few flats sell.\nLine spacing {ROW_M} m; peak height grows "
                          f"{PER_1K} m per 1,000 PLN/m². Source: GUGiK Rejestr Cen Nieruchomości (RCN) 2006–2026, "
                          "OSM, BDOT10k.", color=MUTED, fontsize=7.5, va="bottom")
    mpl.rcParams["savefig.facecolor"] = BG      # summary_charts.save() would use the light page colour
    save(fig, "31_price_ridgelines.png")


if __name__ == "__main__":
    main()
