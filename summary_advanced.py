"""
Advanced summary figures (portrait 4:5, same set as summary_charts.py / summary_maps.py):

  10_district_anatomy.png   each district's price gap to the city split into causes (OLS contributions,
                            feature groups) + the unexplained "reputation" part
  11_transit_x_price.png    bivariate hex map: transit departures x typical-flat price (3 x 3 classes)
  12_price_compass.png      polar heatmap around PKiN: price by bearing x distance
  13_shape_of_value.png     gradient-boosting partial dependence + individual (ICE) curves per feature

Reads output/price_drivers.gpkg (features + value_2025 per building). -> output/summary/
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence

import style
from summary_charts import page, save, wrap
from summary_maps import FEATURES, basemap, load, outlines

style.apply()

GROUPS = {   # feature group -> features (fixed order = fixed colour)
    "Centrality & the river": ["centre_km", "river_km"],
    "Metro & rail lines": ["walk_metro_km", "train_km"],
    "Buses & trams": ["log2_trips", "walk_tram_km"],
    "Cafés, shops & schools": ["walk7_food_drink", "walk7_grocery", "walk7_zabka", "walk7_schools",
                               "walk7_health", "walk7_sport_play", "walk_school_km"],
    "Green & quiet": ["green_ha_500", "walk_park_km", "canopy_100", "noise_road_db", "noise_rail_db",
                      "noise_air_db"],
    "The buildings": ["pct_primary", "new_build", "height_m", "log_footprint", "heritage", "age_prewar",
                      "age_1941_60", "age_1991_00", "age_2001_10", "age_2011_20"],
}
REPUTATION = "Unexplained (reputation)"


# 10 -------------------------------------------------------------------------------------------
def district_anatomy(v):
    X = v[FEATURES].to_numpy(float)
    y = np.log(v["value_2025"]).to_numpy()
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    b = pd.Series(beta[1:], index=FEATURES)
    dev = v[FEATURES] - v[FEATURES].mean()
    contrib = pd.DataFrame({g: (dev[f] * b[f]).sum(axis=1) for g, f in GROUPS.items()})
    contrib[REPUTATION] = resid - resid.mean()
    contrib["district"] = v["district"].values
    total = pd.Series(y - y.mean(), index=v.index).groupby(v["district"]).mean()
    c = contrib.groupby("district").mean() * 100            # log points x 100 ~ %
    c = c.loc[total.sort_values().index]
    tot = total.loc[c.index] * 100

    cols = style.CAT[:6] + ["#9a9890"]
    fig = page("Why a district is expensive",
               "Each district's price gap to the city average, split into its causes",
               "Linear regression of log price on all features (no district variable), 12.7K buildings with "
               "≥ 3 sales. A group's bar = its coefficients × how far the district's buildings are from the city "
               "average on those features; 'unexplained' = the district's mean residual. Values are log points "
               "(≈ %), so bars add up to the dot. " + style.SOURCE)
    ax = fig.add_axes([0.24, 0.17, 0.72, 0.70])
    yy = np.arange(len(c))
    left_pos, left_neg = np.zeros(len(c)), np.zeros(len(c))
    for g, colr in zip(c.columns, cols):
        val = c[g].to_numpy()
        start = np.where(val >= 0, left_pos, left_neg)
        ax.barh(yy, val, left=start, height=0.66, color=colr, edgecolor=style.SURFACE, linewidth=1.2, label=g,
                hatch="////" if g == REPUTATION else None)
        left_pos += np.where(val >= 0, val, 0)
        left_neg += np.where(val < 0, val, 0)
    ax.scatter(tot, yy, s=55, color=style.INK, zorder=5, edgecolor=style.SURFACE, lw=1.5,
               label="Total vs city average")
    for yi, t in zip(yy, tot):
        ax.text(max(left_pos[yi], t) + 1.2, yi, f"{t:+.0f}%", va="center", fontsize=9.5, color=style.INK,
                fontweight="semibold")
    ax.axvline(0, color=style.INK_2, lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels(c.index, fontsize=10.5)
    ax.set_xlim(left_neg.min() - 4, left_pos.max() + 9)
    style.pct(ax, "x")
    style.vgrid_only(ax)
    ax.set_xlabel("Contribution to the price gap vs the city average (≈ %)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.36, -0.07), ncol=3, fontsize=8.8, columnspacing=1.0,
              handlelength=1.4)
    save(fig, "10_district_anatomy.png")
    return c


# 11 -------------------------------------------------------------------------------------------
BIV = [["#e8e8e8", "#ace4e4", "#5ac8c8"],     # rows: price low -> high, cols: transit low -> high
       ["#dfb0d6", "#a5add3", "#5698b9"],
       ["#be64ac", "#8c62aa", "#3b4994"]]


def transit_x_price(v, d, w):
    fig = page("Where good transit is cheap",
               wrap("Bus, tram, metro and rail departures within a 5-minute walk, crossed with what a "
                    "typical flat costs", 80),
               "600 m hexagons with ≥ 5 priced buildings; median departures/day (TransitFrequency, 06–22 h, "
               "within a 5-minute walk) and median typical-flat price (2025 PLN/m²), each cut into thirds of the "
               "hexagons. " + style.SOURCE)
    ax = basemap(fig, d, w)
    pts = v.geometry.representative_point()
    kw = dict(gridsize=(48, 42), extent=(*ax.get_xlim(), *ax.get_ylim()), mincnt=5, reduce_C_function=np.median)
    hp = ax.hexbin(pts.x, pts.y, C=v["value_2025"], **kw)
    ht = ax.hexbin(pts.x, pts.y, C=v["trips"], **kw)
    assert np.allclose(hp.get_offsets(), ht.get_offsets())
    price, trips = hp.get_array(), ht.get_array()
    pq = np.digitize(price, np.quantile(price, [1 / 3, 2 / 3]))
    tq = np.digitize(trips, np.quantile(trips, [1 / 3, 2 / 3]))
    hp.set_array(None)
    hp.set_facecolor([BIV[p][t] for p, t in zip(pq, tq)])
    hp.set_edgecolor(style.SURFACE)
    hp.set_linewidth(0.3)
    hp.set_zorder(3)
    ht.remove()
    outlines(ax, d)
    pb = np.quantile(price, [1 / 3, 2 / 3])
    tb = np.quantile(trips, [1 / 3, 2 / 3])

    lg = fig.add_axes([0.07, 0.38, 0.13, 0.104])
    for r in range(3):
        for c in range(3):
            lg.add_patch(Rectangle((c, r), 1, 1, color=BIV[r][c], ec=style.SURFACE, lw=1.5))
    lg.set_xlim(0, 3)
    lg.set_ylim(0, 3)
    lg.set_xticks([1, 2])
    lg.set_xticklabels([f"{tb[0]:.0f}", f"{tb[1]:.0f}"], fontsize=7.5)
    lg.set_yticks([1, 2])
    lg.set_yticklabels([f"{pb[0] / 1000:.1f}k", f"{pb[1] / 1000:.1f}k"], fontsize=7.5)
    lg.set_xlabel("departures/day →", fontsize=8.5)
    lg.set_ylabel("price →", fontsize=8.5)
    lg.grid(False)
    for sp in lg.spines.values():
        sp.set_visible(False)
    lg.set_aspect("equal")

    # bottom: which districts hold the corners
    off = hp.get_offsets()
    hexd = gpd.sjoin(gpd.GeoDataFrame(geometry=gpd.points_from_xy(off[:, 0], off[:, 1]), crs=2180),
                     d[["district", "geometry"]], predicate="within", how="left")["district"].to_numpy()
    corners = [("Well served, cheap", 0, 2, BIV[0][2]), ("Well served, dear", 2, 2, BIV[2][2]),
               ("Poorly served, dear", 2, 0, BIV[2][0]), ("Poorly served, cheap", 0, 0, BIV[0][0])]
    y0 = 0.25
    for i, (name, p, t, colr) in enumerate(corners):
        m = (pq == p) & (tq == t)
        top = pd.Series(hexd[m]).value_counts().head(3)
        x = 0.05 + (i % 2) * 0.47
        yy = y0 - (i // 2) * 0.08
        fig.patches.append(Rectangle((x, yy - 0.012), 0.022, 0.022, transform=fig.transFigure, color=colr,
                                     ec=style.INK_2, lw=0.4))
        fig.text(x + 0.03, yy, f"{name}  ({m.sum()} hexagons)", fontsize=10.5, fontweight="semibold",
                 va="center")
        fig.text(x + 0.03, yy - 0.03, ", ".join(f"{k} {n}" for k, n in top.items()), fontsize=9.5,
                 color=style.INK_2, va="center")
    save(fig, "11_transit_x_price.png")


# 12 -------------------------------------------------------------------------------------------
def price_compass(v, d):
    from pyproj import Transformer
    cx, cy = Transformer.from_crs(4326, 2180, always_xy=True).transform(21.0060, 52.2318)
    pts = v.geometry.representative_point()
    dx, dy = pts.x - cx, pts.y - cy
    r = np.hypot(dx, dy) / 1000
    th = np.mod(np.arctan2(dx, dy), 2 * np.pi)          # bearing, 0 = north, clockwise
    n_sec, r_max = 24, 12
    sec = (th / (2 * np.pi / n_sec)).astype(int)
    ring = r.astype(int)
    df = pd.DataFrame({"sec": sec, "ring": ring, "p": v["value_2025"].to_numpy()})
    df = df[df["ring"] < r_max]
    g = df.groupby(["ring", "sec"])["p"].agg(["median", "size"])
    grid = np.full((r_max, n_sec), np.nan)
    for (ri, si), row in g.iterrows():
        if row["size"] >= 5:
            grid[ri, si] = row["median"]

    fig = page("The price compass",
               wrap("Typical-flat price by direction and distance from the Palace of Culture", 80),
               f"Median value of a typical flat (2025 PLN/m²) per cell: {n_sec} directions × 1 km rings, cells "
               "with ≥ 5 priced buildings. Directions are compass bearings from PKiN. " + style.SOURCE)
    ax = fig.add_axes([0.08, 0.13, 0.84, 0.70], projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    T, R = np.meshgrid(np.linspace(0, 2 * np.pi, n_sec + 1), np.arange(r_max + 1))
    vmin, vmax = np.nanquantile(grid, [0.03, 0.97])
    pc = ax.pcolormesh(T, R, grid, cmap=style.PRICE_CMAP, vmin=vmin, vmax=vmax, edgecolors=style.SURFACE,
                       linewidth=0.6)
    ax.set_ylim(0, r_max)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2 km", "4", "6", "8", "10"], fontsize=8.5, color=style.INK_2)
    ax.set_rlabel_position(100)
    ax.set_xticks(np.radians(np.arange(0, 360, 45)))
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], fontsize=11, fontweight="semibold")
    ax.grid(color=style.GRID, lw=0.6)
    ax.set_facecolor(style.SURFACE)
    # district labels at their buildings' median bearing / distance
    for dist, s in pd.DataFrame({"d": v["district"], "th": th, "r": r}).groupby("d"):
        t0 = np.angle(np.mean(np.exp(1j * s["th"])))
        ax.text(np.mod(t0, 2 * np.pi), min(s["r"].median(), r_max - 0.6), dist, fontsize=7.5, ha="center",
                va="center", color=style.INK, bbox=dict(boxstyle="round,pad=0.1", fc=style.SURFACE, ec="none",
                                                        alpha=0.75))
    cb = fig.colorbar(pc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.06, shrink=0.6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9, length=0)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}k"))
    cb.set_label("PLN/m², typical flat 2025", fontsize=9.5, color=style.INK_2)
    # same distance, different direction: 8 compass sectors, 4-7 km out
    names = np.array(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    lab = names[((np.degrees(th) + 22.5) // 45).astype(int) % 8]
    band = (r >= 4) & (r < 7)
    med = pd.Series(v["value_2025"].to_numpy()[band], index=lab[band]).groupby(level=0).median()
    hi, lo = med.idxmax(), med.idxmin()
    word = {"N": "north", "NE": "north-east", "E": "east", "SE": "south-east", "S": "south", "SW": "south-west",
            "W": "west", "NW": "north-west"}
    fig.text(0.05, 0.885, wrap(f"Same distance, different direction: 4–7 km from the centre, a typical flat to the "
                               f"{word[hi]} costs {style.fmt_int(med[hi])} PLN/m², to the {word[lo]} "
                               f"{style.fmt_int(med[lo])} ({100 * (med[lo] / med[hi] - 1):+.0f}%).", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "12_price_compass.png")


# 13 -------------------------------------------------------------------------------------------
SHAPES = [   # feature, title, x transform for display, x label
    ("centre_km", "Distance from the centre", lambda x: x, "km from PKiN"),
    ("pct_primary", "New building share", lambda x: 100 * x, "% of sales by the developer"),
    ("walk7_food_drink", "Cafés, restaurants, bars", lambda x: 2 ** x - 1, "venues within a 7-minute walk"),
    ("walk_metro_km", "Walk to the metro", lambda x: x, "km on foot"),
    ("green_ha_500", "Green nearby", lambda x: x, "ha of park/forest within 500 m"),
    ("noise_rail_db", "Tram & train noise", lambda x: x, "Lden at the façade, dB (45 = below 55)"),
    ("log2_trips", "Bus & tram departures", lambda x: 2 ** x - 10, "departures/day within 5 min"),
    ("height_m", "Building height", lambda x: x, "m"),
]


def shape_of_value(v):
    X = v[FEATURES]
    y = np.log(v["value_2025"])
    m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, min_samples_leaf=40,
                                      l2_regularization=1.0, random_state=0).fit(X, y)
    fig = page("The shape of value",
               "How price responds to each feature, everything else held as it is",
               "Gradient boosting on all features (12.7K buildings). Thick line: average effect (partial dependence); "
               "thin lines: 120 individual buildings (ICE), all centred on the median building. Ticks: deciles "
               "of the feature. Effects in % of price. " + style.SOURCE)
    gs = fig.add_gridspec(4, 2, left=0.09, right=0.97, top=0.86, bottom=0.12, hspace=0.62, wspace=0.22)
    rng = np.random.default_rng(0)
    sample = rng.choice(len(X), 120, replace=False)
    for k, (f, title, tf, xl) in enumerate(SHAPES):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        lo, hi = np.quantile(X[f], [0.02, 0.98])
        grid = np.linspace(lo, hi, 40) if X[f].nunique() > 40 else np.unique(X[f])
        pd_res = partial_dependence(m, X.iloc[sample], [FEATURES.index(f)], kind="individual",
                                    custom_values={FEATURES.index(f): grid}) \
            if "custom_values" in partial_dependence.__code__.co_varnames else \
            partial_dependence(m, X.iloc[sample], [FEATURES.index(f)], kind="individual", grid_resolution=40)
        xs = pd_res["grid_values"][0] if "grid_values" in pd_res else pd_res["values"][0]
        ice = pd_res["individual"][0]
        med = np.median(X[f])
        ref = np.array([np.interp(med, xs, row) for row in ice])
        ice_pct = 100 * (np.exp(ice - ref[:, None]) - 1)
        avg = 100 * (np.exp(np.mean(ice - ref[:, None], axis=0)) - 1)
        xd = tf(xs)
        for row in ice_pct:
            ax.plot(xd, row, color=style.BLUE, alpha=0.08, lw=0.8)
        ax.plot(xd, avg, color=style.BLUE, lw=2.6)
        ax.axhline(0, color=style.AXIS, lw=0.8)
        for q in np.quantile(X[f], np.linspace(0.1, 0.9, 9)):
            ax.axvline(tf(q), ymin=0, ymax=0.05, color=style.INK_2, lw=0.8)
        ax.set_ylim(np.clip(np.quantile(ice_pct, 0.02) - 2, -40, 0), np.clip(np.quantile(ice_pct, 0.98) + 2, 0, 60))
        style.pct(ax)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5, steps=[1, 2, 5, 10]))
        style.hgrid_only(ax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xl, fontsize=8.5)
        ax.tick_params(labelsize=8.5)
        if f in ("walk7_food_drink", "log2_trips"):
            ax.set_xscale("symlog", linthresh=10)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax.set_xlim(left=max(0, xd.min()))
    save(fig, "13_shape_of_value.png")


def main():
    v, d, w = load()
    district_anatomy(v)
    transit_x_price(v, d, w)
    price_compass(v, d)
    shape_of_value(v)


if __name__ == "__main__":
    main()
