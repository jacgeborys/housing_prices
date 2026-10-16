"""
Summary maps: four portrait (4:5) figures that add the location data to the summary set
(summary_charts.py makes 01-05). Each has a city map on top and, below, what the feature does to the
price of a typical flat compared with other buildings in the same district.

  06_cheaper_than_expected.png  model residuals (price_drivers.py, spatial CV) on a hex map
  07_metro_walk.png             walking minutes to the nearest metro station (walk_access.py)
  08_cafes.png                  cafés, restaurants and bars (fetch_pois.py)
  09_noise.png                  EU strategic noise map, road + rail Lden (fetch_noise.py)

-> output/summary/
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from rasterio.warp import Resampling, calculate_default_transform, reproject

import style
from summary_charts import DPI, FIG, page, save, wrap

style.apply()

WALK_M_PER_MIN = 80                     # 4.8 km/h
OSM = "D:/QGIS/mapy_warszawy_misc/data/osm"
MAP_BOX = [0.03, 0.35, 0.94, 0.49]      # map axes (figure fractions)
BAR_BOX = [0.14, 0.09, 0.80, 0.19]      # within-district bar panel
GREY = "#e7e6e1"


def load():
    v = gpd.read_file("output/price_drivers.gpkg")
    v["rel"] = np.log(v["value_2025"]) - v.groupby("district")["value_2025"].transform(lambda s: np.log(s).mean())
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    city = d.unary_union
    w = gpd.read_file(f"{OSM}/water.gpkg", bbox=tuple(d.to_crs(4326).total_bounds)).to_crs(2180)
    w = w[w.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    w = gpd.GeoDataFrame(geometry=w.geometry.buffer(0).intersection(city.buffer(500)), crs=2180)
    w = w[~w.is_empty & (w.area > 2e4)]
    return v, d, w


def basemap(fig, d, w, buildings=None):
    ax = fig.add_axes(MAP_BOX)
    d.plot(ax=ax, color=style.SURFACE, edgecolor="none")
    if buildings is not None:
        buildings.plot(ax=ax, color=GREY, linewidth=0)
    w.plot(ax=ax, color="#cfdcec", linewidth=0, zorder=2)
    ax.set_axis_off()
    ax.set_aspect("equal")
    x0, y0, x1, y1 = d.total_bounds
    ax.set_xlim(x0 - 300, x1 + 300)
    ax.set_ylim(y0 - 300, y1 + 300)
    return ax


def outlines(ax, d, labels=True):
    d.boundary.plot(ax=ax, color=style.INK_2, linewidth=0.5, zorder=5)
    if labels:
        for _, r in d.iterrows():
            p = r.geometry.representative_point()
            ax.text(p.x, p.y, r["district"], fontsize=6.5, color=style.INK, ha="center", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.12", fc=style.SURFACE, ec="none", alpha=0.7))


FEATURES = ["log2_trips", "walk_metro_km", "train_km", "walk_tram_km", "walk_park_km", "walk_school_km",
            "noise_road_db", "noise_rail_db", "noise_air_db", "heritage", "centre_km", "river_km", "green_ha_500",
            "height_m", "pct_primary", "new_build", "log_footprint", "walk7_food_drink", "walk7_grocery",
            "walk7_zabka", "walk7_schools", "walk7_health", "walk7_sport_play", "canopy_100",
            "age_prewar", "age_1941_60", "age_1991_00", "age_2001_10", "age_2011_20"]


def adjusted(v, focus):
    """log price net of the district and of every feature except `focus` (OLS residual): what is left
    varies with the focus feature 'other things equal'."""
    X = pd.get_dummies(v["district"]).astype(float)
    X = pd.concat([X, v[[f for f in FEATURES if f != focus]]], axis=1).to_numpy(float)
    y = np.log(v["value_2025"]).to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def within_district_bars(fig, v, col, bins, labels, title, xlabel, min_n=60, focus=None):
    ax = fig.add_axes(BAR_BOX)
    b = pd.cut(v[col], bins, labels=labels)
    rel = adjusted(v, focus or col)
    rel = rel - rel.mean()
    g = pd.DataFrame({"rel": rel, "b": b}).groupby("b", observed=False)["rel"].agg(["mean", "size"])
    g = g[g["size"] >= min_n]
    pct = 100 * (np.exp(g["mean"]) - 1)
    x = np.arange(len(g))
    ax.bar(x, pct, width=0.62, color=[style.BLUE if p >= 0 else style.ORANGE for p in pct])
    for xi, p in zip(x, pct):
        ax.text(xi, p + (0.4 if p >= 0 else -0.4), f"{p:+.1f}%", ha="center",
                va="bottom" if p >= 0 else "top", fontsize=9.5, color=style.INK)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(g.index, fontsize=9.5)
    lo, hi = min(pct.min(), 0), max(pct.max(), 0)
    pad = 0.35 * max(hi - lo, 2)
    ax.set_ylim(lo - pad, hi + pad)
    style.pct(ax)
    ax.yaxis.set_major_locator(plt.MultipleLocator(2))
    style.hgrid_only(ax)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    return ax


def legend_patches(ax, colours, labels, title, loc="lower left"):
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colours, labels)], title=title, loc=loc,
              fontsize=8.5, title_fontsize=9, frameon=True, framealpha=0.9, edgecolor="none")


# 6 --------------------------------------------------------------------------------------------
def cheaper_than_expected(v, d, w):
    fig = page("Reputation has a price",
               wrap("What flats sell for vs what the model predicts from location, transit, amenities, noise "
                    "and the building. The model never sees district names", 80),
               "Median residual per 600 m hexagon (≥ 5 buildings): actual value of a typical flat vs the "
               "out-of-sample prediction (gradient boosting, spatial 5-fold CV on 2 km blocks; the model "
               "never saw the block it predicts; no district variable). Residuals capture what the data can't see: "
               "reputation, condition, finish, views. A blue hexagon is not a bargain by itself. " + style.SOURCE)
    ax = basemap(fig, d, w)
    pts = v.geometry.representative_point()
    norm = TwoSlopeNorm(vmin=-15, vcenter=0, vmax=15)
    hb = ax.hexbin(pts.x, pts.y, C=v["resid_pct"].clip(-40, 40), gridsize=(48, 42), reduce_C_function=np.median,
                   mincnt=5, cmap=style.DIV_CMAP, norm=norm, linewidths=0.3, edgecolors=style.SURFACE, zorder=3)
    outlines(ax, d)
    cax = fig.add_axes([0.08, 0.40, 0.025, 0.14])
    cb = fig.colorbar(hb, cax=cax, ticks=[-15, -10, -5, 0, 5, 10, 15])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0)
    cb.ax.set_yticklabels(["−15%", "−10%", "−5%", "as\nexpected", "+5%", "+10%", "+15%"])
    cb.set_label("vs prediction", fontsize=9, color=style.INK_2)

    # bottom: residual by district (median), as bars
    ax2 = fig.add_axes([0.14, 0.15, 0.80, 0.16])
    g = v.groupby("district")["resid_pct"].median().sort_values()
    x = np.arange(len(g))
    ax2.bar(x, g.values, width=0.65, color=[style.DIV_CMAP(norm(val)) for val in g.values],
            edgecolor=style.INK_2, linewidth=0.3)
    ax2.axhline(0, color=style.AXIS, lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(g.index, rotation=55, ha="right", fontsize=8.5)
    style.pct(ax2)
    style.hgrid_only(ax2)
    ax2.set_title("Median building vs prediction, by district: the part no feature explains", fontsize=11)
    for xi, val in zip(x, g.values):
        if abs(val) >= 4:
            ax2.text(xi, val + (0.8 if val > 0 else -0.8), f"{val:+.0f}%", ha="center",
                     va="bottom" if val > 0 else "top", fontsize=9, color=style.INK)
    ax2.set_ylim(g.min() - 6, g.max() + 6)
    fig.text(0.03, 0.885, wrap(f"Same location, transit, cafés, green and noise: a flat in Żoliborz still sells "
                               f"{g.get('Żoliborz', 0):+.0f}% above the prediction, one in Praga-Północ "
                               f"{g.get('Praga-Północ', 0):+.0f}% below.", 95),
             fontsize=10.5, color=style.INK, fontweight="semibold", va="top")
    save(fig, "06_cheaper_than_expected.png")


# 7 --------------------------------------------------------------------------------------------
def metro_walk(v, d, w, ctx):
    mins = v["walk_metro_km"] * 1000 / WALK_M_PER_MIN
    bins = [0, 5, 10, 15, 20, 30, 1e9]
    labels = ["≤ 5 min", "5–10", "10–15", "15–20", "20–30", "30+"]
    cols = ["#0b2f63", "#1d58a8", "#3987e5", "#7fb1ef", "#b9d4f6", "#dfe9f6"]   # near = dark
    fig = page("Every kilometre to the metro costs about 2 %",
               "Walking time from each building to the nearest metro station, along real footpaths",
               "Walking distances on the OSM footpath and street network (TransitFrequency's Warsaw walking "
               "graph, multi-source Dijkstra), 4.8 km/h. 38 metro stations. Coloured: buildings with ≥ 3 flat "
               "sales. Bars: typical-flat price net of the district and every other feature (OLS residual), "
               "relative to the average building. Regression: −2.1 % per km walked. " + style.SOURCE)
    ax = basemap(fig, d, w, ctx)
    vv = v.assign(m=mins)
    for (lo, hi), c in zip(zip(bins[:-1], bins[1:]), cols):
        sub = vv[(vv["m"] > lo) & (vv["m"] <= hi)]
        sub.buffer(25).plot(ax=ax, color=c, linewidth=0, zorder=3)
    st = gpd.read_file(f"{OSM}/train_stations.gpkg").to_crs(2180)
    metro = st[st["station"] == "subway"]
    ax.scatter(metro.geometry.x, metro.geometry.y, s=26, marker="o", color="white", edgecolor=style.INK,
               lw=1.2, zorder=6)
    outlines(ax, d, labels=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labels)]
              + [Line2D([], [], marker="o", ls="", mfc="white", mec=style.INK, label="metro station")],
              title="Walk to metro", loc="lower left", fontsize=8.5, title_fontsize=9, frameon=True,
              framealpha=0.9, edgecolor="none")
    within_district_bars(fig, vv.assign(m=mins), "m", bins, labels,
                         "Price by walk to the metro, all else equal",
                         "Walking time to the nearest metro station", focus="walk_metro_km")
    save(fig, "07_metro_walk.png")


# 8 --------------------------------------------------------------------------------------------
def cafes(v, d, w):
    p = gpd.read_file("data/osm_pois.gpkg")
    f = p[p["amenity"].isin(["cafe", "restaurant", "bar", "pub"])]
    f = f[f.within(d.unary_union)]
    fig = page("The café effect",
               "A premium appears only where 40 or more venues are within a short walk",
               f"{len(f):,} cafés, restaurants, bars and pubs from OpenStreetMap (Oct 2026), counted per 600 m "
               "hexagon. Bars: typical-flat price net of the district and every other feature, by venues within a "
               "7-minute walk (560 m on footpaths). Regression: +3.4 % per doubling of venues; Żabka shops: no effect. " + style.SOURCE)
    ax = basemap(fig, d, w)
    cmap = ListedColormap(["#fde4cf", "#f9b98a", "#f08a4b", "#d4602a", "#a33f14", "#6e2508"])
    bounds = [1, 3, 6, 12, 25, 50, 1000]
    norm = BoundaryNorm(bounds, cmap.N)
    ax.hexbin(f.geometry.x, f.geometry.y, gridsize=(48, 42), mincnt=1, cmap=cmap, norm=norm,
              extent=(*ax.get_xlim(), *ax.get_ylim()), linewidths=0.3, edgecolors=style.SURFACE, zorder=3)
    outlines(ax, d)
    legend_patches(ax, [cmap(i) for i in range(cmap.N)], ["1–2", "3–5", "6–11", "12–24", "25–49", "50+"],
                   "Venues per hexagon")
    n500 = 2 ** v["walk7_food_drink"] - 1
    within_district_bars(fig, v.assign(n=n500), "n", [-1, 0, 5, 15, 40, 100, 1e9],
                         ["none", "1–5", "6–15", "16–40", "41–100", "100+"],
                         "Price by venues nearby, all else equal",
                         "Cafés, restaurants and bars within a 7-minute walk", focus="walk7_food_drink")
    save(fig, "08_cafes.png")


# 9 --------------------------------------------------------------------------------------------
def noise(v, d, w):
    fig = page("Where Warsaw is loud",
               "EU strategic noise map: road, tram and train, day-evening-night level (Lden)",
               "EEA Environmental Noise Directive data (round 4, update April 2026), 10 m grid; the louder of road "
               "and rail shown ('rail' includes trams: 91 % of tram stops away from railways lie in it). Bars: "
               "typical-flat price net of the district and every other feature, by tram & train noise at the façade "
               "(loudest band within 10 m). Road noise has a much smaller effect (−0.5 % per 10 dB). "
               + style.SOURCE)
    ax = basemap(fig, d, w)
    arrs = []
    for s in ("road", "rail"):
        with rasterio.open(f"data/noise_{s}_lden.tif") as src:
            tr, wd, ht = calculate_default_transform(src.crs, "EPSG:2180", src.width, src.height, *src.bounds,
                                                     resolution=20)
            out = np.zeros((ht, wd), np.uint8)
            reproject(rasterio.band(src, 1), out, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tr, dst_crs="EPSG:2180", resampling=Resampling.max)
            arrs.append(out)
    a = np.maximum(*arrs).astype(float)
    from rasterio.features import geometry_mask
    outside = geometry_mask([d.unary_union], out_shape=a.shape, transform=tr)
    a[(a == 0) | outside] = np.nan
    cmap = ListedColormap(["#fde9a9", "#f8b84e", "#ec7b2c", "#c8431d", "#7f1d12"])
    extent = (tr.c, tr.c + tr.a * a.shape[1], tr.f + tr.e * a.shape[0], tr.f)
    ax.imshow(a, extent=extent, cmap=cmap, vmin=0.5, vmax=5.5, interpolation="nearest", zorder=3)
    outlines(ax, d)
    legend_patches(ax, [cmap(i) for i in range(5)], ["55–59 dB", "60–64", "65–69", "70–74", "75+"], "Lden")
    within_district_bars(fig, v, "noise_rail_db", [0, 50, 60, 65, 70, 100],
                         ["below 55 dB", "55–59", "60–64", "65–69", "70+"],
                         "Price by tram & train noise at the façade, all else equal",
                         "Tram & train noise (Lden) at the building", min_n=40, focus="noise_rail_db")
    save(fig, "09_noise.png")


def trees(v, d, w):
    """Tree cover density map (Copernicus 2018, trees/) + price by tree cover within 100 m, all else equal."""
    from rasterio.features import geometry_mask
    greens = ["#dcedc8", "#aed581", "#7cb342", "#4f8f2f", "#2e6b1f", "#16461a"]
    fig = page("Trees: lovely, barely priced",
               "Only the leafiest surroundings (30 %+ tree cover) add about 2 %; parks nearby matter more",
               "Tree Cover Density 2018: Copernicus Land Monitoring Service (EEA), 10 m, summer Sentinel-2; shown from "
               "10 %. Bars: typical-flat price net of the district and every other feature (incl. parks and forest "
               "within 500 m), by mean tree cover within 100 m of the building. Regression: +0.3 % per 10 points, not "
               "significant; 10 ha of park within 500 m: +2.7 %. " + style.SOURCE)
    ax = basemap(fig, d, w)
    with rasterio.open("trees/data/tree_density_warsaw_10m.tif") as src:
        a = src.read(1).astype(float)
        tr = src.transform
    a[(a > 100) | (a < 10) | geometry_mask([d.unary_union], out_shape=a.shape, transform=tr)] = np.nan
    extent = (tr.c, tr.c + tr.a * a.shape[1], tr.f + tr.e * a.shape[0], tr.f)
    cmap = ListedColormap(greens)
    ax.imshow(a, extent=extent, cmap=cmap, norm=BoundaryNorm([10, 25, 40, 55, 70, 85, 101], cmap.N),
              interpolation="nearest", zorder=3)
    outlines(ax, d)
    legend_patches(ax, greens, ["10–25 %", "25–40", "40–55", "55–70", "70–85", "85–100 %"], "Tree cover")
    bx = within_district_bars(fig, v, "canopy_100", [-1, 5, 10, 15, 20, 30, 101],
                              ["< 5 %", "5–10", "10–15", "15–20", "20–30", "30 %+"],
                              "Price by tree cover within 100 m, all else equal",
                              "Mean tree cover within 100 m of the building", focus="canopy_100")
    bx.set_position([0.14, 0.13, 0.80, 0.17])
    save(fig, "20_trees.png")


def main():
    v, d, w = load()
    ctx = gpd.read_file("output/coverage.gpkg", layer="building_coverage", columns=["geometry"])
    cheaper_than_expected(v, d, w)
    # metro_walk(v, d, w, ctx)   # dropped: duplicates the TransitFrequency metro-access maps
    cafes(v, d, w)
    noise(v, d, w)
    trees(v, d, w)


if __name__ == "__main__":
    main()
