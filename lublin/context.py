"""
What goes with price in Lublin, described (no model): cafés, noise, trees, transit. Each slide = a map of the layer and bars
of the typical-flat value against the district's median building, by class of the layer at the building.
Run from lublin/ after fetch_context.py:  python context.py [--lang pl]

Inputs: output/price_model.gpkg (building_value), data/osm_pois.gpkg, data/noise_{road,rail}_lden.tif (EEA, 3035),
D:/QGIS/lidar/trees/data/hrl_full/lublin.tif (Copernicus HRL tree cover 2018, 10 m, read-only),
D:/QGIS/TransitFrequency/_data/lublin/2026_10_09/buildings_gates.gpkg (max_trips: departures within a 5-minute walk,
each line once at its best stop; read-only).
-> output/report/src[_pl]/r11_cafes, r12_noise, r13_trees, r14_transit (.png); output/analysis/building_context.csv
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from rasterio.warp import Resampling, reproject
from scipy.ndimage import uniform_filter

from common import T, basemap, building_values, districts, fmt, headline, page, pct, rel_bars, save, set_lang, source

TREES = "D:/QGIS/lidar/trees/data/hrl_full/lublin.tif"
GATES = "D:/QGIS/TransitFrequency/_data/lublin/2026_10_09/buildings_gates.gpkg"
FOOD = ["cafe", "restaurant", "bar", "pub", "fast_food"]
MAP = (0.03, 0.36, 0.94, 0.47)


def sample(path, pts, band_filter=None, radius_m=0):
    """Raster value at points (any CRS), optionally the mean over a square of side 2*radius."""
    with rasterio.open(path) as s:
        a = s.read(1).astype(float)
        if s.nodata is not None:
            a[a == s.nodata] = np.nan
        if radius_m:
            k = int(2 * radius_m / s.res[0]) + 1
            valid = np.isfinite(a)
            a = uniform_filter(np.nan_to_num(a), k) / np.maximum(uniform_filter(valid.astype(float), k), 1e-9)
        p = pts.to_crs(s.crs)
        rows, cols = s.index(p.x.to_numpy(), p.y.to_numpy())
        rows, cols = np.clip(rows, 0, a.shape[0] - 1), np.clip(cols, 0, a.shape[1] - 1)
        return a[rows, cols]


def features(bv):
    pts = bv.geometry.representative_point()
    poi = gpd.read_file("data/osm_pois.gpkg")
    food = poi[poi["amenity"].isin(FOOD)]
    ids = gpd.sjoin(gpd.GeoDataFrame(geometry=pts.buffer(500), crs=2180), food[["geometry"]], predicate="contains")
    bv["cafes"] = pd.Series(ids.index).value_counts().reindex(bv.index).fillna(0).to_numpy()
    db = {}
    for src in ("road", "rail"):
        if os.path.exists(f"data/noise_{src}_lden.tif"):
            v = sample(f"data/noise_{src}_lden.tif", pts)
            db[src] = np.where(v > 0, np.nan_to_num(v), 0)
    bv["noise_band"] = np.maximum(db.get("road", 0), db.get("rail", 0))
    bv["trees"] = sample(TREES, pts, radius_m=100)
    g = gpd.read_file(GATES)[["max_trips", "geometry"]].to_crs(2180)
    j = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=pts, crs=2180), g, how="left", max_distance=25)
    bv["trips"] = j[~j.index.duplicated()]["max_trips"].reindex(bv.index).to_numpy()
    os.makedirs("output/analysis", exist_ok=True)
    bv.drop(columns="geometry").to_csv("output/analysis/building_context.csv", index=False)
    return bv, food


def cafes(bv, food, d):
    bins, labels = [-1, 0, 5, 15, 40, 1e9], ["0", "1–5", "6–15", "16–40", "40+"]
    grp = pd.cut(bv["cafes"], bins, labels=labels).astype(str)
    med = {l: bv.loc[grp == l, "rel"].median() for l in labels}
    fig = page(T("Cafés: no price pattern in Lublin", "Kawiarnie: w Lublinie bez wpływu na ceny"),
               T("Cafés, restaurants and bars per hexagon, and the price of flats by how many are within 500 m",
                 "Kawiarnie, restauracje i bary w heksagonach oraz ceny mieszkań wg ich liczby w promieniu 500 m"),
               T(f"{fmt(len(food))} venues from OpenStreetMap (Oct 2026), 500 m hexagons. Bars: typical flat vs the median building of its "
                 "district, by venues within 500 m (straight line). A plain comparison: cafés follow money as much as they attract it. ",
                 f"{fmt(len(food))} lokali z OpenStreetMap (paź 2026), heksagony 500 m. Słupki: typowe mieszkanie wobec mediany budynków "
                 "dzielnicy, wg lokali w promieniu 500 m. Proste porównanie: kawiarnie idą za pieniędzmi tak samo, jak je przyciągają. ")
               + source(", OSM"))
    headline(fig, T(f"Flats with 40+ venues within 500 m sell {pct(med['40+'])} against their district's median building, those "
                    f"with none {pct(med['0'])}, the ones between about 0: no gradient. Venues crowd into Śródmieście and Stare "
                    "Miasto, where the register sees fewest flats.",
                    f"Mieszkania z 40+ lokalami w promieniu 500 m są {pct(med['40+'])} wobec mediany budynków dzielnicy, bez lokali "
                    f"{pct(med['0'])}, pośrednie około 0: brak zależności. Lokale skupiają się w Śródmieściu i na Starym Mieście, gdzie "
                    "rejestr widzi najmniej mieszkań."))
    ax = basemap(fig, d, rect=MAP)
    cols = ["#fde4cf", "#f9b98a", "#f08a4b", "#c4521f", "#7a2a08"]
    hb = ax.hexbin(food.geometry.x, food.geometry.y, gridsize=(34, 30), extent=(*ax.get_xlim(), *ax.get_ylim()), mincnt=1,
                   cmap=ListedColormap(cols), norm=BoundaryNorm([1, 3, 6, 12, 25, 1000], 5), edgecolors="white",
                   linewidths=0.3, zorder=3)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ["1–2", "3–5", "6–11", "12–24", "25+"])],
              title=T("Venues per hexagon", "Lokale w heksagonie"), loc="upper right", fontsize=8.5, title_fontsize=9,
              frameon=False)
    a = fig.add_axes([0.12, 0.09, 0.84, 0.21])
    rel_bars(a, grp, bv["rel"], labels, T("Cafés, restaurants and bars within 500 m", "Kawiarnie, restauracje i bary w promieniu 500 m"))
    a.set_title(T("Typical flat against its district's median building", "Typowe mieszkanie wobec mediany budynków dzielnicy"),
                fontsize=10.5, loc="left")
    save(fig, "r11_cafes.png")


def noise(bv, d):
    labels = [T("quiet (< 55 dB)", "cicho (< 55 dB)"), "55–59", "60–64", "65+"]
    band = bv["noise_band"]
    grp = pd.Series(np.select([band == 0, band == 1, band == 2], labels[:3], labels[3]), index=bv.index)
    med = {l: bv.loc[grp == l, "rel"].median() for l in labels}
    fig = page(T("Noise: loud streets sell a little lower", "Hałas: głośne ulice sprzedają się trochę taniej"),
               T("Road noise (Lden, day-evening-night), and the price of flats by the noise at the building",
                 "Hałas drogowy (Lden, doba) oraz ceny mieszkań wg hałasu przy budynku"),
               T("EU strategic noise maps, round 4 (2022), EEA: road noise at the building (the rail map is empty for Lublin). Bars: typical flat vs the median "
                 "building of its district. A plain comparison: loud streets are also main streets. ",
                 "Mapy akustyczne UE, runda 4 (2022), EEA: hałas drogowy przy budynku (mapa kolejowa dla Lublina jest pusta). Słupki: typowe "
                 "mieszkanie wobec mediany budynków dzielnicy. Proste porównanie: głośne ulice to też główne ulice. ")
               + source(", EEA"))
    headline(fig, T(f"Flats in quiet buildings (below 55 dB) sell {pct(med[labels[0]])} against their district's median building, "
                    f"those at 65 dB or more {pct(med['65+'])}. {(band >= 3).mean():.0%} of priced buildings face 65 dB+.",
                    f"Mieszkania w cichych budynkach (poniżej 55 dB) są {pct(med[labels[0]])} wobec mediany dzielnicy, przy 65 dB "
                    f"i więcej {pct(med['65+'])}. {(band >= 3).mean():.0%} budynków z cenami ma 65 dB+."))
    ax = basemap(fig, d, rect=MAP)
    cols = ["#fbe3c4", "#f6b26b", "#e5733c", "#b8342a", "#6e1424"]
    for src, alpha in (("road", 1.0), ("rail", 1.0)):
        p = f"data/noise_{src}_lden.tif"
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as s:
            x0, y0, x1, y1 = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
            W, H = int((x1 - x0) / 20), int((y1 - y0) / 20)
            dst = np.zeros((H, W), np.uint8)
            reproject(s.read(1), dst, src_transform=s.transform, src_crs=s.crs,
                      dst_transform=rasterio.transform.from_origin(x0, y1, 20, 20), dst_crs="EPSG:2180",
                      resampling=Resampling.nearest)
        m = np.ma.masked_equal(dst, 0)
        ax.imshow(m, extent=(x0, x1, y0, y1), cmap=ListedColormap(cols), norm=BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 5),
                  zorder=2, interpolation="nearest", alpha=alpha)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ["55–59 dB", "60–64", "65–69", "70–74", "75+"])],
              title="Lden", loc="upper right", fontsize=8.5, title_fontsize=9, frameon=False)
    a = fig.add_axes([0.12, 0.09, 0.84, 0.21])
    rel_bars(a, grp, bv["rel"], labels, T("Road noise at the building", "Hałas drogowy przy budynku"))
    a.set_title(T("Typical flat against its district's median building", "Typowe mieszkanie wobec mediany budynków dzielnicy"),
                fontsize=10.5, loc="left")
    save(fig, "r12_noise.png")


def trees(bv, d):
    bins, labels = [-1, 5, 10, 15, 20, 30, 101], ["< 5 %", "5–10", "10–15", "15–20", "20–30", "30 %+"]
    grp = pd.cut(bv["trees"], bins, labels=labels).astype(str)
    med = {l: bv.loc[grp == l, "rel"].median() for l in labels}
    fig = page(T("Trees: lovely, barely priced", "Drzewa: miłe, ale prawie nie w cenie"),
               T("Tree cover, and the price of flats by tree cover within 100 m of the building",
                 "Pokrycie koronami drzew oraz ceny mieszkań wg zadrzewienia w promieniu 100 m"),
               T("Tree cover density 2018 (Copernicus HRL, 10 m). Bars: typical flat vs the median building of its district, by mean tree "
                 "cover within 100 m. A plain comparison, not net of age or location. ",
                 "Gęstość koron drzew 2018 (Copernicus HRL, 10 m). Słupki: typowe mieszkanie wobec mediany budynków dzielnicy, wg średniego "
                 "zadrzewienia w promieniu 100 m. Proste porównanie, bez uwzględnienia wieku i położenia. ")
               + source(", Copernicus"))
    hi, lo = med["30 %+"], med["< 5 %"]
    headline(fig, T(f"The leafiest surroundings (30 %+ cover within 100 m) go with {pct(hi)} against the district's median building, "
                    f"the barest (under 5 %) with {pct(lo)}: no premium for trees. The big forests lie on the southern edge, away from the blocks.",
                    f"Najbardziej zielone otoczenie (30 %+ w promieniu 100 m) to {pct(hi)} wobec mediany budynków dzielnicy, "
                    f"najbardziej łyse (poniżej 5 %) {pct(lo)}: drzewa nie podnoszą ceny. Duże lasy leżą na południowym skraju, z dala od bloków."))
    ax = basemap(fig, d, rect=MAP)
    with rasterio.open(TREES) as s:
        x0, y0, x1, y1 = ax.get_xlim()[0], ax.get_ylim()[0], ax.get_xlim()[1], ax.get_ylim()[1]
        W, H = int((x1 - x0) / 10), int((y1 - y0) / 10)
        dst = np.full((H, W), 255, np.uint8)
        reproject(s.read(1), dst, src_transform=s.transform, src_crs=s.crs, src_nodata=255, dst_nodata=255,
                  dst_transform=rasterio.transform.from_origin(x0, y1, 10, 10), dst_crs="EPSG:2180",
                  resampling=Resampling.nearest)
    from shapely import contains_xy
    gx, gy = np.meshgrid(x0 + (np.arange(W) + 0.5) * 10, y1 - (np.arange(H) + 0.5) * 10)
    inside = contains_xy(d.unary_union, gx, gy)
    m = np.ma.masked_where((dst < 10) | (dst == 255) | ~inside, dst)
    cols = ["#dcebc6", "#b3d68a", "#7fb54f", "#4e8f34", "#2c6b26", "#163f17"]
    ax.imshow(m, extent=(x0, x1, y0, y1), cmap=ListedColormap(cols), norm=BoundaryNorm([10, 25, 40, 55, 70, 85, 101], 6),
              zorder=2, interpolation="nearest")
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, ["10–25 %", "25–40", "40–55", "55–70", "70–85", "85–100 %"])],
              title=T("Tree cover", "Pokrycie koronami"), loc="upper right", fontsize=8.5, title_fontsize=9, frameon=False)
    a = fig.add_axes([0.12, 0.09, 0.84, 0.21])
    rel_bars(a, grp, bv["rel"], labels, T("Mean tree cover within 100 m of the building", "Średnie zadrzewienie w promieniu 100 m"))
    a.set_title(T("Typical flat against its district's median building", "Typowe mieszkanie wobec mediany budynków dzielnicy"),
                fontsize=10.5, loc="left")
    save(fig, "r13_trees.png")


def transit(bv, d):
    bins, labels = [-1, 50, 100, 200, 400, 1e9], ["< 50", "50–100", "100–200", "200–400", "400+"]
    ok = bv["trips"].notna()
    grp = pd.cut(bv["trips"], bins, labels=labels).astype(str).where(ok, "none")
    med = {l: bv.loc[grp == l, "rel"].median() for l in labels}
    fig = page(T("Buses and trolleybuses: more departures, no higher prices", "Autobusy i trolejbusy: więcej odjazdów, nie wyższe ceny"),
               T("Transit departures within a 5-minute walk of each building, and the price of flats by that number",
                 "Odjazdy komunikacji w zasięgu 5 minut pieszo od budynku oraz ceny mieszkań wg ich liczby"),
               T("Departures a day from stops within a 5-minute walk, each line once at its best stop (ZTM Lublin GTFS, Oct 2026; "
                 "TransitFrequency project). Bars: typical flat vs the median building of its district. ",
                 "Odjazdy na dobę z przystanków w zasięgu 5 minut pieszo, każda linia raz na najlepszym przystanku (GTFS ZTM Lublin, "
                 "paź 2026; projekt TransitFrequency). Słupki: typowe mieszkanie wobec mediany budynków dzielnicy. ")
               + source(", ZTM Lublin"))
    headline(fig, T(f"Flats with 400+ departures nearby sell {pct(med['400+'])} against their district's median building, those "
                    f"with under 50 {pct(med['< 50'])}. Lublin has no metro or tram: buses and trolleybuses reach almost every block.",
                    f"Mieszkania z 400+ odjazdami w pobliżu są {pct(med['400+'])} wobec mediany dzielnicy, z mniej niż 50 "
                    f"{pct(med['< 50'])}. Lublin nie ma metra ani tramwaju: autobusy i trolejbusy docierają prawie do każdego bloku."))
    ax = basemap(fig, d, rect=MAP)
    g = gpd.read_file(GATES)[["max_trips", "geometry"]].to_crs(2180)
    g = g[g.representative_point().within(d.unary_union)]
    cols = ["#e9eef6", "#b9cdea", "#7aa3d8", "#3f73c0", "#1c3f86"]
    norm = BoundaryNorm([-1, 50, 100, 200, 400, 1e9], 5)
    g.plot(ax=ax, color=[ListedColormap(cols)(norm(v)) for v in g["max_trips"]], lw=0, zorder=3)
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(cols, labels)],
              title=T("Departures, 5 min walk", "Odjazdy, 5 min pieszo"), loc="upper right", fontsize=8.5, title_fontsize=9,
              frameon=False)
    a = fig.add_axes([0.12, 0.09, 0.84, 0.21])
    rel_bars(a, grp, bv["rel"], labels, T("Departures a day within a 5-minute walk", "Odjazdy na dobę w zasięgu 5 minut pieszo"))
    a.set_title(T("Typical flat against its district's median building", "Typowe mieszkanie wobec mediany budynków dzielnicy"),
                fontsize=10.5, loc="left")
    save(fig, "r14_transit.png")


def main(lang="en", only=None):
    set_lang(lang)
    d = districts()
    bv, food = features(building_values())
    for name, f in (("cafes", lambda: cafes(bv, food, d)), ("noise", lambda: noise(bv, d)), ("trees", lambda: trees(bv, d)),
                    ("transit", lambda: transit(bv, d))):
        if only is None or name in only:
            f()


if __name__ == "__main__":
    args = sys.argv[1:]
    lang = "pl" if "pl" in args else "en"
    only = [a for a in args if a in ("cafes", "noise", "trees", "transit")] or None
    main(lang, only)
