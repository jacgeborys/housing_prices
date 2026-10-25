"""
Sanity check of the building ages (GIS Expert decades, fetch_building_age.py; their map is from ~2020 and has no
2021+ layer) against an independent signal: the year of a building's first developer (primary) sale in RCN, which
is recorded at completion. RCN starts in 2006, so this checks recent buildings only.

Flags per building:
  ok                      dated, no contradiction
  likely_2021_plus        no date, first developer sale 2019+ (built after the map)
  no_data                 no date, no recent developer sale
  newer_than_dated        dated <= 1990 but first developer sale 2016+ (renovated tenement or attic conversion
                          sold by a developer, or a new building where the map still shows the old one)

OSM (osm_dates()): undated buildings with a construction year tagged in OSM (fetch_osm_dates.py) take its decade
(osm_date). Twins (twins()): an undated building takes the decade of its identical-footprint neighbours (prefab estates repeat
the same block) when >= 2 dated twins, >= 75 % of them, agree (twin_imputed); dated buildings whose twins disagree are listed for
review (twin_conflict, decade kept).

Corrected age (column decade_end; the map's own value kept as decade_gisexpert), read by every age-dependent
script: likely_2021_plus -> 2030 ("2021+"), no_data -> -1 (unknown, NOT the prefab reference), hand-checked new
buildings on old plots (OVERRIDES) -> 2030; newer_than_dated stays as dated: checked by eye 2026-10-18, most are
genuinely old (renovations / attic conversions sold by developers).

Outputs: output/age_check.gpkg (layer age_check, every dated or flagged building; open in QGIS over a basemap),
         output/analysis/age_check_centre.png (preview of the centre, 2400 x 3000)
"""
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import style
from price_drivers import pkin_xy

style.apply()
HALF_W, HALF_H = 2600, 3250          # m around PKiN for the preview
DECADE_COL = {1900: "#5a1e0f", 1910: "#7a2a12", 1920: "#983818", 1930: "#b4491f", 1940: "#c9622a",
              1950: "#e08a3c", 1960: "#efb15c", 1970: "#9fb3c8", 1980: "#7f9bb8", 1990: "#5f7fa6",
              2000: "#a9d38f", 2010: "#5fb36b", 2020: "#1f7a4d"}
OVERRIDES = {   # osm_id: new building on an old plot, map still shows the old one (checked by eye 2026-10-18)
    80030649: "Rotunda na Dynasach, Dynasy 22 (old rotunda demolished, rebuilt)",
    152094243: "Moderna Powiśle, Zajęcza 1A",
    97629178: "Koszykowa 49A",
}
FLAG_COL = {"likely_2021_plus": "#ffd400", "no_data": "#e01cd5"}
TWIN_R, TWIN_TOL, TWIN_DH = 600, 0.03, 3       # twins: within 600 m, footprint within 3 %, height within 3 m
TWIN_CSV = "output/analysis/age_twin_conflicts.csv"
OSM_DATES = "data/osm_building_dates.csv"     # fetch_osm_dates.py


def flags():
    a = gpd.read_file("output/building_age.gpkg")
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                       columns=["osm_id", "year", "market"])
    t = tx[tx["osm_id"].notna()].copy()
    t["osm_id"] = t["osm_id"].astype("int64")
    dev = t[t["market"] == "primary"].groupby("osm_id")["year"].agg(first_dev="min", n_dev="size")
    a = a.merge(dev, left_on="osm_id", right_index=True, how="left")
    a = a.merge(t.groupby("osm_id").size().rename("n_sales"), left_on="osm_id", right_index=True, how="left")
    dated = a["decade_end"] > 0
    a["flag"] = "ok"
    a.loc[~dated & (a["first_dev"] >= 2019), "flag"] = "likely_2021_plus"
    a.loc[~dated & ~(a["first_dev"] >= 2019), "flag"] = "no_data"
    a.loc[dated & (a["decade_end"] <= 1990) & (a["first_dev"] >= 2016), "flag"] = "newer_than_dated"
    a.loc[a["osm_id"].isin(list(OVERRIDES)), "flag"] = "override_2021_plus"
    a["decade_gisexpert"] = a["decade_end"]
    a["decade_end"] = np.where(a["flag"].isin(["likely_2021_plus", "override_2021_plus"]), 2030,
                               np.where(a["flag"] == "no_data", -1, a["decade_end"]))
    a = osm_dates(a)
    a = twins(a)
    a["label"] = np.where(a["decade_end"] == 2030, "2021+", np.where(a["decade_end"] < 0, "unknown", a["label"]))
    print(a["flag"].value_counts().to_string())
    priced = a["n_sales"] >= 3
    print("priced buildings (>= 3 sales):\n" + a.loc[priced, "flag"].value_counts().to_string())
    a.to_file("output/age_check.gpkg", layer="age_check", driver="GPKG")
    return a


def osm_dates(a):
    """Undated buildings (no_data) with a construction year tagged in OSM (fetch_osm_dates.py) take its decade
    (flag osm_date). Runs before twins(), so a tagged year wins over a look-alike neighbour."""
    if not os.path.exists(OSM_DATES):
        print(f"osm dates: {OSM_DATES} missing (python fetch_osm_dates.py), skipped")
        return a
    o = pd.read_csv(OSM_DATES).dropna(subset=["year"]).drop_duplicates("osm_id").set_index("osm_id")["year"]
    yr = a["osm_id"].map(o)
    hit = (a["flag"] == "no_data") & yr.notna()
    dec = np.clip((yr - 1) // 10 * 10 + 10, 1900, None)
    a.loc[hit, "decade_end"] = np.where(yr[hit] >= 2021, 2030, dec[hit])
    a.loc[hit, "label"] = [("2021+" if d == 2030 else "before 1901" if d == 1900 else f"{int(d) - 9}–{int(d)}")
                           for d in a.loc[hit, "decade_end"]]
    a.loc[hit, "flag"] = "osm_date"
    print(f"osm dates: {hit.sum()} undated buildings dated from OSM tags ({(hit & (a['n_sales'] >= 3)).sum()} priced)")
    return a


def _sides(g):
    """Short and long side of the minimum rotated rectangle."""
    c = np.asarray(g.minimum_rotated_rectangle.exterior.coords)
    return sorted(np.hypot(*np.diff(c[:3], axis=0).T))


def twins(a):
    """Planar twins: buildings within TWIN_R m whose area, perimeter and rectangle sides match within TWIN_TOL and
    height within TWIN_DH m (prefab estates repeat the same block). An undated building (no_data) with >= 2 dated
    twins, >= 75 % of them agreeing, takes their majority decade (flag twin_imputed). A dated building whose >= 3 twins agree (>= 80 %) on
    another decade is listed in TWIN_CSV (flag twin_conflict, decade kept: a one-decade gap is often a real
    1979 / 1981 split within one estate)."""
    from shapely.strtree import STRtree
    s = np.array([_sides(g) for g in a.geometry])
    area, perim, short, long_ = a.area.to_numpy(), a.length.to_numpy(), s[:, 0], s[:, 1]
    h = a["height_m"].to_numpy(float)
    cen = a.geometry.centroid.values
    tree = STRtree(cen)
    dated = a["flag"].isin(["ok", "newer_than_dated"]).to_numpy()
    dec = a["decade_end"].to_numpy(float)
    n_tw, tw_dec, tw_share = np.zeros(len(a), int), np.full(len(a), np.nan), np.zeros(len(a))
    for i in range(len(a)):
        j = tree.query(cen[i].buffer(TWIN_R))
        j = j[(j != i) & dated[j]]
        m = ((np.abs(area[j] / area[i] - 1) < TWIN_TOL) & (np.abs(perim[j] / perim[i] - 1) < TWIN_TOL)
             & (np.abs(short[j] / short[i] - 1) < TWIN_TOL) & (np.abs(long_[j] / long_[i] - 1) < TWIN_TOL))
        if np.isfinite(h[i]):
            m &= ~np.isfinite(h[j]) | (np.abs(h[j] - h[i]) <= TWIN_DH)
        if m.any():
            vals, cnt = np.unique(dec[j[m]], return_counts=True)
            n_tw[i], tw_dec[i], tw_share[i] = m.sum(), vals[cnt.argmax()], cnt.max() / m.sum()
    a["n_twins"], a["twin_decade"], a["twin_share"] = n_tw, tw_dec, tw_share
    imp = (a["flag"] == "no_data") & (a["n_twins"] >= 2) & (a["twin_share"] >= 0.75)
    a.loc[imp, "decade_end"] = a.loc[imp, "twin_decade"]
    a.loc[imp, "flag"] = "twin_imputed"
    a.loc[imp, "label"] = [f"{int(d) - 9}–{int(d)}" if d > 1900 else "before 1901" for d in a.loc[imp, "decade_end"]]
    conf = (dated & (a["n_twins"] >= 3) & (a["twin_share"] >= 0.8) & (a["twin_decade"] != a["decade_end"])
            & (a["flag"] == "ok"))
    a.loc[conf, "flag"] = "twin_conflict"
    a.loc[conf, ["osm_id", "addr_street", "addr_housenumber", "decade_end", "twin_decade", "n_twins",
                 "twin_share", "height_m", "n_sales"]].assign(area_m2=a.loc[conf].area.round()) \
        .sort_values("n_sales", ascending=False).to_csv(TWIN_CSV, index=False)
    print(f"twins: {imp.sum()} undated buildings dated from their twins ({(imp & (a['n_sales'] >= 3)).sum()} priced), "
          f"{conf.sum()} dated buildings disagree with their twins -> {TWIN_CSV}")
    return a


def preview(a):
    cx, cy = pkin_xy()
    box = (cx - HALF_W, cy - HALF_H, cx + HALF_W, cy + HALF_H)
    a = a.cx[box[0]:box[2], box[1]:box[3]]
    allb = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all", bbox=box)
    roads = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/roads.shp", bbox=gpd.GeoSeries(
        [gpd.GeoSeries.from_xy([box[0], box[2]], [box[1], box[3]], crs=2180).to_crs(4326).unary_union.envelope],
        crs=4326)).to_crs(2180)
    roads = roads[roads["highway"].isin(["motorway", "trunk", "primary", "secondary"])]

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.79])
    allb.plot(ax=ax, color="#e4e2da", lw=0)
    roads.plot(ax=ax, color="#c9c6bb", lw=0.6)
    dated = a[(a["decade_end"] > 0) & (a["decade_end"] < 2030)]
    dated.plot(ax=ax, color=dated["decade_end"].map(DECADE_COL).fillna("#999999"), lw=0)
    FLAG_COL["override_2021_plus"] = FLAG_COL["likely_2021_plus"]
    for f, c in FLAG_COL.items():
        s = a[a["flag"] == f]
        if len(s):
            s.plot(ax=ax, color=c, lw=0)
    nd = a[a["flag"] == "newer_than_dated"]
    if len(nd):
        nd.plot(ax=ax, facecolor="none", edgecolor="#e8000b", lw=1.1)
    for nm, grp in roads[roads["name"].notna()].groupby("name"):
        if grp.length.sum() > 1500:
            p = grp.unary_union.interpolate(0.5, normalized=True) if grp.unary_union.geom_type == "LineString" \
                else grp.geometry.iloc[len(grp) // 2].interpolate(0.5, normalized=True)
            ax.text(p.x, p.y, nm, fontsize=4.5, color="#6b6860", ha="center", va="center")
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_aspect("equal")
    ax.axis("off")
    style.header(fig, "Building age check: the centre",
                 "GIS Expert construction decade (map from ~2020) vs RCN developer sales", y=0.985)
    h = [Patch(color=DECADE_COL[y], label=("before 1901" if y == 1900 else f"{y - 9}–{y}")) for y in DECADE_COL]
    h += [Patch(color=FLAG_COL["likely_2021_plus"], label="no date, developer sales 2019+ (likely 2021+)"),
          Patch(color=FLAG_COL["no_data"], label="no date"),
          Patch(facecolor="none", edgecolor="#e8000b", label="dated ≤ 1990, developer sales 2016+")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.025))
    path = "output/analysis/age_check_centre.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}, output/age_check.gpkg")


if __name__ == "__main__":
    preview(flags())
