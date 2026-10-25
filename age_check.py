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
(osm_date), after Maksim Pashnin's dates (makscape(), fetch_makscape.py: heritage registers, project documents;
undated -> makscape, official date >= 20 years from GIS Expert's -> makscape_rebuilt). Developer sales (dev_dates()): undated buildings with >= 5 developer sales take the decade of
the year before the first one (dev_sale). New on an old plot (infill()): flagged newer_than_dated buildings sold whole by their developer, or
whose outline the ~2020 map does not cover, take the completion decade (infill_new / infill_or_renovated, listed in
output/analysis/age_infill.csv). Flats on sale now in a developer price list (on_sale(), offers/): undated, or dated <= 1990 with no sale -> 2021+.
BDOT10k creation date (bdot_new()): undated or newer_than_dated buildings first drawn in 2021+ -> new; undated
buildings on neither GIS Expert's map nor in BDOT -> 2021+ (absent_new).
Hand dating (manual(), manual/age_manual.gpkg, in git): `year` filled in by hand wins over everything.
Twins (twins()): an undated building takes the decade of its identical-footprint neighbours (prefab estates repeat
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
DEV_MIN = 5                                   # developer sales needed to date a building by its first one
MAKSCAPE = "data/makscape/makscape_dates.gpkg"  # fetch_makscape.py (Maksim Pashnin, warsaw.makscape.com; local only)
MK_OFFICIAL = ["official-heritage", "official-address-date", "primary-source-date"]
MK_GAP = 20                                   # years: an official date this far from GIS Expert's replaces it
INFILL_SHARE, INFILL_COVER = 0.5, 0.6         # new on an old plot: developer sold >= half the flats / outline covered
INFILL_CSV = "output/analysis/age_infill.csv"
OFFERS = "offers/data/offers_warszawa.gpkg"   # offers/: developer price lists (flats on sale now)
BDOT_SHP = "D:/QGIS/bdot_basemap/Warszawa/PL.PZGiK.330.1465__OT_BUBD_A.shp"   # read only; X_DATAUTW = object created
BDOT_NEW_YEAR = 2021                          # created in BDOT from this year: drawn in the 2022 update (built ~2018-22)
MANUAL = "manual/age_manual.gpkg"             # hand-dated buildings (in git): fill `year`, applied last


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
    a = makscape(a)
    a = osm_dates(a)
    a = dev_dates(a)
    a = infill(a)
    a = on_sale(a)
    a = bdot_new(a)
    a = twins(a)
    a = manual(a)
    a["label"] = np.where(a["decade_end"] == 2030, "2021+", np.where(a["decade_end"] < 0, "unknown", a["label"]))
    print(a["flag"].value_counts().to_string())
    priced = a["n_sales"] >= 3
    print("priced buildings (>= 3 sales):\n" + a.loc[priced, "flag"].value_counts().to_string())
    a.to_file("output/age_check.gpkg", layer="age_check", driver="GPKG")
    return a


def _decade(year):
    year = np.asarray(year, float)
    return np.where(year >= 2021, 2030, np.clip((year - 1) // 10 * 10 + 10, 1900, None))


def _label(d):
    return "2021+" if d == 2030 else "before 1901" if d == 1900 else f"{int(d) - 9}–{int(d)}"


def makscape(a):
    """Maksim Pashnin's dates (fetch_makscape.py), matched by his building's centre inside our footprint (largest of
    his footprints when several fall in one). Decade of the end of his date (completion; a range like 1948-1953 ->
    1951-60). Undated buildings (no_data) take any of his dates except a vague century (flag makscape). Dated
    buildings take his official or primary-source date (heritage registers, mostly post-war reconstructions) when it
    is >= MK_GAP years from GIS Expert's (flag makscape_rebuilt); smaller gaps are left, mostly range boundaries."""
    if not os.path.exists(MAKSCAPE):
        print(f"makscape: {MAKSCAPE} missing (python fetch_makscape.py), skipped")
        return a
    m = gpd.read_file(MAKSCAPE)
    m = m[m["kind"] != "century"]
    j = gpd.sjoin(m, a[["osm_id", "geometry"]], predicate="within", how="inner")
    j = j.sort_values("footprint_m2", ascending=False).drop_duplicates("osm_id").set_index("osm_id")
    yr = a["osm_id"].map(j["date_end"].fillna(j["date_start"]))
    q = a["osm_id"].map(j["quality"])
    dec = pd.Series(_decade(yr), index=a.index)
    a["mk_quality"], a["mk_source"], a["mk_raw"] = q, a["osm_id"].map(j["source"]), a["osm_id"].map(j["raw"])
    fill = (a["flag"] == "no_data") & yr.notna()
    rebuilt = (a["flag"].isin(["ok", "newer_than_dated"]) & q.isin(MK_OFFICIAL)
               & ((dec - a["decade_end"]).abs() >= MK_GAP))
    for mask, flag in ((fill, "makscape"), (rebuilt, "makscape_rebuilt")):
        a.loc[mask, "decade_end"] = dec[mask]
        a.loc[mask, "label"] = [_label(d) for d in dec[mask]]
        a.loc[mask, "flag"] = flag
    print(f"makscape: {len(j):,} of our buildings matched; {fill.sum()} undated dated ({(fill & (a['n_sales'] >= 3)).sum()} "
          f"priced), {rebuilt.sum()} re-dated from heritage registers ({(rebuilt & (a['n_sales'] >= 3)).sum()} priced)")
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


def gisexpert_cover(geoms, n=40, seed=0):
    """Share of each footprint that GIS Expert's map (cached tiles, any decade layer) draws as a building: a new
    outline on an old plot is often only partly covered by the building the ~2020 map still shows there."""
    from PIL import Image
    import fetch_building_age as fb
    rng, tiles, out = np.random.default_rng(seed), {}, []
    for g in geoms:
        pts = gpd.GeoSeries(fb.sample_points(g, n, rng), crs=2180).to_crs(3857)
        px, py = (pts.x + fb.ORIGIN) / fb.RES, (fb.ORIGIN - pts.y) / fb.RES
        hit = 0
        for x, y in zip(px, py):
            for layer in fb.LAYERS:
                k = (layer, int(x // 256), int(y // 256))
                if k not in tiles:
                    f = f"{fb.CACHE}/{layer}/{fb.Z}/{k[2]}/{k[1]}.png"
                    tiles[k] = (np.array(Image.open(f).convert("RGBA"))[..., 3]
                                if os.path.exists(f) and os.path.getsize(f) else None)
                if tiles[k] is not None and tiles[k][int(y % 256), int(x % 256)] > 0:
                    hit += 1
                    break
        out.append(hit / len(pts))
    return np.array(out)


def infill(a):
    """New buildings on old plots among newer_than_dated (dated <= 1990 by the ~2020 map, developer sales 2016+): the
    developer sold >= INFILL_SHARE of the estimated flats (coverage.py) with no resale before the first developer sale,
    or the outline is < INFILL_COVER covered by the building the map shows, with >= DEV_MIN developer sales. They take
    the completion decade (year before the first developer sale; 2021+ -> 2030): flag infill_new, or
    infill_or_renovated when the map dates the plot before 1941 and still covers the outline (>= 90 %): new behind
    an old outline or a gutted tenement, sold whole as new flats either way; listed in INFILL_CSV for review."""
    c = a["flag"] == "newer_than_dated"
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                       columns=["osm_id", "year", "market"])
    tx = tx[tx["osm_id"].isin(a.loc[c, "osm_id"])]
    tx = tx[(tx["market"] != "primary") & (tx["year"] < tx["osm_id"].map(a.set_index("osm_id")["first_dev"]))]
    before = a["osm_id"].map(tx.groupby("osm_id").size()).fillna(0)
    cov = gpd.read_file("output/coverage.gpkg", layer="building_coverage", ignore_geometry=True,
                        columns=["osm_id", "est_flats"]).drop_duplicates("osm_id").set_index("osm_id")["est_flats"]
    share = a["n_dev"] / a["osm_id"].map(cov)
    gx = pd.Series(np.nan, index=a.index)
    gx[c] = gisexpert_cover(a.loc[c, "geometry"])
    new = c & (a["n_dev"] >= DEV_MIN) & (((share >= INFILL_SHARE) & (before == 0)) | (gx < INFILL_COVER))
    amb = new & (a["decade_end"] <= 1940) & (gx >= 0.9)
    dec = pd.Series(_decade(a["first_dev"] - 1), index=a.index)
    for mask, flag in ((new & ~amb, "infill_new"), (amb, "infill_or_renovated")):
        a.loc[mask, "decade_end"] = dec[mask]
        a.loc[mask, "label"] = [_label(d) for d in dec[mask]]
        a.loc[mask, "flag"] = flag
    a.loc[new, ["osm_id", "addr_street", "addr_housenumber", "flag", "decade_gisexpert", "decade_end", "first_dev",
                "n_dev", "n_sales"]].assign(est_flats=a.loc[new, "osm_id"].map(cov), resales_before_dev=before[new],
                                            gisexpert_cover=gx[new].round(2))         .sort_values(["flag", "n_dev"], ascending=[True, False]).to_csv(INFILL_CSV, index=False)
    print(f"infill: {new.sum()} new buildings on old plots ({amb.sum()} of them new or gutted behind a pre-war outline) "
          f"-> {INFILL_CSV}")
    return a


def bdot_new(a):
    """BDOT10k creation date (X_DATAUTW) of the footprint under the building's point: objects come in update batches
    (2012 initial load, 2014, 2018, 2022); the 2022 batch is ~80 % buildings from 2011 on, but also old buildings
    redrawn, so it only decides uncertain cases. Undated (no_data) -> 2021+; newer_than_dated -> completion decade
    from the first developer sale (2021+ without one). Flag bdot_new."""
    if not os.path.exists(BDOT_SHP):
        return a
    b = gpd.read_file(BDOT_SHP, columns=["X_DATAUTW"]).set_crs(2180, allow_override=True)
    b["yr"] = pd.to_datetime(b["X_DATAUTW"], errors="coerce").dt.year
    pt = gpd.GeoDataFrame({"osm_id": a["osm_id"]}, geometry=a.geometry.representative_point(), crs=2180)
    yr = a["osm_id"].map(gpd.sjoin(pt, b[["yr", "geometry"]], predicate="within").drop_duplicates("osm_id")
                         .set_index("osm_id")["yr"])
    hit = a["flag"].isin(["no_data", "newer_than_dated"]) & (yr >= BDOT_NEW_YEAR)
    dec = pd.Series(np.where(a["first_dev"].notna(), _decade(a["first_dev"] - 1), 2030), index=a.index)
    dec[a["flag"] == "no_data"] = 2030
    a.loc[hit, "decade_end"] = dec[hit]
    a.loc[hit, "label"] = [_label(d) for d in dec[hit]]
    a.loc[hit, "flag"] = "bdot_new"
    print(f"BDOT creation date: {hit.sum()} uncertain buildings drawn in BDOT {BDOT_NEW_YEAR}+ dated as new "
          f"({(hit & (a['n_sales'] >= 3)).sum()} priced)")
    # in OSM now, but on neither GIS Expert's ~2020 map (no layer drawn at all, not even "no data") nor in this BDOT
    # (< 10 % of the footprint under any BDOT building): built after both -> 2021+ (flag absent_new); roofs excluded
    u = (a["flag"] == "no_data") & (a["votes"] == 0) & (a["building"] != "roof")
    ov = gpd.overlay(a.loc[u, ["osm_id", "geometry"]], b[["geometry"]], how="intersection", keep_geom_type=True)
    share = a["osm_id"].map(ov.dissolve("osm_id").area / a.loc[u].set_index("osm_id").area).fillna(0)
    absent = u & (share < 0.1)
    a.loc[absent, ["decade_end", "label", "flag"]] = [2030, "2021+", "absent_new"]
    print(f"absent from GIS Expert and BDOT: {absent.sum()} undated buildings dated 2021+ "
          f"({(absent & (a['n_sales'] >= 3)).sum()} priced)")
    return a


def on_sale(a):
    """Buildings with flats on sale in a developer's price list now (offers/, geocoded to the house number) are new or
    under construction: 2021+ (flag on_sale) when undated, or when dated <= 1990 with no recorded sale (the ~2020
    map still shows what stood on the plot)."""
    if not os.path.exists(OFFERS):
        return a
    o = gpd.read_file(OFFERS).to_crs(2180)
    o = o[o["geo"] == "number"]
    hit_ids = gpd.sjoin(o[["geometry"]], a[["osm_id", "geometry"]], predicate="within")["osm_id"].unique()
    hit = a["osm_id"].isin(hit_ids) & ((a["flag"] == "no_data")
                                       | (a["flag"].isin(["ok", "twin_conflict"]) & (a["decade_end"] <= 1990)
                                          & ~(a["n_sales"] > 0)))
    a.loc[hit, ["decade_end", "label", "flag"]] = [2030, "2021+", "on_sale"]
    print(f"on sale: {hit.sum()} buildings with flats in a developer price list dated 2021+ "
          f"({(hit & (a['n_sales'] >= 3)).sum()} priced)")
    return a


def dev_dates(a):
    """Undated buildings sold by their developer (>= DEV_MIN developer sales, the first in 2007+, RCN starts in 2006)
    take the decade of the year before that first sale: completion. Checked on dated buildings: same decade as GIS
    Expert for 91 % (the sale year itself: 78 %). Flag dev_sale."""
    hit = (a["flag"] == "no_data") & (a["first_dev"] >= 2007) & (a["n_dev"] >= DEV_MIN)
    a.loc[hit, "decade_end"] = (a.loc[hit, "first_dev"] - 2) // 10 * 10 + 10
    a.loc[hit, "label"] = [f"{int(d) - 9}–{int(d)}" for d in a.loc[hit, "decade_end"]]
    a.loc[hit, "flag"] = "dev_sale"
    print(f"developer sales: {hit.sum()} undated buildings dated from their first developer sale "
          f"({(hit & (a['n_sales'] >= 3)).sum()} priced)")
    return a


def _sides(g):
    """Short and long side of the minimum rotated rectangle."""
    c = np.asarray(g.minimum_rotated_rectangle.exterior.coords)
    return sorted(np.hypot(*np.diff(c[:3], axis=0).T))


def manual(a):
    """Hand dating, applied last (wins over every rule). MANUAL holds one row per building: osm_id (key, don't edit),
    address / district / n_sales / flag_before (for reference), year (fill in: the year the current building was
    completed, or rebuilt after the war; empty = still unknown), note (optional, e.g. a source). Rows with a year
    set decade_end (2021+ -> 2030) and flag manual. Each run appends the buildings that are still unknown and not in
    the file yet, priced ones first; existing rows and edits are never overwritten. To fix a building that is not
    listed, copy its feature from output/age_check.gpkg into the layer and fill year."""
    m = gpd.read_file(MANUAL) if os.path.exists(MANUAL) else None
    if m is not None and len(m):
        m["year"] = pd.to_numeric(m["year"], errors="coerce")
        yr = a["osm_id"].map(m.dropna(subset=["year"]).drop_duplicates("osm_id", keep="last").set_index("osm_id")["year"])
        hit = yr.notna()
        dec = pd.Series(_decade(yr), index=a.index)
        a.loc[hit, "decade_end"] = dec[hit]
        a.loc[hit, "label"] = [_label(d) for d in dec[hit]]
        a.loc[hit, "flag"] = "manual"
        print(f"manual: {hit.sum()} buildings hand-dated ({(hit & (a['n_sales'] >= 3)).sum()} priced)")
    todo = a[(a["flag"] == "no_data") & ~(a["osm_id"].isin(m["osm_id"]) if m is not None else False)]
    if len(todo):
        d = gpd.read_file("data/districts.gpkg").to_crs(2180)
        pts = gpd.GeoDataFrame(geometry=todo.geometry.representative_point(), crs=2180)
        dist = gpd.sjoin(pts, d[["district", "geometry"]], predicate="within", how="left")
        new = gpd.GeoDataFrame({
            "osm_id": todo["osm_id"].to_numpy(),
            "address": (todo["addr_street"].fillna("") + " " + todo["addr_housenumber"].fillna("")).str.strip().to_numpy(),
            "district": dist[~dist.index.duplicated()]["district"].to_numpy(),
            "n_sales": todo["n_sales"].fillna(0).astype(int).to_numpy(),
            "flag_before": "no_data", "year": pd.array([pd.NA] * len(todo), dtype="Int64"), "note": None},
            geometry=todo.geometry.to_numpy(), crs=2180).sort_values("n_sales", ascending=False)
        out = new if m is None else pd.concat([m, new], ignore_index=True)
        os.makedirs(os.path.dirname(MANUAL), exist_ok=True)
        from shapely.geometry import MultiPolygon
        out["geometry"] = [MultiPolygon([g]) if g.geom_type == "Polygon" else g for g in out.geometry]
        out.to_file(MANUAL, layer="age_manual", driver="GPKG", schema={
            "geometry": "MultiPolygon", "properties": {"osm_id": "int", "address": "str", "district": "str", "n_sales": "int",
                                                  "flag_before": "str", "year": "int", "note": "str"}})
        print(f"manual: {len(new)} still-unknown buildings added to {MANUAL} (fill `year`)")
    return a


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
