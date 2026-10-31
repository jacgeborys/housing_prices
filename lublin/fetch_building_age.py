"""
Lublin building age (decade of construction) from GIS Expert's public map "Rozwój zabudowy Lublina w XX i XXI wieku"
(https://mapy.gis-expert.pl/zabudowa-lublina/; decades to 2011-20, no 2021+ layer). Same method as Warsaw's
fetch_building_age.py; layer names from the map's main.*.js (layerInfos: budynki_do_1900, rok_1910 ... rok_2020,
budynki_brak_danych). Credit them in anything published; the raw layer is not redistributed (tiles cache in data/raw/,
which is not in git). An email asking for their permission/data is the proper route; this is a stop-gap.

Each decade is a separate tile layer (transparent PNG; drawn pixels = buildings of that decade). For every
building we sample up to N_PTS points inside the footprint, look up which decade layers are drawn there
(alpha > 0) and take the majority. Only tiles containing our buildings are fetched, once, at a polite rate.

  python fetch_building_age.py  (from lublin/)  -> output/building_age.gpkg (osm_id, decade_end, label, votes)
Buildings: every BDOT10k residential building in Lublin plus any other with a recorded flat sale.
"""
import io
import math
import os
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image
from shapely.geometry import Point

BASE = "https://tiles.arcgis.com/tiles/mQcAehnytds8jMvo/arcgis/rest/services/{layer}/MapServer/tile/{z}/{y}/{x}"
LAYERS = {"budynki_do_1900": 1900, **{f"rok_{y}": y for y in range(1910, 2030, 10)}}   # value = last year of the decade
LAYERS["budynki_brak_danych"] = -1                                                       # "no data"
LABEL = {1900: "before 1901", -1: "no data", **{y: f"{y - 9}–{y}" for y in range(1910, 2030, 10)}}
Z = 15
CACHE = "data/raw/gisexpert_tiles"
OUT = "output/building_age.gpkg"
N_PTS = 5
PAUSE_S = 0.25
HEADERS = {"User-Agent": "ceny-mieszkan research (non-commercial; credit GIS Expert)",
           "Referer": "https://mapy.gis-expert.pl/zabudowa-lublina/"}
ORIGIN = 20037508.342787
RES = 2 * ORIGIN / (256 * 2 ** Z)


def buildings():
    b = gpd.read_file("data/buildings.gpkg")
    val = gpd.read_file("output/price_model.gpkg", layer="building_value", columns=["osm_id"])
    b = b[b["residential"] | b["osm_id"].isin(val["osm_id"])][["osm_id", "geometry"]].reset_index(drop=True)
    print(f"{len(b):,} buildings (residential + priced)")
    return b


def sample_points(geom, n, rng):
    pts = [geom.representative_point()]
    x0, y0, x1, y1 = geom.bounds
    tries = 0
    while len(pts) < n and tries < 60:
        p = Point(rng.uniform(x0, x1), rng.uniform(y0, y1))
        if geom.contains(p):
            pts.append(p)
        tries += 1
    return pts


def tile(layer, tx, ty, session):
    path = f"{CACHE}/{layer}/{Z}/{ty}/{tx}.png"
    if os.path.exists(path):
        return np.array(Image.open(path).convert("RGBA"))[..., 3] if os.path.getsize(path) else None
    r = session.get(BASE.format(layer=layer, z=Z, y=ty, x=tx), headers=HEADERS, timeout=60)
    time.sleep(PAUSE_S)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if r.status_code != 200 or not r.headers.get("content-type", "").startswith(("image", "application/octet")):
        open(path, "wb").close()                     # empty file = no tile here (don't ask again)
        return None
    open(path, "wb").write(r.content)
    return np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))[..., 3]


def main():
    b = buildings()
    rng = np.random.default_rng(0)
    rows = []
    for oid, g in zip(b["osm_id"], b.geometry):
        for p in sample_points(g, N_PTS, rng):
            rows.append((oid, p.x, p.y))
    pts = gpd.GeoDataFrame(pd.DataFrame(rows, columns=["osm_id", "x", "y"]),
                           geometry=gpd.points_from_xy([r[1] for r in rows], [r[2] for r in rows]), crs=2180)
    m = pts.to_crs(3857).geometry
    px = (m.x + ORIGIN) / RES
    py = (ORIGIN - m.y) / RES
    pts["tx"], pts["ty"] = (px // 256).astype(int), (py // 256).astype(int)
    pts["ix"], pts["iy"] = (px % 256).astype(int), (py % 256).astype(int)
    tiles = pts[["tx", "ty"]].drop_duplicates().to_numpy()
    print(f"{len(pts):,} sample points in {len(tiles):,} tiles x {len(LAYERS)} layers "
          f"= {len(tiles) * len(LAYERS):,} requests at most (cached ones are skipped)")

    s = requests.Session()
    hits = {}
    done = 0
    for layer in LAYERS:
        h = np.zeros(len(pts), bool)
        for tx, ty in tiles:
            a = tile(layer, tx, ty, s)
            done += 1
            if a is None:
                continue
            sel = np.flatnonzero((pts["tx"].to_numpy() == tx) & (pts["ty"].to_numpy() == ty))
            h[sel] = a[pts["iy"].to_numpy()[sel], pts["ix"].to_numpy()[sel]] > 0
            if done % 500 == 0:
                print(f"  {done:,} tiles ({layer})", flush=True)
        hits[layer] = h
        print(f"{layer}: {h.mean():.1%} of points", flush=True)

    H = pd.DataFrame(hits)
    H["osm_id"] = pts["osm_id"].to_numpy()
    votes = H.groupby("osm_id").sum()
    n_pts = pts.groupby("osm_id").size()
    best = votes.idxmax(axis=1)
    top = votes.max(axis=1)
    out = b.merge(pd.DataFrame({"layer": best, "votes": top, "n_pts": n_pts}), left_on="osm_id", right_index=True)
    out.loc[out["votes"] == 0, "layer"] = None
    out["decade_end"] = out["layer"].map(LAYERS)
    out["label"] = out["decade_end"].map(LABEL)
    out.to_file(OUT, layer="building_age", driver="GPKG")
    print(out["label"].value_counts(dropna=False).to_string())
    print(f"Saved {OUT}  (source: GIS Expert, mapy.gis-expert.pl/zabudowa-lublina)")


if __name__ == "__main__":
    main()
