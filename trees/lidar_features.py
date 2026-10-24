"""
Trees around each building from 2025 LiDAR -> trees/output/tree_features_lidar.gpkg

Canopy source: D:/QGIS/lidar/trees (read only): fetch_lidar.py's 1 m canopy tiles (>= 3 m above ground, outside
buildings and bridges, not flat), cleaned with render_zoom.clean_canopy (unmapped roofs, wall / eave lines, rings),
the same canopy as its Warsaw LiDAR maps. Imported, never edited; no bytecode is written there.

    python trees/lidar_features.py            canopy raster (if missing) + features
    python trees/lidar_features.py --raster   rebuild the raster

1. trees/data/lidar_canopy_warsaw_1m.tif: whole city at 1 m, uint8 canopy height in m (0 = no canopy), EPSG:2180
2. per building with a model value (output/price_model.gpkg / building_value), footprint excluded as in
   tree_features.py:
     canopy_50, canopy_100   share (0-1) of the ring within 50 / 100 m under canopy
     tall_100                share of the 100 m ring under canopy >= 15 m (mature trees)
"""
import os
import sys

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, from_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIDAR = "D:/QGIS/lidar/trees"                                    # read only
RASTER = os.path.join(HERE, "data", "lidar_canopy_warsaw_1m.tif")
OUT = os.path.join(HERE, "output", "tree_features_lidar.gpkg")
X0, Y0, X1, Y1 = 625500, 471500, 656500, 504500                 # fetch_lidar's Warsaw sheets, 500 m grid
TALL_M = 15


def build_raster():
    sys.dont_write_bytecode = True
    sys.path.insert(0, LIDAR)
    import render_zoom as rz
    with rasterio.open(os.path.join(LIDAR, "data", "hrl_full", "warszawa.tif")) as s:
        h10 = s.read(1)
        h10[h10 > 100] = 0
        hrl = (h10.astype(np.uint8), s.transform, s.crs)
    W, H = X1 - X0, Y1 - Y0
    B, P = rz.BLOCK, rz.BLOCK_PAD
    prof = dict(driver="GTiff", width=W, height=H, count=1, dtype="uint8", crs="EPSG:2180", nodata=None,
                transform=rasterio.transform.from_origin(X0, Y1, 1, 1), tiled=True, blockxsize=512,
                blockysize=512, compress="deflate", predictor=2, BIGTIFF="IF_SAFER")
    os.makedirs(os.path.dirname(RASTER), exist_ok=True)
    blocks = [(r, c) for r in range(0, H, B) for c in range(0, W, B)]
    with rasterio.open(RASTER, "w", **prof) as dst:
        for i, (r, c) in enumerate(blocks):
            h_, w_ = min(B, H - r), min(B, W - c)
            is_c, hgt = rz.clean_canopy("warszawa", X0 + c - P, Y1 - r + P, max(h_, w_) + 2 * P, hrl)
            core = (slice(P, P + h_), slice(P, P + w_))
            m = np.where(is_c[core], np.clip(np.round(hgt[core] / 10), 1, 255), 0).astype(np.uint8)
            dst.write(m, 1, window=Window(c, r, w_, h_))
            print(f"  block {i + 1}/{len(blocks)}: canopy {is_c[core].mean():.1%}", flush=True)
    print(f"Saved {RASTER}")


def features():
    b = gpd.read_file(os.path.join(ROOT, "output", "price_model.gpkg"), layer="building_value",
                      columns=["osm_id"]).drop_duplicates("osm_id").reset_index(drop=True)
    res = {"canopy_50": [], "canopy_100": [], "tall_100": []}
    with rasterio.open(RASTER) as src:
        for i, g in enumerate(b.geometry):
            ring100 = g.buffer(100).difference(g)
            win = from_bounds(*ring100.bounds, transform=src.transform).round_offsets().round_lengths()
            a = src.read(1, window=win, boundless=True, fill_value=0)
            tr = src.window_transform(win)
            for r, key in ((50, "canopy_50"), (100, "canopy_100")):
                inside = ~geometry_mask([g.buffer(r).difference(g)], out_shape=a.shape, transform=tr)
                vals = a[inside]
                res[key].append((vals > 0).mean() if vals.size else np.nan)
                if r == 100:
                    res["tall_100"].append((vals >= TALL_M).mean() if vals.size else np.nan)
            if i % 4000 == 0:
                print(f"  {i:,}/{len(b):,}", flush=True)
    for k, v in res.items():
        b[k] = v
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    b.to_file(OUT, layer="tree_features", driver="GPKG")
    print(b[list(res)].describe().round(3).T[["mean", "25%", "50%", "75%", "max"]])
    print(f"Saved {OUT}")


if __name__ == "__main__":
    if "--raster" in sys.argv or not os.path.exists(RASTER):
        build_raster()
    features()
