"""
Trees around each building (straight-line rings) -> trees/output/tree_features.gpkg

For every building with a model value (output/price_model.gpkg / building_value):
  canopy_50, canopy_100   mean tree cover density (0-1) within 50 / 100 m of the footprint
  dense_100               share of the ring with dense tree cover (>= 50 %)
Source: trees/data/tree_density_warsaw_10m.tif (Copernicus HRL Tree Cover Density 2018, 10 m).
The footprint itself is excluded (a roof is not a tree, and its pixels would dilute the share).
"""
import os

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CHM = os.path.join(HERE, "data", "tree_density_warsaw_10m.tif")
OUT = os.path.join(HERE, "output", "tree_features.gpkg")
DENSE = 50


def main():
    b = gpd.read_file(os.path.join(ROOT, "output", "price_model.gpkg"), layer="building_value",
                      columns=["osm_id"]).drop_duplicates("osm_id").reset_index(drop=True)
    res = {"canopy_50": [], "canopy_100": [], "dense_100": []}
    with rasterio.open(CHM) as src:
        for i, g in enumerate(b.geometry):
            ring100 = g.buffer(100).difference(g)
            win = from_bounds(*ring100.bounds, transform=src.transform).round_offsets().round_lengths()
            a = src.read(1, window=win, boundless=True, fill_value=255).astype(float)
            a[a > 100] = np.nan
            tr = src.window_transform(win)
            for r, key in ((50, "canopy_50"), (100, "canopy_100")):
                ring = g.buffer(r).difference(g)
                inside = ~geometry_mask([ring], out_shape=a.shape, transform=tr, all_touched=True)
                vals = a[inside]
                vals = vals[~np.isnan(vals)]
                res[key].append(vals.mean() / 100 if vals.size else np.nan)
                if r == 100:
                    res["dense_100"].append((vals >= DENSE).mean() if vals.size else np.nan)
            if i % 4000 == 0:
                print(f"  {i:,}/{len(b):,}", flush=True)
    for k, v in res.items():
        b[k] = v
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    b.to_file(OUT, layer="tree_features", driver="GPKG")
    print(b[list(res)].describe().round(3).T[["mean", "25%", "50%", "75%", "max"]])
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
