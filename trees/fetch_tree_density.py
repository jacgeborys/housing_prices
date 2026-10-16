"""
Tree cover density for Warsaw at 10 m -> trees/data/tree_density_warsaw_10m.tif (EPSG:2180, uint8 %, 255 = no data).

Source: Copernicus Land Monitoring Service, High Resolution Layer Tree Cover Density 2018 (10 m, from summer
Sentinel-2 imagery; EEA ImageServer GioLandPublic/HRL_TreeCoverDensity_2018). Leaf-on, so deciduous city
parks and courtyards show; it is a density (share of the 10 m pixel under tree crowns), not a height.
(The Meta/WRI canopy-height layer, fetch_canopy.py, was tried first and rejected: its imagery for Warsaw's
tile reads deciduous city trees as a few metres tall or absent; only conifer forests come out right.)

    python trees/fetch_tree_density.py
"""
import os

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "tree_density_warsaw_10m.tif")
URL = ("https://image.discomap.eea.europa.eu/arcgis/rest/services/GioLandPublic/HRL_TreeCoverDensity_2018/"
       "ImageServer/exportImage")
WARSAW_LONLAT = (20.83, 52.08, 21.29, 52.38)
PIXEL = 10
TILE = 2000


def export(x0, y0, x1, y1):
    w, h = int(round((x1 - x0) / PIXEL)), int(round((y1 - y0) / PIXEL))
    r = requests.get(URL, params={"bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3035, "imageSR": 3035,
                                  "size": f"{w},{h}", "format": "tiff", "pixelType": "U8",
                                  "interpolation": "RSP_NearestNeighbor", "f": "image"}, timeout=300)
    r.raise_for_status()
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        return src.read(1)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    x0, y0, x1, y1 = Transformer.from_crs(4326, 3035, always_xy=True).transform_bounds(*WARSAW_LONLAT)
    x0, y0 = np.floor(x0 / PIXEL) * PIXEL, np.floor(y0 / PIXEL) * PIXEL
    x1, y1 = np.ceil(x1 / PIXEL) * PIXEL, np.ceil(y1 / PIXEL) * PIXEL
    W, H = int((x1 - x0) / PIXEL), int((y1 - y0) / PIXEL)
    a = np.full((H, W), 255, np.uint8)
    for r0 in range(0, H, TILE):
        for c0 in range(0, W, TILE):
            r1, c1 = min(r0 + TILE, H), min(c0 + TILE, W)
            a[r0:r1, c0:c1] = export(x0 + c0 * PIXEL, y1 - r1 * PIXEL, x0 + c1 * PIXEL, y1 - r0 * PIXEL)[:r1 - r0, :c1 - c0]
    src_tr = from_origin(x0, y1, PIXEL, PIXEL)
    bx0, by0, bx1, by1 = Transformer.from_crs(3035, 2180, always_xy=True).transform_bounds(x0, y0, x1, y1)
    dw, dh = int((bx1 - bx0) / PIXEL), int((by1 - by0) / PIXEL)
    dst_tr = from_origin(bx0, by1, PIXEL, PIXEL)
    out = np.full((dh, dw), 255, np.uint8)
    reproject(a, out, src_transform=src_tr, src_crs="EPSG:3035", dst_transform=dst_tr, dst_crs="EPSG:2180",
              resampling=Resampling.nearest, src_nodata=255, dst_nodata=255)
    with rasterio.open(OUT, "w", driver="GTiff", width=dw, height=dh, count=1, dtype="uint8", crs="EPSG:2180",
                       transform=dst_tr, nodata=255, compress="deflate") as dst:
        dst.write(out, 1)
    v = out[out <= 100]
    print(f"Saved {OUT}: {dw} x {dh} px; mean density {v.mean():.1f}%, pixels >= 30 %: {(v >= 30).mean():.1%}")


if __name__ == "__main__":
    main()
