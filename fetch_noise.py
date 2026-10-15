"""
Strategic noise maps (EU Environmental Noise Directive, round 4 / 2022, EEA update April 2026) for Warsaw:
road, rail and air Lden as 10 m rasters -> data/noise_<source>_lden.tif (EPSG:3035).

Pixel values are EEA noise bands: 1 = 55-59 dB, 2 = 60-64, 3 = 65-69, 4 = 70-74, 5 = >= 75; 0 = below 55
or not mapped. Source: EEA Copernicus ImageServers noiseStoryMap/NoiseContours_<source>_lden.
"""
import sys

import numpy as np
import rasterio
import requests
from pyproj import Transformer
from rasterio.transform import from_origin

BASE = "https://copernicus.discomap.eea.europa.eu/arcgis/rest/services/noiseStoryMap"
SOURCES = ["road", "rail", "air"]
WARSAW_LONLAT = (20.84, 52.09, 21.28, 52.37)     # city bbox with a small margin
PIXEL = 10
TILE = 3000                                      # px per request side (server max 15000 x 4100)
DB = {1: 57, 2: 62, 3: 67, 4: 72, 5: 77}         # band midpoints, dB


def export(source, x0, y0, x1, y1):
    w, h = int((x1 - x0) / PIXEL), int((y1 - y0) / PIXEL)
    r = requests.get(f"{BASE}/NoiseContours_{source}_lden/ImageServer/exportImage", params={
        "bbox": f"{x0},{y0},{x1},{y1}", "bboxSR": 3035, "imageSR": 3035, "size": f"{w},{h}",
        "format": "tiff", "pixelType": "U8", "noData": 0, "interpolation": "RSP_NearestNeighbor",
        "f": "image"}, timeout=300)
    r.raise_for_status()
    if not r.headers.get("content-type", "").startswith("image"):
        sys.exit(f"{source}: unexpected response {r.text[:300]}")
    with rasterio.MemoryFile(r.content) as mf, mf.open() as src:
        return src.read(1)


def main():
    t = Transformer.from_crs(4326, 3035, always_xy=True)
    x0, y0, x1, y1 = t.transform_bounds(*WARSAW_LONLAT)
    x0, y0 = np.floor(x0 / PIXEL) * PIXEL, np.floor(y0 / PIXEL) * PIXEL
    x1, y1 = np.ceil(x1 / PIXEL) * PIXEL, np.ceil(y1 / PIXEL) * PIXEL
    W, H = int((x1 - x0) / PIXEL), int((y1 - y0) / PIXEL)
    for source in SOURCES:
        full = np.zeros((H, W), np.uint8)
        for r0 in range(0, H, TILE):
            for c0 in range(0, W, TILE):
                r1, c1 = min(r0 + TILE, H), min(c0 + TILE, W)
                tile = export(source, x0 + c0 * PIXEL, y1 - r1 * PIXEL, x0 + c1 * PIXEL, y1 - r0 * PIXEL)
                full[r0:r1, c0:c1] = tile[: r1 - r0, : c1 - c0]
        out = f"data/noise_{source}_lden.tif"
        with rasterio.open(out, "w", driver="GTiff", width=W, height=H, count=1, dtype="uint8", crs="EPSG:3035",
                           transform=from_origin(x0, y1, PIXEL, PIXEL), nodata=0, compress="deflate") as dst:
            dst.write(full, 1)
        vals, n = np.unique(full, return_counts=True)
        print(f"{out}: {W}x{H} px, bands " + ", ".join(f"{v}:{c / full.size:.1%}" for v, c in zip(vals, n)))


if __name__ == "__main__":
    main()
