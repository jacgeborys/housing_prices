# Trees

Tree cover around Warsaw's buildings, kept separate from the price pipeline.

| Script | Output |
|---|---|
| `fetch_tree_density.py` | `data/tree_density_warsaw_10m.tif`: Copernicus HRL Tree Cover Density 2018 (10 m, summer Sentinel-2), % per pixel, EPSG:2180 |
| `render_canopy.py` | `output/canopy_warsaw.png` (whole city, 300 dpi), `output/canopy_centre.png` (5 x 5 km around PKiN): denser cover = deeper green |
| `tree_features.py` | `output/tree_features.gpkg`: per building, mean tree cover within 50 / 100 m (straight-line rings, footprint excluded) and share of dense (>= 50 %) cover; used by `price_drivers.py` |

Tried and rejected: Meta & WRI 1 m canopy height (Tolan et al. 2024). Its imagery for Warsaw's tile reads
deciduous city trees as a few metres tall or absent (Ogród Saski: median 4 m); only conifer forests come out right.
The downloaded tiles (`data/chm_*.tif`, ~1 GB) are not used.

Result (2026-10-16): tree crowns cover ~26 % of Warsaw; tree cover within 100 m of a building adds nothing to its
price beyond what parks already explain (+0.3 % per 10 points, not significant).
