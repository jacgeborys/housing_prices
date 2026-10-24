# Trees

Tree cover around Warsaw's buildings, kept separate from the price pipeline.

| Script | Output |
|---|---|
| `fetch_tree_density.py` | `data/tree_density_warsaw_10m.tif`: Copernicus HRL Tree Cover Density 2018 (10 m, summer Sentinel-2), % per pixel, EPSG:2180 |
| `render_canopy.py` | `output/canopy_warsaw.png` (whole city, 300 dpi), `output/canopy_centre.png` (5 x 5 km around PKiN): denser cover = deeper green |
| `tree_features.py` | `output/tree_features.gpkg`: per building, mean tree cover within 50 / 100 m (straight-line rings, footprint excluded) and share of dense (>= 50 %) cover; used by `price_drivers.py` |
| `lidar_features.py` | `data/lidar_canopy_warsaw_1m.tif` (whole city, 1 m, canopy height in m, 0 = none) + `output/tree_features_lidar.gpkg` (share of the 50 / 100 m ring under canopy, share under canopy >= 15 m). Canopy = GUGiK 2025 LiDAR from `D:/QGIS/lidar/trees` (read only: its `fetch_lidar.py` tiles, cleaned by its `render_zoom.clean_canopy`, the same canopy as its Warsaw maps). Used by `price_drivers.py --trees lidar` |

Tried and rejected: Meta & WRI 1 m canopy height (Tolan et al. 2024). Its imagery for Warsaw's tile reads
deciduous city trees as a few metres tall or absent (Ogród Saski: median 4 m); only conifer forests come out right.
The downloaded tiles (`data/chm_*.tif`, ~1 GB) are not used.

Result (2026-10-16): tree crowns cover ~26 % of Warsaw; tree cover within 100 m of a building adds nothing to its
price beyond what parks already explain (+0.3 % per 10 points, not significant).

LiDAR check (2026-10-24, `price_drivers.py --trees lidar`, separate `*_lidar` outputs): 2025 LiDAR finds about twice
the cover around buildings (median 24 % within 100 m vs 11 % from Copernicus; it sees street trees and single crowns),
but ranks buildings much the same (Spearman 0.83). The price association does not change: within the same district
+0.7 % per 10 points (95 % CI -0.2..+1.6) vs +0.9 % (-0.1..+1.8); per interquartile step both about +1 %. Out-of-sample
R² 0.578 vs 0.574, no other coefficient moves by more than 0.5 points. Copernicus stays the default.
