# Warsaw Housing Price Analysis

Spatial analysis of residential real estate transactions in Warsaw, combining official transaction records with building footprints, address points, and OpenStreetMap data.

## Data Sources

| Source | Records | Auto-fetch | Description |
|--------|---------|------------|-------------|
| **RCN transactions** | ~381K | `fetch_rcn.py` | Closed residential transactions from GUGiK WFS (Rejestr Cen Nieruchomości) |
| **OSM buildings** | ~320K+ | `fetch_geodata.py` | Building footprints + `building:part` with 3D tags (height, levels, roof) via Overpass API |
| **BDOT buildings** | 155K | `fetch_geodata.py` | Official GUGiK BDOT10k building footprints (auto-downloaded from opendata.geoportal.gov.pl) |
| **Address points** | 127K | `fetch_geodata.py` | Warsaw city WFS address points, partitioned by district |

## Pipeline

```
fetch_rcn.py          ->  output/rcn_lokale_warszawa.gpkg
fetch_geodata.py      ->  data/osm_buildings.gpkg
                      ->  data/bdot_buildings.gpkg
                      ->  data/address_points.gpkg
build_viz.py          ->  output/warsaw_prices_viz.gpkg (buildings + prices)
                      ->  output/warsaw_prices_viz_transactions.gpkg
                      ->  output/warsaw_prices_viz_unmatched.gpkg
```

### Two-tier data architecture

`fetch_geodata.py` uses a two-tier approach:
1. **Fetch** full datasets to canonical folders on `D:/QGIS/` (reusable across projects)
2. **Extract** to `ceny_mieszkan/data/` for this project

| Dataset | Canonical location | Source |
|---------|-------------------|--------|
| OSM buildings | `D:/QGIS/osm_basemap/warsaw/buildings.gpkg` | Overpass API |
| BDOT buildings | `D:/QGIS/bdot_basemap/Warszawa/*.shp` | GUGiK opendata ZIP |
| Address points | `D:/QGIS/punkty_adresowe/output/punkty_adresowe_warszawa.gpkg` | Warsaw city WFS |

## Usage

```bash
# 1. Fetch transaction data
python fetch_rcn.py

# 2. Fetch/extract spatial reference data
python fetch_geodata.py                    # all sources
python fetch_geodata.py --only osm         # just OSM buildings
python fetch_geodata.py --only bdot        # just BDOT (downloads 78 MB ZIP from GUGiK)
python fetch_geodata.py --only addresses   # just address points
python fetch_geodata.py --force            # re-fetch even if cached
python fetch_geodata.py --extract-only     # skip fetching, just extract from D:/QGIS/

# 3. Build visualization
python build_viz.py
```

## Status

- RCN data: fetched (381K transactions, last refresh 2026-09-27)
- OSM buildings: re-fetching with `building:part` and full 3D tags
- BDOT buildings: auto-downloadable, but GUGiK server data is from Feb 2024
- Address points: 127K fetched via district-partitioned WFS (fixed broken pagination)
- Visualization: `build_viz.py` joins transactions to BDOT buildings; pending update to use OSM + address matching

## Notes

- All spatial data uses EPSG:2180 (Poland CS2000 zone 7)
- BDOT WFS requires GUGiK authentication; auto-download uses the same opendata URL pattern as the `pobieracz_danych_gugik` QGIS plugin
- Warsaw address WFS ignores `startIndex` and `bbox` params; workaround uses OGC filter by district + ID-based pagination
