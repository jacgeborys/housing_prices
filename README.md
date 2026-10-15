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
build_viz.py          ->  output/warsaw_prices_viz.gpkg
                            layer: buildings_all (320K polygons, all with height_m)
                            layer: buildings_prices (26K with median price/m² stats)
                            layer: transactions (297K points + match status)
                            layer: buildings_3d (327K unified: simple + parts, for 3D)
analyze_rcn.py        ->  output/analysis/01-11 PNG charts
                            price distribution, rooms/area, floors, districts,
                            market segments, outliers, micro-apartments,
                            apartments vs houses, price trends, value analysis,
                            repeat sales (52K resale pairs)
price_index.py        ->  output/price_model.gpkg
                            layer: transactions_adj (258K apartment sales + price in 2025 PLN)
                            layer: building_value (20.8K buildings, hedonic "typical flat" value)
                      ->  output/analysis/price_index_quarterly.csv, district_index_annual.csv,
                          model_coefficients.csv
                      ->  output/analysis/12-15 PNG charts
                            quality-adjusted index (nominal + real), adjusted district
                            ranking + district indices, floor premium, size & new-build effects
make_map.py           ->  output/analysis/map_01-06, map_08 PNG maps
                            city overview, district choropleth, center zoom,
                            price/density hexbin, apartment sizes, growth vs city,
                            building-level hedonic value
coverage.py           ->  output/coverage.gpkg (layer building_coverage: 22.9K multi-family buildings,
                          est. flats, recorded sales, coverage class none/thin/ok)
                      ->  output/analysis/coverage_by_district.csv, map_09_coverage.png
make_3d.py            ->  output/analysis/map_07_3d_center.png (static, 900 m around PKiN)
                      ->  output/web/warsaw_3d.html (interactive deck.gl, 3 km around PKiN)
style.py                  shared chart theme, palettes, GUS CPI series
web/warsaw_3d_template.html   HTML/JS template the 3D scene data is injected into
```

Run order after fetching: `build_viz.py` -> `price_index.py` -> `analyze_rcn.py`, `make_map.py`, `make_3d.py`.
`price_index.py` caches its input in `output/cache/model_input.pkl`; delete it after refreshing RCN data.

### Hedonic model (`price_index.py`)

`log(price/m²) = building FE + quarter FE + area bin + rooms + floor bin × height class + top floor × height class + primary × era`,
estimated by within-building demeaning, SEs clustered by building (transactions outside a footprint use a 100 m grid cell).
Within-R² 0.65, residual SD 0.18. Quarters with < 300 sales (the 2017–20 RCN gap) are not estimated; the index is
interpolated across them. Building values are empirical-Bayes shrunk toward the district mean. Real prices use
GUS annual CPI (hardcoded in `style.py`; 2025–26 values are estimates — update when GUS publishes).

### 3D scenes (`make_3d.py`)

RCN floor numbers start at 1 = ground floor. Every building:part inherits the floor grid of its parent footprint
(so floors are counted from the ground), storey height = height / max(height / 3 m, highest floor sold).
Each storey with sales is coloured by the median time-adjusted price (2025 PLN). The web scene uses no tile
server: streets, rail, parks and the Vistula come from `D:/QGIS/osm_basemap/warsaw/*.gpkg`.

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
- OSM buildings: 329K features (320K main + 9.3K building:part with full 3D tags)
- BDOT buildings: 155K auto-downloadable (GUGiK server data from Feb 2024)
- Address points: 127K fetched via district-partitioned WFS (fixed broken pagination)
- Visualization: `build_viz.py` matches 94.9% of residential transactions to OSM buildings (spatial + buffer + address), BDOT fallback for remainder. 26.4K buildings enriched with price stats.
- Analysis: 15 chart sets + 8 maps. All charts share `style.py` (magma-based price ramp, blue↔red diverging for change).
- Key findings (hedonic model, 2026-09-29):
  - A constant-quality flat costs 2.2× its 2013 price nominally, but in real terms is only back to the 2008 bubble peak (~17K PLN/m² in 2025 PLN).
  - Pooled raw medians mis-rank districts: Włochy looks #2 but is #9 once date/flat mix is controlled; Wilanów and Żoliborz rise sharply (Żoliborz #2 at ~19.5K).
  - Within the same building: ground floors −2% vs floors 2–3; tower floors 18+ ≈ +20%; being the top floor has no measurable effect.
  - Studios <25 m² carry +21%/m² vs 45–60 m²; developer (primary) deeds record 18–23% below resales in the same building (deed price agreed at preliminary contract for an unfinished flat).
- 3D: static render + interactive deck.gl scene (13.6K priced storeys in ~3K buildings within 3 km of PKiN).
- District boundaries: official BDOT ADJA_A (`data/districts.gpkg`).
- Fixed 2026-09-29: district ranking mixed eras; "YoY +43%" was a 5-year change across the gap; map_06 change panel was a copy of panel 2; 3D floors were off by one and parts counted from their own base; `build_viz.py` market mapping never matched (`pct_primary` always 0 — re-run `build_viz.py` to refresh that column).
- Coverage audit (`coverage.py`, 2026-10-15): multi-family universe = BDOT10k "3+ flats" or OSM `apartments`
  (or >= 3 recorded sales), inside the city boundary. Flats estimated from footprint x storeys, calibrated on
  1,064 new builds (primary sales ~ all flats): ~127 m² GFA per flat (high: probably not every developer sale
  reaches RCN, so absolute flat counts are low; use them relatively). Results: **19 % of flats sit in buildings with
  no recorded sale, 14 % thin**. Why: RCN holds **no cooperative-right sales** (NIER_PRAWO has only full
  ownership), city-owned/rental blocks rarely sell. Ursynów 23 % none + 24 % thin; Praga-Północ 38 % none;
  Wesoła 79 % none (only 524 sales in 20 years: probably under-recording, unexplained). Matching failures inside
  the city are negligible (86); the ~15K "unmatched" sales lie outside Warsaw (fetch bbox pulls neighbours).
- Plan (2026-10-15): prices in 3D with the Blender pipeline of `D:/QGIS/TransitFrequency/3d`; then a building
  feature table (transit, walk isochrones to parks/schools/metro, age, noise, OSM POIs) and a stage-2 model on
  `value_2025` -> residual "cheaper/dearer than expected" map. Only `coverage == ok` buildings get coloured.
  Leads: Warsaw open-data API (api.um.warszawa.pl, needs a key) had building years; city WFS has
  `ZIELEN_DRZEWA` (tree inventory) and `OBWODY_SZKOL_PODSTAWOWYCH` (school catchments).
- Not yet done: charts 01–11 still use their original per-chart colours inside the shared theme.

## Notes

- All spatial data uses EPSG:2180 (Poland CS2000 zone 7)
- BDOT WFS requires GUGiK authentication; auto-download uses the same opendata URL pattern as the `pobieracz_danych_gugik` QGIS plugin
- Warsaw address WFS ignores `startIndex` and `bbox` params; workaround uses OGC filter by district + ID-based pagination
