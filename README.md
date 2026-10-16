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
price_drivers.py      ->  output/price_drivers.gpkg (layer building_drivers: features, out-of-fold prediction,
                          resid_pct = cheaper/dearer than expected), output/analysis/price_drivers_coefficients.csv,
                          16_price_drivers.png (stage 2: what explains value_2025 across buildings)
walk_amenities.py     ->  output/walk_amenities.gpkg: POIs within a 7-minute walk (560 m on footpaths) per building:
                          food & drink, Żabka, grocery, schools, health, sport & play (bounded Dijkstra per building)
reputation_time.py    ->  output/reputation_time.gpkg + summary/14: within-building resale change 2010-14 -> 2021-25
                          vs the city (no-primary buildings), by hexagon / MSI neighbourhood / district
rcn_clean.py              shared RCN cleaning (build_viz.py, analyze_rcn.py -> price_index.py): drops sales of a share of a
                          flat (nier_udzial 1/2 ...) and flats carrying a multi-flat deed's total price (7.7K records)
fetch_building_age.py ->  output/building_age.gpkg: construction decade per building from GIS Expert's public map
                          (mapy.gis-expert.pl/zabudowa-warszawy; tiles cached in data/raw/, not redistributed; credit them)
building_age_story.py ->  summary/18: price by construction decade, new-vs-prefab premium by place
infill.py             ->  output/infill.gpkg + summary/21: 2001-20 buildings among pre-1961 ones (>= 60 % of neighbours within
                          150 m) vs new estates; premium over old neighbours, model residual
walkups.py            ->  summary/22: within-building price by floor for walk-ups (pre-war 3-5 storeys, 1961-90 4-5
                          storeys) vs modern buildings with lifts (2001-20, 4-8 storeys); resales only
market_rhythms.py     ->  summary/19: seasonality, district momentum, company buyers
trees/                    tree cover density (Copernicus 2018) map + per-building features, see trees/README.md
flips.py              ->  output/flips.gpkg + summary/16: flats resold within 14-365 days. Flat = land-register id
                          LOK_ID_LOKALU (attached from raw RCN; 99 % of sales since 2018), before 2018 address+floor+area
                          with resale->resale pairs only (identical new-build flats share that key: false flips)
estates.py            ->  output/analysis/estates_ranking.csv + summary/15: MSI neighbourhoods ranked by mean model
                          residual, empirical-Bayes shrunk (data/msi.gpkg = city WFS layer MSI, 143 areas)
summary_maps.py       ->  output/summary/06, 08, 09: reputation (model residual hex map + by district), cafés
                          (density + price by venue band), noise (EEA road/rail Lden + price by rail-noise band);
                          bars = price net of district and all other features (OLS residual)
summary_advanced.py   ->  output/summary/10-13: district price gap decomposed into feature groups + reputation;
                          bivariate transit x price hex map; polar price compass around PKiN; partial dependence
                          + ICE curves ("shape of value")
summary_charts.py     ->  output/summary/01-05 PNG (4:5 portrait, 1200x1500, carousel-ready): prices over time,
                          districts, what drives price, transit, data blind spot (reads saved outputs only)
transit_story.py      ->  output/analysis/17_transit_price.png (price by departures band: new vs older buildings,
                          older vs their district average)
walk_access.py        ->  output/walk_access.gpkg (network walking distance to metro, tram stop, school, park >= 2 ha;
                          TransitFrequency's Warsaw walking graph loaded read-only, multi-source Dijkstra)
fetch_noise.py        ->  data/noise_{road,rail,air}_lden.tif (EEA END round-4 noise contours, 10 m, EPSG:3035)
fetch_pois.py         ->  data/osm_pois.gpkg (OSM food & drink, shops, schools, health, culture, leisure;
                          via D:/QGIS/osm_basemap/overpass_polite.py, imported read-only)
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

### 3D price renders (`3d/`, Blender)

Self-contained copy of the Blender pipeline from `D:/QGIS/TransitFrequency/3d` (that project is only
read, never edited): same heights, dark ground and cameras, so price renders pair with the transit ones.

```
python 3d/export_scene.py --name prices_wide --bounds=-7000,-15000,15000,9000 --detail-bounds=-7000,-6500,7000,9000
blender --background --factory-startup --python 3d/build_blend.py -- --name prices_wide --view center
blender --background 3d/scenes/prices_wide/scene.blend --python 3d/render.py -- --view mokotow --out 3d/renders/x.png
```
(`blender` = `C:/Program Files/Blender Foundation/Blender 5.2/blender.exe`.)
- `export_scene.py` (standalone): OSM buildings + parts from `D:/QGIS/osm_basemap/warsaw`, ground layers from
  `D:/QGIS/mapy_warszawy_misc/data/osm`, transit-poster colours hardcoded. Colour = `--set price_m2`:
  hedonic `value_2025` (`price_model.gpkg` / `building_value`) for buildings with >= 3 sales, 11 classes
  (breaks 12k..28k PLN/m²), viridis (dark blue = cheap, yellow = dear; deliberately not the transit palette).
  No value = dark grey. Parts take the parent's value. Origin = PKiN, so `3d/views/*.json` fit.
- `build_blend.py`: copy, plus a fix: degenerate/duplicate faces are dropped before the mesh is made
  (`me.validate()` removed them afterwards and shifted every per-face colour: "internal error setting the array").
- `render.py`, `save_view.py`, `views/{center,mokotow,zoliborz}.json`: copies. Scenes + renders are gitignored.

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

**2026-10-16 (evening):** district anatomy (fig. 10) splits building age from building form. New: fig. 20 trees
(Copernicus density map; only 30 %+ tree cover within 100 m adds ~+2 %), fig. 21 infill (388 new buildings among
pre-1961 ones sell +32 % over the old buildings next door, model residual +1.6 % vs -0.8 % for new estates; smallest
premium in Praga-Północ +17 %, largest Mokotów/Śródmieście +45 %), fig. 22 walk-ups (top floor of a 4-5-storey block
without a lift -3 % vs the 1st floor, +2.4 % with a lift; ground floors -3 % in old blocks, 0 in new ones).

**2026-10-16 (later):** data cleaning + building age + trees. Shared cleaning removes 2,377 partial-share sales and
5,367 multi-flat deed totals; price_index.py now trims sales > 4 SD from their building's prediction (2,377, 0.9 %)
and refits: within-R² 0.647 -> 0.786. Building age (GIS Expert decades, 92 % of priced buildings) in the model:
out-of-sample R² 0.48 -> 0.58. Within a district vs 1961-90 prefab: 1991-2000 +11 %, 2001-10 +17 %, 2011-20 +17 %,
1941-60 -2 %, pre-war +1 % (n.s.); metro -3.4 %/km walked; bus/tram frequency ~0 once age is in; tree cover within
100 m ~0 beyond parks. Reputation survives age: Żoliborz +15 %, Praga-Północ -19 %. Fig. 18: U-shape by decade
(pre-war +5 %, prefab -7..-11 %, 2011-20 +14 % vs district median); new-vs-prefab premium largest in the centre
(Śródmieście +97 %) and smallest on the edges (+17 %). Fig. 19: seasonality within +-1 %; district leads partly
reverse next year; companies buy ~3 % of flats. Flips: flipped flats that sell again keep the markup (+4.7 % vs the
flipper's price, net of market, 538 flats). Repeat-sales index dropped: since 2021 the same-flat resales are mostly
flips (median gap 248 days), which measures renovation, not the market. Meta canopy height rejected (see trees/).

**2026-10-16 (resumed):** walking-network amenities replace 500 m circles (cafés +3.4 %/doubling within a district,
unchanged -> robust; Żabka no effect; more grocery shops slightly negative = dense estates). Walking searches now
snap only to the main connected component (0.6 % of graph nodes were isolated fragments: 125 buildings reached
nothing). Reputation over time (fig. 14): central areas pull away (Praga-Północ +9.8 %, Wola +8.3 % vs city),
outskirts lose (Ursynów -6.8 %, Ursus -6.3 %); no general catch-up of low-reputation areas (corr +0.04), but the two
Pragas gain most. Estates (fig. 15): Nowa Praga -22 %, Stara Praga -16 %, Muranów -11 %; Stary Żoliborz +23 %,
Służewiec +21 %, Wyględów +20 %. EEA "rail" noise includes trams.
Flippers (fig. 16): 2021-25 about 5.8 flips per 100 resales (2-3 a decade earlier), median markup +30 % (+25 % net of
the market) after ~150 days; flip share correlates +0.57 with a neighbourhood's gain on the city (fig. 14). Building age: not openly available (city WMS
GEODEZJA_BUDYNKI / budynki return no attributes); options = BGiK EGiB request, ask gis-expert, historical orthophotos.

**Paused 2026-10-16.** Wrap-up: `summary_charts.py` -> `output/summary/` (five portrait charts telling the story:
index 2.2x since 2013 but real prices back at the 2008 peak; district ranking of a typical flat; within-district
drivers; transit premium only for new buildings; one flat in five invisible to RCN). Maps & advanced figures
06-13 (`summary_maps.py`, `summary_advanced.py`): reputation residual (Żoliborz +15 %, Praga-Północ -18 % vs
the feature-based prediction), district anatomy, transit x price bivariate map, price compass (4-7 km out: south
16.5k vs east 13.8k, -17 %), shape of value (café effect is a threshold at ~25-40 venues within 500 m; metro
effect beyond ~2.5 km on foot). Building age dropped (not openly available). Possible next steps: residual ("cheaper than expected") map, floor-by-floor Blender
slabs, LinkedIn post.

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
- Stage-2 model (`price_drivers.py`, 2026-10-16): 12.7K buildings (>= 3 sales, transit matched by osm_id or nearest
  footprint <= 25 m). Features: transit departures, **network walking distances** (metro, tram, park, school),
  façade noise (EEA road/rail/air Lden, loudest band within 10 m), BDOT heritage (ZABYTEK), centrality, river,
  green area, POIs, building form, developer-sale share. **47.5 % out of sample** (spatial 5-fold CV on 2 km
  blocks) vs 29 % for district alone; OLS + district 52 %. Within a district: new building (developer share)
  +32 %, cafés/bars x2 +3.5 %, -2.1 % per km walked to metro, tram & train noise -2.4 %/10 dB (EEA "rail" includes trams), road noise only -0.5 %,
  green +3.4 %/10 ha, heritage ~0. `--index district` (district price indices instead of the city index) changes
  no coefficient by more than 0.1 pt except first-sale-2015+.
- Transit (`transit_story.py`, chart 17): no premium for bus/tram frequency in older buildings (within a district
  the best-served streets are 3-6 % cheaper: arterials, prefab estates built around bus loops); new
  developments rise +24 % from the least to the best served band (partly centrality). Only 18 % of priced
  buildings have <= 150 departures/day: sparse service mostly means houses, which are outside this sample.
- Matching: 99.97 % of in-city flat sales attach to a building (270,655 of 270,741); 26.8K fetched sales lie in
  neighbouring towns. Cooperative-right sales (spółdzielcze własnościowe prawo) are absent from RCN altogether;
  only flats converted to full ownership appear (the "thin" buildings of the coverage audit).
- Building age: not openly available (BDOT, city GeoServer, KIEG have none; api.um.warszawa.pl dropped it).
  Request to BGiK pending; API key for api.um.warszawa.pl in `.secrets/` (gitignored).
- 3D price renders (`3d/`): quantile classes (12.5k..21k), parts matched to the parent they overlap most
  (Cosmopolitan's podium roof went grey under a separate building=roof canopy), blocks of flats without
  enough sales get a desaturated neighbourhood estimate (median of priced buildings within 300 m, >= 5).
- Not yet done: charts 01–11 still use their original per-chart colours inside the shared theme.

## Notes

- All spatial data uses EPSG:2180 (Poland CS2000 zone 7)
- BDOT WFS requires GUGiK authentication; auto-download uses the same opendata URL pattern as the `pobieracz_danych_gugik` QGIS plugin
- Warsaw address WFS ignores `startIndex` and `bbox` params; workaround uses OGC filter by district + ID-based pagination
