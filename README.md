# Warsaw Housing Price Analysis

Spatial analysis of residential real estate transactions in Warsaw, combining official transaction records with building footprints, address points, and OpenStreetMap data.

## Overview (2026-10-17)

Twenty years of Warsaw flat sales (RCN, ~270K in-city flat deeds) attached to buildings, turned into a
constant-quality price per building (`price_index.py`), then explained by location and building features
(`price_drivers.py`). What is left over, the price a place earns beyond what its features predict, is read as
**reputation**. Developer price lists (`offers/`) add what is on sale now; `access/` adds door-to-door transit
travel times (r5py). All findings are associations within a district, not causal effects; ranges come from a
2 km spatial block bootstrap where given.

**Headline findings**
- A constant-quality flat costs 2.2x its 2013 price, but in real terms is only back at the 2008 peak (fig. 01).
- Building age matters most: within the same district vs 1961-90 prefab, 1991-2000 +16 %, 2001-10 +26 %, 2011-20
  +31 %, 2021+ +30 %, pre-war +4.5 % (fig. 03, 18); infill among pre-1961 blocks (420 buildings, now incl. 2021+)
  sells +33 % over its old neighbours (fig. 21).
- Location: -2..-5 % per km walked to the metro; reach to cafés, restaurants and culture +4..+9 % per doubling,
  about as informative as distance to PKiN + metro (fig. 04, 08, 25). Bus/tram frequency adds ~nothing once age
  is known (fig. 05); tram and train noise costs ~2 % per 10 dB, road noise much less (fig. 09).
- Reputation: Żoliborz about +13 %, Praga-Północ about -18 % vs the prediction (fig. 06), unchanged when café and
  amenity counts are dropped from the model; central areas are pulling
  away, the two Pragas gain most (fig. 14); estate-level extremes in fig. 15.
- Market behaviour: flips are ~6 per 100 resales since 2021 with a +25 % net markup (fig. 16); new flats on offer ask
  about what a 5-15-year-old flat nearby sells for (fig. 26); 68 % of flats on offer never change price in a year,
  half are still listed after 12 months (fig. 27); new flats shrank from 60 to 47 m² (fig. 29).
- Forward test (fig. 30): reputation measured on 2010-14 resales predicts 2021-25 reputation (r = +0.91 across 43
  neighbourhoods) but not the price change (r = -0.11): stable, so already priced in. What moved the same buildings
  vs Warsaw: pre-war +6 %, 1941-60 +2.5 %, 1991-2010 -5.5 % (the new-build premium wears off), new M2 extension
  station within 800 m +2.9 %; cafés, trees, parks, tram stops ~0.
- Blind spot: 19 % of flats are in buildings with no recorded sale, mostly cooperative-right flats that RCN
  does not record (fig. 23).

**The report** (`report/`, the deliverable): 27 slides in story order -> `output/report/NN_*.png` +
`output/report/warsaw_flats.pdf`. `python report/build_report.py` assembles it from built figures; `--rebuild` reruns
the figure scripts first (pipeline order, ~1 h). Principles: one question per slide, title = the answer, at most two
panels, describe before explaining, no model internals on slides (they go to the methods page).
- Opening: 31 price ridgelines, 32 every priced building as a dot
- 1 What's in the data: r01 sales per year / district (new, resale), 23 blind spot
- 2 Sizes: r02 size categories by district + share over time, r03 new vs resale (price as paid, sizes), 29 smaller flats
- 3 Prices: 01 over time, 33 what 1 million buys, 34 room for a family, 35 house or flat
- 4 What goes with price: 03, 18 age, 21 infill, 08 cafés, 09 noise, 20 trees, 11 transit x price, 12 compass, 24 jobs
- 5 Reputation: 10 district anatomy, 06 reputation map, 15 estates, 14 over time
- 6 Market: 16 flippers, 26 new flats on offer; end: r99 how to read these charts (yardstick flat, sources, limits)
Not in the report (kept in output/summary/): 02, 04, 05, 13, 19, 22, 25, 27, 28, 30, 36.
`report/descriptive.py` builds r01-r03 and r99 (plain counts, shares and prices as paid; no model).

**Figures** (`output/summary/`, contact sheet in `00_contact_sheet.png`, built by `contact_sheet.py`)

| # | Figure | Script |
|---|--------|--------|
| 01-03 | prices over time, districts, what drives price | `summary_charts.py` |
| 04 | metro / rail / tram / bus walking distance | `transit_modes.py` |
| 05 | transit frequency | `frequency_story.py` |
| 06, 08, 09, 20 | reputation map, cafés, noise, trees | `summary_maps.py` |
| 10-13 | district anatomy, transit x price, price compass, shape of value | `summary_advanced.py` |
| 14 | reputation over time | `reputation_time.py` |
| 15 | estates | `estates.py` |
| 16 | flippers | `flips.py` |
| 18 | building age | `building_age_story.py` |
| 19 | market rhythms | `market_rhythms.py` |
| 21 | infill | `infill.py` |
| 22 | walk-ups and lifts | `walkups.py` |
| 23 | blind spot (coverage) | `summary_charts.py` |
| 24-25 | transit access, what reach pays | `access/access_preview.py`, `access/gravity_story.py` |
| 26-28 | new flats on offer, asking prices over time, projects map | `offers/offers_story.py`, `offers/history_story.py`, `offers/projects_map.py` |
| 29 | flat sizes and rooms | `flat_sizes.py` |
| 30 | forward test: reputation 2010-14 vs 2021-25, what moved prices | `forward_test.py` |
| 31 | opening map: price ridgelines (dark, "Unknown Pleasures") | `price_ridgelines.py` |
| 32 | companion to 31: every building with >= 3 sales as a dot (price colour, size = sales), 2400x3000 | `price_points.py` |
| 33-34 | what 1 million PLN buys (size of 0.9-1.1M resales 2024-26); room for a family (share of 70 m²+ resales, price of a 65-80 m² flat) | `family_flats.py` |
| 36 | where the big flats are: median size and share of 4+ room flats, all sales since 2016, top MSI neighbourhoods | `family_flats.py` |
| 35 | house or flat: house sales (RCN budynki layer) vs flat resales by district, per m² and in total | `houses.py` |

Rerun order after new data: `fetch_rcn.py` / `fetch_geodata.py` -> `price_index.py` -> `walk_access.py` ->
`price_drivers.py` -> figure scripts -> `contact_sheet.py`. Subprojects have their own READMEs or headers:
`access/` (Python 3.12 venv, r5py, portable JDK), `offers/` (dane.gov.pl price lists), `trees/`, `3d/` (Blender).

**Data sources added since the first version:** BDOT10k June 2025 buildings
(`D:/QGIS/bdot_basemap/Warszawa/*OT_BUBD_A.gpkg`; heritage flag ZABYTEK only exists in the Feb 2024 shapefile at
`D:/QGIS/mapy_warszawy_misc/BDOT10k/`), GIS Expert building decades, EEA noise (Lden), Copernicus tree cover,
Geofabrik mazowieckie + WTP/KM/WKD GTFS, dane.gov.pl developer price lists.

## Data Sources

| Source | Records | Auto-fetch | Description |
|--------|---------|------------|-------------|
| **RCN transactions** | ~381K | `fetch_rcn.py` | Closed residential transactions from GUGiK WFS (Rejestr Cen Nieruchomości) |
| **OSM buildings** | ~320K+ | `fetch_geodata.py` | Building footprints + `building:part` with 3D tags (height, levels, roof) via Overpass API |
| **BDOT buildings** | 167K | `fetch_geodata.py` | GUGiK BDOT10k building footprints, June 2025 gpkg from `D:/QGIS/bdot_basemap` (falls back to the opendata download) |
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
frequency_story.py    ->  summary/05: TransitFrequency departures within a 5-min walk vs price, all else equal (district
                          and neighbourhood fixed effects) + raw medians new vs older; blind spot moved to summary/23
transit_modes.py      ->  summary/04: price by walking distance to metro / rail / tram / bus, within the same MSI
                          neighbourhood (fixed effects), all else equal; metro and rail only in neighbourhoods with a station
flat_sizes.py         ->  summary/29: median size of flats sold by year (new vs resale), rooms in new flats, price per
                          m² by size (hedonic coefficients), map of median resold-flat size 2021-26
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
  (fixed breaks 12K..40K PLN/m²), `PRICE_RAMP` (indigo = cheap ... yellow ~19K ... magenta = 40K+).
  No value = light clay grey (`NODATA_RGB`). Parts take the parent's value. Origin = PKiN, so `3d/views/*.json` fit.
- `build_blend.py`: copy, plus a fix: degenerate/duplicate faces are dropped before the mesh is made
  (`me.validate()` removed them afterwards and shifted every per-face colour: "internal error setting the array").
- `--set reputation`: `resid_pct` from `price_drivers.gpkg` (price vs model prediction), fixed breaks, diverging ramp:
  `python 3d/export_scene.py --name reputation_wide --set reputation --bounds=... --detail-bounds=...` (same bounds).
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
| BDOT buildings | `D:/QGIS/bdot_basemap/Warszawa/*OT_BUBD_A.gpkg` (June 2025) | GUGiK opendata ZIP |
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

**2026-10-18 (report overhaul):** one deck instead of a growing pile of figures (user: newer charts "obscure what
the dataset holds"). New `report/` folder: build_report.py (slide list, --rebuild pipeline, PDF) and descriptive.py
(r01 data overview, r02 sizes by district and over time = clean redo of analysis/07, r03 new vs resale = redo of
analysis/05, r99 methods page). Trimmed: 24 (evening map and "price model fit" panel dropped, minutes to the centre by
district added), 29 (rooms panel and map dropped), 35 (map dropped), 10 subtitle ("what goes with it", not "causes").
linkedin_pdf.py removed (superseded by report/build_report.py). In the register 2026 holds only deeds registered by
late September (months of lag), not a full year.

**2026-10-18 (buyer's figures):** `family_flats.py` -> summary/33 (1 million PLN buys 49-50 m² in Żoliborz,
Śródmieście, Wola and 74 m² in Wawer; 3,955 resales at 0.9-1.1M 2024-26) and summary/34 (17 % of resales since 2021
are 70 m²+; a 65-80 m² flat costs 0.80M in Rembertów to 1.63M in Żoliborz). Actual sales, resales only (developer
deeds carry older prices). RCN rooms are "izby" (a separate kitchen counts, a kitchen annex does not): fig. 29's room
panel now says so, and size in m² is used for families. Figures dropped from posting by the user: 22 (lift status not
known per building), 02 (needs the benchmark explained). Fig. 36 (successor of analysis/map_05, cleaned, in-city): biggest
flats in Las (Wawer, terraced homes sold as flats, 100 m²), Nadwiśle, Zerzeń, Wyględów, Stegny; many rooms is partly a
prefab layout signal (Piaski: 40 % of flats 4+ izby at a median 47 m²). House sales: `fetch_rcn.py --layer budynki` -> output/rcn_budynki_warszawa.gpkg (386K rows: every transaction
touching a building, mostly flats again; TERYT 1465 = Warsaw, others neighbours; no date field, year from the deed
number). `houses.py` -> output/houses.gpkg + summary/35: single-house deeds (built-up land, 1/1, free market, one
building per deed), 2,272 in 2023-26; usable area recorded in only ~1 deed in 3 (where present ~1.6x footprint), so
per-m² uses those 717, totals use all. Result: in all six districts with enough sales a house costs less per m² than
a resold flat, plot included (Wawer 9.4K vs 13.3K, Białołęka 8.3K vs 12.9K), but a median house costs ~2x a median
flat in total (Wawer 1.35M, 149 m², 360 m² plot). Houses only recorded from 2017; 2021-22 nearly empty. The old
analysis/08 "houses" chart was a heuristic on the flats layer: superseded.

**2026-10-18 (age check):** GIS Expert's map (~2020) has no 2021+ layer; undated buildings fell into the
reference class (1961-90 prefab). `age_check.py` -> output/age_check.gpkg (flag per building) +
output/analysis/age_check_centre.png. Priced buildings: 935 undated with developer sales 2019+ (likely 2021+), 185
undated otherwise, 283 dated <= 1990 with developer sales 2016+ (renovated tenements / attic conversions, or stale
map). Test refit with the two undated groups as own classes: 2011-20 premium +17.5 -> +30.7 %, 2001-10 +18.3 ->
+25.6 %, pre-war +1.2 -> +4.7 %, developer share +20 -> +13 %; metro and cafés unchanged. User checked the centre by eye:
yellow flags right; red mostly genuinely old (kept as dated) except Rotunda na Dynasach, Moderna Powiśle
(Zajęcza 1A), Koszykowa 49A (age_check.OVERRIDES -> 2021+). Applied: age_check.gpkg holds the corrected decade
(2021+ = 2030, unknown = -1, map value kept as decade_gisexpert) and replaces building_age.gpkg in price_drivers,
building_age_story, infill (now 2001+ incl. 2021+), walkups, offers_story; new dummies age_2021_plus, age_unknown
(also in summary_maps.FEATURES). Run order: price_index -> age_check -> price_drivers -> figures. Results: age
premiums as in the overview; metro -3.5 %/km, cafés +2.7 %, reputation Żoliborz +13 %, Praga-Północ -18 %, R² 0.574.

**2026-10-17 (audit before posting):** reputation no longer leaks the district. `price_index.py` also saves the
unshrunk building value (`value_2025_raw`) and its sampling variance (`fe_var`); `price_drivers.py` models the unshrunk
value (the shrunk one was pulled toward the district mean, so thinly sold buildings borrowed their district's level),
weights buildings by 1/(tau² + fe_var) and shrinks the residual toward the model's prediction instead. Effect small:
Żoliborz +15.7 -> +14.1 %, Praga-Północ -18.8 -> -18.3 %. The bigger Żoliborz gap between thinly and heavily sold
buildings is composition (pre-war villas vs new estates), not the method. Robustness: `resid_pct_no_amen` (model
without the walk7_* amenity counts) gives the same district pattern, so cafés do not absorb reputation. Out-of-sample
R² 0.571 (precision-weighted). Listed heritage now +3.5 % within a district (was ~0 on the shrunk target).
Wording: every factor is an association ("buildings with X sell for Y % more than similar ones in the district");
cafés follow wealth as much as they attract it, transit supply is planned and slow to follow demand, so neither
coefficient is a cause. Titles changed (fig. 06, 08, 25); flip markup marked as before renovation, tax and fees.
**Known issue, fig. 27:** 23 % of developer files followed 10+ months never drop a flat (sold flats most likely kept
on the list), so "still for sale after 12 months" and "never changed price" are overstated; not for publication
until those files are excluded. Snapshot of the previous figures: `output/snapshots/2026-10-17/` (untracked).
Same day, later: `forward_test.py` -> summary/30 (Kahneman test: what 2010-14 data knew vs 2021-25; same 2,233
buildings with >= 4 resales in both periods, so older stock only; new M2 stations by opening: central 2015, extensions
2019-22, 800 m straight line). `price_ridgelines.py` -> summary/31, the opening map: Gaussian price surface (350 m) of
value_2025, one line per 250 m row, heights above 12K PLN/m², faded where data is thin and tapered at the city edge and
the Vistula (no cliffs); renders in ~10 min (per-segment colours). `make_map.py` stalled after map_02 (> 30 min on map_03,
legacy exploratory maps): maps 03-09 in output/analysis are from before the fix. Not rebuilt: Blender renders (price colours unchanged; reputation render uses the old residual).
Fig. 30 judged too hard to read for posting (kept as evidence that reputation is stable). Fig. 32
`price_points.py`: same frame and colours as 31, 13.6K priced buildings over all city buildings in grey.
Fig. 31 fix: peak height now needs ~4 priced buildings within ~350 m (was: sales weight), so a lone building
cannot make a mountain (Wawer peak was Planetowa 33 alone: 6 sales, mostly large developer flats, extrapolated
to a 25.5K typical-flat value vs Wawer median 13.8K).
Next: Bemowo metro extension before/after (measure the "before" now); tram to Wilanów as a lead (check dates).

**2026-10-17 (tidy-up):** main pipeline on BDOT10k June 2025 (`fetch_geodata.py` reads the gpkg, maps the 2025
function names to the old codes, drops destroyed buildings: 166.7K; heritage/ZABYTEK still from the Feb 2024
shapefile in `D:/QGIS/mapy_warszawy_misc/BDOT10k/`). Model unchanged in substance: out-of-sample R² 0.571 (was 0.58),
within a district metro -3.4 %/km, cafés x2 +2.8 %, 2011-20 buildings +16.5 % vs prefab. Overview + figure index at the
top of this README; `contact_sheet.py` -> summary/00_contact_sheet.png; shared `save()` retries while a viewer locks the
PNG. 3D: `export_scene.py --set reputation` (model residual, fixed breaks +-3/6/10/15 %, blue-neutral-red) ->
3d/renders/04_reputation_{center,zoliborz}.png. Dropped as the hero image: the diverging ramp's neutral middle is the brightest
class on the dark ground. Hero = `--set price_m2` (`prices_v2`, viridis) with unpriced buildings (offices, shops)
in light clay grey (`NODATA_RGB`, was near-ground dark) -> 3d/renders/05_prices_{center,zoliborz}.png.
Price scale then made explicit: fixed breaks 12K..40K (quantiles had lumped every building >= 21.5K into one
yellow) on `PRICE_RAMP` indigo - teal - green - yellow - amber - orange - red - crimson - magenta (40K+), glow off
(`build_blend.py --glow 0`). Złota 44 and Cosmopolitan now stand out from Śródmieście's 20-25K tenements.

**2026-10-17:** fig. 28 full-page map of all 305 projects on sale (offers/projects_map.py; 20 largest named in
callout columns; projects listed under two street spellings merged); fig. 27 colours swapped (cuts red, raises
blue), coverage note (history datasets = 62 % of flats on offer, asking ~6 % more). Fig. 29 flat sizes: new flats
sold shrank from 60 m² (2006-13) to 47 m² (2025-26), now no bigger than resales (48 m², flat for 20 years); 4+-room
share of new flats fell from ~40 % to ~16 %; under 25 m² costs +20 % per m² vs 45-60 m² in the same building.

**2026-10-16 (night, 2):** asking-price history (offers/fetch_price_history.py, parse_price_history.py,
history_story.py -> summary/27): last snapshot per month of the 675 Warsaw datasets (3,551 files, 3 min); 313
keep history (Sep 2025 - Oct 2026). Files listing one flat several times: latest-dated row wins, undecidable flats
(> 5 % apart) dropped (991). 8,611 flats followed >= 3 months: 68 % never changed price, 25 % pricier (typically
+3 %), 7 % cheaper (-6 %); the same flats +0.5 % over the year; of newly listed flats 49 % are still for sale after
12 months (Kaplan-Meier, leaving = vanishing while the developer keeps publishing). Fig. 25 headline corrected
after a 2 km block bootstrap: café/culture reach explains prices about as well as PKiN + metro distance (beats it
in 38 % of resamples), effect +4..+9 % per doubling; metro -1.7..-5.4 %/km.

**2026-10-16 (night):** offers/offers_story.py -> summary/26 new flats on offer: 8.6K flats in 336 investments,
median 16.6K PLN/m², 50 m²; the median investment asks 3 % less per m² than a 5-15-year-old flat within 1 km
(typical-flat value of 2011-20 buildings); Wola's median (38K) is driven by towers near Rondo Daszyńskiego
(Grzybowska, NOHO Karolkowa, Towarowa 22, Prosta, 30-50K); supply concentrates in Białołęka, Ursus and Wola.
Office cores: ONZ and Central Station merged (37K); five second-tier centres drawn on fig. 24. All map legends and
colour bars moved to the upper right (the empty corner of Warsaw's outline).

**2026-10-16 (late):** developer price lists (offers/): all 13,032 dane.gov.pl datasets fetched in 13.6 min with 8
parallel workers (first version: one request at a time + 0.5 s pause, ~9 h). parse_price_lists.py keeps rows whose
INVESTMENT location is Warsaw: 752 datasets, 37.7K rows, 9.6K distinct flats on offer (median 16.5K PLN/m²,
54 m²). geocode_offers.py places 8.6K (90 %) via the city's address points (5.9K at a house number, 2.8K at the
street's midpoint) -> offers/data/offers_warszawa.gpkg, 204 investments. First comparison: in 33 buildings with
offers and >= 3 developer deeds since 2025, asking is -0.9 % vs deed (IQR -6..+7 %); deeds lag the price agreed
at the preliminary contract by 1-2 years. Access (access/): figures 24-25 rebuilt on BDOT June 2025 + SimCity
categories; office cores Daszyńskiego 68k, ONZ 25k, Służewiec 22k, Central Station 12k; best price
predictor = reach to cafés/restaurants/culture with a 5-min half-life (+6.9 % per doubling within a district).

**2026-10-16 (night):** walk_access.py adds walking distance to rail stations (SKM/KM/WKD/PKP, replaces the
straight-line train_km) and bus stops. Fig. 04 rebuilt (transit_modes.py): the metro lifts whole neighbourhoods
(-3.4 % per km walked, within a district) but inside a neighbourhood with a station the doorstep adds only +1.6 %
(<300 m vs 900-1500 m; +0.8 % at 300-600 m); within 300-600 m of a rail station -2.1 % (track noise); tram and bus
stop distance ~0 (within +-1.5 %). Raw data shows the reverse in Ursynów (newer, quieter buildings lie deeper in the
estates). Old fig. 04 (new vs old by departures) removed from summary_charts.py; output/analysis/17 keeps it.

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
- BDOT buildings: 166.7K (June 2025 gpkg; destroyed buildings dropped; heritage still from the Feb 2024 shapefile)
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
