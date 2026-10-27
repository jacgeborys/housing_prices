> **Resolved 2026-10-27**: the horse race was run (README, 2026-10-27 centrality test) and `centre_km` stays; see `PLAN_FINAL.md`.

# Next task: what lies under "distance to the Palace of Culture"?

*Written 2026-10-27 to resume in a new conversation. Read `MODEL_EXPLAINED.md` first (plain-language model), then
README "Model state (2026-10-25)".*

## The question

`centre_km` (straight-line distance to PKiN) is the strongest feature of the reputation model (24 % of permutation
importance, 42 % before the river side was added). It describes but doesn't explain, and it can't say why **remote
Wilanów sells above relatively close Praga**. Goal: find which concrete, interpretable measures carry that signal, and
whether one or a few of them can replace `centre_km` without losing accuracy.

**Expectation:** accuracy will barely move (every candidate measures "centrality"); the gain is in meaning: a
slide that says *why* central flats cost more. One bounded round, roughly half a day; if nothing beats the benchmark,
keep PKiN and say so.

## Where the model stands (2026-10-25)

- `price_drivers.py`: 13,607 buildings (>= 3 sales), target = log typical-flat value 2025 (unshrunk). Transit from
  TransitFrequency `buildings_modes.gpkg` (default). Features incl. `centre_km`, `river_km`, `right_bank` (new,
  -13.5 % city-wide; districts never cross the Vistula, so no within-district tag), `log2_jobs_transit` (new; corr
  with `centre_km` -0.75; +1.3 % per doubling city-wide, +0.3 % within district), walking distances, 7-min-walk amenity
  counts, noise, LiDAR-checked trees, building age dummies.
- Out-of-sample R² 0.600 (gradient boosting, spatial 5-fold CV on 2 km blocks, precision-weighted). Linear regression
  + district 0.649 in sample.
- Reputation by district (median, vs prediction): Praga-Północ -13 %, Wola -6 %, Żoliborz +4.5 %, Wilanów +1.6 %.
- `ml_compare.py` has the reusable pieces: `load()`, spatial folds, `oof()`, `score()`, `cluster_slope()` (double ML).

## Candidates (horse race: swap each in for `centre_km`, one at a time)

| Candidate | Source (exists?) | Story it would tell |
|---|---|---|
| Benchmark: `centre_km` | price_drivers.py | — |
| Jobs reachable by transit, AM peak | `access/output/access_am_peak.csv` `jobs_grav` (yes) | access to work |
| Leisure / culture / restaurants reachable in 20 min by transit | `access/output/gravity_am_peak.csv` (yes; fig. 25 found +4..+9 % per doubling, as informative as PKiN + metro) | access to city life |
| Office-cluster travel times | `access_am_peak.csv` `t_centrum`, `t_daszynskiego`, `t_sluzewiec`, ... (yes) | which job centre matters |
| Amenities within a wider walk (15 min) | rerun `walk_amenities.py` with a larger radius (no) | walkable urban life |
| Pre-war urban fabric around the building | `output/age_check.gpkg` decades, share of pre-1941 footprint within 500 m (no) | historic city |
| Greenery / low density around (Wilanów-type) | trees, `green_ha_500`, building density (partly) | suburban quality |

Wilanów vs Praga is the test case: a good replacement should put Wilanów's predicted price close to its actual one
*without* `centre_km`, and leave Praga-Północ's reputation gap mostly intact.

## Method

1. Same sample, folds and target as `ml_compare.py`. For each candidate set: gradient boosting out-of-sample R²,
   linear tag within district, double ML tag, median reputation per district (esp. Wilanów, Praga-Północ, Żoliborz).
2. Partial dependence of the winner (is it linear, a threshold, a plateau like PKiN's: steep within 2 km, flat
   4-10 km, drop at the edge?).
3. Keep the winner only if R² >= benchmark - 0.005 and the story is clearer; then rebuild
   (`python report/build_report.py --rebuild`, then the extra scripts listed below).

## Conventions to keep

- Every factor framed as an association ("buildings with X sell for Y % more than similar ones in the same district");
  "reputation" = model residual, keep the word, define it.
- Public slides: plain descriptive charts, title = the answer, at most two panels; model internals on a methods page.
- After every change: git add, commit, push (no Co-Authored-By); keep README status current.
- Other projects (TransitFrequency, lidar/trees, osm_basemap) are read-only.
- Age data includes Maksim Pashnin's dates (no open licence, `data/makscape/`, not in git): credit him and tell him
  before publishing.

## Rebuild checklist (what `build_report.py --rebuild` does not run)

After `report/build_report.py --rebuild`: `price_drivers.py --trees lidar`, `explain_comparison.py`, `tram_test.py`,
`ml_compare.py`, `transit_modes.py`, `frequency_story.py`, `forward_test.py`, `walkups.py`, `market_rhythms.py`,
`summary_charts.py`, `contact_sheet.py`, then `report/build_report.py` once more to reassemble the PDF.

## Ideas parked (tested 2026-10-25, not worth adding)

Trams vs buses, gated estates, nuisances (Radiowo, Czajka, Siekierki), crowding / patodeweloperka, per-building job
access, driving times, churches, escarpment, school results, regeneration areas (would explain reputation away).
