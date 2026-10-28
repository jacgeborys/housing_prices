# Final iteration of the report

*Written 2026-10-27 evening for the next session. Read `README.md` (items dated 2026-10-27) first; `PLAN_CENTRALITY.md`
is resolved by the test below. One model rerun is expected; the whole thing is about half a day.*

## What was decided today (and why)

| Question | Decision | Evidence |
|---|---|---|
| Houses in the story | Descriptive only, three slides: 38 where houses sell, 39 house price per m² by 300 m cell, 35 house or flat (overlapping curves). No house model in this round. | 6,913 clean house sales vs 251,896 flat sales; living area estimated for 97 % (BDOT10k outline x floors, typical miss 14 %) |
| Wesoła "empty" | Not a bug: 1.1 % of its buildings have a flat sale (city lowest), it is almost all houses. Fig. 32 now draws house sales too; fig. 31 stays flats only. | 8,575 buildings, 97 with a flat sale; 222 house sales |
| Distance to PKiN (`centre_km`) | **Keep it.** Add nothing new to the model. | Replacing it by leisure reach / pre-war fabric / jobs / density / amenities: R² 0.597-0.610 vs 0.589-0.600 (3 fold seeds, sd 0.01): no real gain; adding all of them on top: 0.594 |
| River side (`right_bank`) | **Drop from the model, keep `river_km`.** Keep "right-bank flats sell about 13 % below similar left-bank ones" as a descriptive fact (fig. 10). | Without it: Żoliborz +6 -> +14.5 %, Praga-Północ -12 -> -15 %, Wilanów +3 -> +7 %, Wesoła +6.8 -> +4.6 %; R² 0.589 -> 0.563. The river side is mostly reputation (image of the right bank), so counting it as a feature explains reputation away |
| Small districts | Mark or shrink: Wesoła (38 buildings) and Rembertów (148) must not top a ranking. Shrink district medians by sample size (as `estates.py` does) and grey districts under 100 buildings. | fig. 06 currently shows Wesoła +7 % on 38 buildings |
| 1-2 sale buildings | Dot maps only (figs. 31, 32). Not in the model or in reputation. | `--min-sales 1` moved Wilanów +4 -> +10 % (README 2026-10-25) |
| 2017-2020 flat gap | Label it (done on 23, 38), no recovery. | The `budynki` layer has the deeds (prices match 99 %) but no flat size, floor or function |
| Footnotes | Two lines at most on every slide; anything longer goes to the methods page (r99). | "Nobody will read that much" |

## Status 2026-10-28 evening: steps 1-6 done (see README); open: slide count, parked items

## Update 2026-10-28

The full rerun on complete data is done (README, 2026-10-28): Wesoła is no longer an outlier (171 buildings, about 0 %),
R² 0.616. River side on the new data: with it Praga-Północ -4.8 / Żoliborz +4.8, without it -12.0 / +12.7 (3 fold splits);
the no-river version is the stable one (barely moved with the data), so dropping `right_bank` stays the recommendation.
Steps 2-6 below still apply; the rebuild now takes about 1 h 40 (data steps are not needed again).

## Steps

1. **Model.** Remove `right_bank` from `FEATURES` (summary_maps.py) and from price_drivers.py (the `RIGHT_BANK` list,
   the label, the NaN line 310); clean the duplicated lines there (jobs block computed twice around lines 230-245,
   lines 55/56 and 67/68). Keep `river_km`, `centre_km`, `log2_jobs_transit`.
2. **Rebuild.** `python report/build_report.py --rebuild` (about 1 h, includes houses.py, house_map.py, house_cells.py),
   then the extras `price_drivers.py --trees lidar`, `explain_comparison.py`, `tram_test.py`, `ml_compare.py`,
   `transit_modes.py`, `frequency_story.py`, `forward_test.py`, `walkups.py`, `market_rhythms.py`, `summary_charts.py`,
   `contact_sheet.py`, then `report/build_report.py` once more. `build_viz.py` / `price_index.py` are not rerun (the
   address-fallback fix touches 43 sales; it takes effect at the next data refresh).
3. **Reputation slides (06, 10).** Small-district rule above. On fig. 06 show how much the river side moves each district
   (dots on the district bars: "if the river side were counted as a feature"), instead of hiding it. Wesoła gets a note:
   "mostly houses, see fig. 39". Re-check every number in titles and the README headline against the rerun.
4. **Footnote diet.** Shorten to <= 2 lines: figs. 31, 35, 38, 39 first (the long ones), then the rest. Move the details
   (estimated house area, house price index, 1-2 sale buildings, outlier rule, register gaps) to r99. Add a houses
   paragraph and the river-side paragraph (R² cost 0.026, reputation sensitivity) to r99 in `report/descriptive.py`.
5. **Length.** Decide: 29 slides is long for a LinkedIn document. Proposal for ~15: 31, 32, r01 data, 23 blind spot,
   01 prices over time, 33 what 1 million buys, 38 where houses sell, 35 house or flat, 03 what drives price, 18 building
   age, 08 cafes, 24 transit access, 06 reputation, 10 district anatomy, r99 methods; the rest as an appendix PDF.
6. **README.** Update Status, headline findings and the report section after the rerun.

## Parked (after publishing)

- A house model: sale-level (6.1K sales), plot, estimated area, type, floors, location features at the house. Needs
  `walk_access.py` / `walk_amenities.py` for about 5K house buildings (flats only so far). Worth it only if a house
  reputation slide is wanted.
- Recovering 2017-2020 flat sizes (guess from the same building's other years): noisy, not worth the risk.
- Network walk of 15 min for amenities (the centrality test used straight-line 1 km).

## Conventions (unchanged)

Every factor framed as an association; "reputation" = model residual, keep the word, define it; fig. 27 not for
publication; plain descriptive charts, title = the answer, at most two panels; after every change git add, commit, push
(no Co-Authored-By); other projects (TransitFrequency, lidar/trees, osm_basemap) are read-only.
