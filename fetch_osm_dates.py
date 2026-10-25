"""
Construction years tagged in OpenStreetMap (start_date, building:year, construction_date, year_of_construction) for
Warsaw's buildings: a second source for buildings GIS Expert has no date for (age_check.py). One Overpass request
via D:/QGIS/osm_basemap/overpass_polite.py (imported read-only, no bytecode written there).

    python fetch_osm_dates.py   -> data/osm_building_dates.csv (osm_id, type, start_date, year)

Coverage is thin (~500 dated buildings city-wide, 2026-10-25); where both exist, the decade matches GIS Expert's for
86 % of buildings (92 % within ten years).
"""
import os
import re
import sys

import pandas as pd

sys.dont_write_bytecode = True
sys.path.insert(0, "D:/QGIS/osm_basemap")
import overpass_polite  # noqa: E402

OUT = "data/osm_building_dates.csv"
TAGS = ("start_date", "building:year", "construction_date", "year_of_construction")
QUERY = ('[out:json][timeout:300];area["name"="Warszawa"]["boundary"="administrative"]["admin_level"="6"]->.a;'
         f'(way["building"][~"^({"|".join(TAGS)})$"~"."](area.a);'
         f'relation["building"][~"^({"|".join(TAGS)})$"~"."](area.a););out tags;')


def main():
    data = overpass_polite.post(QUERY, headers={}, timeout=360)
    if data is None:
        sys.exit("Overpass: all attempts failed")
    rows = []
    for e in data["elements"]:
        raw = next((e["tags"][t] for t in TAGS if t in e["tags"]), "")
        m = re.search(r"(1[5-9]\d\d|20[0-2]\d)", raw)            # first 4-digit year: "1912", "~1930", "1965..1967"
        rows.append({"osm_id": e["id"], "type": e["type"], "start_date": raw, "year": int(m.group(1)) if m else None})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"{len(df):,} tagged buildings, {df['year'].notna().sum():,} with a year -> {OUT}")


if __name__ == "__main__":
    main()
