"""
Building dates from Maksim Pashnin's "Warsaw, through time" (https://warsaw.makscape.com/, 2026-10-25): GIS Expert's
decades plus his own additions, each with its source: the city's municipal heritage inventory (ARKA / GEZ), the
Mazovian heritage register (MWKZ, incl. post-war reconstruction dates), National Heritage Board entries, developer
and project documents, and secondary listings (SonarHome) or community sources, which he labels as such.

His data carries no open licence for the dates: kept local (data/ is not in git), credited in README, and he should
be told before anything built on it is published.

    python fetch_makscape.py   -> data/makscape/buildings.json.gz (his file, downloaded once)
                                  data/makscape/makscape_dates.gpkg (points, EPSG:2180: every building with a date)

Columns: mk_id (BDOT id), quality (his ageQuality: base-raster = GIS Expert; official-heritage, official-address-date,
primary-source-date, secondary-source-date, community-source-date, contextual-heritage-period, ...), kind (exact,
range, circa, period, ...), date_start, date_end, footprint_m2, source, raw (the source's own wording).
"""
import gzip
import json
import os

import geopandas as gpd
import pandas as pd
import requests

URL = ("https://warsaw.makscape.com/assets/88008bb9c1b0d8c0f645705e77777c2cc50b4c0ba7742378337b851e066b5bb3.json.gz"
       "?v=20261025-reconstruction")          # buildings.json.gz, from geometry/streaming/manifest.json
RAW = "data/makscape/buildings.json.gz"
OUT = "data/makscape/makscape_dates.gpkg"
X0, N0 = 638000, 486000                        # his local frame: x = easting - X0, z = -(northing - N0), EPSG:2180


def main():
    os.makedirs(os.path.dirname(RAW), exist_ok=True)
    if not os.path.exists(RAW):
        r = requests.get(URL, headers={"User-Agent": "ceny-mieszkan research (credit Makscape)"}, timeout=120)
        r.raise_for_status()
        open(RAW, "wb").write(r.content)
    d = json.load(gzip.open(RAW))
    s = d["schema"]
    m = pd.DataFrame(d["records"], columns=s["record"])
    m["quality"] = m["ageQuality"].map(dict(enumerate(s["ageQuality"])))
    m["kind"] = m["dateKind"].map(dict(enumerate(s["dateKind"])))
    add = d["additions"]
    m["source"] = [add.get(str(i), {}).get("source") for i in range(len(m))]
    m["raw"] = [add.get(str(i), {}).get("rawDate") for i in range(len(m))]
    m = m[m["dateStart"].notna() | m["dateEnd"].notna()]
    g = gpd.GeoDataFrame(m.rename(columns={"id": "mk_id", "dateStart": "date_start", "dateEnd": "date_end",
                                           "footprintAreaM2": "footprint_m2"})
                         [["mk_id", "quality", "kind", "date_start", "date_end", "footprint_m2", "source", "raw"]],
                         geometry=gpd.points_from_xy(m["x"] + X0, N0 - m["z"]), crs=2180)
    g.to_file(OUT, layer="makscape_dates", driver="GPKG")
    print(f"{len(d['records']):,} buildings in his file, {len(g):,} dated -> {OUT}")
    print(g["quality"].value_counts().to_string())


if __name__ == "__main__":
    main()
