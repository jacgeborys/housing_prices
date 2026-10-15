"""
Fetch neighbourhood points of interest in Warsaw from OSM -> data/osm_pois.gpkg (layer pois).

Uses D:/QGIS/osm_basemap/overpass_polite.py (imported read-only, no bytecode written there):
one request at a time, waits for a free Overpass slot, backs off after errors.

Categories: food & drink (cafe, restaurant, bar, pub, fast_food), schools, kindergartens,
health (pharmacy, doctors, clinic), culture (library, cinema, theatre), every shop=*, and
playgrounds / fitness / sports centres.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "D:/QGIS/osm_basemap")
import geopandas as gpd  # noqa: E402
from shapely.geometry import Point  # noqa: E402

import overpass_polite  # noqa: E402

AREA = 'area["name"="Warszawa"]["boundary"="administrative"]["admin_level"="6"]->.a;'
PARTS = {
    "amenity": 'nwr["amenity"~"^(cafe|restaurant|bar|pub|fast_food|school|kindergarten|pharmacy|'
               'doctors|clinic|library|cinema|theatre)$"](area.a);',
    "shop": 'nwr["shop"](area.a);',
    "leisure": 'nwr["leisure"~"^(playground|fitness_centre|sports_centre)$"](area.a);',
}
OUT = "data/osm_pois.gpkg"


def main():
    rows = []
    for name, body in PARTS.items():
        print(f"{name}: ", end="", flush=True)
        data = overpass_polite.post(f"[out:json][timeout:180];{AREA}({body});out center tags;",
                                    headers={}, timeout=240)
        if data is None:
            sys.exit(f"\n{name}: all attempts failed")
        for e in data["elements"]:
            c = e.get("center", {})
            lat, lon = e.get("lat", c.get("lat")), e.get("lon", c.get("lon"))
            if lat is None:
                continue
            t = e.get("tags", {})
            rows.append({"osm_id": e["id"], "osm_type": e["type"], "amenity": t.get("amenity"),
                         "shop": t.get("shop"), "leisure": t.get("leisure"), "name": t.get("name"),
                         "geometry": Point(lon, lat)})
        print(f"{len(data['elements']):,} elements")
    g = gpd.GeoDataFrame(rows, crs=4326).drop_duplicates(["osm_type", "osm_id"]).to_crs(2180)
    g.to_file(OUT, layer="pois", driver="GPKG")
    print(f"Saved {OUT}: {len(g):,} POIs")
    print(g["amenity"].value_counts().head(15).to_dict())
    print(g["shop"].value_counts().head(15).to_dict())


if __name__ == "__main__":
    main()
