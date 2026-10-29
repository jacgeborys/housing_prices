"""
Gdańsk districts (dzielnice) from OSM -> gdansk/data/districts.gpkg (columns district, geometry, EPSG:2180).
Uses D:/QGIS/osm_basemap/overpass_polite.py (imported read-only). Tries admin_level 9, then 10.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "D:/QGIS/osm_basemap")
import geopandas as gpd  # noqa: E402
from shapely.geometry import LineString  # noqa: E402
from shapely.ops import polygonize, unary_union  # noqa: E402

import overpass_polite  # noqa: E402

OUT = "gdansk/data/districts.gpkg"


def relation_polygon(rel):
    lines = [LineString([(p["lon"], p["lat"]) for p in m["geometry"]])
             for m in rel.get("members", []) if m["type"] == "way" and m.get("role") == "outer" and m.get("geometry")]
    polys = list(polygonize(unary_union(lines)))
    return unary_union(polys) if polys else None


def main():
    for level in (9, 10):
        q = ('[out:json][timeout:180];area["name"="Gdańsk"]["boundary"="administrative"]["admin_level"="7"]->.a;'
             f'rel["boundary"="administrative"]["admin_level"="{level}"](area.a);out geom;')
        data = overpass_polite.post(q, {})
        rels = [r for r in (data or {}).get("elements", []) if r["type"] == "relation"]
        print(f"admin_level {level}: {len(rels)} relations")
        if len(rels) >= 15:
            break
    rows = []
    for r in rels:
        g = relation_polygon(r)
        if g is not None:
            rows.append({"district": r["tags"].get("name"), "geometry": g})
    gdf = gpd.GeoDataFrame(rows, crs=4326).to_crs(2180)
    gdf.to_file(OUT, driver="GPKG")
    print(gdf.assign(km2=gdf.area / 1e6)[["district", "km2"]].round(1).to_string())


if __name__ == "__main__":
    main()
