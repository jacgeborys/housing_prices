"""
Lublin buildings from the national BDOT10k shapefile (D:/QGIS/BODT10k/budynki.shp, Feb 2025, read-only) -> data/buildings.gpkg.
Run from the lublin/ folder:  python fetch_buildings.py   (~2 min: the shapefile has no spatial index)

(Same source as gdansk/; the OSM basemap D:/QGIS/osm_basemap/lublin has no storeys.)
Columns: osm_id (a running number; the name is kept so build_prices.py needs no change), residential (FOBUD = budynki
mieszkalne), kind (FSBUD), storeys (LICZ_KONDY), geometry (EPSG:2180). Only buildings whose centre lies in a Lublin district.
"""
import geopandas as gpd

SRC = "D:/QGIS/BODT10k/budynki.shp"


def main():
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    x0, y0, x1, y1 = d.total_bounds
    b = gpd.read_file(SRC, bbox=(x0 - 500, y0 - 500, x1 + 500, y1 + 500))
    b = b.set_crs(2180, allow_override=True)
    print(f"{len(b):,} buildings in the box")
    c = gpd.GeoDataFrame(geometry=b.geometry.representative_point(), crs=2180)
    inside = gpd.sjoin(c, d[["geometry"]], predicate="within").index.unique()
    b = b.loc[b.index.isin(inside)].reset_index(drop=True)
    out = gpd.GeoDataFrame({"osm_id": range(1, len(b) + 1), "residential": b["FOBUD"].eq("budynki mieszkalne"),
                            "kind": b["FSBUD"], "storeys": b["LICZ_KONDY"]}, geometry=b.geometry, crs=2180)
    out.to_file("data/buildings.gpkg", driver="GPKG")
    print(f"{len(out):,} in Lublin ({int(out['residential'].sum()):,} residential) -> data/buildings.gpkg")


if __name__ == "__main__":
    main()
