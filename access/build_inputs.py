"""
Inputs for the transit accessibility subproject (run with the main Python; routing runs in access/.venv).

  data/gtfs_warsaw.zip     the merged Warsaw GTFS (ZTM + KM + WKD) from D:/QGIS/TransitFrequency/_data/warsaw
                           (read only, copied and zipped; shapes.txt left out, routing doesn't use it)
  data/destinations.gpkg   250 m grid cells with opportunity weights (EPSG:4326 centroids):
                             jobs      floor area / m² per job by BDOT10k building function (office 18, retail &
                                       services 40, hotels 60, schools & universities 50, hospitals 40,
                                       museums & libraries 60, industry & warehouses 100)
                             shop_gfa  retail & services floor area (m²)
                             leisure   cafés, restaurants, bars, pubs, cinemas, theatres (OSM, data/osm_pois.gpkg)
  data/origins.gpkg        every building with a model value (output/price_model.gpkg / building_value), inner point

Floor area = BDOT10k footprint area x storeys (LKOND). No open register of jobs by building exists in Poland,
so floor area by function is the proxy.
"""
import os
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(HERE, "data")
GTFS_DIR = "D:/QGIS/TransitFrequency/_data/warsaw/2026_10_04"          # read only
GTFS_FILES = ["agency.txt", "stops.txt", "routes.txt", "trips.txt", "stop_times.txt", "calendar_dates.txt",
              "frequencies.txt", "feed_info.txt"]
BDOT = "D:/QGIS/bdot_basemap/Warszawa/PL.PZGiK.330.1465__OT_BUBD_A.shp"   # read only
M2_PER_JOB = {"budynkiBiurowe": 18, "budynkiHandlowoUslugowe": 40, "budynkiHoteli": 60,
              "budynkiSzkolIInstytucjiBadawczych": 50, "budynkiSzpitaliIZakladowOpiekiMedycznej": 40,
              "budynkiMuzeowIBibliotek": 60, "budynkiPrzemyslowe": 100, "zbiornikSilosIBudynkiMagazynowe": 100}
CELL = 250


def gtfs():
    """Zip a repaired copy of the merged GTFS. The merge in TransitFrequency left Koleje Mazowieckie trips
    pointing at route ids that routes.txt lacks (KM_R1, KM_RE6, KM_R8_BUS...); R5 rejects that, so the copy gets a
    KM agency and those routes (rail, or bus for *_BUS replacement services). The originals are not touched."""
    out = os.path.join(DATA, "gtfs_warsaw.zip")
    if os.path.exists(out):
        return out
    routes = pd.read_csv(os.path.join(GTFS_DIR, "routes.txt"), dtype=str)
    agency = pd.read_csv(os.path.join(GTFS_DIR, "agency.txt"), dtype=str)
    trips = pd.read_csv(os.path.join(GTFS_DIR, "trips.txt"), dtype=str, usecols=["route_id"])
    missing = sorted(set(trips["route_id"]) - set(routes["route_id"]))
    if missing:
        km = agency.iloc[[0]].copy()
        km["agency_id"], km["agency_name"] = "KM", "Koleje Mazowieckie"
        agency = pd.concat([agency, km], ignore_index=True)
        add = pd.DataFrame({"route_id": missing, "agency_id": "KM",
                            "route_short_name": [m.replace("KM_", "") for m in missing], "route_long_name": "",
                            "route_type": ["3" if "bus" in m.lower() else "2" for m in missing]})
        routes = pd.concat([routes, add], ignore_index=True)
        print(f"  GTFS copy: added {len(missing)} missing routes ({', '.join(missing[:5])}...)")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("routes.txt", routes.to_csv(index=False))
        z.writestr("agency.txt", agency.to_csv(index=False))
        for f in GTFS_FILES:
            p = os.path.join(GTFS_DIR, f)
            if f not in ("routes.txt", "agency.txt") and os.path.exists(p):
                z.write(p, arcname=f)
    print(f"Saved {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    return out


def destinations():
    b = gpd.read_file(BDOT, columns=["FUNOGBUD", "LKOND"]).to_crs(2180)
    b = b[b["FUNOGBUD"].isin(M2_PER_JOB)].copy()
    # shopping centres BDOT files as offices (Arkadia...): footprints holding an OSM shop=mall point count as retail
    malls = gpd.read_file(os.path.join(ROOT, "data", "osm_pois.gpkg"))
    malls = malls[malls["shop"] == "mall"][["geometry"]]
    hit = gpd.sjoin(b[["geometry"]], malls, predicate="contains", how="inner").index.unique()
    b.loc[hit, "FUNOGBUD"] = "budynkiHandlowoUslugowe"
    print(f"  {len(hit)} mall footprints reclassified as retail")
    b["gfa"] = b.geometry.area * b["LKOND"].fillna(1).clip(lower=1)
    b["jobs"] = b["gfa"] / b["FUNOGBUD"].map(M2_PER_JOB)
    b["shop_gfa"] = np.where(b["FUNOGBUD"] == "budynkiHandlowoUslugowe", b["gfa"], 0.0)
    c = b.geometry.centroid
    b["gx"], b["gy"] = (c.x // CELL).astype(int), (c.y // CELL).astype(int)
    g = b.groupby(["gx", "gy"])[["jobs", "shop_gfa"]].sum()
    p = gpd.read_file(os.path.join(ROOT, "data", "osm_pois.gpkg"))
    p = p[p["amenity"].isin(["cafe", "restaurant", "bar", "pub", "cinema", "theatre"])]
    p["gx"], p["gy"] = (p.geometry.x // CELL).astype(int), (p.geometry.y // CELL).astype(int)
    g = g.join(p.groupby(["gx", "gy"]).size().rename("leisure"), how="outer").fillna(0).reset_index()
    g = g[(g["jobs"] >= 10) | (g["leisure"] >= 1)]
    pts = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy((g["gx"] + 0.5) * CELL, (g["gy"] + 0.5) * CELL), crs=2180)
    pts["id"] = np.arange(len(pts))
    pts = pts.to_crs(4326)
    out = os.path.join(DATA, "destinations.gpkg")
    pts[["id", "jobs", "shop_gfa", "leisure", "geometry"]].to_file(out, layer="destinations", driver="GPKG")
    print(f"Saved {out}: {len(pts):,} cells, {pts['jobs'].sum() / 1e6:.2f} M job equivalents, "
          f"{pts['leisure'].sum():,.0f} leisure venues")
    top = pts.sort_values("jobs", ascending=False).head(8).to_crs(2180)
    print("top job cells (EPSG:2180):", [(round(x), round(y), round(j)) for x, y, j in
                                          zip(top.geometry.x, top.geometry.y, top["jobs"])])


def origins():
    v = gpd.read_file(os.path.join(ROOT, "output", "price_model.gpkg"), layer="building_value", columns=["osm_id"])
    v = v.drop_duplicates("osm_id").reset_index(drop=True)
    pts = gpd.GeoDataFrame({"id": v["osm_id"].astype("int64")}, geometry=v.geometry.representative_point(),
                           crs=v.crs).to_crs(4326)
    out = os.path.join(DATA, "origins.gpkg")
    pts.to_file(out, layer="origins", driver="GPKG")
    print(f"Saved {out}: {len(pts):,} buildings")


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    gtfs()
    destinations()
    origins()
