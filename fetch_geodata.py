"""
Fetch and prepare spatial reference data for the housing price analysis.

Two-tier architecture:
  1. FETCH full datasets to canonical folders on D:/QGIS/ (reusable across projects)
  2. EXTRACT & CLIP to ceny_mieszkan/data/ for this project

Sources:
  - OSM buildings    — Overpass API -> D:/QGIS/osm_basemap/<city>/buildings.gpkg
  - BDOT buildings   — GUGiK opendata download -> D:/QGIS/bdot_basemap/<city>/
  - Address points   — Warsaw city WFS -> D:/QGIS/punkty_adresowe/output/

Usage:
  python fetch_geodata.py                    # fetch + extract all
  python fetch_geodata.py --only osm         # just OSM buildings
  python fetch_geodata.py --only addresses   # just address points
  python fetch_geodata.py --only bdot        # just BDOT (extract only)
  python fetch_geodata.py --extract-only     # skip fetching, just clip from canonical
  python fetch_geodata.py --force            # re-fetch even if files exist
"""

import argparse
import io
import os
import time
import xml.etree.ElementTree as ET
import zipfile

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, Polygon, MultiPolygon

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QGIS_ROOT = os.path.join("D:", os.sep, "QGIS")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

# Warsaw bbox (WGS84: south, west, north, east)
BBOX_WGS84 = {"south": 52.0977, "west": 20.8519, "north": 52.3690, "east": 21.2711}

TARGET_CRS = "EPSG:2180"

# Canonical source locations
OSM_BUILDINGS_GPKG = os.path.join(QGIS_ROOT, "osm_basemap", "warsaw", "buildings.gpkg")
BDOT_SHP = os.path.join(QGIS_ROOT, "bdot_basemap", "Warszawa",
                        "PL.PZGiK.330.1465__OT_BUBD_A.shp")
BDOT_ADJA_SHP = os.path.join(QGIS_ROOT, "bdot_basemap", "Warszawa",
                             "PL.PZGiK.330.1465__OT_ADJA_A.shp")
ADDR_GPKG = os.path.join(QGIS_ROOT, "punkty_adresowe", "output",
                         "punkty_adresowe_warszawa.gpkg")

# BDOT10k download (GUGiK opendata)
# URL pattern: {base}{teryt[:2]}/{teryt}_{format}.zip
BDOT_DOWNLOAD_BASE = "https://opendata.geoportal.gov.pl/bdot10k/schemat2021/SHP/"
BDOT_TERYT = "1465"  # Warsaw powiat
BDOT_DIR = os.path.join(QGIS_ROOT, "bdot_basemap", "Warszawa")

# Overpass API
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_HEADERS = {"User-Agent": "ceny-mieszkan-fetcher/1.0"}

# Warsaw address WFS
ADDR_WFS_URL = "https://wfs.um.warszawa.pl/serwis"
ADDR_PAGE_SIZE = 10000

# Warsaw districts — used to partition address fetching (startIndex is broken)
WARSAW_DISTRICTS = [
    "Bemowo", "Białołęka", "Bielany", "Mokotów", "Ochota",
    "Praga-Południe", "Praga-Północ", "Rembertów", "Targówek",
    "Ursus", "Ursynów", "Wawer", "Wesoła", "Wilanów",
    "Wola", "Włochy", "Śródmieście", "Żoliborz",
]

BUILDINGS_QUERY = """
[out:json][timeout:180];
(
  way["building"]({bbox});
  relation["building"]({bbox});
  way["building:part"]({bbox});
  relation["building:part"]({bbox});
);
out geom;
"""

# Tags to preserve from OSM elements
OSM_KEEP_TAGS = [
    "building", "building:part", "building:levels", "building:min_level",
    "building:colour", "building:material",
    "height", "min_height",
    "roof:shape", "roof:colour", "roof:material",
    "name", "addr:street", "addr:housenumber", "addr:city",
    "man_made",
]

# Tags that indicate non-building features to exclude
OSM_EXCLUDE_TAGS = {"man_made": {"bridge", "pier", "tower"}}


# ---------------------------------------------------------------------------
# OSM buildings — fetch to canonical location via Overpass
# ---------------------------------------------------------------------------

def _parse_osm_geometry(element):
    """Convert an Overpass JSON element to a Shapely geometry."""
    if element["type"] == "way" and "geometry" in element:
        coords = [(n["lon"], n["lat"]) for n in element["geometry"]]
        if len(coords) >= 4 and coords[0] == coords[-1]:
            return Polygon(coords)
    if element["type"] == "relation" and "members" in element:
        outer, inner = [], []
        for m in element["members"]:
            if m.get("type") != "way" or "geometry" not in m:
                continue
            coords = [(n["lon"], n["lat"]) for n in m["geometry"]]
            if len(coords) < 4:
                continue
            if coords[0] != coords[-1]:
                coords.append(coords[0])
            role = m.get("role", "outer")
            (outer if role == "outer" else inner).append(coords)
        if outer:
            try:
                if len(outer) == 1:
                    return Polygon(outer[0], inner or None)
                polys = [Polygon(ring) for ring in outer]
                return MultiPolygon([(p.exterior.coords, []) for p in polys if p.is_valid])
            except Exception:
                pass
    return None


def _overpass_query(query):
    """Execute one Overpass query with retries."""
    for attempt in range(4):
        try:
            resp = requests.post(OVERPASS_URL, data={"data": query},
                                 headers=OVERPASS_HEADERS, timeout=180)
            if resp.status_code == 429:
                wait = 60 * (2 ** attempt)
                print(f"  rate-limited, waiting {wait}s...", flush=True)
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                print(f"  HTTP {resp.status_code}", end=" ", flush=True)
                time.sleep(30)
                continue
            return resp.json()
        except requests.exceptions.Timeout:
            print(f"  timeout", end=" ", flush=True)
            time.sleep(30 * (attempt + 1))
        except Exception as e:
            print(f"  error: {e}", end=" ", flush=True)
            time.sleep(15)
    return None


def fetch_osm_buildings(force=False):
    """Fetch OSM buildings to canonical location via tiled Overpass queries."""
    if os.path.exists(OSM_BUILDINGS_GPKG) and not force:
        print(f"  OSM buildings already at {OSM_BUILDINGS_GPKG}")
        return True

    print(f"  Fetching OSM buildings via Overpass -> {OSM_BUILDINGS_GPKG}")

    n = 5
    bbox = BBOX_WGS84
    lat_step = (bbox["north"] - bbox["south"]) / n
    lon_step = (bbox["east"] - bbox["west"]) / n
    tiles = []
    for i in range(n):
        for j in range(n):
            tiles.append({
                "south": bbox["south"] + i * lat_step,
                "north": bbox["south"] + (i + 1) * lat_step,
                "west": bbox["west"] + j * lon_step,
                "east": bbox["west"] + (j + 1) * lon_step,
            })

    all_rows = []
    for idx, tile in enumerate(tiles, 1):
        bbox_str = f"{tile['south']},{tile['west']},{tile['north']},{tile['east']}"
        print(f"    [{idx}/{len(tiles)}]...", end=" ", flush=True)
        query = BUILDINGS_QUERY.format(bbox=bbox_str)
        data = _overpass_query(query)
        if data is None:
            print("FAILED")
            continue

        count = 0
        for el in data.get("elements", []):
            geom = _parse_osm_geometry(el)
            if geom is None:
                continue
            tags = el.get("tags", {})
            # Skip non-building features (bridges, piers, etc.)
            skip = False
            for excl_tag, excl_vals in OSM_EXCLUDE_TAGS.items():
                if tags.get(excl_tag) in excl_vals:
                    skip = True
                    break
            if skip:
                continue
            row = {"osm_id": el["id"], "geometry": geom}
            for tag in OSM_KEEP_TAGS:
                row[tag.replace(":", "_")] = tags.get(tag)
            all_rows.append(row)
            count += 1
        print(f"{count:,} features")
        if idx < len(tiles):
            time.sleep(5)

    if not all_rows:
        print("  ERROR: No OSM buildings fetched!")
        return False

    gdf = gpd.GeoDataFrame(all_rows, crs="EPSG:4326")
    before = len(gdf)
    gdf = gdf.drop_duplicates(subset=["osm_id"], keep="first")
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    print(f"  Dedup: {before:,} -> {len(gdf):,}")

    os.makedirs(os.path.dirname(OSM_BUILDINGS_GPKG), exist_ok=True)
    gdf.to_file(OSM_BUILDINGS_GPKG, driver="GPKG", layer="buildings")
    print(f"  Saved -> {OSM_BUILDINGS_GPKG}")
    return True


# ---------------------------------------------------------------------------
# Address points — fetch to canonical location via Warsaw WFS
# ---------------------------------------------------------------------------

ADDR_NS = {
    "gml": "http://www.opengis.net/gml",
    "wfs": "http://fdo.osgeo.org/schemas/feature/wfs",
}


def _parse_address_page(xml_bytes):
    """Parse one page of address point WFS response."""
    root = ET.fromstring(xml_bytes)
    records = []
    for member in root.findall(".//gml:featureMember", ADDR_NS):
        feat = member.find("wfs:PUNKTY_ADRESOWE_WGS84", ADDR_NS)
        if feat is None:
            feat = member[0] if len(member) > 0 else None
        if feat is None:
            continue

        pos_el = feat.find(".//gml:pos", ADDR_NS)
        if pos_el is None or not pos_el.text:
            continue
        coords = pos_el.text.strip().split()
        if len(coords) < 2:
            continue

        def _txt(tag):
            el = feat.find(f"wfs:{tag}", ADDR_NS)
            return el.text.strip() if el is not None and el.text else None

        # Warsaw WFS returns lon, lat in gml:pos
        records.append({
            "geometry": Point(float(coords[0]), float(coords[1])),
            "addr_id": _txt("ID"),
            "street": _txt("NAZWA_PODST"),
            "street_short": _txt("NAZWA_SKROC"),
            "district": _txt("DZIELNICA"),
            "number": _txt("NR_PORZADK"),
        })
    return records


def _fetch_district_addresses(district):
    """Fetch all address points for one district, with ID-based pagination."""
    all_records = []
    seen_ids = set()
    last_id = None

    while True:
        # Build OGC filter: district + optional ID > last_id for pagination
        if last_id is None:
            ogc_filter = (
                '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">'
                "<ogc:PropertyIsEqualTo>"
                "<ogc:PropertyName>DZIELNICA</ogc:PropertyName>"
                f"<ogc:Literal>{district}</ogc:Literal>"
                "</ogc:PropertyIsEqualTo>"
                "</ogc:Filter>"
            )
        else:
            ogc_filter = (
                '<ogc:Filter xmlns:ogc="http://www.opengis.net/ogc">'
                "<ogc:And>"
                "<ogc:PropertyIsEqualTo>"
                "<ogc:PropertyName>DZIELNICA</ogc:PropertyName>"
                f"<ogc:Literal>{district}</ogc:Literal>"
                "</ogc:PropertyIsEqualTo>"
                "<ogc:PropertyIsGreaterThan>"
                "<ogc:PropertyName>ID</ogc:PropertyName>"
                f"<ogc:Literal>{last_id}</ogc:Literal>"
                "</ogc:PropertyIsGreaterThan>"
                "</ogc:And>"
                "</ogc:Filter>"
            )

        params = {
            "service": "WFS",
            "version": "1.1.0",
            "request": "GetFeature",
            "typeName": "wfs:PUNKTY_ADRESOWE_WGS84",
            "maxFeatures": str(ADDR_PAGE_SIZE),
            "Filter": ogc_filter,
            "sortBy": "ID",
        }

        resp = None
        for attempt in range(3):
            try:
                resp = requests.get(ADDR_WFS_URL, params=params, timeout=120)
                resp.raise_for_status()
                break
            except Exception as e:
                if attempt == 2:
                    print(f" FAILED: {e}", end="")
                time.sleep(5 * (attempt + 1))

        if resp is None or resp.status_code != 200:
            break

        records = _parse_address_page(resp.content)
        if not records:
            break

        for r in records:
            aid = r.get("addr_id")
            if aid and aid not in seen_ids:
                seen_ids.add(aid)
                all_records.append(r)
                last_id = aid

        # If we got less than page size, we're done with this district
        if len(records) < ADDR_PAGE_SIZE:
            break
        time.sleep(1)

    return all_records


def fetch_address_points(force=False):
    """Fetch address points from Warsaw city WFS, partitioned by district."""
    if os.path.exists(ADDR_GPKG) and not force:
        print(f"  Address points already at {ADDR_GPKG}")
        return True

    print(f"  Fetching address points from Warsaw WFS -> {ADDR_GPKG}")
    print(f"  Partitioning by {len(WARSAW_DISTRICTS)} districts...")

    # Test connectivity
    test_params = {
        "service": "WFS", "version": "1.1.0", "request": "GetFeature",
        "typeName": "wfs:PUNKTY_ADRESOWE_WGS84", "maxFeatures": "1",
    }
    try:
        test_resp = requests.get(ADDR_WFS_URL, params=test_params, timeout=30)
        test_root = ET.fromstring(test_resp.content)
        if not test_root.findall(".//gml:featureMember", ADDR_NS):
            print("  ERROR: WFS returned 0 results, may be down")
            return False
        print("  Connection OK")
    except Exception as e:
        print(f"  ERROR connecting to WFS: {e}")
        return False

    all_records = []
    for i, district in enumerate(WARSAW_DISTRICTS, 1):
        print(f"  [{i}/{len(WARSAW_DISTRICTS)}] {district}...", end=" ", flush=True)
        records = _fetch_district_addresses(district)
        all_records.extend(records)
        print(f"{len(records):,} pts (total: {len(all_records):,})")
        time.sleep(1)

    if not all_records:
        print("  ERROR: No address points fetched!")
        return False

    gdf = gpd.GeoDataFrame(all_records, crs="EPSG:4326")

    # Deduplicate by addr_id
    if "addr_id" in gdf.columns:
        before = len(gdf)
        gdf = gdf.drop_duplicates(subset=["addr_id"], keep="first")
        if before != len(gdf):
            print(f"  Dedup: {before:,} -> {len(gdf):,}")

    out_dir = os.path.dirname(ADDR_GPKG)
    os.makedirs(out_dir, exist_ok=True)
    gdf.to_file(ADDR_GPKG, driver="GPKG", layer="punkty_adresowe")
    print(f"  Saved {len(gdf):,} address points -> {ADDR_GPKG}")
    return True


# ---------------------------------------------------------------------------
# BDOT — no auto-fetch (requires manual download), just check existence
# ---------------------------------------------------------------------------

def fetch_bdot_buildings(force=False):
    """Download BDOT10k SHP from GUGiK opendata to canonical location."""
    if os.path.exists(BDOT_SHP) and not force:
        import datetime
        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(BDOT_SHP))
        print(f"  BDOT source: {BDOT_SHP}")
        print(f"  File date:   {mtime.strftime('%Y-%m-%d')}")
        print(f"  (use --force to re-download)")
        return True

    teryt = BDOT_TERYT
    url = f"{BDOT_DOWNLOAD_BASE}{teryt[:2]}/{teryt}_SHP.zip"
    print(f"  Downloading BDOT10k from GUGiK opendata...")
    print(f"  URL: {url}")

    try:
        resp = requests.get(url, timeout=300, stream=True)
        resp.raise_for_status()
    except Exception as e:
        print(f"  ERROR downloading: {e}")
        return False

    total_mb = int(resp.headers.get("content-length", 0)) / 1024 / 1024
    print(f"  Size: {total_mb:.0f} MB")

    # Download to memory then extract
    print(f"  Downloading...", end=" ", flush=True)
    chunks = []
    downloaded = 0
    for chunk in resp.iter_content(chunk_size=1024 * 1024):
        chunks.append(chunk)
        downloaded += len(chunk)
        print(f"\r  Downloading... {downloaded/1024/1024:.0f}/{total_mb:.0f} MB", end="", flush=True)
    print()

    content = b"".join(chunks)
    print(f"  Extracting to {BDOT_DIR}...")
    os.makedirs(BDOT_DIR, exist_ok=True)

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        zf.extractall(BDOT_DIR)
        print(f"  Extracted {len(zf.namelist())} files")

    if os.path.exists(BDOT_SHP):
        print(f"  OK: {BDOT_SHP}")
        return True
    else:
        # The zip may contain a subfolder — find the shapefile
        for root, dirs, files in os.walk(BDOT_DIR):
            for f in files:
                if "BUBD_A" in f and f.endswith(".shp"):
                    print(f"  Found: {os.path.join(root, f)}")
                    return True
        print(f"  WARNING: BUBD_A shapefile not found after extraction")
        return False


# ---------------------------------------------------------------------------
# Extract & clip to project data/ directory
# ---------------------------------------------------------------------------

def extract_osm(force=False):
    """Extract OSM buildings from canonical location to data/."""
    out_path = os.path.join(DATA_DIR, "osm_buildings.gpkg")
    if os.path.exists(out_path) and not force:
        gdf = gpd.read_file(out_path)
        print(f"  Extracted OSM buildings: {len(gdf):,} (cached)")
        return gdf

    if not os.path.exists(OSM_BUILDINGS_GPKG):
        print(f"  Source not found: {OSM_BUILDINGS_GPKG}")
        return None

    print(f"  Reading {OSM_BUILDINGS_GPKG}...")
    gdf = gpd.read_file(OSM_BUILDINGS_GPKG)
    print(f"  Source: {len(gdf):,} features")

    # Reproject to target CRS
    if gdf.crs and gdf.crs.to_epsg() != 2180:
        gdf = gdf.to_crs(TARGET_CRS)

    # Filter out non-building features (bridges etc.)
    if "man_made" in gdf.columns:
        exclude_vals = set()
        for excl_vals in OSM_EXCLUDE_TAGS.values():
            exclude_vals |= excl_vals
        mask = gdf["man_made"].isin(exclude_vals)
        if mask.any():
            print(f"  Excluding {mask.sum()} non-building features (bridges etc.)")
            gdf = gdf[~mask].copy()

    # Summary of building vs building:part
    if "building_part" in gdf.columns:
        n_parts = gdf["building_part"].notna().sum()
        n_buildings = gdf["building"].notna().sum()
        print(f"  Buildings: {n_buildings:,}, Building parts: {n_parts:,}")

    os.makedirs(DATA_DIR, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="osm_buildings")
    print(f"  -> {out_path} ({len(gdf):,} features)")
    return gdf


def _find_bdot_shp():
    """Find the BUBD_A shapefile, searching subdirectories if needed."""
    if os.path.exists(BDOT_SHP):
        return BDOT_SHP
    for root, dirs, files in os.walk(BDOT_DIR):
        for f in files:
            if "BUBD_A" in f and f.endswith(".shp"):
                return os.path.join(root, f)
    return None


def extract_bdot(force=False):
    """Extract BDOT buildings from canonical shapefile to data/."""
    out_path = os.path.join(DATA_DIR, "bdot_buildings.gpkg")
    if os.path.exists(out_path) and not force:
        gdf = gpd.read_file(out_path)
        print(f"  Extracted BDOT buildings: {len(gdf):,} (cached)")
        return gdf

    shp_path = _find_bdot_shp()
    if shp_path is None:
        print(f"  Source not found: {BDOT_SHP}")
        return None

    print(f"  Reading {shp_path}...")
    gdf = gpd.read_file(shp_path)
    print(f"  Source: {len(gdf):,} features")

    # Rename to friendly names (handle both old and new column names)
    rename_map = {
        "LOKALNYID": "bdot_id",
        "FUNOGBUD": "building_type",
        "FUNSZCZ": "building_subtype",
        "LKOND": "n_floors",
        "NAZWA": "name",
    }
    gdf = gdf.rename(columns={k: v for k, v in rename_map.items() if k in gdf.columns})

    keep = ["bdot_id", "building_type", "building_subtype", "n_floors", "name", "geometry"]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()

    gdf = gdf.to_crs(TARGET_CRS)

    os.makedirs(DATA_DIR, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="bdot_buildings")
    print(f"  -> {out_path} ({len(gdf):,} buildings)")
    return gdf


def extract_addresses(force=False):
    """Extract address points from canonical location to data/."""
    out_path = os.path.join(DATA_DIR, "address_points.gpkg")
    if os.path.exists(out_path) and not force:
        gdf = gpd.read_file(out_path)
        print(f"  Extracted address points: {len(gdf):,} (cached)")
        return gdf

    if not os.path.exists(ADDR_GPKG):
        print(f"  Source not found: {ADDR_GPKG}")
        return None

    print(f"  Reading {ADDR_GPKG}...")
    gdf = gpd.read_file(ADDR_GPKG)
    print(f"  Source: {len(gdf):,} features")

    # Reproject to target CRS
    if gdf.crs and gdf.crs.to_epsg() != 2180:
        gdf = gdf.to_crs(TARGET_CRS)

    os.makedirs(DATA_DIR, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="address_points")
    print(f"  -> {out_path} ({len(gdf):,} address points)")
    return gdf


def extract_districts(force=False):
    """Extract Warsaw district boundaries from BDOT ADJA_A shapefile."""
    out_path = os.path.join(DATA_DIR, "districts.gpkg")
    if os.path.exists(out_path) and not force:
        gdf = gpd.read_file(out_path)
        print(f"  Extracted districts: {len(gdf):,} (cached)")
        return gdf

    if not os.path.exists(BDOT_ADJA_SHP):
        print(f"  Source not found: {BDOT_ADJA_SHP}")
        return None

    print(f"  Reading {BDOT_ADJA_SHP}...")
    gdf = gpd.read_file(BDOT_ADJA_SHP)

    # Filter to dzielnica (district) level only
    gdf = gdf[gdf["RODZAJ"] == "dzielnica"].copy()
    gdf = gdf.rename(columns={"NAZWA": "district", "TERYT": "teryt"})
    gdf = gdf[["district", "teryt", "geometry"]].copy()
    gdf = gdf.to_crs(TARGET_CRS)

    os.makedirs(DATA_DIR, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG", layer="districts")
    print(f"  -> {out_path} ({len(gdf):,} districts)")
    return gdf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SOURCES = {
    "osm": {
        "label": "OSM buildings",
        "fetch": fetch_osm_buildings,
        "extract": extract_osm,
    },
    "bdot": {
        "label": "BDOT buildings",
        "fetch": fetch_bdot_buildings,
        "extract": extract_bdot,
    },
    "addresses": {
        "label": "Address points",
        "fetch": fetch_address_points,
        "extract": extract_addresses,
    },
    "districts": {
        "label": "District boundaries",
        "fetch": lambda force=False: None,  # no fetch needed, comes with BDOT
        "extract": extract_districts,
    },
}


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and prepare spatial reference data for housing price analysis")
    parser.add_argument("--only", choices=list(SOURCES.keys()),
                        help="Process only one dataset")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip fetching, only extract from canonical locations")
    parser.add_argument("--force", action="store_true",
                        help="Re-fetch/re-extract even if files exist")
    args = parser.parse_args()

    print("=" * 60)
    print("Geodata Pipeline — Warsaw Housing Analysis")
    print("=" * 60)
    print(f"Canonical data:  {QGIS_ROOT}")
    print(f"Project data:    {DATA_DIR}")
    print(f"Target CRS:      {TARGET_CRS}")
    print()

    targets = {args.only: SOURCES[args.only]} if args.only else SOURCES

    # Phase 1: Fetch to canonical locations
    if not args.extract_only:
        print("Phase 1: Fetch to canonical locations")
        print("-" * 40)
        for key, src in targets.items():
            print(f"\n[{src['label']}]")
            try:
                src["fetch"](force=args.force)
            except Exception as e:
                print(f"  ERROR: {e}")
        print()

    # Phase 2: Extract & clip to project data/
    print("Phase 2: Extract to project data/")
    print("-" * 40)
    results = {}
    for key, src in targets.items():
        print(f"\n[{src['label']}]")
        try:
            gdf = src["extract"](force=args.force)
            results[key] = len(gdf) if gdf is not None else 0
        except Exception as e:
            print(f"  ERROR: {e}")
            results[key] = 0

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for key, count in results.items():
        status = f"{count:,} features" if count > 0 else "FAILED"
        print(f"  {SOURCES[key]['label']:20s}: {status}")
    print(f"\nProject data dir: {DATA_DIR}")


if __name__ == "__main__":
    main()
