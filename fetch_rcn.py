"""
Pobiera dane z Rejestru Cen Nieruchomości (RCN) przez WFS GUGiK.

Warstwy: lokale (mieszkania), budynki, dzialki.
Domyślnie pobiera lokale dla Warszawy (bbox EPSG:2178).
Wynik: GeoPackage w output/rcn_lokale_warszawa.gpkg

Użycie:
  python fetch_rcn.py                     # Warsaw lokale
  python fetch_rcn.py --layer budynki     # Warsaw budynki
  python fetch_rcn.py --bbox-name polska  # cała Polska (uwaga: ~2.6M rekordów)
"""

import argparse
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import geopandas as gpd
import requests
from shapely.geometry import Point, Polygon
from tqdm import tqdm

# --- CONFIG ---

WFS_URL = "https://mapy.geoportal.gov.pl/wss/service/rcn"
PAGE_SIZE = 5000  # safe page size for the WFS
MAX_WORKERS = 6   # parallel requests (be gentle with government servers)

BBOXES = {
    "warszawa": "5770000,7485000,5810000,7515000,urn:ogc:def:crs:EPSG::2178",
    "polska": None,  # no bbox = entire country
}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


# --- WFS HELPERS ---

def get_total_count(layer, bbox):
    """Get total number of features matching the query."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": f"ms:{layer}",
        "resultType": "hits",
    }
    if bbox:
        params["bbox"] = bbox

    resp = requests.get(WFS_URL, params=params, timeout=60)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    return int(root.attrib.get("numberMatched", 0))


def fetch_page(layer, bbox, start_index):
    """Fetch one page of WFS features. Returns (start_index, raw_bytes) or (start_index, None) on error."""
    params = {
        "service": "WFS",
        "version": "2.0.0",
        "request": "GetFeature",
        "typeNames": f"ms:{layer}",
        "count": str(PAGE_SIZE),
        "STARTINDEX": str(start_index),
    }
    if bbox:
        params["bbox"] = bbox

    for attempt in range(3):
        try:
            resp = requests.get(WFS_URL, params=params, timeout=120)
            resp.raise_for_status()
            return (start_index, resp.content)
        except Exception as e:
            if attempt == 2:
                print(f"\n  FAILED page at offset {start_index}: {e}")
                return (start_index, None)
            time.sleep(2 ** attempt)


# --- GML PARSING ---

NS = {
    "wfs": "http://www.opengis.net/wfs/2.0",
    "ms": "http://mapserver.gis.umn.edu/mapserver",
    "gml": "http://www.opengis.net/gml/3.2",
}


def parse_lokale_page(xml_bytes):
    """Parse a WFS response page for ms:lokale into a list of dicts."""
    root = ET.fromstring(xml_bytes)
    records = []

    for member in root.findall("wfs:member", NS):
        feat = member.find("ms:lokale", NS)
        if feat is None:
            continue

        rec = {}

        # Geometry — Point (GML urn:ogc axis order is northing, easting; swap for Shapely x,y)
        pt = feat.find(".//gml:pos", NS)
        if pt is not None and pt.text:
            coords = pt.text.strip().split()
            if len(coords) >= 2:
                rec["geometry"] = Point(float(coords[1]), float(coords[0]))
        if "geometry" not in rec:
            continue

        # All text fields
        for child in feat:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ("boundedBy", "msGeometry"):
                continue
            rec[tag] = child.text

        records.append(rec)

    return records


def parse_budynki_page(xml_bytes):
    """Parse a WFS response page for ms:budynki into a list of dicts."""
    root = ET.fromstring(xml_bytes)
    records = []

    for member in root.findall("wfs:member", NS):
        feat = member.find("ms:budynki", NS)
        if feat is None:
            continue

        rec = {}

        # Geometry — Polygon (GML urn:ogc axis order is northing, easting; swap for Shapely x,y)
        pos_list = feat.find(".//gml:posList", NS)
        if pos_list is not None and pos_list.text:
            coords_flat = list(map(float, pos_list.text.strip().split()))
            pairs = [(coords_flat[i + 1], coords_flat[i]) for i in range(0, len(coords_flat), 2)]
            if len(pairs) >= 4:
                rec["geometry"] = Polygon(pairs)
        if "geometry" not in rec:
            # Try Point fallback
            pt = feat.find(".//gml:pos", NS)
            if pt is not None and pt.text:
                coords = pt.text.strip().split()
                if len(coords) >= 2:
                    rec["geometry"] = Point(float(coords[1]), float(coords[0]))
        if "geometry" not in rec:
            continue

        for child in feat:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ("boundedBy", "msGeometry"):
                continue
            rec[tag] = child.text

        records.append(rec)

    return records


def parse_dzialki_page(xml_bytes):
    """Parse a WFS response page for ms:dzialki into a list of dicts."""
    root = ET.fromstring(xml_bytes)
    records = []

    for member in root.findall("wfs:member", NS):
        feat = member.find("ms:dzialki", NS)
        if feat is None:
            continue

        rec = {}

        pos_list = feat.find(".//gml:posList", NS)
        if pos_list is not None and pos_list.text:
            coords_flat = list(map(float, pos_list.text.strip().split()))
            pairs = [(coords_flat[i + 1], coords_flat[i]) for i in range(0, len(coords_flat), 2)]
            if len(pairs) >= 4:
                rec["geometry"] = Polygon(pairs)
        if "geometry" not in rec:
            continue

        for child in feat:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag in ("boundedBy", "msGeometry"):
                continue
            rec[tag] = child.text

        records.append(rec)

    return records


PARSERS = {
    "lokale": parse_lokale_page,
    "budynki": parse_budynki_page,
    "dzialki": parse_dzialki_page,
}


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser(description="Fetch RCN data from GUGiK WFS")
    parser.add_argument("--layer", default="lokale", choices=["lokale", "budynki", "dzialki"])
    parser.add_argument("--bbox-name", default="warszawa", choices=list(BBOXES.keys()))
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    layer = args.layer
    bbox = BBOXES[args.bbox_name]
    parse_fn = PARSERS[layer]

    print(f"{'=' * 60}")
    print(f"RCN WFS Fetcher — {layer} / {args.bbox_name}")
    print(f"{'=' * 60}")

    # Step 1: count
    print("Counting features...")
    total = get_total_count(layer, bbox)
    print(f"Total features: {total:,}")

    if total == 0:
        print("Nothing to fetch.")
        return

    # Step 2: build page offsets
    offsets = list(range(0, total, PAGE_SIZE))
    print(f"Pages to fetch: {len(offsets)} (page size {PAGE_SIZE})")
    print(f"Parallel workers: {args.max_workers}")
    print()

    # Step 3: fetch all pages in parallel
    all_records = []
    failed_offsets = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetch_page, layer, bbox, offset): offset
            for offset in offsets
        }

        with tqdm(total=len(offsets), desc="Fetching", unit="page") as pbar:
            for future in as_completed(futures):
                offset = futures[future]
                start_idx, xml_bytes = future.result()

                if xml_bytes is None:
                    failed_offsets.append(start_idx)
                else:
                    records = parse_fn(xml_bytes)
                    all_records.extend(records)

                pbar.update(1)
                pbar.set_postfix(records=f"{len(all_records):,}")

    elapsed = time.time() - t0
    print(f"\nFetched {len(all_records):,} records in {elapsed:.0f}s")

    if failed_offsets:
        print(f"WARNING: {len(failed_offsets)} pages failed: {failed_offsets}")
        print("Retrying failed pages sequentially...")
        for offset in tqdm(failed_offsets, desc="Retrying"):
            _, xml_bytes = fetch_page(layer, bbox, offset)
            if xml_bytes:
                records = parse_fn(xml_bytes)
                all_records.extend(records)
        print(f"Total after retry: {len(all_records):,}")

    # Step 4: build GeoDataFrame and save
    print("\nBuilding GeoDataFrame...")
    gdf = gpd.GeoDataFrame(all_records, crs="EPSG:2180")

    # Convert numeric columns
    numeric_cols = [
        "tran_cena_brutto", "tran_vat", "nier_cena_brutto", "nier_vat",
        "lok_liczba_izb", "lok_nr_kond", "lok_pow_uzyt", "lok_pow_przyn",
        "lok_cena_brutto", "lok_vat", "nier_pow_gruntu",
    ]
    for col in numeric_cols:
        if col in gdf.columns:
            gdf[col] = gdf[col].apply(lambda x: float(x) if x else None)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, f"rcn_{layer}_{args.bbox_name}.gpkg")
    gdf.to_file(out_path, driver="GPKG", layer=layer)

    print(f"\nSaved {len(gdf):,} features -> {out_path}")
    print(f"Columns: {list(gdf.columns)}")

    # Quick stats for lokale
    if layer == "lokale" and "lok_cena_brutto" in gdf.columns:
        prices = gdf["lok_cena_brutto"].dropna()
        areas = gdf["lok_pow_uzyt"].dropna()
        print(f"\nPrice stats (PLN):")
        print(f"  Count:  {len(prices):,}")
        print(f"  Median: {prices.median():,.0f}")
        print(f"  Mean:   {prices.mean():,.0f}")
        if len(areas) > 0:
            print(f"Area stats (m²):")
            print(f"  Median: {areas.median():.1f}")
            print(f"  Mean:   {areas.mean():.1f}")

        # Date range
        if "dok_data" in gdf.columns:
            dates = gdf["dok_data"].dropna()
            print(f"Date range: {dates.min()} — {dates.max()}")


if __name__ == "__main__":
    main()
