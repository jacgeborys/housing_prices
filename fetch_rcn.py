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
    "warszawa": "5770000,7485000,5810000,7521000,urn:ogc:def:crs:EPSG::2178",   # east edge 7,521,000 (was 7,515,000, 2026-10-27)
    # The "warszawa" box stops at 21.22 E (EPSG:2180 x = 652,044) but the city reaches x = 655,260: half of Wesoła and the
    # east of Wawer were missing until 2026-10-27. This strip (1.5 km overlap) is fetched on its own and merged by gid.
    "warszawa_east": "5770000,7514000,5810000,7521000,urn:ogc:def:crs:EPSG::2178",
    # Gdańsk (2026-10-29): 18.38-18.98 E, 54.25-54.53 N, also takes in Sopot, Gdynia and the gminas round the city
    "gdansk": "6016000,7329000,6047000,7370000,urn:ogc:def:crs:EPSG::2178",
    # Lublin (2026-10-31): 22.44-22.71 E, 51.13-51.31 N (city 22.45-22.68, 51.14-51.30), takes in a strip of Świdnik and the gminas
    "lublin": "5667098,7600791,5687524,7619223,urn:ogc:def:crs:EPSG::2178",
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


def count_retry(layer, bbox):
    for attempt in range(4):
        try:
            return get_total_count(layer, bbox)
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def split_cells(layer, bbox, max_hits, workers):
    """Cut the box into cells that each hold <= max_hits features, so every cell is one request. WFS paging with
    STARTINDEX and no sort order returned duplicate and missing records (2026-10-27: 3,900 duplicate gids and ~8,700
    records short of the service's own count in the old paged download)."""
    parts = bbox.split(",")
    n0, e0, n1, e1 = map(float, parts[:4])
    crs = parts[4]
    level, leaves = [(n0, e0, n1, e1)], []
    fmt = lambda c: f"{c[0]:.0f},{c[1]:.0f},{c[2]:.0f},{c[3]:.0f},{crs}"
    with ThreadPoolExecutor(max_workers=workers) as ex:
        while level:
            counts = list(ex.map(lambda c: count_retry(layer, fmt(c)), level))
            nxt = []
            for c, h in zip(level, counts):
                if h == 0:
                    continue
                if h <= max_hits or (c[2] - c[0]) < 120:
                    leaves.append((fmt(c), h))
                else:
                    nm, em = (c[0] + c[2]) / 2, (c[1] + c[3]) / 2
                    nxt += [(c[0], c[1], nm, em), (c[0], em, nm, c[3]), (nm, c[1], c[2], em), (nm, em, c[2], c[3])]
            level = nxt
    return leaves


def element_text(el):
    """Text of a WFS field. Dates (DOK_DATA) come as <ms:DOK_DATA><gml:timePosition>2015-02-27</gml:timePosition>, so
    the text is in a child: el.text alone returned None for every deed date (found 2026-10-27)."""
    t = el.text
    if t is None or not t.strip():
        t = "".join(el.itertext()).strip()
    return t or None


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
            rec[tag] = element_text(child)

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
            rec[tag] = element_text(child)

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
            rec[tag] = element_text(child)

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
    parser.add_argument("--paged", action="store_true", help="old STARTINDEX paging (loses and duplicates records)")
    args = parser.parse_args()

    layer = args.layer
    bbox = BBOXES[args.bbox_name]
    parse_fn = PARSERS[layer]

    print(f"{'=' * 60}")
    print(f"RCN WFS Fetcher — {layer} / {args.bbox_name}")
    print(f"{'=' * 60}")

    all_records, failed_offsets = [], []
    t0 = time.time()
    if not args.paged:
        total = count_retry(layer, bbox)
        print(f"Service count for the box: {total:,}; cutting into cells of <= 4,500 ...")
        leaves = split_cells(layer, bbox, 4500, args.max_workers)
        print(f"{len(leaves)} cells, {sum(h for _, h in leaves):,} features (sum of cell counts, border features count twice)")
        jobs = [(cell, off) for cell, h in leaves for off in range(0, h, PAGE_SIZE)]
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(fetch_page, layer, cell, off): (cell, off) for cell, off in jobs}
            with tqdm(total=len(jobs), desc="Fetching", unit="cell") as pbar:
                for future in as_completed(futures):
                    _, xml_bytes = future.result()
                    if xml_bytes is None:
                        failed_offsets.append(futures[future])
                    else:
                        all_records.extend(parse_fn(xml_bytes))
                    pbar.update(1)
        for cell, off in failed_offsets:
            _, xml_bytes = fetch_page(layer, cell, off)
            if xml_bytes:
                all_records.extend(parse_fn(xml_bytes))
        print(f"Fetched {len(all_records):,} records in {time.time() - t0:.0f}s")
        seen, uniq = set(), []
        for r in all_records:
            if r.get("gid") not in seen:
                seen.add(r.get("gid"))
                uniq.append(r)
        print(f"Unique gid: {len(uniq):,} (service count {total:,}; difference {total - len(uniq):,})")
        all_records = uniq
    else:
        print("Counting features...")
        total = get_total_count(layer, bbox)
        print(f"Total features: {total:,}")
        if total == 0:
            print("Nothing to fetch.")
            return
        offsets = list(range(0, total, PAGE_SIZE))
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(fetch_page, layer, bbox, offset): offset for offset in offsets}
            with tqdm(total=len(offsets), desc="Fetching", unit="page") as pbar:
                for future in as_completed(futures):
                    start_idx, xml_bytes = future.result()
                    if xml_bytes is None:
                        failed_offsets.append(start_idx)
                    else:
                        all_records.extend(parse_fn(xml_bytes))
                    pbar.update(1)
        for offset in failed_offsets:
            _, xml_bytes = fetch_page(layer, bbox, offset)
            if xml_bytes:
                all_records.extend(parse_fn(xml_bytes))
        print(f"Fetched {len(all_records):,} records in {time.time() - t0:.0f}s")

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
