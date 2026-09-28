"""
Builds a visualization-ready GeoPackage by joining RCN transactions to buildings.

Uses three matching strategies in order:
  1. Spatial join (point within OSM building polygon)
  2. Buffer match (25m) for near-miss spatial joins
  3. Address-based match via address points as intermediary

Data sources:
  - OSM buildings (329K, includes building:part with 3D attributes)
  - BDOT buildings (155K, official GUGiK footprints) — used as fallback
  - Address points (127K, Warsaw WFS)
  - RCN transactions (381K lokale)

Output (separate files due to fiona version):
  1. buildings_prices — building polygons with median price/m², stats, 3D height
  2. transactions — individual points with price/m² and match status
  3. unmatched — transactions that didn't match any building
"""

import os
import re

import geopandas as gpd
import numpy as np
import pandas as pd

# --- Paths ---
RCN_PATH = os.path.join("output", "rcn_lokale_warszawa.gpkg")
OSM_PATH = os.path.join("data", "osm_buildings.gpkg")
BDOT_PATH = os.path.join("data", "bdot_buildings.gpkg")
ADDR_PATH = os.path.join("data", "address_points.gpkg")
OUTPUT_PATH = os.path.join("output", "warsaw_prices_viz.gpkg")

FLOOR_HEIGHT_M = 3.0
NEAR_BUFFER_M = 25


def load_transactions():
    """Load and clean residential transactions."""
    print("Loading RCN transactions...")
    gdf = gpd.read_file(RCN_PATH)
    # Normalize column names to lowercase
    gdf.columns = [c.lower() for c in gdf.columns]
    print(f"  Total: {len(gdf):,}")

    # Convert numeric columns from strings
    for col in ["lok_cena_brutto", "lok_pow_uzyt", "lok_nr_kond", "lok_liczba_izb"]:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    # Residential only
    gdf = gdf[gdf["lok_funkcja"] == "mieszkalna"].copy()
    print(f"  Residential: {len(gdf):,}")

    # Compute price per m²
    gdf["price_m2"] = gdf["lok_cena_brutto"] / gdf["lok_pow_uzyt"]

    # Filter outliers
    mask = gdf["price_m2"].between(1000, 100000) & gdf["lok_pow_uzyt"].between(10, 500)
    gdf = gdf[mask].copy()
    print(f"  After filtering: {len(gdf):,}")

    # Parse year
    gdf["year"] = gdf["dok_data"].str[:4].apply(
        lambda x: int(x) if x and x.isdigit() and int(x) > 2000 else None
    )

    # Clean floor
    gdf["floor"] = gdf["lok_nr_kond"]
    gdf.loc[gdf["floor"] > 50, "floor"] = None

    # Market type
    gdf["market"] = gdf["tran_rodzaj_rynku"].map({
        "pierwotny": "primary",
        "wtorny": "secondary",
    })

    # Normalize address for matching
    gdf["addr_norm"] = gdf["lok_adres"].apply(normalize_address)

    keep = [
        "geometry", "lok_cena_brutto", "lok_pow_uzyt", "price_m2",
        "lok_liczba_izb", "floor", "year", "market",
        "lok_adres", "addr_norm", "tran_lokalny_id_iip",
    ]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()
    gdf.rename(columns={
        "lok_cena_brutto": "price",
        "lok_pow_uzyt": "area_m2",
        "lok_liczba_izb": "rooms",
    }, inplace=True)

    return gdf


def normalize_address(addr):
    """Normalize RCN address string for matching with address points.

    RCN format: 'MSC:Warszawa;UL:ulica Ciasna;NR_PORZ:15'
    Address points: street='ulica Ciasna', number='15'
    """
    if not addr or not isinstance(addr, str):
        return None
    # Parse structured RCN format
    parts = {}
    for part in addr.split(";"):
        if ":" in part:
            key, val = part.split(":", 1)
            parts[key.strip()] = val.strip()
    street = parts.get("UL", "")
    number = parts.get("NR_PORZ", "")
    if not street or not number:
        return None
    # Strip just the leading numeric part of the number (ignore letter suffixes like 15A)
    num_match = re.match(r"(\d+)", number)
    if not num_match:
        return None
    return f"{street.lower()}|{num_match.group(1)}"


def load_osm_buildings():
    """Load OSM building polygons (main buildings only, not parts)."""
    print("\nLoading OSM buildings...")
    gdf = gpd.read_file(OSM_PATH)
    print(f"  Total features: {len(gdf):,}")

    # Separate main buildings from building:part
    main_mask = gdf["building"].notna() & (gdf["building"] != "")
    parts_mask = gdf["building_part"].notna() & (gdf["building_part"] != "")

    main = gdf[main_mask].copy()
    parts = gdf[parts_mask & ~main_mask].copy()
    print(f"  Main buildings: {len(main):,}")
    print(f"  Building parts: {len(parts):,}")

    # Compute height for main buildings
    main["height_m"] = main["height"].apply(_parse_height)
    # Fallback to levels * floor height
    no_height = main["height_m"].isna()
    levels = main.loc[no_height, "building_levels"].apply(
        lambda x: float(x) if x and str(x).replace(".", "").isdigit() else None
    )
    main.loc[no_height, "height_m"] = levels * FLOOR_HEIGHT_M
    # Default: 1 floor
    main["height_m"] = main["height_m"].fillna(FLOOR_HEIGHT_M)

    # Build address key for address-based matching
    main["osm_addr"] = main.apply(
        lambda r: f"{r['addr_street']}|{r['addr_housenumber']}"
        if pd.notna(r.get("addr_street")) and pd.notna(r.get("addr_housenumber"))
        else None,
        axis=1,
    )
    main["osm_addr"] = main["osm_addr"].str.lower()

    return main, parts


def _parse_height(val):
    """Parse height string like '237' or '12.5 m' to float."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    val = str(val).strip().replace(" m", "").replace("m", "")
    try:
        return float(val)
    except ValueError:
        return None


def load_bdot_buildings():
    """Load BDOT building polygons as fallback."""
    print("\nLoading BDOT buildings...")
    gdf = gpd.read_file(BDOT_PATH)
    gdf.columns = [c.lower() for c in gdf.columns]
    print(f"  Total: {len(gdf):,}")

    if "lkond" in gdf.columns:
        gdf["height_m"] = gdf["lkond"].fillna(1).clip(lower=1) * FLOOR_HEIGHT_M
    else:
        gdf["height_m"] = FLOOR_HEIGHT_M

    return gdf


def load_address_points():
    """Load address points for address-based matching."""
    print("\nLoading address points...")
    gdf = gpd.read_file(ADDR_PATH)
    print(f"  Total: {len(gdf):,}")

    # Build normalized address key (strip letter suffix from number)
    def make_addr_key(r):
        street = r.get("street")
        number = r.get("number")
        if pd.isna(street) or pd.isna(number):
            return None
        num_match = re.match(r"(\d+)", str(number))
        if not num_match:
            return None
        return f"{street.lower()}|{num_match.group(1)}"

    gdf["addr_key"] = gdf.apply(make_addr_key, axis=1)
    gdf = gdf[gdf["addr_key"].notna()].copy()
    print(f"  With valid address key: {len(gdf):,}")

    return gdf


def spatial_join(transactions, buildings, label="OSM"):
    """Join transactions to buildings with two-pass spatial matching.
    Returns (matched_map, transactions with 'match' column updated).
    matched_map: Series mapping transaction index -> building index.
    """
    print(f"\nSpatial join: transactions -> {label} buildings...")

    # Pass 1: exact "within"
    joined = gpd.sjoin(transactions, buildings[["geometry"]], how="left", predicate="within")
    mask_matched = joined["index_right"].notna()
    exact_map = joined.loc[mask_matched, "index_right"].astype(int)
    exact_map = exact_map[~exact_map.index.duplicated(keep="first")]
    print(f"  Pass 1 (within):    {len(exact_map):,} / {len(transactions):,}")

    # Pass 2: buffer
    unmatched_idx = transactions.index.difference(exact_map.index)
    unmatched_txn = transactions.loc[unmatched_idx]

    if len(unmatched_txn) > 0:
        buildings_buf = buildings[["geometry"]].copy()
        buildings_buf["geometry"] = buildings_buf.geometry.buffer(NEAR_BUFFER_M)
        joined_buf = gpd.sjoin(unmatched_txn, buildings_buf, how="left", predicate="within")
        joined_buf = joined_buf[~joined_buf.index.duplicated(keep="first")]
        near_map = joined_buf.loc[joined_buf["index_right"].notna(), "index_right"].astype(int)
        print(f"  Pass 2 ({NEAR_BUFFER_M}m buf): {len(near_map):,} more")
    else:
        near_map = pd.Series(dtype=int)

    combined = pd.concat([exact_map, near_map])
    combined.name = "bld_idx"

    # Tag transactions
    transactions["match"] = "unmatched"
    transactions.loc[exact_map.index, "match"] = "exact"
    transactions.loc[near_map.index, "match"] = "buffer"

    total = len(combined)
    print(f"  Total spatial: {total:,} ({100*total/len(transactions):.1f}%)")

    return combined


def address_match(transactions, buildings, address_points):
    """Match remaining unmatched transactions via address points.

    Strategy:
    1. For each unmatched transaction, find address points with matching normalized address
    2. For each matching address point, find the building it falls within
    3. Assign the transaction to that building
    """
    unmatched = transactions[transactions["match"] == "unmatched"]
    if len(unmatched) == 0 or unmatched["addr_norm"].isna().all():
        print("\n  No address matching possible.")
        return pd.Series(dtype=int)

    print(f"\nAddress matching for {len(unmatched):,} unmatched transactions...")

    # Build address point -> building mapping
    print("  Joining address points to buildings...")
    addr_in_bldg = gpd.sjoin(address_points, buildings[["geometry"]], how="inner", predicate="within")
    addr_in_bldg = addr_in_bldg[~addr_in_bldg["addr_key"].duplicated(keep="first")]
    addr_to_bldg = addr_in_bldg.set_index("addr_key")["index_right"]
    print(f"  Address points matched to buildings: {len(addr_to_bldg):,}")

    # Match transactions via address
    txn_with_addr = unmatched[unmatched["addr_norm"].notna()].copy()
    matched_bldg = txn_with_addr["addr_norm"].map(addr_to_bldg)
    matched_bldg = matched_bldg.dropna().astype(int)
    print(f"  Transactions matched via address: {len(matched_bldg):,}")

    # Update match status
    transactions.loc[matched_bldg.index, "match"] = "address"
    matched_bldg.name = "bld_idx"

    return matched_bldg


def aggregate_to_buildings(transactions, buildings, match_map):
    """Aggregate transaction stats per building."""
    print("\n  Aggregating per building...")
    matched_txn = transactions.loc[match_map.index].copy()
    matched_txn["bld_idx"] = match_map

    stats = matched_txn.groupby("bld_idx").agg(
        median_price_m2=("price_m2", "median"),
        mean_price_m2=("price_m2", "mean"),
        min_price_m2=("price_m2", "min"),
        max_price_m2=("price_m2", "max"),
        n_transactions=("price_m2", "count"),
        median_price=("price", "median"),
        median_area=("area_m2", "median"),
        median_floor=("floor", "median"),
        latest_year=("year", "max"),
        pct_primary=("market", lambda x: (x == "primary").mean() * 100),
    ).reset_index()

    enriched = buildings.merge(stats, left_index=True, right_on="bld_idx", how="inner")
    enriched.drop(columns=["bld_idx"], inplace=True)
    enriched = gpd.GeoDataFrame(enriched, geometry="geometry", crs=buildings.crs)

    # Price category
    enriched["price_cat"] = pd.cut(
        enriched["median_price_m2"],
        bins=[0, 6000, 8000, 10000, 12000, 15000, 20000, 200000],
        labels=["1_under_6k", "2_6k-8k", "3_8k-10k", "4_10k-12k", "5_12k-15k", "6_15k-20k", "7_over_20k"],
    ).astype(str)

    print(f"  Buildings with transactions: {len(enriched):,}")
    return enriched


def save_gpkg(gdf, path, layer):
    """Save a GeoDataFrame to a GeoPackage file (overwrite)."""
    if os.path.exists(path):
        os.remove(path)
    gdf.to_file(path, driver="GPKG", layer=layer)


def main():
    print("=" * 60)
    print("Building Warsaw Price Visualization")
    print("=" * 60)

    transactions = load_transactions()
    osm_buildings, osm_parts = load_osm_buildings()
    bdot_buildings = load_bdot_buildings()
    address_points = load_address_points()

    # --- Matching pipeline ---

    # Step 1: Spatial match to OSM buildings
    spatial_map = spatial_join(transactions, osm_buildings, label="OSM")

    # Step 2: Address-based match for remaining
    addr_map = address_match(transactions, osm_buildings, address_points)

    # Step 3: Spatial match remaining to BDOT as fallback
    still_unmatched = transactions[transactions["match"] == "unmatched"].copy()
    if len(still_unmatched) > 0:
        print(f"\nFallback: matching {len(still_unmatched):,} remaining to BDOT...")
        bdot_map = spatial_join(still_unmatched, bdot_buildings, label="BDOT")
        # Update match status on main transactions df
        for idx in bdot_map.index:
            status = still_unmatched.loc[idx, "match"]
            if status in ("exact", "buffer"):
                transactions.loc[idx, "match"] = f"bdot_{status}"
    else:
        bdot_map = pd.Series(dtype=int)

    # Summary
    match_counts = transactions["match"].value_counts()
    total = len(transactions)
    print(f"\n{'=' * 40}")
    print("Match summary:")
    for status, count in match_counts.sort_index().items():
        print(f"  {status:>15s}: {count:>7,} ({100*count/total:5.1f}%)")
    print(f"{'=' * 40}")

    # Aggregate OSM matches
    osm_all_map = pd.concat([spatial_map, addr_map])
    enriched_osm = aggregate_to_buildings(transactions, osm_buildings, osm_all_map)

    # Aggregate BDOT matches (if any)
    if len(bdot_map) > 0:
        enriched_bdot = aggregate_to_buildings(transactions, bdot_buildings, bdot_map)
    else:
        enriched_bdot = gpd.GeoDataFrame()

    # Combine OSM + BDOT enriched buildings
    # Keep only columns that exist in both
    if len(enriched_bdot) > 0:
        common_cols = ["geometry", "height_m", "median_price_m2", "mean_price_m2",
                       "min_price_m2", "max_price_m2", "n_transactions", "median_price",
                       "median_area", "median_floor", "latest_year", "pct_primary", "price_cat"]
        osm_out = enriched_osm[[c for c in common_cols if c in enriched_osm.columns]]
        osm_out = osm_out.copy()
        osm_out["source"] = "osm"
        bdot_out = enriched_bdot[[c for c in common_cols if c in enriched_bdot.columns]]
        bdot_out = bdot_out.copy()
        bdot_out["source"] = "bdot"
        enriched = gpd.GeoDataFrame(
            pd.concat([osm_out, bdot_out], ignore_index=True),
            geometry="geometry", crs=osm_buildings.crs,
        )
    else:
        enriched = enriched_osm.copy()
        enriched["source"] = "osm"

    # --- Save outputs ---
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\nSaving...")

    save_gpkg(enriched, OUTPUT_PATH, "buildings_prices")
    print(f"  buildings_prices: {len(enriched):,} polygons -> {OUTPUT_PATH}")

    # Drop internal columns before saving transactions
    txn_out = transactions.drop(columns=["addr_norm"], errors="ignore")
    txn_path = OUTPUT_PATH.replace(".gpkg", "_transactions.gpkg")
    save_gpkg(txn_out, txn_path, "transactions")
    print(f"  transactions: {len(txn_out):,} points -> {txn_path}")

    unmatched = txn_out[txn_out["match"] == "unmatched"]
    unm_path = OUTPUT_PATH.replace(".gpkg", "_unmatched.gpkg")
    save_gpkg(unmatched, unm_path, "unmatched")
    print(f"  unmatched: {len(unmatched):,} points -> {unm_path}")

    # Save building parts separately for 3D viz
    if len(osm_parts) > 0:
        parts_path = OUTPUT_PATH.replace(".gpkg", "_parts.gpkg")
        osm_parts_out = osm_parts.copy()
        osm_parts_out["height_m"] = osm_parts_out["height"].apply(_parse_height)
        osm_parts_out["min_height_m"] = osm_parts_out["min_height"].apply(_parse_height)
        save_gpkg(osm_parts_out, parts_path, "building_parts")
        print(f"  building_parts: {len(osm_parts_out):,} polygons -> {parts_path}")

    # --- Summary stats ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Price per m² distribution (building medians):")
    for cat, cnt in enriched["price_cat"].value_counts().sort_index().items():
        print(f"  {cat:>12s}: {cnt:,} buildings")
    print(f"\nMedian price/m² across all buildings: "
          f"{enriched['median_price_m2'].median():,.0f} PLN")
    print(f"Buildings with 10+ transactions: "
          f"{(enriched['n_transactions'] >= 10).sum():,}")

    print(f"\n3D in QGIS:")
    print(f"  1. Add {os.path.basename(OUTPUT_PATH)} -> buildings_prices")
    print(f"  2. View -> 3D Map View -> New 3D Map View")
    print(f"  3. Layer Properties -> 3D View:")
    print(f"     - Altitude clamping: Relative to ground")
    print(f"     - Extrusion height: field 'height_m'")
    print(f"  4. Symbology -> Graduated -> Column: median_price_m2")
    print(f"     - Color ramp: Spectral (inverted) or RdYlGn (inverted)")
    if len(osm_parts) > 0:
        parts_path = OUTPUT_PATH.replace(".gpkg", "_parts.gpkg")
        print(f"  5. For detailed 3D (PKiN etc): add {os.path.basename(parts_path)}")
        print(f"     - Extrusion height: height_m, base height: min_height_m")


if __name__ == "__main__":
    main()
