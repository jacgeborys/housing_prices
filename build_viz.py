"""
Builds a visualization-ready GeoPackage by joining RCN transactions to BDOT buildings.

Creates two layers:
  1. buildings_prices — BDOT building polygons enriched with median price/m², transaction
     count, floor count, and price category for styling
  2. transactions — individual RCN lokale points with price/m² for detailed inspection

Output: output/warsaw_prices_viz.gpkg
"""

import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

# --- Paths ---
RCN_PATH = os.path.join("output", "rcn_lokale_warszawa.gpkg")
BDOT_PATH = os.path.join("D:", os.sep, "QGIS", "bdot_basemap", "Warszawa",
                         "PL.PZGiK.330.1465__OT_BUBD_A.shp")
OUTPUT_PATH = os.path.join("output", "warsaw_prices_viz.gpkg")


def load_transactions():
    """Load and clean residential transactions."""
    print("Loading RCN transactions...")
    gdf = gpd.read_file(RCN_PATH)
    print(f"  Total: {len(gdf):,}")

    # Residential only
    gdf = gdf[gdf["lok_funkcja"] == "mieszkalna"].copy()
    print(f"  Residential: {len(gdf):,}")

    # Compute price per m²
    gdf["price_m2"] = gdf["lok_cena_brutto"] / gdf["lok_pow_uzyt"]

    # Filter outliers (keep 1K-100K PLN/m²)
    mask = gdf["price_m2"].between(1000, 100000) & gdf["lok_pow_uzyt"].between(10, 500)
    gdf = gdf[mask].copy()
    print(f"  After filtering: {len(gdf):,}")

    # Parse year from dok_data
    gdf["year"] = gdf["dok_data"].str[:4].apply(
        lambda x: int(x) if x and x.isdigit() and int(x) > 2000 else None
    )

    # Clean up floor
    gdf["floor"] = gdf["lok_nr_kond"]
    gdf.loc[gdf["floor"] > 50, "floor"] = None  # data errors

    # Market type shorthand
    gdf["market"] = gdf["tran_rodzaj_rynku"].map({
        "pierwotny": "primary",
        "wtorny": "secondary",
    })

    # Keep useful columns
    keep = [
        "geometry", "lok_cena_brutto", "lok_pow_uzyt", "price_m2",
        "lok_liczba_izb", "floor", "year", "market",
        "lok_adres", "tran_lokalny_id_iip",
    ]
    gdf = gdf[[c for c in keep if c in gdf.columns]].copy()
    gdf.rename(columns={
        "lok_cena_brutto": "price",
        "lok_pow_uzyt": "area_m2",
        "lok_liczba_izb": "rooms",
    }, inplace=True)

    return gdf


def load_buildings():
    """Load BDOT building polygons."""
    print("\nLoading BDOT buildings...")
    gdf = gpd.read_file(BDOT_PATH,
                        columns=["LOKALNYID", "FUNOGBUD", "FUNSZCZ", "LKOND", "NAZWA", "geometry"])
    print(f"  Total: {len(gdf):,}")

    gdf.rename(columns={
        "LOKALNYID": "bdot_id",
        "FUNOGBUD": "building_type",
        "FUNSZCZ": "building_subtype",
        "LKOND": "n_floors",
        "NAZWA": "name",
    }, inplace=True)

    # Ensure CRS is recognized as EPSG:2180
    gdf = gdf.to_crs("EPSG:2180")

    return gdf


def spatial_join(transactions, buildings):
    """Join transactions to buildings, compute per-building stats."""
    print("\nSpatial join: transactions → buildings...")

    # Join each transaction to the building polygon it falls within
    joined = gpd.sjoin(transactions, buildings, how="left", predicate="within")
    matched = joined["index_right"].notna().sum()
    print(f"  Matched: {matched:,} / {len(transactions):,} ({100*matched/len(transactions):.1f}%)")

    # Per-building aggregation
    print("  Aggregating per building...")
    building_ids = joined[joined["index_right"].notna()]["index_right"].astype(int)

    stats = joined[joined["index_right"].notna()].groupby("index_right").agg(
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

    stats["index_right"] = stats["index_right"].astype(int)

    # Merge stats back to buildings
    enriched = buildings.merge(stats, left_index=True, right_on="index_right", how="inner")
    enriched.drop(columns=["index_right"], inplace=True)
    enriched = gpd.GeoDataFrame(enriched, geometry="geometry", crs=buildings.crs)

    # Price category for easy styling
    enriched["price_cat"] = pd.cut(
        enriched["median_price_m2"],
        bins=[0, 6000, 8000, 10000, 12000, 15000, 20000, 200000],
        labels=["<6k", "6-8k", "8-10k", "10-12k", "12-15k", "15-20k", ">20k"],
    ).astype(str)

    print(f"  Buildings with transactions: {len(enriched):,}")
    return enriched, joined


def main():
    print("=" * 60)
    print("Building Warsaw Price Visualization")
    print("=" * 60)

    transactions = load_transactions()
    buildings = load_buildings()
    enriched_buildings, joined = spatial_join(transactions, buildings)

    # Save
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    print(f"\nSaving to {OUTPUT_PATH}...")

    enriched_buildings.to_file(OUTPUT_PATH, driver="GPKG", layer="buildings_prices")
    print(f"  buildings_prices: {len(enriched_buildings):,} polygons")

    txn_path = OUTPUT_PATH.replace(".gpkg", "_transactions.gpkg")
    if os.path.exists(txn_path):
        os.remove(txn_path)
    transactions.to_file(txn_path, driver="GPKG", layer="transactions")
    print(f"  transactions: {len(transactions):,} points -> {txn_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Price per m² distribution (building medians):")
    for cat in enriched_buildings["price_cat"].value_counts().sort_index().items():
        print(f"  {cat[0]:>8s}: {cat[1]:,} buildings")
    print(f"\nMedian price/m² across all buildings: "
          f"{enriched_buildings['median_price_m2'].median():,.0f} PLN")
    print(f"Buildings with 10+ transactions: "
          f"{(enriched_buildings['n_transactions'] >= 10).sum():,}")

    print(f"\nOpen in QGIS: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
