"""
Exploratory analysis of Warsaw residential transactions (RCN data).

Generates charts covering:
  - Price/m² distribution and trends
  - Room counts, floor numbers, apartment areas
  - District-level comparisons (small vs spacious apartments)
  - Outlier detection and data quality
  - Market segmentation (primary vs secondary)

Output: output/analysis/ folder with PNG charts
"""

import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

OUTPUT_DIR = os.path.join("output", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})


def load_data():
    """Load and prepare transaction data with district assignment."""
    print("Loading data...")
    rcn = gpd.read_file("output/rcn_lokale_warszawa.gpkg")
    rcn.columns = [c.lower() for c in rcn.columns]
    for col in ["lok_cena_brutto", "lok_pow_uzyt", "lok_nr_kond", "lok_liczba_izb"]:
        rcn[col] = pd.to_numeric(rcn[col], errors="coerce")

    res = rcn[rcn["lok_funkcja"] == "mieszkalna"].copy()
    res["price_m2"] = res["lok_cena_brutto"] / res["lok_pow_uzyt"]
    res["year"] = res["dok_data"].str[:4].apply(
        lambda x: int(x) if x and x.isdigit() and int(x) > 2000 else None
    )
    res["rooms"] = res["lok_liczba_izb"]
    res["floor"] = res["lok_nr_kond"]
    res["area"] = res["lok_pow_uzyt"]
    res["price"] = res["lok_cena_brutto"]
    res["market"] = res["tran_rodzaj_rynku"].map({
        "rynekPierwotny": "primary",
        "rynekWtorny": "secondary",
    })

    # Assign districts via nearest address point
    addr = gpd.read_file("data/address_points.gpkg")
    print("  Assigning districts via nearest address point...")
    res_pts = res[["geometry"]].copy()
    nearest = gpd.sjoin_nearest(res_pts, addr[["geometry", "district"]], how="left", max_distance=500)
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    res["district"] = nearest["district"]

    print(f"  Residential transactions: {len(res):,}")
    print(f"  With district: {res['district'].notna().sum():,}")
    return res


def plot_price_distribution(df):
    """Price/m² histogram and box plot."""
    valid = df[df["price_m2"].between(1000, 100000)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram
    ax = axes[0]
    ax.hist(valid["price_m2"], bins=100, color="#2196F3", alpha=0.7, edgecolor="white")
    ax.axvline(valid["price_m2"].median(), color="red", linestyle="--", label=f'Median: {valid["price_m2"].median():,.0f}')
    ax.axvline(valid["price_m2"].mean(), color="orange", linestyle="--", label=f'Mean: {valid["price_m2"].mean():,.0f}')
    ax.set_xlabel("Price per m² (PLN)")
    ax.set_ylabel("Number of transactions")
    ax.set_title("Price per m² distribution")
    ax.legend()
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # By year box plot
    ax = axes[1]
    yearly = valid[valid["year"].between(2006, 2026)]
    yearly.boxplot(column="price_m2", by="year", ax=ax, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor="#2196F3", alpha=0.5))
    ax.set_title("Price per m² by year")
    ax.set_xlabel("Year")
    ax.set_ylabel("PLN / m²")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.suptitle("")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_price_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  01_price_distribution.png")


def plot_rooms_and_area(df):
    """Room count and area distributions."""
    valid = df[df["price_m2"].between(1000, 100000) & df["area"].between(10, 300)]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Room count distribution
    ax = axes[0, 0]
    rooms = valid["rooms"].dropna()
    rooms = rooms[rooms.between(1, 7)]
    counts = rooms.value_counts().sort_index()
    ax.bar(counts.index, counts.values, color="#4CAF50", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Number of rooms")
    ax.set_ylabel("Transactions")
    ax.set_title(f"Room count distribution (n={len(rooms):,})")
    for i, v in zip(counts.index, counts.values):
        ax.text(i, v + 500, f"{v:,}", ha="center", fontsize=9)

    # Area histogram
    ax = axes[0, 1]
    ax.hist(valid["area"], bins=80, color="#FF9800", alpha=0.7, edgecolor="white", range=(10, 200))
    ax.axvline(valid["area"].median(), color="red", linestyle="--",
               label=f'Median: {valid["area"].median():.0f} m²')
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Transactions")
    ax.set_title("Apartment area distribution")
    ax.legend()

    # Area by room count
    ax = axes[1, 0]
    room_area = valid[valid["rooms"].between(1, 6)].copy()
    room_area["rooms_label"] = room_area["rooms"].astype(int).astype(str) + " rooms"
    room_area.boxplot(column="area", by="rooms", ax=ax, showfliers=False,
                      patch_artist=True, boxprops=dict(facecolor="#FF9800", alpha=0.5))
    ax.set_title("Area by room count")
    ax.set_xlabel("Rooms")
    ax.set_ylabel("Area (m²)")
    fig.suptitle("")

    # Price/m² by room count
    ax = axes[1, 1]
    room_area.boxplot(column="price_m2", by="rooms", ax=ax, showfliers=False,
                      patch_artist=True, boxprops=dict(facecolor="#2196F3", alpha=0.5))
    ax.set_title("Price/m² by room count")
    ax.set_xlabel("Rooms")
    ax.set_ylabel("PLN / m²")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.suptitle("")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_rooms_and_area.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  02_rooms_and_area.png")


def plot_floors(df):
    """Floor number analysis."""
    valid = df[df["price_m2"].between(1000, 100000)]
    floors = valid[valid["floor"].between(0, 30)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Floor distribution
    ax = axes[0]
    fcounts = floors["floor"].value_counts().sort_index()
    ax.bar(fcounts.index, fcounts.values, color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Floor number")
    ax.set_ylabel("Transactions")
    ax.set_title(f"Floor distribution (n={len(floors):,})")

    # Price/m² by floor
    ax = axes[1]
    floor_prices = floors.groupby("floor")["price_m2"].agg(["median", "count"])
    floor_prices = floor_prices[floor_prices["count"] >= 50]
    ax.bar(floor_prices.index, floor_prices["median"], color="#9C27B0", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Floor number")
    ax.set_ylabel("Median price/m² (PLN)")
    ax.set_title("Price premium by floor")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_floors.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  03_floors.png")


def plot_districts(df):
    """District-level analysis: prices, areas, density."""
    valid = df[df["price_m2"].between(1000, 100000) & df["district"].notna()].copy()

    # Sort districts by median price
    district_order = valid.groupby("district")["price_m2"].median().sort_values(ascending=False).index

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Median price/m² by district
    ax = axes[0, 0]
    med_prices = valid.groupby("district")["price_m2"].median().reindex(district_order)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(district_order)))
    ax.barh(range(len(district_order)), med_prices.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(district_order)))
    ax.set_yticklabels(district_order)
    ax.invert_yaxis()
    ax.set_xlabel("Median price/m² (PLN)")
    ax.set_title("Median price per m² by district")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, v in enumerate(med_prices.values):
        ax.text(v + 100, i, f"{v:,.0f}", va="center", fontsize=9)

    # Median area by district
    ax = axes[0, 1]
    med_area = valid.groupby("district")["area"].median().reindex(district_order)
    ax.barh(range(len(district_order)), med_area.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(district_order)))
    ax.set_yticklabels(district_order)
    ax.invert_yaxis()
    ax.set_xlabel("Median area (m²)")
    ax.set_title("Median apartment area by district")
    for i, v in enumerate(med_area.values):
        ax.text(v + 0.5, i, f"{v:.0f}", va="center", fontsize=9)

    # Small apartments (<35m²) percentage by district
    ax = axes[1, 0]
    small_pct = valid.groupby("district").apply(
        lambda g: 100 * (g["area"] < 35).mean()
    ).reindex(district_order)
    ax.barh(range(len(district_order)), small_pct.values, color="#E91E63", alpha=0.7, edgecolor="white")
    ax.set_yticks(range(len(district_order)))
    ax.set_yticklabels(district_order)
    ax.invert_yaxis()
    ax.set_xlabel("% of transactions")
    ax.set_title("Small apartments (<35 m²) share by district")
    for i, v in enumerate(small_pct.values):
        ax.text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    # Room distribution by district (stacked)
    ax = axes[1, 1]
    room_dist = valid[valid["rooms"].between(1, 5)].groupby(
        ["district", "rooms"]
    ).size().unstack(fill_value=0)
    room_dist = room_dist.div(room_dist.sum(axis=1), axis=0) * 100
    room_dist = room_dist.reindex(district_order)
    room_colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
    room_dist.plot(kind="barh", stacked=True, ax=ax, color=room_colors, edgecolor="white", width=0.8)
    ax.set_xlabel("% of transactions")
    ax.set_title("Room count distribution by district")
    ax.legend(title="Rooms", bbox_to_anchor=(1.02, 1), labels=["1", "2", "3", "4", "5"])
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_districts.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  04_districts.png")


def plot_market_segments(df):
    """Primary vs secondary market comparison."""
    valid = df[df["price_m2"].between(1000, 100000) & df["market"].notna()].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Price/m² by market type over time
    ax = axes[0]
    for mkt, color in [("primary", "#2196F3"), ("secondary", "#FF9800")]:
        yearly = valid[valid["market"] == mkt].groupby("year")["price_m2"].median()
        yearly = yearly[yearly.index.to_series().between(2006, 2026)]
        ax.plot(yearly.index, yearly.values, "o-", color=color, label=mkt, markersize=4)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median price/m² (PLN)")
    ax.set_title("Primary vs secondary market prices")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Area distribution by market
    ax = axes[1]
    for mkt, color in [("primary", "#2196F3"), ("secondary", "#FF9800")]:
        data = valid[valid["market"] == mkt]["area"]
        data = data[data.between(10, 200)]
        ax.hist(data, bins=60, alpha=0.5, color=color, label=mkt, density=True, edgecolor="white")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Density")
    ax.set_title("Area distribution by market type")
    ax.legend()

    # Room count by market
    ax = axes[2]
    room_mkt = valid[valid["rooms"].between(1, 5)].groupby(
        ["market", "rooms"]
    ).size().unstack(fill_value=0)
    room_mkt_pct = room_mkt.div(room_mkt.sum(axis=1), axis=0) * 100
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w/2, room_mkt_pct.loc["primary"].values, w, label="primary", color="#2196F3", alpha=0.7)
    ax.bar(x + w/2, room_mkt_pct.loc["secondary"].values, w, label="secondary", color="#FF9800", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(["1", "2", "3", "4", "5"])
    ax.set_xlabel("Rooms")
    ax.set_ylabel("% of transactions")
    ax.set_title("Room count by market type")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_market_segments.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  05_market_segments.png")


def plot_outliers(df):
    """Outlier detection and data quality analysis."""
    res = df.copy()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Price vs area scatter (log scale)
    ax = axes[0, 0]
    sample = res[res["price"].between(10000, 50_000_000) & res["area"].between(5, 500)].sample(
        min(30000, len(res)), random_state=42
    )
    ax.scatter(sample["area"], sample["price"], alpha=0.05, s=3, c="#2196F3")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Total price (PLN)")
    ax.set_title("Price vs area (30K sample)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    # Highlight outliers: price/m² < 1000 or > 50000
    outliers_cheap = res[res["price_m2"].between(100, 1000)]
    outliers_expensive = res[res["price_m2"] > 50000]
    ax.scatter(outliers_cheap["area"], outliers_cheap["price"], c="red", s=8, alpha=0.3, label=f"<1K PLN/m² ({len(outliers_cheap):,})")
    ax.scatter(outliers_expensive["area"], outliers_expensive["price"], c="orange", s=8, alpha=0.3, label=f">50K PLN/m² ({len(outliers_expensive):,})")
    ax.legend(fontsize=9)

    # Price/m² distribution with tails
    ax = axes[0, 1]
    all_prices = res["price_m2"].dropna()
    ax.hist(all_prices[all_prices.between(0, 80000)], bins=200, color="#2196F3", alpha=0.7, edgecolor="none")
    ax.axvline(1000, color="red", linestyle="--", alpha=0.7, label="Filter: 1K")
    ax.axvline(100000, color="red", linestyle="--", alpha=0.7, label="Filter: 100K")
    ax.set_xlabel("Price/m² (PLN)")
    ax.set_ylabel("Count")
    ax.set_title("Full price/m² distribution (with tails)")
    ax.legend()

    # Suspicious patterns: area distribution
    ax = axes[1, 0]
    areas = res["area"].dropna()
    ax.hist(areas[areas.between(0, 300)], bins=150, color="#4CAF50", alpha=0.7, edgecolor="none")
    ax.axvline(10, color="red", linestyle="--", label="Min filter: 10 m²")
    ax.axvline(15, color="orange", linestyle="--", alpha=0.7)
    ax.axvline(25, color="orange", linestyle="--", alpha=0.7, label="Micro range: 15-25 m²")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Count")
    ax.set_title("Area distribution — micro-apartments visible")
    ax.legend(fontsize=9)

    # Data quality: missing values
    ax = axes[1, 1]
    cols = ["price_m2", "area", "rooms", "floor", "year", "market", "district"]
    missing = [(c, res[c].isna().mean() * 100) for c in cols]
    missing.sort(key=lambda x: x[1], reverse=True)
    bars = ax.barh([m[0] for m in missing], [m[1] for m in missing],
                   color="#F44336", alpha=0.7, edgecolor="white")
    ax.set_xlabel("% missing")
    ax.set_title("Missing data by field")
    for bar, (_, pct) in zip(bars, missing):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_outliers_quality.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  06_outliers_quality.png")


def plot_micro_apartments(df):
    """Deep dive into micro/aparthotel vs spacious apartments by district."""
    valid = df[df["price_m2"].between(1000, 100000) & df["district"].notna()].copy()

    # Define segments
    valid["size_cat"] = pd.cut(
        valid["area"],
        bins=[0, 25, 35, 50, 70, 100, 500],
        labels=["micro (<25)", "small (25-35)", "compact (35-50)",
                "medium (50-70)", "spacious (70-100)", "large (100+)"],
    )

    district_order = valid.groupby("district")["area"].median().sort_values().index

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Size category distribution by district
    ax = axes[0]
    size_dist = valid.groupby(["district", "size_cat"]).size().unstack(fill_value=0)
    size_dist = size_dist.div(size_dist.sum(axis=1), axis=0) * 100
    size_dist = size_dist.reindex(district_order)
    colors = ["#E91E63", "#FF5722", "#FF9800", "#FFC107", "#4CAF50", "#2196F3"]
    size_dist.plot(kind="barh", stacked=True, ax=ax, color=colors, edgecolor="white", width=0.8)
    ax.set_xlabel("% of transactions")
    ax.set_title("Apartment size categories by district")
    ax.legend(title="Size", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.invert_yaxis()

    # Micro apartment trend over time
    ax = axes[1]
    yearly_valid = valid[valid["year"].between(2006, 2026)]
    for cat, color, ls in [("micro (<25)", "#E91E63", "-"), ("small (25-35)", "#FF5722", "--"),
                            ("compact (35-50)", "#FF9800", "-.")]:
        pct = yearly_valid.groupby("year").apply(
            lambda g: 100 * (g["size_cat"] == cat).mean()
        )
        ax.plot(pct.index, pct.values, ls, color=color, label=cat, linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of transactions")
    ax.set_title("Micro & small apartment share over time")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_micro_apartments.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  07_micro_apartments.png")


def print_outlier_report(df):
    """Print statistical outlier analysis."""
    valid = df[df["price_m2"].between(1000, 100000)].copy()
    print("\n" + "=" * 60)
    print("OUTLIER & DATA QUALITY REPORT")
    print("=" * 60)

    # Extreme prices
    all_res = df[df["price_m2"].notna()]
    print(f"\nTotal residential with price/m²: {len(all_res):,}")
    print(f"  < 500 PLN/m²:    {(all_res['price_m2'] < 500).sum():>6,}  (likely errors / related party)")
    print(f"  500-1000 PLN/m²: {all_res['price_m2'].between(500, 1000).sum():>6,}  (suspicious)")
    print(f"  1K-100K PLN/m²:  {all_res['price_m2'].between(1000, 100000).sum():>6,}  (kept)")
    print(f"  > 100K PLN/m²:   {(all_res['price_m2'] > 100000).sum():>6,}  (likely errors)")

    # Suspicious areas
    print(f"\nArea outliers:")
    print(f"  < 10 m²:    {(all_res['area'] < 10).sum():>6,}  (not real apartments)")
    print(f"  10-15 m²:   {all_res['area'].between(10, 15).sum():>6,}  (micro-studios)")
    print(f"  15-25 m²:   {all_res['area'].between(15, 25).sum():>6,}  (aparthotels)")
    print(f"  > 300 m²:   {(all_res['area'] > 300).sum():>6,}  (penthouses / errors)")

    # Floor outliers
    floors = all_res["floor"].dropna()
    print(f"\nFloor outliers:")
    print(f"  Floor 0 (ground):  {(floors == 0).sum():>6,}")
    print(f"  Floor > 20:        {(floors > 20).sum():>6,}")
    print(f"  Floor > 40:        {(floors > 40).sum():>6,}  (likely errors)")

    # Price = 1 PLN or round numbers
    print(f"\nSuspicious prices (likely non-market):")
    print(f"  Price = 1 PLN:     {(all_res['price'] == 1).sum():>6,}")
    print(f"  Price < 1000 PLN:  {(all_res['price'] < 1000).sum():>6,}")

    # Duplicate transactions (same location, price, date)
    dedup_cols = ["geometry", "price", "area"]
    n_before = len(valid)
    n_after = valid.drop_duplicates(subset=["price", "area", "rooms", "floor"]).shape[0]
    print(f"\nPotential duplicates (same price+area+rooms+floor): {n_before - n_after:,}")

    # IQR-based outliers per district
    print(f"\nIQR outliers by district (1.5*IQR rule):")
    for dist in sorted(valid["district"].dropna().unique()):
        d = valid[valid["district"] == dist]["price_m2"]
        q1, q3 = d.quantile(0.25), d.quantile(0.75)
        iqr = q3 - q1
        outliers = d[(d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)]
        print(f"  {dist:>20s}: {len(outliers):>5,} / {len(d):>6,} ({100*len(outliers)/len(d):.1f}%)"
              f"  range: {q1-1.5*iqr:>7,.0f} - {q3+1.5*iqr:>7,.0f}")


def main():
    df = load_data()

    print(f"\nGenerating charts -> {OUTPUT_DIR}/")
    plot_price_distribution(df)
    plot_rooms_and_area(df)
    plot_floors(df)
    plot_districts(df)
    plot_market_segments(df)
    plot_outliers(df)
    plot_micro_apartments(df)
    print_outlier_report(df)

    print(f"\nDone. Charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
