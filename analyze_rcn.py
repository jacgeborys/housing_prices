"""
Exploratory analysis of Warsaw residential transactions (RCN data).

Generates charts covering:
  - Price/m² distribution and trends
  - Room counts, floor numbers, apartment areas
  - District-level comparisons (small vs spacious apartments)
  - Outlier detection and data quality
  - Market segmentation (primary vs secondary)
  - Apartments vs houses

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

    # Classify property type: apartment vs house
    # Houses have nier_prawo = ownership of building, or very large area + low floor
    res["prop_type"] = "apartment"
    is_building = res["nier_prawo"] == "wlasnoscBudynkuWrazZPrawemZwiazanym"
    is_ground = res["nier_prawo"] == "wlasnoscNieruchomosciGruntowej"
    res.loc[is_building | is_ground, "prop_type"] = "house"

    # Assign districts via nearest address point
    addr = gpd.read_file("data/address_points.gpkg")
    print("  Assigning districts via nearest address point...")
    res_pts = res[["geometry"]].copy()
    nearest = gpd.sjoin_nearest(res_pts, addr[["geometry", "district"]], how="left", max_distance=500)
    nearest = nearest[~nearest.index.duplicated(keep="first")]
    res["district"] = nearest["district"]

    print(f"  Residential transactions: {len(res):,}")
    print(f"  With district: {res['district'].notna().sum():,}")
    print(f"  Apartments: {(res['prop_type'] == 'apartment').sum():,}, "
          f"Houses: {(res['prop_type'] == 'house').sum():,}")
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
    # Rotate year labels and ensure integer ticks
    ax.tick_params(axis="x", rotation=45)
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


def _sorted_barh(ax, series, color="#4CAF50", fmt="{v:,.0f}"):
    """Horizontal bar chart sorted by value with labels."""
    s = series.sort_values(ascending=True)
    bars = ax.barh(range(len(s)), s.values, color=color, edgecolor="white")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels(s.index)
    offset = s.max() * 0.02
    for i, v in enumerate(s.values):
        ax.text(v + offset, i, fmt.format(v=v), va="center", fontsize=9)


def plot_districts(df):
    """District-level analysis: prices, areas, density. Each subplot sorted independently."""
    valid = df[df["price_m2"].between(1000, 100000) & df["district"].notna()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Median price/m² — sorted by price
    ax = axes[0, 0]
    med_prices = valid.groupby("district")["price_m2"].median()
    n = len(med_prices)
    price_sorted = med_prices.sort_values(ascending=True)
    colors_price = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, n))
    ax.barh(range(n), price_sorted.values, color=colors_price, edgecolor="white")
    ax.set_yticks(range(n))
    ax.set_yticklabels(price_sorted.index)
    ax.set_xlabel("Median price/m² (PLN)")
    ax.set_title("Median price per m² by district")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, v in enumerate(price_sorted.values):
        ax.text(v + 100, i, f"{v:,.0f}", va="center", fontsize=9)

    # Median area — sorted by area
    ax = axes[0, 1]
    _sorted_barh(ax, valid.groupby("district")["area"].median(), color="#4CAF50", fmt="{v:.0f}")
    ax.set_xlabel("Median area (m²)")
    ax.set_title("Median apartment area by district")

    # Small apartments (<35m²) — sorted by %
    ax = axes[1, 0]
    small_pct = valid.groupby("district").apply(lambda g: 100 * (g["area"] < 35).mean())
    _sorted_barh(ax, small_pct, color="#E91E63", fmt="{v:.1f}%")
    ax.set_xlabel("% of transactions")
    ax.set_title("Small apartments (<35 m²) share by district")

    # Room distribution — sorted by 1-room share
    ax = axes[1, 1]
    room_dist = valid[valid["rooms"].between(1, 5)].groupby(
        ["district", "rooms"]
    ).size().unstack(fill_value=0)
    room_dist = room_dist.div(room_dist.sum(axis=1), axis=0) * 100
    room_order = room_dist[1.0].sort_values(ascending=True).index
    room_dist = room_dist.reindex(room_order)
    room_colors = ["#E91E63", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
    room_dist.plot(kind="barh", stacked=True, ax=ax, color=room_colors, edgecolor="white", width=0.8)
    ax.set_xlabel("% of transactions")
    ax.set_title("Room count distribution by district (sorted by 1-room %)")
    ax.legend(title="Rooms", bbox_to_anchor=(1.02, 1), labels=["1", "2", "3", "4", "5"])

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
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

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

    # Price vs area scatter — ALL points, outliers highlighted
    ax = axes[0, 0]
    plotable = res[res["price"].between(100, 100_000_000) & res["area"].between(1, 1000)].copy()
    plotable["outlier"] = "normal"
    plotable.loc[plotable["price_m2"] < 1000, "outlier"] = "cheap"
    plotable.loc[plotable["price_m2"] > 50000, "outlier"] = "expensive"

    normal = plotable[plotable["outlier"] == "normal"].sample(min(30000, len(plotable)), random_state=42)
    cheap = plotable[plotable["outlier"] == "cheap"]
    expensive = plotable[plotable["outlier"] == "expensive"]

    ax.scatter(normal["area"], normal["price"], alpha=0.03, s=3, c="#2196F3", zorder=1)
    ax.scatter(cheap["area"], cheap["price"], c="red", s=10, alpha=0.4, zorder=2,
               label=f"<1K PLN/m² ({len(cheap):,})")
    ax.scatter(expensive["area"], expensive["price"], c="#FF8F00", s=10, alpha=0.4, zorder=2,
               label=f">50K PLN/m² ({len(expensive):,})")
    # Reference lines for 1K and 50K PLN/m²
    xx = np.array([5, 500])
    ax.plot(xx, xx * 1000, "r--", alpha=0.3, linewidth=1)
    ax.plot(xx, xx * 50000, "--", color="#FF8F00", alpha=0.3, linewidth=1)
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Total price (PLN)")
    ax.set_title("Price vs area — outliers highlighted")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend(fontsize=9, loc="upper left")

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

    # Sort by median area
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
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_micro_apartments.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  07_micro_apartments.png")


def plot_apartments_vs_houses(df):
    """Apartment vs house comparison."""
    valid = df[df["price_m2"].between(1000, 100000)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Counts
    ax = axes[0]
    counts = valid["prop_type"].value_counts()
    ax.bar(counts.index, counts.values, color=["#2196F3", "#4CAF50"], edgecolor="white")
    ax.set_ylabel("Transactions")
    ax.set_title("Transaction count")
    for i, (k, v) in enumerate(counts.items()):
        ax.text(i, v + 1000, f"{v:,}", ha="center", fontsize=11)

    # Price/m² comparison
    ax = axes[1]
    for pt, color in [("apartment", "#2196F3"), ("house", "#4CAF50")]:
        data = valid[valid["prop_type"] == pt]["price_m2"]
        ax.hist(data, bins=80, alpha=0.5, color=color, label=pt, density=True,
                edgecolor="white", range=(1000, 40000))
    ax.set_xlabel("Price/m² (PLN)")
    ax.set_ylabel("Density")
    ax.set_title("Price/m² distribution")
    ax.legend()

    # Area comparison
    ax = axes[2]
    for pt, color in [("apartment", "#2196F3"), ("house", "#4CAF50")]:
        data = valid[valid["prop_type"] == pt]["area"]
        data = data[data.between(10, 300)]
        ax.hist(data, bins=80, alpha=0.5, color=color, label=pt, density=True, edgecolor="white")
    ax.set_xlabel("Area (m²)")
    ax.set_ylabel("Density")
    ax.set_title("Area distribution")
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "08_apartments_vs_houses.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  08_apartments_vs_houses.png")


def plot_price_trends(df):
    """Year-over-year price trends by district, and price appreciation analysis."""
    valid = df[df["price_m2"].between(1000, 100000) & df["year"].between(2010, 2026)
               & df["district"].notna()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Price trend by top districts
    ax = axes[0, 0]
    top_districts = valid.groupby("district").size().nlargest(8).index
    for dist in top_districts:
        d = valid[valid["district"] == dist].groupby("year")["price_m2"].median()
        ax.plot(d.index, d.values, "o-", label=dist, markersize=3, linewidth=1.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Median price/m² (PLN)")
    ax.set_title("Price trend — top 8 districts by volume")
    ax.legend(fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Year-over-year % change (city-wide)
    ax = axes[0, 1]
    yearly_med = valid.groupby("year")["price_m2"].median()
    yoy = yearly_med.pct_change() * 100
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in yoy.values]
    ax.bar(yoy.index, yoy.values, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("YoY change (%)")
    ax.set_title("Year-over-year price change (city-wide)")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for i, (yr, v) in enumerate(zip(yoy.index, yoy.values)):
        if not np.isnan(v):
            ax.text(yr, v + (1 if v >= 0 else -2), f"{v:+.1f}%", ha="center", fontsize=8)

    # Price appreciation 2018 vs 2024+ by district
    ax = axes[1, 0]
    old_med = valid[valid["year"].between(2018, 2020)].groupby("district")["price_m2"].median()
    new_med = valid[valid["year"].between(2024, 2026)].groupby("district")["price_m2"].median()
    change = ((new_med - old_med) / old_med * 100).dropna().sort_values()
    colors_ch = ["#F44336" if v < 0 else "#4CAF50" for v in change.values]
    ax.barh(range(len(change)), change.values, color=colors_ch, edgecolor="white")
    ax.set_yticks(range(len(change)))
    ax.set_yticklabels(change.index)
    ax.set_xlabel("Price change (%)")
    ax.set_title("Price appreciation: 2018-20 → 2024-26")
    for i, v in enumerate(change.values):
        ax.text(v + 1, i, f"{v:+.0f}%", va="center", fontsize=9)

    # Transaction volume over time
    ax = axes[1, 1]
    vol = valid.groupby("year").size()
    ax.bar(vol.index, vol.values, color="#2196F3", edgecolor="white")
    ax.set_xlabel("Year")
    ax.set_ylabel("Transactions")
    ax.set_title("Transaction volume over time")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.suptitle("Price Trends & Appreciation", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "09_price_trends.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  09_price_trends.png")


def plot_value_analysis(df):
    """Value for money: total price vs area, price per room, efficiency metrics."""
    valid = df[df["price_m2"].between(1000, 100000) & df["area"].between(10, 200)
               & df["rooms"].between(1, 6) & df["district"].notna()].copy()

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # Area per room by district
    ax = axes[0, 0]
    valid["area_per_room"] = valid["area"] / valid["rooms"]
    district_apr = valid.groupby("district")["area_per_room"].median().sort_values(ascending=True)
    colors = plt.cm.RdYlBu(np.linspace(0.15, 0.85, len(district_apr)))
    ax.barh(range(len(district_apr)), district_apr.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(district_apr)))
    ax.set_yticklabels(district_apr.index)
    ax.set_xlabel("Median area per room (m²)")
    ax.set_title("Space efficiency — area per room by district")
    for i, v in enumerate(district_apr.values):
        ax.text(v + 0.3, i, f"{v:.1f} m²", va="center", fontsize=9)

    # Price per room by district
    ax = axes[0, 1]
    valid["price_per_room"] = valid["price"] / valid["rooms"]
    district_ppr = valid.groupby("district")["price_per_room"].median().sort_values(ascending=True)
    colors = plt.cm.RdYlGn_r(np.linspace(0.15, 0.85, len(district_ppr)))
    ax.barh(range(len(district_ppr)), district_ppr.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(district_ppr)))
    ax.set_yticklabels(district_ppr.index)
    ax.set_xlabel("Median price per room (PLN)")
    ax.set_title("Cost per room by district")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    for i, v in enumerate(district_ppr.values):
        ax.text(v + 5000, i, f"{v:,.0f}", va="center", fontsize=9)

    # Area distribution by room count (violin-like)
    ax = axes[1, 0]
    room_cats = sorted(valid["rooms"].unique())
    data_by_room = [valid[valid["rooms"] == r]["area"].values for r in room_cats]
    parts = ax.violinplot(data_by_room, positions=room_cats, showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#2196F3")
        pc.set_alpha(0.6)
    ax.set_xlabel("Number of rooms")
    ax.set_ylabel("Area (m²)")
    ax.set_title("Area distribution by room count")

    # Scatter: area per room vs price/m² (colored by rooms)
    ax = axes[1, 1]
    sample = valid.sample(min(10000, len(valid)), random_state=42)
    scatter = ax.scatter(sample["area_per_room"], sample["price_m2"],
                        c=sample["rooms"], cmap="viridis", alpha=0.3, s=5)
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
    cbar.set_label("Rooms")
    ax.set_xlabel("Area per room (m²)")
    ax.set_ylabel("Price/m² (PLN)")
    ax.set_title("Space efficiency vs price")
    ax.set_xlim(5, 60)
    ax.set_ylim(1000, 30000)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.suptitle("Value Analysis — Space & Cost Efficiency", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "10_value_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  10_value_analysis.png")


def plot_repeat_sales(df):
    """Repeat sales analysis: same apartment sold multiple times."""
    valid = df[df["price_m2"].between(1000, 100000) & df["year"].notna()].copy()

    # Identify units by address + area (more complete than lok_id_lokalu)
    valid["unit_key"] = valid["lok_adres"].astype(str) + "|" + valid["area"].round(1).astype(str)
    unit_counts = valid.groupby("unit_key").size()
    repeat_keys = unit_counts[unit_counts > 1].index
    repeats = valid[valid["unit_key"].isin(repeat_keys)].copy()
    repeats = repeats.sort_values(["unit_key", "year"])

    print(f"  Repeat sales: {len(repeat_keys):,} units, {len(repeats):,} transactions")

    # Compute per-unit stats
    pairs = []
    for key, grp in repeats.groupby("unit_key"):
        grp = grp.sort_values("year")
        if len(grp) < 2 or len(grp) > 10:  # skip very high repeat counts (likely common address)
            continue
        for i in range(len(grp) - 1):
            r1, r2 = grp.iloc[i], grp.iloc[i + 1]
            y1, y2 = r1["year"], r2["year"]
            if y2 <= y1:
                continue
            gap_years = y2 - y1
            if gap_years < 0.5:
                continue
            p1, p2 = r1["price_m2"], r2["price_m2"]
            pct_change = 100 * (p2 - p1) / p1
            annual_return = 100 * ((p2 / p1) ** (1 / gap_years) - 1)
            pairs.append({
                "year_buy": y1, "year_sell": y2,
                "gap_years": gap_years,
                "price_m2_buy": p1, "price_m2_sell": p2,
                "pct_change": pct_change, "annual_return": annual_return,
                "area": r1["area"],
                "district": r1.get("district"),
            })

    if not pairs:
        print("  No valid repeat sale pairs found")
        return

    pdf = pd.DataFrame(pairs)
    pdf = pdf[pdf["annual_return"].between(-50, 100)]  # trim extreme outliers
    print(f"  Valid repeat sale pairs: {len(pdf):,}")

    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    # Holding period distribution
    ax = axes[0, 0]
    ax.hist(pdf["gap_years"], bins=np.arange(0.5, 21.5, 1), color="#2196F3",
            edgecolor="white", alpha=0.7)
    ax.axvline(pdf["gap_years"].median(), color="red", linestyle="--",
               label=f'Median: {pdf["gap_years"].median():.1f} yrs')
    ax.set_xlabel("Years between sales")
    ax.set_ylabel("Number of pairs")
    ax.set_title("Holding period distribution")
    ax.legend()

    # Annual return distribution
    ax = axes[0, 1]
    ax.hist(pdf["annual_return"], bins=80, color="#4CAF50", edgecolor="white", alpha=0.7,
            range=(-30, 50))
    ax.axvline(pdf["annual_return"].median(), color="red", linestyle="--",
               label=f'Median: {pdf["annual_return"].median():+.1f}%/yr')
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Annualized return (%)")
    ax.set_ylabel("Count")
    ax.set_title("Annual price appreciation (repeat sales)")
    ax.legend()

    # Return by holding period
    ax = axes[0, 2]
    bins = [(1, 2), (2, 3), (3, 5), (5, 8), (8, 12), (12, 20)]
    labels = ["1-2yr", "2-3yr", "3-5yr", "5-8yr", "8-12yr", "12-20yr"]
    medians = []
    for lo, hi in bins:
        sub = pdf[pdf["gap_years"].between(lo, hi)]
        medians.append(sub["annual_return"].median() if len(sub) > 10 else 0)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in medians]
    ax.bar(labels, medians, color=colors, edgecolor="white")
    ax.set_xlabel("Holding period")
    ax.set_ylabel("Median annual return (%)")
    ax.set_title("Return by holding period")
    for i, v in enumerate(medians):
        ax.text(i, v + 0.3, f"{v:+.1f}%", ha="center", fontsize=9)

    # Return by buy year
    ax = axes[1, 0]
    by_buy = pdf.groupby("year_buy")["annual_return"].median()
    by_buy = by_buy[by_buy.index >= 2006]
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in by_buy.values]
    ax.bar(by_buy.index, by_buy.values, color=colors, edgecolor="white")
    ax.set_xlabel("Year of purchase")
    ax.set_ylabel("Median annual return (%)")
    ax.set_title("Return by purchase year")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Return by district
    ax = axes[1, 1]
    dist_return = pdf[pdf["district"].notna()].groupby("district")["annual_return"].median()
    dist_return = dist_return.sort_values(ascending=True)
    colors = ["#4CAF50" if v >= 0 else "#F44336" for v in dist_return.values]
    ax.barh(range(len(dist_return)), dist_return.values, color=colors, edgecolor="white")
    ax.set_yticks(range(len(dist_return)))
    ax.set_yticklabels(dist_return.index)
    ax.set_xlabel("Median annual return (%)")
    ax.set_title("Return by district (repeat sales)")
    for i, v in enumerate(dist_return.values):
        ax.text(v + 0.2, i, f"{v:+.1f}%", va="center", fontsize=9)

    # % that lost money
    ax = axes[1, 2]
    loss_pct = pdf.groupby("year_buy").apply(lambda g: 100 * (g["pct_change"] < 0).mean())
    loss_pct = loss_pct[loss_pct.index >= 2006]
    ax.bar(loss_pct.index, loss_pct.values, color="#F44336", edgecolor="white", alpha=0.7)
    ax.set_xlabel("Year of purchase")
    ax.set_ylabel("% sold at a loss")
    ax.set_title("Share of loss-making resales")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    for i, (yr, v) in enumerate(zip(loss_pct.index, loss_pct.values)):
        ax.text(yr, v + 1, f"{v:.0f}%", ha="center", fontsize=8)

    plt.suptitle("Repeat Sales Analysis — Same Apartment Resold", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "11_repeat_sales.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  11_repeat_sales.png")


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

    # Suspicious prices (likely non-market)
    print(f"\nSuspicious prices (likely non-market):")
    print(f"  Price = 1 PLN:     {(all_res['price'] == 1).sum():>6,}")
    print(f"  Price < 1000 PLN:  {(all_res['price'] < 1000).sum():>6,}")

    # Duplicates
    n_before = len(valid)
    n_after = valid.drop_duplicates(subset=["price", "area", "rooms", "floor"]).shape[0]
    print(f"\nPotential duplicates (same price+area+rooms+floor): {n_before - n_after:,}")

    # Apartments vs houses
    print(f"\nProperty type:")
    for pt in ["apartment", "house"]:
        sub = valid[valid["prop_type"] == pt]
        print(f"  {pt:>10s}: {len(sub):>7,}  median price/m²={sub['price_m2'].median():>8,.0f}  "
              f"median area={sub['area'].median():>5.0f} m²")

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
    plot_apartments_vs_houses(df)
    plot_price_trends(df)
    plot_value_analysis(df)
    plot_repeat_sales(df)
    print_outlier_report(df)

    print(f"\nDone. Charts saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
