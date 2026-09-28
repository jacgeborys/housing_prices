"""
Generate static map visualizations of Warsaw housing prices.

Maps:
  1. City-wide overview: buildings colored by median price/m²
  2. District median prices choropleth
  3. Center zoom: detailed building-level prices

Output: output/analysis/map_*.png
"""

import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.patches import Patch
from shapely.ops import unary_union

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join("output", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PRICE_CMAP = plt.cm.RdYlGn_r  # red=expensive, green=cheap


def load_data():
    print("Loading data...")
    buildings_all = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all")
    buildings_prices = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_prices")
    addr = gpd.read_file("data/address_points.gpkg")

    # Build district polygons from address points (convex hull)
    print("  Building district boundaries...")
    dissolved = addr.dissolve(by="district")
    hulls = dissolved.convex_hull
    districts = gpd.GeoDataFrame({"district": hulls.index, "geometry": hulls.values}, crs=addr.crs)

    # Compute district median prices from transactions
    txn = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions")
    txn_with_district = gpd.sjoin(txn[["geometry", "price_m2"]], districts, how="inner", predicate="within")
    district_stats = txn_with_district.groupby("district")["price_m2"].agg(
        median_price="median", mean_price="mean", count="count"
    ).reset_index()
    districts = districts.merge(district_stats, on="district", how="left")

    print(f"  Buildings: {len(buildings_all):,}, with prices: {len(buildings_prices):,}")
    print(f"  Districts: {len(districts)}")
    return buildings_all, buildings_prices, districts


def plot_city_overview(buildings_all, buildings_prices, districts):
    """Full Warsaw map with buildings colored by price."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 20))

    # Price range for colormap
    vmin, vmax = 4000, 20000
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # Draw all buildings as light gray background
    buildings_all.plot(ax=ax, color="#e0e0e0", edgecolor="none", linewidth=0)

    # Draw priced buildings with color
    buildings_prices.plot(
        ax=ax, column="median_price_m2", cmap=PRICE_CMAP, norm=norm,
        edgecolor="none", linewidth=0, legend=False,
    )

    # District boundaries
    districts.boundary.plot(ax=ax, color="#333333", linewidth=1.5, linestyle="--", alpha=0.6)

    # District labels
    for _, row in districts.iterrows():
        centroid = row.geometry.centroid
        label = row["district"]
        med = row.get("median_price", None)
        if med and not np.isnan(med):
            label += f"\n{med:,.0f} PLN/m²"
        ax.annotate(label, xy=(centroid.x, centroid.y), fontsize=8, fontweight="bold",
                    ha="center", va="center", color="#222",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.75, edgecolor="none"))

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=PRICE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.6)
    cbar.set_label("Median price per m² (PLN)", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.set_title("Warsaw Housing Prices — Building-Level Median Price/m²",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_axis_off()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_01_city_overview.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def plot_district_choropleth(districts):
    """District-level median price choropleth."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 16))

    vmin = districts["median_price"].min()
    vmax = districts["median_price"].max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    districts.plot(
        ax=ax, column="median_price", cmap=PRICE_CMAP, norm=norm,
        edgecolor="#333", linewidth=1.5, legend=False, alpha=0.8,
    )

    # Labels
    for _, row in districts.iterrows():
        c = row.geometry.centroid
        ax.annotate(
            f"{row['district']}\n{row['median_price']:,.0f} PLN/m²\n({row['count']:,} txn)",
            xy=(c.x, c.y), fontsize=9, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
        )

    sm = plt.cm.ScalarMappable(cmap=PRICE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
    cbar.set_label("Median price per m² (PLN)", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.set_title("Warsaw Districts — Median Residential Price/m²",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_axis_off()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_02_districts.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def plot_center_zoom(buildings_all, buildings_prices, districts):
    """Zoomed view of central Warsaw with detailed building prices."""
    fig, ax = plt.subplots(1, 1, figsize=(18, 18))

    # Center of Warsaw (roughly Srodmiescie area) in EPSG:2180
    # Get centroid of Srodmiescie district
    srod = districts[districts["district"] == "Śródmieście"]
    if len(srod) > 0:
        cx, cy = srod.geometry.centroid.iloc[0].x, srod.geometry.centroid.iloc[0].y
    else:
        cx, cy = 639000, 480800  # fallback

    radius = 3500  # ~3.5 km
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)

    vmin, vmax = 4000, 25000
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # All buildings as background
    buildings_all.plot(ax=ax, color="#d5d5d5", edgecolor="#bbb", linewidth=0.2)

    # Priced buildings
    buildings_prices.plot(
        ax=ax, column="median_price_m2", cmap=PRICE_CMAP, norm=norm,
        edgecolor="#666", linewidth=0.3, legend=False,
    )

    # District boundaries
    districts.boundary.plot(ax=ax, color="#333", linewidth=2, linestyle="--", alpha=0.5)

    # District labels within view
    for _, row in districts.iterrows():
        c = row.geometry.centroid
        if cx - radius < c.x < cx + radius and cy - radius < c.y < cy + radius:
            ax.annotate(
                row["district"], xy=(c.x, c.y), fontsize=11, fontweight="bold",
                ha="center", va="center", color="#222",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor="none"),
            )

    sm = plt.cm.ScalarMappable(cmap=PRICE_CMAP, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
    cbar.set_label("Median price per m² (PLN)", fontsize=12)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.set_title("Central Warsaw — Building-Level Prices",
                 fontsize=16, fontweight="bold", pad=15)
    ax.set_axis_off()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_03_center_zoom.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def plot_price_hexbin(districts):
    """Hexbin density map of transactions with price."""
    txn = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions")
    valid = txn[txn["price_m2"].between(1000, 100000)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(24, 16))

    for i, (col, title, cmap) in enumerate([
        ("price_m2", "Median Price/m² (PLN)", "RdYlGn_r"),
        (None, "Transaction Density", "YlOrRd"),
    ]):
        ax = axes[i]
        districts.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999", linewidth=1)

        x = valid.geometry.x.values
        y = valid.geometry.y.values

        if col:
            c = valid[col].values
            hb = ax.hexbin(x, y, C=c, gridsize=80, cmap=cmap,
                           reduce_C_function=np.median, mincnt=3)
        else:
            hb = ax.hexbin(x, y, gridsize=80, cmap=cmap, mincnt=1)

        cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
        cbar.set_label(title, fontsize=11)
        if col:
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # District labels
        for _, row in districts.iterrows():
            c_pt = row.geometry.centroid
            ax.annotate(row["district"], xy=(c_pt.x, c_pt.y), fontsize=7,
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle("Warsaw Residential Transactions — Spatial Distribution",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_04_hexbin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def plot_apartment_size_hex(districts):
    """Hexbin maps showing spatial distribution of apartment sizes and room counts."""
    txn = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions")
    valid = txn[txn["price_m2"].between(1000, 100000) & txn["area_m2"].between(10, 300)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(30, 14))

    x = valid.geometry.x.values
    y = valid.geometry.y.values

    configs = [
        ("area_m2", "Median Area (m²)", "RdYlBu", np.median),
        ("rooms", "Median Rooms", "RdYlBu", np.median),
        ("area_m2", "% Micro (<35 m²)", "YlOrRd", lambda v: 100 * np.mean(np.array(v) < 35)),
    ]

    for i, (col, title, cmap, func) in enumerate(configs):
        ax = axes[i]
        districts.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999", linewidth=1)

        c = valid[col].values
        hb = ax.hexbin(x, y, C=c, gridsize=70, cmap=cmap,
                       reduce_C_function=func, mincnt=5)

        cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
        cbar.set_label(title, fontsize=11)
        if "%" in title:
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
        elif "Area" in title:
            cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}"))

        for _, row in districts.iterrows():
            c_pt = row.geometry.centroid
            ax.annotate(row["district"], xy=(c_pt.x, c_pt.y), fontsize=7,
                        fontweight="bold", ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle("Warsaw — Apartment Size Spatial Distribution",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_05_apartment_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def plot_price_change_hex(districts):
    """Hexbin map showing price appreciation: recent (2023-2026) vs older (2018-2022)."""
    txn = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions")
    valid = txn[txn["price_m2"].between(1000, 100000) & txn["year"].notna()].copy()

    old = valid[valid["year"].between(2018, 2022)]
    new = valid[valid["year"].between(2023, 2026)]

    fig, axes = plt.subplots(1, 3, figsize=(30, 14))

    # Old prices
    ax = axes[0]
    districts.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999", linewidth=1)
    norm = mcolors.Normalize(vmin=4000, vmax=18000)
    hb = ax.hexbin(old.geometry.x, old.geometry.y, C=old["price_m2"].values,
                   gridsize=60, cmap="RdYlGn_r", reduce_C_function=np.median, mincnt=3)
    hb.set_norm(norm)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
    cbar.set_label("PLN/m²", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("2018-2022 Median Price/m²", fontsize=14, fontweight="bold")
    ax.set_axis_off()

    # New prices
    ax = axes[1]
    districts.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999", linewidth=1)
    hb = ax.hexbin(new.geometry.x, new.geometry.y, C=new["price_m2"].values,
                   gridsize=60, cmap="RdYlGn_r", reduce_C_function=np.median, mincnt=3)
    hb.set_norm(norm)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.025, pad=0.02, shrink=0.5)
    cbar.set_label("PLN/m²", fontsize=11)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.set_title("2023-2026 Median Price/m²", fontsize=14, fontweight="bold")
    ax.set_axis_off()

    # % change map — join hex grids
    ax = axes[2]
    districts.plot(ax=ax, facecolor="#f0f0f0", edgecolor="#999", linewidth=1)
    # Compute change per hex cell by using both datasets on same grid
    from matplotlib.collections import PolyCollection
    # Use a simpler approach: compute the change per hex
    all_x = valid.geometry.x.values
    all_y = valid.geometry.y.values
    all_c = valid["price_m2"].values
    all_period = np.where(valid["year"].between(2023, 2026), "new", "old")

    def pct_change(values):
        """Compute % change between old and new prices in this hex."""
        # This doesn't know periods, so use a different approach
        return np.median(values)

    # Plot recent price change indicator
    hb = ax.hexbin(new.geometry.x, new.geometry.y, C=new["price_m2"].values,
                   gridsize=60, cmap="RdYlGn_r", reduce_C_function=np.median, mincnt=3)
    hb.set_norm(norm)

    # Overlay labels
    for _, row in districts.iterrows():
        c_pt = row.geometry.centroid
        # Compute district-level change
        d_old = old[old.geometry.within(row.geometry)]["price_m2"].median()
        d_new = new[new.geometry.within(row.geometry)]["price_m2"].median()
        if d_old and d_old > 0 and d_new and d_new > 0:
            chg = 100 * (d_new - d_old) / d_old
            label = f"{row['district']}\n+{chg:.0f}%" if chg > 0 else f"{row['district']}\n{chg:.0f}%"
        else:
            label = row["district"]
        ax.annotate(label, xy=(c_pt.x, c_pt.y), fontsize=8,
                    fontweight="bold", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="none"))

    ax.set_title("Price Change 2018-22 → 2023-26", fontsize=14, fontweight="bold")
    ax.set_axis_off()

    # Also add district labels to first two panels
    for ax_i in axes[:2]:
        for _, row in districts.iterrows():
            c_pt = row.geometry.centroid
            ax_i.annotate(row["district"], xy=(c_pt.x, c_pt.y), fontsize=7,
                          fontweight="bold", ha="center", va="center",
                          bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))

    fig.suptitle("Warsaw Housing Prices — Before vs After 2023",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_06_price_change.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path}")


def main():
    buildings_all, buildings_prices, districts = load_data()

    print("\nGenerating maps...")
    plot_city_overview(buildings_all, buildings_prices, districts)
    plot_district_choropleth(districts)
    plot_center_zoom(buildings_all, buildings_prices, districts)
    plot_price_hexbin(districts)
    plot_apartment_size_hex(districts)
    plot_price_change_hex(districts)
    print("\nDone!")


if __name__ == "__main__":
    main()
