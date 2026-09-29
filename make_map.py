"""
Generate static map visualizations of Warsaw housing prices.

Maps:
  1. City-wide overview: buildings colored by median price/m²
  2. District median prices choropleth
  3. Center zoom: detailed building-level prices
  4. Price / transaction density hexbins
  5. Apartment size hexbins
  6. 2014-16 vs 2024-26 prices + growth relative to the city
  8. Building-level hedonic value (needs price_index.py)

The 3D renders live in make_3d.py.

Output: output/analysis/map_*.png
"""

import os
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import numpy as np

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join("output", "analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import style

style.apply()
PRICE_CMAP = style.PRICE_CMAP


def load_data():
    print("Loading data...")
    buildings_all = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all")
    buildings_prices = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_prices")

    # Official district boundaries from BDOT ADJA_A
    print("  Loading district boundaries (BDOT)...")
    districts = gpd.read_file("data/districts.gpkg")

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
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=style.SURFACE, alpha=0.75, edgecolor="none"))

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
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=style.SURFACE)
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
            bbox=dict(boxstyle="round,pad=0.3", facecolor=style.SURFACE, alpha=0.8, edgecolor="none"),
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
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=style.SURFACE)
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
                bbox=dict(boxstyle="round,pad=0.3", facecolor=style.SURFACE, alpha=0.7, edgecolor="none"),
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
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=style.SURFACE)
    plt.close(fig)
    print(f"  {path}")


def plot_price_hexbin(districts):
    """Hexbin density map of transactions with price."""
    txn = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions")
    valid = txn[txn["price_m2"].between(1000, 100000)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(24, 16))

    for i, (col, title, cmap) in enumerate([
        ("price_m2", "Median Price/m² (PLN)", style.PRICE_CMAP),
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
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=style.SURFACE, alpha=0.7, edgecolor="none"))

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle("Warsaw Residential Transactions — Spatial Distribution",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_04_hexbin.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=style.SURFACE)
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
                        bbox=dict(boxstyle="round,pad=0.2", facecolor=style.SURFACE, alpha=0.7, edgecolor="none"))

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_axis_off()

    fig.suptitle("Warsaw — Apartment Size Spatial Distribution",
                 fontsize=16, fontweight="bold", y=0.98)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "map_05_apartment_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=style.SURFACE)
    plt.close(fig)
    print(f"  {path}")


def _hex_label_districts(ax, districts, fontsize=7.5):
    for _, row in districts.iterrows():
        c_pt = row.geometry.representative_point()
        ax.annotate(row["district"], xy=(c_pt.x, c_pt.y), fontsize=fontsize, color=style.INK_2,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=style.SURFACE, alpha=0.75, edgecolor="none"))


def plot_price_change_hex(districts):
    """Hexbin maps: 2014-16 vs 2024-26 medians, and where prices grew faster or
    slower than the city as a whole."""
    txn = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions")
    valid = txn[txn["price_m2"].between(1000, 100000) & txn["year"].notna()].copy()
    valid["period"] = np.select([valid["year"].between(2014, 2016), valid["year"].between(2024, 2026)],
                                ["old", "new"], "")
    valid = valid[valid["period"] != ""].reset_index(drop=True)
    old, new = valid[valid["period"] == "old"], valid[valid["period"] == "new"]
    city_chg = 100 * (new["price_m2"].median() / old["price_m2"].median() - 1)

    xmin, ymin, xmax, ymax = districts.total_bounds
    extent = (xmin - 3000, xmax + 3000, ymin - 3000, ymax + 3000)
    grid = 55

    fig, axes = plt.subplots(1, 3, figsize=(27, 11.5))
    fig.subplots_adjust(top=0.9, bottom=0.03, left=0.01, right=0.97, wspace=0.04)
    norm = mcolors.Normalize(vmin=4000, vmax=20000)

    for ax, sub, title in [(axes[0], old, "Median price/m², 2014–16"),
                           (axes[1], new, "Median price/m², 2024–26")]:
        districts.plot(ax=ax, facecolor=style.SURFACE, edgecolor=style.AXIS, linewidth=0.8)
        hb = ax.hexbin(sub.geometry.x, sub.geometry.y, C=sub["price_m2"].values, gridsize=grid,
                       extent=extent, cmap=style.PRICE_CMAP, norm=norm,
                       reduce_C_function=np.median, mincnt=5, linewidths=0.2, edgecolors=style.SURFACE)
        style.colorbar(fig, hb, ax, "PLN/m²", shrink=0.45)
        _hex_label_districts(ax, districts)
        ax.set_title(title, fontsize=14)
        ax.set_axis_off()

    # Change per hex: bin row ids, then compare the two periods inside each cell
    period = valid["period"].values
    price = valid["price_m2"].values

    def rel_change(ids):
        ids = np.asarray(ids, dtype=int)
        o, n = price[ids][period[ids] == "old"], price[ids][period[ids] == "new"]
        if len(o) < 5 or len(n) < 5:
            return np.nan
        return 100 * (np.median(n) / np.median(o) - 1) - city_chg

    ax = axes[2]
    districts.plot(ax=ax, facecolor=style.SURFACE, edgecolor=style.AXIS, linewidth=0.8)
    hb = ax.hexbin(valid.geometry.x, valid.geometry.y, C=np.arange(len(valid)), gridsize=grid,
                   extent=extent, cmap=style.DIV_CMAP, reduce_C_function=rel_change,
                   mincnt=10, linewidths=0.2, edgecolors=style.SURFACE)
    hb.set_array(np.ma.masked_invalid(hb.get_array()))
    hb.set_norm(mcolors.TwoSlopeNorm(vcenter=0, vmin=-60, vmax=60))
    style.colorbar(fig, hb, ax, f"pp vs city (+{city_chg:.0f}%)", fmt="pct", shrink=0.45)

    for _, row in districts.iterrows():
        c_pt = row.geometry.representative_point()
        inside_old = old[old.geometry.within(row.geometry)]["price_m2"].median()
        inside_new = new[new.geometry.within(row.geometry)]["price_m2"].median()
        label = row["district"]
        if inside_old > 0 and inside_new > 0:
            label += f"\n{100 * (inside_new / inside_old - 1):+.0f}%"
        ax.annotate(label, xy=(c_pt.x, c_pt.y), fontsize=7.5, ha="center", va="center", color=style.INK,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor=style.SURFACE, alpha=0.8, edgecolor="none"))
    ax.set_title("Growth vs the city average", fontsize=14)
    ax.set_axis_off()

    style.header(fig, "Where did prices grow fastest?",
                 f"City-wide the median price per m² rose {city_chg:+.0f}% between 2014–16 and 2024–26. "
                 f"Red cells outpaced that, blue cells lagged (raw medians — the mix of flats sold also shifts).",
                 y=0.985)
    style.footer(fig)
    path = os.path.join(OUTPUT_DIR, "map_06_price_change.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  {path}")


def plot_building_value(buildings_all, districts):
    """Every building with sales, coloured by the hedonic model's value of a
    typical flat there in 2025 — comparable across buildings and years."""
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value")
    fig, ax = plt.subplots(figsize=(16, 18))
    fig.subplots_adjust(top=0.95, bottom=0.02, left=0.01, right=0.93)
    buildings_all.plot(ax=ax, color=style.CONTEXT, edgecolor="none", linewidth=0)
    vmin, vmax = np.percentile(bv["value_2025"], [3, 97]).round(-3)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    # Buffer slightly so small blocks remain visible at city scale
    bv = bv.sort_values("value_2025")
    bv.assign(geometry=bv.geometry.buffer(6)).plot(
        ax=ax, column="value_2025", cmap=style.PRICE_CMAP, norm=norm, edgecolor="none", linewidth=0)
    districts.boundary.plot(ax=ax, color=style.INK_2, linewidth=0.8, alpha=0.6)
    for _, row in districts.iterrows():
        sub = bv[bv["district"] == row["district"]]
        c = row.geometry.representative_point()
        lab = row["district"]
        if len(sub):
            lab += "\n" + style.fmt_int(np.exp(np.average(np.log(sub["value_2025"]), weights=sub["n"])))
        ax.annotate(lab, xy=(c.x, c.y), fontsize=9, ha="center", va="center", color=style.INK,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor=style.SURFACE, alpha=0.8, edgecolor="none"))
    sm = plt.cm.ScalarMappable(cmap=style.PRICE_CMAP, norm=norm)
    style.colorbar(fig, sm, ax, "PLN/m², typical flat, 2025", shrink=0.4)
    ax.set_axis_off()
    style.header(fig, "What a typical flat costs, building by building",
                 "Hedonic model value in 2025 PLN/m² for a 45–60 m², 2-room, 2nd–3rd floor resale — "
                 "so a block last sold in 2012 is directly comparable with one sold last month.", y=0.985)
    style.footer(fig, style.SOURCE + " Values shrunk toward the district mean where a building has few sales.")
    path = os.path.join(OUTPUT_DIR, "map_08_building_value.png")
    fig.savefig(path, dpi=150)
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
    plot_building_value(buildings_all, districts)
    print("\nDone!")


if __name__ == "__main__":
    main()
