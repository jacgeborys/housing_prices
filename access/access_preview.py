"""
Intermediate look at transit accessibility (run with the main Python, after travel_times.py's AM / midday /
evening windows): where "the centre" is, who reaches it, where service collapses in the evening, and whether
door-to-door access explains prices better than straight-line distance to PKiN.

Output: output/summary/24_transit_access.png  (+ printed model test)
"""
import os
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import style  # noqa: E402
from summary_charts import page, save, wrap  # noqa: E402
from summary_maps import FEATURES, basemap, legend_patches, load, outlines  # noqa: E402

style.apply()
OUT = os.path.join(HERE, "output")
CLUSTERS = [  # gravity_run.py's detection on the SimCity office layer, BDOT June 2025 (cells >= 4000, >= 2 cells)
    ("Rondo Daszyńskiego", 20.9846, 52.2307, 67591), ("Centrum / ONZ", 20.9998, 52.2341, 25241),
    ("Służewiec", 20.9934, 52.1828, 21714), ("Centrum south (Marszałkowska)", 21.0024, 52.2278, 11579)]

LABEL_OFF = {"Rondo Daszyńskiego": (-30, 32), "Centrum / ONZ": (22, 14), "Służewiec": (14, -6), "Centrum south (Marszałkowska)": (18, -14)}


def within_r2(v, cols):
    y = np.log(v["value_2025"]).to_numpy()
    X = np.column_stack([pd.get_dummies(v["district"]).astype(float).to_numpy(), v[cols].to_numpy(float)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - e.var() / y.var(), dict(zip(cols, b[-len(cols):]))


def main():
    v, d, w = load()
    a = {s: pd.read_csv(os.path.join(OUT, f"access_{s}.csv"), index_col=0) for s in ("am_peak", "midday", "evening")}
    v = v.merge(a["am_peak"][["jobs_30", "jobs_grav", "t_centrum"]], left_on="osm_id", right_index=True, how="left")
    v["eve_ratio"] = v["osm_id"].map(a["evening"]["jobs_30"] / a["am_peak"]["jobs_30"].replace(0, np.nan))
    v = v.dropna(subset=["jobs_grav"])
    v["log_grav"] = np.log(v["jobs_grav"].clip(lower=100))

    # model test (OLS with district fixed effects): does access replace distance to PKiN and the metro?
    base = [f for f in FEATURES]
    r_base, _ = within_r2(v, base)
    r_add, c_add = within_r2(v, base + ["log_grav"])
    swap = [f for f in base if f not in ("centre_km", "walk_metro_km")] + ["log_grav"]
    r_swap, c_swap = within_r2(v, swap)
    r_dist, _ = within_r2(v, [f for f in base if f not in ("centre_km", "walk_metro_km")])
    print(f"R² with district FE: features {r_base:.3f} | + access {r_add:.3f} | access instead of PKiN & metro "
          f"{r_swap:.3f} | neither {r_dist:.3f}")
    print(f"access coefficient (log gravity jobs): {100 * c_add['log_grav']:+.1f}% per log unit with PKiN/metro kept; "
          f"centre_km then {100 * c_add['centre_km']:+.1f}%/km, metro {100 * c_add['walk_metro_km']:+.1f}%/km")
    gain = 100 * (np.exp(c_swap["log_grav"] * np.log(2)) - 1)
    print(f"doubling access to jobs (gravity) = {gain:+.1f}% price, within a district")

    fig = page("Where is the centre, and who can reach it?",
               "Jobs reachable by public transport, door to door: walking, waiting, riding, changing",
               "Jobs = floor area (BDOT10k footprint x storeys) per m² per job by building category of the SimCity map "
               "(offices 18, public offices 25, retail & hospitals 40, schools 50...); 250 m cells. Travel: R5 routing on the Warsaw GTFS (ZTM, KM, WKD, timetable of Wed "
               "7.10.2026) and OSM paths, median over every departure minute 07–09 / 19–21. Office clusters: "
               "neighbouring cells with ≥ 4,000 office job equivalents. Preliminary (3 of 4 time windows). "
               + style.SOURCE)
    # 1 destinations
    ax1 = basemap(fig, d, w)
    ax1.set_position([0.01, 0.45, 0.48, 0.35])
    dest = gpd.read_file(os.path.join(HERE, "data", "destinations.gpkg")).to_crs(2180)
    dest = dest[dest["jobs"] > 50]
    cm1 = ListedColormap(["#fde4cf", "#f9b98a", "#f08a4b", "#c4521f", "#7a2a08"])
    nb1 = BoundaryNorm([50, 500, 1500, 4000, 10000, 1e9], 5)
    ax1.scatter(dest.geometry.x, dest.geometry.y, c=dest["jobs"], cmap=cm1, norm=nb1, s=4, marker="s", lw=0,
                zorder=3)
    from pyproj import Transformer
    t = Transformer.from_crs(4326, 2180, always_xy=True)
    for name, lon, lat, jobs in CLUSTERS:
        x, y = t.transform(lon, lat)
        ax1.scatter([x], [y], s=jobs / 250, facecolor="none", edgecolor=style.INK, lw=1.2, zorder=5)
        ax1.annotate(f"{name}\n{jobs / 1000:.0f}k", (x, y), xytext=LABEL_OFF[name], textcoords="offset points", fontsize=7.5,
                     zorder=6, arrowprops=dict(arrowstyle="-", color=style.INK_2, lw=0.6), bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.8))
    outlines(ax1, d, labels=False)
    legend_patches(ax1, [cm1(i) for i in range(5)], ["50–500", "500–1.5k", "1.5–4k", "4–10k", "10k+"],
                   "Jobs per 250 m cell")
    ax1.set_title("Where the jobs are (circles: office clusters)", fontsize=10.5)

    # 2 jobs within 30 min
    ax2 = basemap(fig, d, w)
    ax2.set_position([0.51, 0.45, 0.48, 0.35])
    cm2 = ListedColormap(["#e6effa", "#b9d3f3", "#83b2ea", "#4c8bd9", "#2461b0", "#0d3670"])
    nb2 = BoundaryNorm([0, 25e3, 75e3, 150e3, 300e3, 500e3, 2e6], 6)
    pts = v.geometry.representative_point()
    ax2.scatter(pts.x, pts.y, c=v["jobs_30"], cmap=cm2, norm=nb2, s=3, lw=0, zorder=3)
    outlines(ax2, d, labels=False)
    legend_patches(ax2, [cm2(i) for i in range(6)], ["< 25k", "25–75k", "75–150k", "150–300k", "300–500k", "500k+"],
                   "Jobs within 30 min")
    ax2.set_title("Jobs within 30 minutes, morning peak", fontsize=10.5)

    # 3 evening loss
    ax3 = basemap(fig, d, w)
    ax3.set_position([0.01, 0.07, 0.48, 0.33])
    cm3 = ListedColormap(["#7a2a08", "#c4521f", "#f08a4b", "#f9d7bd", "#eceae4"])
    nb3 = BoundaryNorm([0, 0.6, 0.75, 0.85, 0.95, 10], 5)
    ax3.scatter(pts.x, pts.y, c=v["eve_ratio"].clip(0, 2), cmap=cm3, norm=nb3, s=3, lw=0, zorder=3)
    outlines(ax3, d, labels=False)
    legend_patches(ax3, [cm3(i) for i in range(5)], ["< 60 %", "60–75 %", "75–85 %", "85–95 %", "95 %+"],
                   "Evening vs morning")
    ax3.set_title("Jobs reachable at 19–21 h vs 07–09 h", fontsize=10.5)

    # 4 model test
    ax4 = fig.add_axes([0.62, 0.12, 0.34, 0.26])
    labels = ["neither", "PKiN +\nmetro", "door-to-door\naccess", "both"]
    vals = [r_dist, r_base, r_swap, r_add]
    ax4.bar(range(4), vals, color=["#c9c7bf", style.CONTEXT_DARK, style.BLUE, "#5a8fd8"], width=0.6)
    for i, val in enumerate(vals):
        ax4.text(i, val + 0.003, f"{val:.3f}", ha="center", fontsize=9)
    ax4.set_xticks(range(4))
    ax4.set_xticklabels(labels, fontsize=8.5)
    ax4.set_ylim(min(vals) - 0.03, max(vals) + 0.02)
    style.hgrid_only(ax4)
    ax4.set_title("Price model fit (R², within districts)", fontsize=10.5)
    med = v.groupby("district")["eve_ratio"].median().sort_values()
    fig.text(0.03, 0.885, wrap(f"The centre is two office cores: Rondo Daszyńskiego ({CLUSTERS[0][3] / 1000:.0f}k office "
                               f"jobs) and Centrum/ONZ ({CLUSTERS[1][3] / 1000:.0f}k), with Służewiec ({CLUSTERS[2][3] / 1000:.0f}k) as the third core. The median "
                               f"building reaches {v['jobs_30'].median() / 1000:.0f}k jobs in 30 minutes; in the evening "
                               f"{med.index[0]} keeps only {med.iloc[0]:.0%} of its morning reach. Doubling job access "
                               f"is worth {gain:+.1f}% in price within a district.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "24_transit_access.png")
    print(med.round(2).to_string())


if __name__ == "__main__":
    main()
