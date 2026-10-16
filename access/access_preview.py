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
CLUSTERS = [  # gravity_run.py's detection on the SimCity office layer, BDOT June 2025; ONZ and Central Station
    # (2 + 2 cells, 25k + 12k, ~700 m apart) merged into one core at their job-weighted centre
    ("Rondo Daszyńskiego", 20.9846, 52.2307, 67591), ("ONZ / Central Station", 21.0007, 52.2320, 36820),
    ("Służewiec", 20.9934, 52.1828, 21714)]

LABEL_OFF = {"Rondo Daszyńskiego": (-30, 32), "ONZ / Central Station": (22, 14), "Służewiec": (14, -6)}
SECOND = [  # second tier: clusters at a looser threshold (cells >= 1500, >= 6k) that lie outside the cores
    ("Domaniewska", 21.0081, 52.1853, 20435, (26, -16)), ("Powiśle", 21.0096, 52.2328, 19363, (14, 8)),
    ("Al. Jerozolimskie\n(Szczęśliwice)", 20.9768, 52.2217, 15236, (-70, -22)),
    ("Dw. Gdański", 20.9953, 52.2551, 11496, (10, 8)), ("Sielce", 21.0467, 52.2012, 9276, (12, 4))]


def within_r2(v, cols):
    y = np.log(v["value_2025"]).to_numpy()
    X = np.column_stack([pd.get_dummies(v["district"]).astype(float).to_numpy(), v[cols].to_numpy(float)])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    return 1 - e.var() / y.var(), dict(zip(cols, b[-len(cols):]))


def main():
    v, d, w = load()
    # reach and model test: gravity_run.py (AM peak, BDOT June 2025, SimCity categories)
    g = pd.read_csv(os.path.join(OUT, "gravity_am_peak.csv"), index_col=0)
    v = v.merge(g[["n_jobs_30", "grav_jobs_5"]], left_on="osm_id", right_index=True, how="left")
    v["jobs_30"] = v["n_jobs_30"].fillna(0)
    v["log_grav"] = np.log1p(v["grav_jobs_5"].fillna(0))
    # evening loss: travel_times.py windows (a ratio, so the earlier destination weights largely cancel)
    a = {s: pd.read_csv(os.path.join(OUT, f"access_{s}.csv"), index_col=0) for s in ("am_peak", "evening")}
    v["eve_ratio"] = v["osm_id"].map(a["evening"]["jobs_30"] / a["am_peak"]["jobs_30"].replace(0, np.nan))

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
               "neighbouring cells with ≥ 4,000 office job equivalents (smaller centres: ≥ 1,500). Model test: gravity access to jobs (5-min "
               "half-life) vs distance to PKiN and the metro; figure 25 compares purposes and half-lives. "
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
    # "sun" labels: each centre's name on a ring around PKiN at its own bearing, a ray back to the centre;
    # bearings closer than MIN_GAP degrees are pushed apart so the labels never overlap
    cx, cy = t.transform(21.0060, 52.2318)
    two = {"Rondo Daszyńskiego": "Rondo\nDaszyńskiego", "ONZ / Central Station": "ONZ /\nCentral Station"}
    items = [(two.get(n, n), *t.transform(lo, la), j, True) for n, lo, la, j in CLUSTERS]
    for _, lo, la, j, _ in SECOND:                        # smaller centres: dashed circle, no label
        x, y = t.transform(lo, la)
        ax1.scatter([x], [y], s=j / 250, facecolor="none", edgecolor=style.INK_2, lw=0.8, ls=(0, (2, 2)), zorder=5)
    ang = np.array([np.degrees(np.arctan2(y - cy, x - cx)) for _, x, y, _, _ in items])
    ang = np.where((ang > 15) & (ang < 80), np.where(ang < 47, 15.0, 80.0), ang)   # keep clear of the legend
    ang = (ang - 47) % 360                                # count from inside the legend sector: no ±180° wrap
    order = np.argsort(ang)
    MIN_GAP = 40
    spread = ang[order].copy()
    for _ in range(50):                                   # relax: push neighbours apart, keep the mean
        for k in range(1, len(spread)):
            if spread[k] - spread[k - 1] < MIN_GAP:
                mid = (spread[k] + spread[k - 1]) / 2
                spread[k - 1], spread[k] = mid - MIN_GAP / 2, mid + MIN_GAP / 2
    R_EAST, R_WEST = 10500, 6500                         # ring radius (m from PKiN); the west edge is closer
    for k, i in enumerate(order):
        name, x, y, jobs, core = items[i]
        a = np.radians(spread[k] + 47)
        r = R_EAST if np.cos(a) >= 0 else R_WEST
        lx, ly = cx + r * np.cos(a), cy + r * np.sin(a)
        ax1.scatter([x], [y], s=jobs / 250, facecolor="none", edgecolor=style.INK if core else style.INK_2,
                    lw=1.2 if core else 0.8, ls="-" if core else (0, (2, 2)), zorder=5)
        ax1.plot([x, lx], [y, ly], color=style.INK_2 if core else style.MUTED, lw=0.6, zorder=4)
        ax1.text(lx, ly, f"{name}\n{jobs / 1000:.0f}k" if "\n" in name else f"{name} {jobs / 1000:.0f}k", fontsize=7.5 if core else 6.8,
                 fontweight="semibold" if core else "normal", color=style.INK if core else style.INK_2,
                 ha="left" if np.cos(a) >= 0 else "right", va="center", zorder=6,
                 bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.85))
    outlines(ax1, d, labels=False)
    legend_patches(ax1, [cm1(i) for i in range(5)], ["50–500", "500–1.5k", "1.5–4k", "4–10k", "10k+"],
                   "Jobs per 250 m cell")
    ax1.set_title("Where the jobs are (bold: office cores, dashed: smaller centres)", fontsize=9.5)

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
    labels = ["neither", "PKiN +\nmetro", "access\nto jobs", "both"]
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
    fig.text(0.03, 0.885, wrap(f"Offices cluster at Rondo Daszyńskiego ({CLUSTERS[0][3] / 1000:.0f}k jobs), ONZ / Central "
                               f"Station ({CLUSTERS[1][3] / 1000:.0f}k) and Służewiec ({CLUSTERS[2][3] / 1000:.0f}k). The median "
                               f"building reaches {v['jobs_30'].median() / 1000:.0f}k jobs in 30 min; doubling job access "
                               f"is worth {gain:+.1f}% within a district.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "24_transit_access.png")
    print(med.round(2).to_string())


if __name__ == "__main__":
    main()
