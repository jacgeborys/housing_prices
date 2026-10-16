"""
Does being close to transit pay, and does the mode matter? Within the same neighbourhood, all else equal.

For each mode (metro, rail, tram, bus) the typical-flat value (price_drivers.gpkg) is regressed on MSI
neighbourhood fixed effects (the city's 143 official areas, data/msi.gpkg) and every model feature except that
mode's walking distance; the residual, averaged by walking-distance band, is what proximity to that mode is worth
compared with other buildings of the same neighbourhood. Walking distances: walk_access.py (OSM footpath graph).

Output: output/summary/04_transit.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap
from summary_maps import FEATURES

style.apply()

MODES = {   # feature, title, band edges (m), colour, station type for "neighbourhoods with a station" (or None)
    "walk_metro_km": ("Metro", [0, 300, 600, 900, 1500], style.BLUE, "subway"),
    "walk_rail_km": ("Rail (SKM, KM, WKD)", [0, 300, 600, 900, 1500], style.VIOLET, "rail"),
    "walk_tram_km": ("Tram", [0, 200, 400, 600, 900, 1500, 9999], style.ORANGE, None),
    "walk_bus_km": ("Bus", [0, 100, 200, 300, 500, 9999], style.AQUA, None),
}
MIN_N = 60


def band_labels(edges):
    out = [f"< {edges[1]}"] + [f"{a}–{b}" for a, b in zip(edges[1:-1], edges[2:])]
    if edges[-1] == 9999:
        out[-1] = f"{edges[-2]}+"
    return out


def main():
    v = gpd.read_file("output/price_drivers.gpkg")
    msi = gpd.read_file("data/msi.gpkg")
    pts = gpd.GeoDataFrame(geometry=v.geometry.representative_point(), crs=2180)
    j = gpd.sjoin(pts, msi[["name", "geometry"]], predicate="within", how="left")
    v["msi"] = j[~j.index.duplicated()]["name"].fillna(v["district"]).to_numpy()
    st = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/train_stations.gpkg").to_crs(2180)
    st["kind"] = np.where(st["station"] == "subway", "subway", "rail")
    m2 = msi.rename(columns={"name": "msi_name"})[["msi_name", "geometry"]]
    has = {k: set(gpd.sjoin(st[st["kind"] == k][["geometry"]], m2, predicate="within")["msi_name"])
           for k in ("subway", "rail")}
    print(f"{len(v):,} buildings in {v['msi'].nunique()} neighbourhoods; with a metro station "
          f"{len(has['subway'])}, with a rail station {len(has['rail'])}")

    fig = page("The metro lifts neighbourhoods, not doorsteps",
               "Price by walking distance to each kind of transit, vs buildings in the same neighbourhood",
               "Typical-flat value 2025 (hedonic model, ≥ 3 sales) net of the neighbourhood (143 MSI areas as fixed "
               "effects) and every other feature (centrality, the other modes, amenities, noise, green, building age "
               f"and form); bands with ≥ {MIN_N} buildings, bars = 95 % CI, relative to the farthest band. Metro and "
               "rail: only neighbourhoods with a station, buildings within 1.5 km of it. Walking "
               "distance along OSM footpaths and streets. " + style.SOURCE)
    gs = fig.add_gridspec(2, 2, left=0.10, right=0.97, top=0.765, bottom=0.15, hspace=0.62, wspace=0.25)
    headline = {}
    for k, (feat, (title, edges, colr, station)) in enumerate(MODES.items()):
        w = v if station is None else v[v["msi"].isin(has[station]) & (v[feat] * 1000 < edges[-1])]
        y = np.log(w["value_2025"]).to_numpy()
        X = np.column_stack([pd.get_dummies(w["msi"]).astype(float).to_numpy(),
                             w[[f for f in FEATURES if f != feat]].to_numpy(float)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        b = pd.cut(w[feat] * 1000, edges, right=False, labels=band_labels(edges))
        g = pd.DataFrame({"r": r, "b": b}).groupby("b", observed=False)["r"].agg(["mean", "sem", "size"])
        g = g[g["size"] >= MIN_N]
        ref = g["mean"].iloc[-1]
        pct = 100 * (np.exp(g["mean"] - ref) - 1)
        ci = 196 * g["sem"]
        headline[title] = pct.iloc[0]
        ax = fig.add_subplot(gs[k // 2, k % 2])
        x = np.arange(len(g))
        ax.bar(x, pct.values, color=colr, width=0.65, alpha=0.9)
        ax.errorbar(x, pct.values, yerr=ci.values, fmt="none", ecolor=style.INK_2, elinewidth=0.9)
        for xi, val in zip(x, pct.values):
            ax.text(xi + 0.18, val, f"{val:+.1f}%", ha="left", va="center", fontsize=8.5, color=style.INK)
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(g.index, fontsize=8, rotation=30, ha="right")
        scope = f"\n{len(w):,} buildings, station neighbourhoods" if station else ""
        ax.set_title(title + scope, fontsize=11 if station else 11.5, color=colr, fontweight="semibold")
        ax.set_xlabel("walking distance to the nearest stop (m)", fontsize=8.5)
        ax.yaxis.set_major_locator(plt.MultipleLocator(1))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:+.0f}%"))
        style.hgrid_only(ax)
        print(title, pct.round(1).to_dict())
    coef = pd.read_csv("output/analysis/price_drivers_coefficients.csv").set_index("feature")
    per_km = coef.loc["walk_metro_km", "pct_district"]
    fig.text(0.03, 0.885, wrap(f"Neighbourhoods on the metro line are dearer ({per_km:+.1f}% per km walked to a station, "
                               "comparing buildings within a district). But inside a neighbourhood, the building at the "
                               f"station sells only {headline['Metro']:+.1f}% more than one 900–1500 m away. Near a rail station "
                               f"it's {headline['Rail (SKM, KM, WKD)']:+.1f}% (the tracks are loud); by a tram stop "
                               f"{headline['Tram']:+.1f}%, by a bus stop {headline['Bus']:+.1f}%.", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "04_transit.png")


if __name__ == "__main__":
    main()
