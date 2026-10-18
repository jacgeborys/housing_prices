"""
Sanity check of the building ages (GIS Expert decades, fetch_building_age.py; their map is from ~2020 and has no
2021+ layer) against an independent signal: the year of a building's first developer (primary) sale in RCN, which
is recorded at completion. RCN starts in 2006, so this checks recent buildings only.

Flags per building:
  ok                      dated, no contradiction
  likely_2021_plus        no date, first developer sale 2019+ (built after the map)
  no_data                 no date, no recent developer sale
  newer_than_dated        dated <= 1990 but first developer sale 2016+ (renovated tenement or attic conversion
                          sold by a developer, or a new building where the map still shows the old one)

Outputs: output/age_check.gpkg (layer age_check, every dated or flagged building; open in QGIS over a basemap),
         output/analysis/age_check_centre.png (preview of the centre, 2400 x 3000)
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

import style
from price_drivers import pkin_xy

style.apply()
HALF_W, HALF_H = 2600, 3250          # m around PKiN for the preview
DECADE_COL = {1900: "#5a1e0f", 1910: "#7a2a12", 1920: "#983818", 1930: "#b4491f", 1940: "#c9622a",
              1950: "#e08a3c", 1960: "#efb15c", 1970: "#9fb3c8", 1980: "#7f9bb8", 1990: "#5f7fa6",
              2000: "#a9d38f", 2010: "#5fb36b", 2020: "#1f7a4d"}
FLAG_COL = {"likely_2021_plus": "#ffd400", "no_data": "#e01cd5"}


def flags():
    a = gpd.read_file("output/building_age.gpkg")
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                       columns=["osm_id", "year", "market"])
    t = tx[tx["osm_id"].notna()].copy()
    t["osm_id"] = t["osm_id"].astype("int64")
    dev = t[t["market"] == "primary"].groupby("osm_id")["year"].agg(first_dev="min", n_dev="size")
    a = a.merge(dev, left_on="osm_id", right_index=True, how="left")
    a = a.merge(t.groupby("osm_id").size().rename("n_sales"), left_on="osm_id", right_index=True, how="left")
    dated = a["decade_end"] > 0
    a["flag"] = "ok"
    a.loc[~dated & (a["first_dev"] >= 2019), "flag"] = "likely_2021_plus"
    a.loc[~dated & ~(a["first_dev"] >= 2019), "flag"] = "no_data"
    a.loc[dated & (a["decade_end"] <= 1990) & (a["first_dev"] >= 2016), "flag"] = "newer_than_dated"
    print(a["flag"].value_counts().to_string())
    priced = a["n_sales"] >= 3
    print("priced buildings (>= 3 sales):\n" + a.loc[priced, "flag"].value_counts().to_string())
    a.to_file("output/age_check.gpkg", layer="age_check", driver="GPKG")
    return a


def preview(a):
    cx, cy = pkin_xy()
    box = (cx - HALF_W, cy - HALF_H, cx + HALF_W, cy + HALF_H)
    a = a.cx[box[0]:box[2], box[1]:box[3]]
    allb = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all", bbox=box)
    roads = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/roads.shp", bbox=gpd.GeoSeries(
        [gpd.GeoSeries.from_xy([box[0], box[2]], [box[1], box[3]], crs=2180).to_crs(4326).unary_union.envelope],
        crs=4326)).to_crs(2180)
    roads = roads[roads["highway"].isin(["motorway", "trunk", "primary", "secondary"])]

    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.79])
    allb.plot(ax=ax, color="#e4e2da", lw=0)
    roads.plot(ax=ax, color="#c9c6bb", lw=0.6)
    dated = a[a["decade_end"] > 0]
    dated.plot(ax=ax, color=dated["decade_end"].map(DECADE_COL).fillna("#999999"), lw=0)
    for f, c in FLAG_COL.items():
        s = a[a["flag"] == f]
        if len(s):
            s.plot(ax=ax, color=c, lw=0)
    nd = a[a["flag"] == "newer_than_dated"]
    if len(nd):
        nd.plot(ax=ax, facecolor="none", edgecolor="#e8000b", lw=1.1)
    for nm, grp in roads[roads["name"].notna()].groupby("name"):
        if grp.length.sum() > 1500:
            p = grp.unary_union.interpolate(0.5, normalized=True) if grp.unary_union.geom_type == "LineString" \
                else grp.geometry.iloc[len(grp) // 2].interpolate(0.5, normalized=True)
            ax.text(p.x, p.y, nm, fontsize=4.5, color="#6b6860", ha="center", va="center")
    ax.set_xlim(box[0], box[2])
    ax.set_ylim(box[1], box[3])
    ax.set_aspect("equal")
    ax.axis("off")
    style.header(fig, "Building age check: the centre",
                 "GIS Expert construction decade (map from ~2020) vs RCN developer sales", y=0.985)
    h = [Patch(color=DECADE_COL[y], label=("before 1901" if y == 1900 else f"{y - 9}–{y}")) for y in DECADE_COL]
    h += [Patch(color=FLAG_COL["likely_2021_plus"], label="no date, developer sales 2019+ (likely 2021+)"),
          Patch(color=FLAG_COL["no_data"], label="no date"),
          Patch(facecolor="none", edgecolor="#e8000b", label="dated ≤ 1990, developer sales 2016+")]
    fig.legend(handles=h, loc="lower center", ncol=4, fontsize=6.5, frameon=False, bbox_to_anchor=(0.5, 0.025))
    path = "output/analysis/age_check_centre.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}, output/age_check.gpkg")


if __name__ == "__main__":
    preview(flags())
