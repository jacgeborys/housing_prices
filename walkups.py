"""
The walk-up penalty: does a high floor cost or pay, with and without a lift?

No open register of lifts exists, so building type stands in for it:
  pre-war walk-up     built before 1941, 3-5 storeys (lifts rare)
  prefab walk-up      built 1961-1990, 4-5 storeys (Polish rules required lifts only in taller buildings)
  modern, with lift   built 2001-2020, 4-8 storeys
Construction decade: fetch_building_age.py (GIS Expert); storeys: coverage.gpkg (BDOT10k / OSM).
Resales only (developers price floors themselves). Within-building comparison: each sale's
log(price_m2_2025) minus its building's mean, then the mean by floor, relative to floor 2 (first floor up;
RCN floor 1 = ground floor).

Output: output/summary/22_walkups.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap

style.apply()

GROUPS = {   # name: (decade range, storeys range, colour)
    "Pre-war walk-up": ((0, 1940), (3, 5), "#8c6d46"),
    "Prefab walk-up (1961–90)": ((1961, 1990), (4, 5), "#7d7d7d"),
    "Modern, with lift (2001–20)": ((2001, 2020), (4, 8), style.BLUE),
}
FLOORS = range(1, 7)
MIN_N = 150


def main():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                      columns=["osm_id", "floor", "market", "price_m2_2025"])
    t = t[(t["market"] == "secondary")].dropna(subset=["osm_id", "floor"])
    age = gpd.read_file("output/age_check.gpkg", ignore_geometry=True)[["osm_id", "decade_end"]]
    cov = gpd.read_file("output/coverage.gpkg", layer="building_coverage", ignore_geometry=True)[["osm_id", "floors"]]
    t = t.merge(age, on="osm_id").merge(cov, on="osm_id")
    t["lp"] = np.log(t["price_m2_2025"])
    t = t[t.groupby("osm_id")["lp"].transform("size") >= 4]
    t["rel"] = t["lp"] - t.groupby("osm_id")["lp"].transform("mean")

    res = {}
    for name, ((d0, d1), (s0, s1), _) in GROUPS.items():
        g = t[t["decade_end"].between(d0, d1) & t["floors"].between(s0, s1) & (t["floor"] <= t["floors"])]
        m = g.groupby("floor")["rel"].agg(["mean", "sem", "size"]).reindex(FLOORS)
        m = m[m["size"] >= MIN_N]
        base = m.loc[2, "mean"]
        m["pct"] = 100 * (np.exp(m["mean"] - base) - 1)
        m["ci"] = 196 * m["sem"]
        res[name] = m
        print(name, f"{g['osm_id'].nunique():,} buildings, {len(g):,} sales")
        print(m[["pct", "ci", "size"]].round(2).to_string())

    fig = page("The walk-up penalty",
               "Price by floor within the same building, with and without a lift",
               "Resales in buildings with ≥ 4 sales; each sale vs its building's mean price per m² (city index removed), "
               f"relative to the first floor above ground; floors with ≥ {MIN_N} sales; bars = 95 % CI. No lift register "
               "exists: pre-war 3–5-storey tenements and 1961–90 blocks of 4–5 storeys (built without lifts) stand in for "
               "walk-ups. Construction decade: GIS Expert; storeys: BDOT10k / OSM. " + style.SOURCE)
    ax = fig.add_axes([0.11, 0.14, 0.84, 0.64])
    for (name, (_, _, colr)), off in zip(GROUPS.items(), (-0.12, 0, 0.12)):
        m = res[name]
        x = np.array(m.index, float) + off
        ax.errorbar(x, m["pct"], yerr=m["ci"], color=colr, lw=2.4, marker="o", ms=7, mec=style.SURFACE, mew=1.5,
                    capsize=0, elinewidth=1, label=name)
        ax.annotate(name, (x[-1], m["pct"].iloc[-1]), xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=9.5, color=colr, fontweight="semibold")
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(list(FLOORS))
    ax.set_xticklabels(["ground", "1st", "2nd", "3rd", "4th", "5th"])
    ax.set_xlim(0.7, 7.6)
    ax.set_xlabel("Floor (Polish count: 1st = first floor up)")
    style.pct(ax)
    style.hgrid_only(ax)
    ax.legend(loc="lower left", fontsize=9.5)
    pw, mod = res["Prefab walk-up (1961–90)"], res["Modern, with lift (2001–20)"]
    top_pw = pw["pct"].iloc[-1]
    top_mod = mod.loc[pw.index[-1], "pct"] if pw.index[-1] in mod.index else mod["pct"].iloc[-1]
    fig.text(0.03, 0.885, wrap(f"In a block without a lift the top floor ({['', 'ground', '1st', '2nd', '3rd', '4th', '5th'][pw.index[-1]]}) "
                               f"sells {top_pw:+.1f}% vs the 1st floor; with a lift the same floor sells {top_mod:+.1f}%. "
                               f"Ground floors cost {pw.loc[1, 'pct']:+.0f}% in old blocks but "
                               f"{mod.loc[1, 'pct']:+.1f}% in new ones, where they often come with a garden.", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "22_walkups.png")


if __name__ == "__main__":
    main()
