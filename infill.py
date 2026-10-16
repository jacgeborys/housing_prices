"""
Infill ("plomby"): new buildings squeezed into old neighbourhoods vs new buildings in new estates.

A building of 2001-2020 (fetch_building_age.py, GIS Expert decades) is
  infill       if >= 60 % of the dated buildings within NEAR_M were built before 1961 (>= MIN_NEIGH of them)
  new estate   if >= 60 % of them were built 2001 or later
  mixed        otherwise
For priced buildings (price_drivers.gpkg): the model residual (price vs the prediction from location, amenities,
noise, age) and, for infills, the premium over the old priced buildings within NEAR_M.

Outputs:
  output/infill.gpkg                       layer new_buildings
  output/summary/21_infill.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load, outlines

style.apply()

NEAR_M = 150
MIN_NEIGH = 4
SHARE = 0.6


def main():
    v, d, w = load()                                         # priced buildings with resid_pct, value_2025
    age = gpd.read_file("output/building_age.gpkg")
    age = age[age["decade_end"] > 0].copy()
    ap = age.geometry.representative_point()
    tree = cKDTree(np.c_[ap.x, ap.y])
    dec = age["decade_end"].to_numpy()

    new = age[age["decade_end"].between(2001, 2020)].copy()
    npnt = new.geometry.representative_point()
    kinds, shares = [], []
    for i, nb in zip(new.index, tree.query_ball_point(np.c_[npnt.x, npnt.y], NEAR_M)):
        nb = [j for j in nb if age.index[j] != i]
        if len(nb) < MIN_NEIGH:
            kinds.append("isolated")
            shares.append(np.nan)
            continue
        old_share = np.mean(dec[nb] <= 1960)
        new_share = np.mean(dec[nb] >= 2001)
        shares.append(old_share)
        kinds.append("infill" if old_share >= SHARE else "new estate" if new_share >= SHARE else "mixed")
    new["kind"], new["old_share"] = kinds, shares
    print(new["kind"].value_counts().to_string())

    # price: residual and premium over old priced neighbours
    vv = v[["osm_id", "resid_pct", "value_2025", "district", "geometry"]].merge(
        age[["osm_id", "decade_end"]], on="osm_id")
    vp = vv.geometry.representative_point()
    vtree = cKDTree(np.c_[vp.x, vp.y])
    old_v = (vv["decade_end"] <= 1960).to_numpy()
    lv = np.log(vv["value_2025"].to_numpy())
    pn = new.merge(vv[["osm_id", "resid_pct", "value_2025", "district"]], on="osm_id")
    pnp = pn.geometry.representative_point()
    prem = []
    for (x, y), val in zip(np.c_[pnp.x, pnp.y], pn["value_2025"]):
        nb = [j for j in vtree.query_ball_point([x, y], NEAR_M) if old_v[j]]
        prem.append(100 * (np.exp(np.log(val) - np.median(lv[nb])) - 1) if len(nb) >= 2 else np.nan)
    pn["prem_old"] = prem
    summ = pn.groupby("kind").agg(n=("resid_pct", "size"), resid=("resid_pct", "median"),
                                  prem_old=("prem_old", "median"))
    print(summ.round(1).to_string())
    inf = pn[pn["kind"] == "infill"]
    by_d = inf.groupby("district").agg(n=("prem_old", "size"), prem=("prem_old", "median")).query("n >= 8") \
        .sort_values("prem")
    print(by_d.round(1).to_string())
    new.merge(pn[["osm_id", "resid_pct", "prem_old"]], on="osm_id", how="left") \
        .to_file("output/infill.gpkg", layer="new_buildings", driver="GPKG")

    fig = page("New among the old",
               "Infill buildings (plomby) vs new buildings in new estates",
               f"Buildings of 2001–2020 (construction decade: GIS Expert, mapy.gis-expert.pl). Infill: ≥ {SHARE:.0%} "
               f"of dated buildings within {NEAR_M} m built before 1961 (≥ {MIN_NEIGH} neighbours); new estate: ≥ "
               f"{SHARE:.0%} built 2001 or later. Price: typical-flat value 2025 (≥ 3 sales); vs prediction = model "
               "residual (location, amenities, noise, age); vs old neighbours = median of priced pre-1961 buildings "
               f"within {NEAR_M} m. " + style.SOURCE)
    ax = basemap(fig, d, w)
    ax.set_position([0.03, 0.40, 0.60, 0.46])
    col = {"infill": style.MAGENTA, "new estate": style.BLUE, "mixed": "#b9b7ae"}
    for k in ("mixed", "new estate", "infill"):
        p = new[new["kind"] == k].geometry.representative_point()
        ax.scatter(p.x, p.y, s=9 if k != "infill" else 14, color=col[k], alpha=0.8, lw=0, zorder=4,
                   label=f"{k} ({(new['kind'] == k).sum():,})")
    outlines(ax, d)
    ax.legend(loc="lower left", fontsize=8.5, markerscale=1.8, frameon=True, framealpha=0.9, edgecolor="none",
              title="Buildings of 2001–2020", title_fontsize=9)

    a2 = fig.add_axes([0.72, 0.60, 0.24, 0.20])
    s2 = summ.loc[[k for k in ("infill", "mixed", "new estate") if k in summ.index]]
    a2.bar(np.arange(len(s2)), s2["resid"], color=[col[k] for k in s2.index], width=0.6)
    for xi, val in enumerate(s2["resid"]):
        a2.text(xi, val + (0.15 if val >= 0 else -0.15), f"{val:+.1f}%", ha="center",
                va="bottom" if val >= 0 else "top", fontsize=9)
    a2.axhline(0, color=style.AXIS, lw=1)
    a2.set_xticks(np.arange(len(s2)))
    a2.set_xticklabels(s2.index, fontsize=9)
    a2.set_ylim(min(s2["resid"].min(), 0) - 1, max(s2["resid"].max(), 0) + 1)
    a2.yaxis.set_major_locator(plt.MultipleLocator(1))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    style.hgrid_only(a2)
    a2.set_title("Price vs prediction", fontsize=10.5)

    a3 = fig.add_axes([0.20, 0.12, 0.32, 0.21])
    a3.hist(inf["prem_old"].dropna().clip(-40, 150), bins=np.arange(-40, 151, 10), color=style.MAGENTA, alpha=0.85,
            edgecolor=style.SURFACE)
    a3.axvline(0, color=style.AXIS, lw=1)
    style.pct(a3, "x")
    style.hgrid_only(a3)
    a3.set_title("Infill vs the old buildings around it", fontsize=10.5)
    a3.set_ylabel("buildings", fontsize=9)

    a4 = fig.add_axes([0.70, 0.12, 0.26, 0.36])
    a4.barh(np.arange(len(by_d)), by_d["prem"], color=style.MAGENTA, height=0.6)
    a4.set_yticks(np.arange(len(by_d)))
    a4.set_yticklabels(by_d.index, fontsize=9)
    for yi, val in enumerate(by_d["prem"]):
        a4.text(val + 1, yi, f"{val:+.0f}%", va="center", fontsize=8.5, color=style.INK_2)
    a4.set_xlim(min(0, by_d["prem"].min()) - 3, by_d["prem"].max() * 1.3)
    style.pct(a4, "x")
    style.vgrid_only(a4)
    a4.set_title("Infill premium over old neighbours", fontsize=10.5)

    fig.text(0.03, 0.885, wrap(f"{(new['kind'] == 'infill').sum():,} new buildings sit among pre-1961 ones. They sell "
                               f"{summ.loc['infill', 'prem_old']:+.0f}% above the old buildings next door, and "
                               f"{summ.loc['infill', 'resid']:+.1f}% vs what the model predicts (new estates "
                               f"{summ.loc['new estate', 'resid']:+.1f}%).", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "21_infill.png")


if __name__ == "__main__":
    main()
