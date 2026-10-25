"""
How the two comparisons work, on real buildings (explainer, methods page; not in the report).

  top     trees (one factor): Wola buildings, typical-flat price as sold vs corrected to the average Wola building in
          everything but trees (OLS on every model feature + district, the canopy term kept), by LiDAR tree cover
          within 100 m; Wolska 69 / 75 arrowed from sold to corrected
  bottom  reputation (the place): Żoliborz and Praga-Północ, price predicted from features alone (boosting, no district)
          vs measured; Krasińskiego 20 and Bródnowska 3: same prediction, ~40 % apart

Reads output/price_drivers_lidar.gpkg (price_drivers.py --trees lidar). Output: output/summary/37_two_comparisons.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save
from summary_maps import FEATURES

style.apply()

BANDS = [0, 10, 20, 30, 40, 100]
PAIR_TREES = ("Wolska", "69", "75")
PAIR_REP = (("Zygmunta Krasińskiego", "20"), ("Bródnowska", "3"))
TEAL, PRAGA, ZOL = style.BLUE, style.ORANGE, style.GREEN


def zl(x):
    return f"{x:,.0f}".replace(",", " ")


def find(v, street, no):
    return v.index[(v["addr_street"] == street) & (v["addr_housenumber"] == no)][0]


def corrected(v):
    """Typical-flat price if the building were the average one of its district in every feature but trees."""
    y = np.log(v["value_2025"]).to_numpy()
    D = pd.get_dummies(v["district"]).astype(float)
    b, *_ = np.linalg.lstsq(np.column_stack([D.to_numpy(), v[FEATURES].to_numpy(float)]), y, rcond=None)
    bf = pd.Series(b[D.shape[1]:], index=FEATURES)
    oth = [f for f in FEATURES if f != "canopy_100"]
    dev = v[oth] - v.groupby("district")[oth].transform("mean")
    return v["value_2025"] * np.exp(-(dev @ bf[oth])), 100 * (np.exp(10 * bf["canopy_100"]) - 1)


def trees_panel(ax, v):
    w = v[v["district"] == "Wola"].copy()
    w["band"] = pd.cut(w["canopy_100"], BANDS)
    mid = [5, 15, 25, 35, 45]                                          # band centres (40 %+ drawn at 45)
    ax.scatter(w["canopy_100"], w["corr"], s=7, color=TEAL, alpha=0.18, lw=0, zorder=1)
    for col, colr, ls, lab in (("value_2025", style.CONTEXT_DARK, "--", "As sold"),
                               ("corr", TEAL, "-", "Corrected: same age, metro, noise, cafés, parks … as the average Wola building")):
        med = w.groupby("band", observed=False)[col].median()
        ax.plot(mid, med.values, ls=ls, color=colr, lw=2.6, marker="o", ms=7, mec=style.SURFACE, zorder=4, label=lab)
    for no, dx in ((PAIR_TREES[1], -1.2), (PAIR_TREES[2], 1.2)):
        r = w.loc[find(w, PAIR_TREES[0], no)]
        ax.annotate("", (r["canopy_100"], r["corr"]), (r["canopy_100"], r["value_2025"]),
                    arrowprops=dict(arrowstyle="->", color=style.INK, lw=1.4), zorder=5)
        ax.scatter([r["canopy_100"]], [r["value_2025"]], s=46, color=style.SURFACE, ec=style.INK, lw=1.4, zorder=6)
        ax.scatter([r["canopy_100"]], [r["corr"]], s=60, color=TEAL, ec=style.INK, lw=1.4, zorder=6)
        ax.text(r["canopy_100"] + dx, r["value_2025"] - 250, f"Wolska {no}, {r['canopy_100']:.0f} % trees\n"
                f"sold {zl(r['value_2025'])}\ncorrected {zl(r['corr'])}",
                ha="right" if dx < 0 else "left", va="top", fontsize=9.5, color=style.INK, zorder=7,
                bbox=dict(fc=style.SURFACE, ec="none", alpha=0.85, pad=1.5))
    ax.set_xlim(0, 60)
    ax.set_ylim(9000, 26000)
    ax.set_xlabel("Tree cover within 100 m of the building (share of the ground under crowns ≥ 3 m, LiDAR 2025)")
    ax.set_ylabel("Typical flat, zł/m² (2025)")
    style.pln(ax)
    style.hgrid_only(ax)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f} %"))
    ax.legend(loc="upper right", fontsize=9.5)
    return w


def reputation_panel(ax, v):
    for d, colr in (("Żoliborz", ZOL), ("Praga-Północ", PRAGA)):
        s = v[v["district"] == d]
        meas = s["pred_2025"] * (1 + s["resid_pct"] / 100)
        ax.scatter(s["pred_2025"], meas, s=10, color=colr, alpha=0.35, lw=0, zorder=2)
        med = s["resid_pct"].median()
        xs = np.array([10000, 24000])
        ax.plot(xs, xs * (1 + med / 100), color=colr, lw=2.2, zorder=3)
        ax.text(xs[1] + 250, xs[1] * (1 + med / 100), f"{d}\ntypically {med:+.0f} %\nvs prediction", color=colr,
                fontsize=10.5, fontweight="bold", ha="left", va="center",
                bbox=dict(fc=style.SURFACE, ec="none", alpha=0.85, pad=1.5), zorder=8)
    xs = np.array([10000, 30000])
    ax.plot(xs, xs, color=style.MUTED, lw=1.2, ls=":", zorder=1)
    ax.text(29700, 29300, "exactly as\npredicted", color=style.MUTED, fontsize=9, ha="right", va="top",
            bbox=dict(fc=style.SURFACE, ec="none", alpha=0.85, pad=1.5))
    for (street, no), colr, dy in ((PAIR_REP[0], ZOL, 1), (PAIR_REP[1], PRAGA, -1)):
        r = v.loc[find(v, street, no)]
        meas = r["pred_2025"] * (1 + r["resid_pct"] / 100)
        ax.scatter([r["pred_2025"]], [meas], s=80, color=colr, ec=style.INK, lw=1.4, zorder=6)
        ax.annotate(f"{street.split()[-1]} {no}: built 1950s, 5 floors, 0.6 km to metro, 4.4 km to centre\n"
                    f"predicted {zl(r['pred_2025'])}, measured {zl(meas)} ({r['resid_pct']:+.0f} %)",
                    (r["pred_2025"], meas), (r["pred_2025"] - 3800, meas + dy * 3200), fontsize=9.5,
                    ha="left", va="center", color=style.INK, bbox=dict(fc=style.SURFACE, ec="none", alpha=0.85, pad=1.5),
                    arrowprops=dict(arrowstyle="-", color=style.INK, lw=0.9), zorder=7)
    ax.set_xlim(10000, 30000)
    ax.set_ylim(8000, 32000)
    ax.set_xlabel("Price predicted from the building's features alone, the place not given (zł/m²)")
    ax.set_ylabel("Price measured, zł/m²")
    style.pln(ax)
    style.pln(ax, "x")
    style.hgrid_only(ax)


def main():
    v = gpd.read_file("output/price_drivers_lidar.gpkg", ignore_geometry=True)
    v["corr"], per10 = corrected(v)
    fig = page("Two different comparisons",
               "Trees: correct for everything else, then compare. Reputation: predict without the place, then group by place",
               "Typical flat = model price of a 2-room ~50 m² flat on a middle floor in 2025, buildings with ≥ 3 flat "
               "sales. Top: Wola's 1 079 buildings; corrected = price if the building had its district's average of "
               "every other feature (age, distance to metro and centre, noise, cafés and shops within a 7-minute walk, "
               "parks, height …), from one regression on all buildings with the district as a factor; lines = medians "
               f"per 10-point band; city-wide, +10 points of tree cover goes with {per10:+.1f} % within a district. "
               "Bottom: prediction from the same features with no district or neighbourhood given (gradient boosting, "
               "out of sample); reputation = measured vs predicted. All associations, not causal effects. " + style.SOURCE)
    a1 = fig.add_axes([0.11, 0.565, 0.84, 0.29])
    a1.set_title("1. Trees: leafier Wola buildings look cheaper as sold, because Wola's new towers have few trees.\n"
                 "Corrected to the same age, location and surroundings, they cost about the same", loc="left",
                 fontsize=11.5, pad=8)
    trees_panel(a1, v)
    a2 = fig.add_axes([0.11, 0.155, 0.84, 0.29])
    a2.set_title("2. Reputation: two near-identical 1950s blocks, same predicted price;\n"
                 "the Żoliborz one sells 39 % above the Praga-Północ one", loc="left", fontsize=11.5, pad=8)
    reputation_panel(a2, v)
    save(fig, "37_two_comparisons.png")


if __name__ == "__main__":
    main()
