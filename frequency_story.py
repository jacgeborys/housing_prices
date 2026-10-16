"""
Does frequent transit pay? Departures within a 5-minute walk (TransitFrequency: real walking isochrones with
fences and gates, all bus / tram / metro / rail departures 06-22 h), against the typical-flat value 2025.

  top     price by departure band, all else equal: OLS residual of log value on every model feature except
          departures, with district fixed effects and, separately, neighbourhood (MSI) fixed effects
  bottom  the raw picture: median value by departure band, new buildings vs older ones

Output: output/summary/05_frequency.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap
from summary_maps import FEATURES

style.apply()

BANDS = [-1, 150, 300, 500, 800, 1200, 100000]
LABELS = ["≤ 150", "150–300", "300–500", "500–800", "800–1200", "1200+"]
MIN_N = 80


def adjusted(v, fe):
    y = np.log(v["value_2025"]).to_numpy()
    X = np.column_stack([pd.get_dummies(v[fe]).astype(float).to_numpy(),
                         v[[f for f in FEATURES if f != "log2_trips"]].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    g = pd.DataFrame({"r": r, "b": pd.cut(v["trips"], BANDS, labels=LABELS)}).groupby("b", observed=False)["r"] \
        .agg(["mean", "sem", "size"])
    g = g[g["size"] >= MIN_N]
    return 100 * (np.exp(g["mean"] - g["mean"].iloc[0]) - 1), 196 * g["sem"]


def main():
    v = gpd.read_file("output/price_drivers.gpkg")
    msi = gpd.read_file("data/msi.gpkg").rename(columns={"name": "msi_name"})
    pts = gpd.GeoDataFrame(geometry=v.geometry.representative_point(), crs=2180)
    j = gpd.sjoin(pts, msi[["msi_name", "geometry"]], predicate="within", how="left")
    v["msi"] = j[~j.index.duplicated()]["msi_name"].fillna(v["district"]).to_numpy()
    coef = pd.read_csv("output/analysis/price_drivers_coefficients.csv").set_index("feature").loc["log2_trips"]

    fig = page("Does frequent transit pay?",
               "Departures within a 5-minute walk (walking isochrones, all modes) against the price of a typical flat",
               "Departures/day 06–22 h within a 5-minute walk along real paths incl. fences and gates (TransitFrequency, "
               "all bus, tram, metro and rail). Top: typical-flat value 2025 net of every other feature (centrality, metro, "
               "amenities, noise, green, building age and form) and of the district or the neighbourhood (143 MSI areas), "
               f"relative to the least-served band; bands with ≥ {MIN_N} buildings, bars = 95 % CI. Bottom: raw medians; "
               "new = mostly developer sales. " + style.SOURCE)
    a1 = fig.add_axes([0.11, 0.53, 0.84, 0.27])
    x = np.arange(len(LABELS))
    for fe, lab, colr, off in (("district", "Within the same district", style.BLUE, -0.08),
                               ("msi", "Within the same neighbourhood", style.ORANGE, 0.08)):
        pct, ci = adjusted(v, fe)
        xi = np.array([LABELS.index(k) for k in pct.index]) + off
        a1.errorbar(xi, pct.values, yerr=ci.values, color=colr, lw=2.2, marker="o", ms=6, mec=style.SURFACE,
                    capsize=0, elinewidth=1, label=lab)
        print(lab, pct.round(1).to_dict())
    a1.axhline(0, color=style.AXIS, lw=1)
    a1.set_xticks(x)
    a1.set_xticklabels(LABELS)
    a1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, _: f"{val:+.1f}%"))
    style.hgrid_only(a1)
    a1.legend(loc="lower left", fontsize=9.5)
    a1.set_title("All else equal (vs the least-served buildings)", fontsize=11)

    a2 = fig.add_axes([0.11, 0.12, 0.84, 0.29])
    v["kind"] = np.where(v["pct_primary"] > 0.3, "new", "old")
    v["band"] = pd.cut(v["trips"], BANDS, labels=LABELS)
    for kind, lab, colr in (("new", "New buildings", style.ORANGE), ("old", "Older buildings", style.BLUE)):
        g = v[v["kind"] == kind].groupby("band", observed=False)["value_2025"].agg(["median", "size"])
        g = g[g["size"] >= MIN_N]
        xi = np.array([LABELS.index(k) for k in g.index])
        a2.plot(xi, g["median"], "-o", color=colr, lw=2.2, ms=6, mec=style.SURFACE, label=lab)
    a2.set_xticks(x)
    a2.set_xticklabels(LABELS)
    a2.set_xlabel("Departures per day within a 5-minute walk")
    style.pln(a2)
    style.hgrid_only(a2)
    a2.legend(loc="upper left", fontsize=9.5)
    a2.set_title("Raw medians, PLN/m²", fontsize=11)
    fig.text(0.03, 0.885, wrap(f"Once location, the metro, amenities, noise and building age are accounted for, more "
                               f"departures add almost nothing ({coef['pct_district']:+.1f}% per doubling within a "
                               "district). In the raw data well-served new buildings look dearer because they sit in "
                               "better locations: nearer the centre, the metro and the cafés.", 95), fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "05_frequency.png")


if __name__ == "__main__":
    main()
