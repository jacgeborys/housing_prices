"""
Does frequent transit raise flat prices? Two views of the same buildings (price_drivers.gpkg):

  left   median value of a typical flat (2025 PLN/m²) by departures/day within a 5-min walk,
         new buildings (mostly developer sales) vs older ones (resales)
  right  within the district: older buildings' price vs their district average, same bands

Output: output/analysis/17_transit_price.png
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

style.apply()

OUT = "output/analysis/17_transit_price.png"
BANDS = [-1, 60, 150, 300, 500, 800, 1200, 5000]
BAND_LABELS = ["≤ 60", "60–150", "150–300", "300–500", "500–800", "800–1200", "1200+"]
MIN_N = 80
SERIES = {"new": ("New buildings (mostly developer sales)", style.ORANGE),
          "old": ("Older buildings (resales)", style.BLUE)}


def main():
    v = gpd.read_file("output/price_drivers.gpkg", ignore_geometry=True)
    v["kind"] = np.where(v["pct_primary"] > 0.3, "new", "old")
    v["band"] = pd.cut(v["trips"], BANDS, labels=BAND_LABELS)
    v["y"] = np.log(v["value_2025"])
    v["rel"] = v["y"] - v.groupby("district")["y"].transform("mean")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6.6))
    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.07, right=0.97, wspace=0.28)
    x = np.arange(len(BAND_LABELS))
    for kind, (label, colr) in SERIES.items():
        g = v[v["kind"] == kind].groupby("band", observed=False)["value_2025"].agg(["median", "size"])
        ok = g["size"] >= MIN_N
        a1.plot(x[ok], g["median"][ok], "-o", color=colr, lw=2, ms=8, mec=style.SURFACE, mew=2, label=label)
        last = np.where(ok)[0][-1]
        a1.annotate(f"{g['median'].iloc[last]:,.0f}".replace(",", " "), (x[last], g["median"].iloc[last]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9.5, color=style.INK_2)
    a1.set_xticks(x)
    a1.set_xticklabels(BAND_LABELS)
    a1.set_xlabel("Departures per day within a 5-minute walk")
    a1.set_ylabel("Typical flat, PLN/m² (2025)")
    style.pln(a1)
    style.hgrid_only(a1)
    a1.legend(loc="upper left")
    a1.set_title("Across the city", fontsize=12)

    o = v[v["kind"] == "old"].groupby("band", observed=False).agg(rel=("rel", "mean"), n=("rel", "size"))
    pct = 100 * (np.exp(o["rel"]) - 1)
    ok = (o["n"] >= MIN_N).to_numpy()
    a2.bar(x[ok], pct[ok], width=0.7, color=[style.BLUE if p >= 0 else style.CONTEXT_DARK for p in pct[ok]])
    for xi, p in zip(x[ok], pct[ok]):
        a2.text(xi, p + (0.25 if p >= 0 else -0.25), f"{p:+.1f}%", ha="center",
                va="bottom" if p >= 0 else "top", fontsize=9.5, color=style.INK_2)
    a2.axhline(0, color=style.AXIS, lw=1)
    a2.set_ylim(min(pct[ok].min() * 1.25, -1), max(pct[ok].max() * 1.6, 1))
    a2.set_xticks(x)
    a2.set_xticklabels(BAND_LABELS)
    a2.set_xlabel("Departures per day within a 5-minute walk")
    style.pct(a2)
    style.hgrid_only(a2)
    a2.set_title("Older buildings, vs the average of their own district", fontsize=12)

    share_low = (v["trips"] <= 150).mean()
    style.header(fig, "Frequent buses and trams lift prices only in new developments",
                 "New buildings with better service sell up to 24 % higher; among older buildings the "
                 "best-served streets are 3–6 % cheaper than the rest of their district")
    style.footer(fig, f"Buildings with ≥ 3 flat sales ({len(v):,}); value of a typical flat from the hedonic model "
                      f"(date, floor, size removed). Bands with < {MIN_N} buildings hidden; only "
                      f"{share_low:.0%} of buildings have ≤ 150 departures/day.\nDepartures: TransitFrequency "
                      "(all bus, tram, metro, rail departures within a 5-min walk, 06-22 h). " + style.SOURCE)
    fig.savefig(OUT, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}")
    print(v.groupby(["band", "kind"], observed=False)["value_2025"].agg(["median", "size"]).unstack().round(0))


if __name__ == "__main__":
    main()
