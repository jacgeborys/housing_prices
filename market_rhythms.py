"""
Market rhythms: seasonality, momentum and company buyers.

  seasonality  resales 2021-25: each sale's log(price_m2_2025) minus the mean of the OTHER sales in the same
               building (leave-one-out; the quarterly city index is already removed), median by month of deed
  momentum     district price indices (district_index_annual.csv): a district's change vs the city in year t
               against the same in t+1, consecutive years only (no RCN-gap years)
  companies    share of flats bought by companies (TRAN_KUPUJACY = osobaPrawna) by year, and 2021-25 by district

Output: output/summary/19_market_rhythms.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap

style.apply()

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def seasonality():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                      columns=["osm_id", "date", "year", "market", "price_m2_2025"])
    t = t[(t["market"] == "secondary") & t["year"].between(2021, 2025)].dropna(subset=["osm_id"])
    t["lp"] = np.log(t["price_m2_2025"])
    g = t.groupby("osm_id")["lp"]
    n, s = g.transform("size"), g.transform("sum")
    t = t[n >= 4]
    loo = (s[t.index] - t["lp"]) / (n[t.index] - 1)
    t["rel"] = t["lp"] - loo
    t["month"] = pd.to_datetime(t["date"]).dt.month
    m = t.groupby("month")["rel"].agg(["mean", "sem", "size"])
    m["pct"] = 100 * (np.exp(m["mean"] - m["mean"].mean()) - 1)
    m["ci"] = 196 * m["sem"]
    return m


def momentum():
    di = pd.read_csv("output/analysis/district_index_annual.csv").sort_values(["district", "year"])
    q = pd.read_csv("output/analysis/price_index_quarterly.csv")
    q["year"] = q["quarter"].str[:4].astype(int)
    city = q.groupby("year")["gamma"].mean()
    di["rel"] = di["gamma"] - di["year"].map(city)
    di["chg"] = di.groupby("district")["rel"].diff()
    di["prev_year"] = di.groupby("district")["year"].shift()
    di = di[di["year"] - di["prev_year"] == 1]
    di["next"] = di.groupby("district")["chg"].shift(-1)
    nxt_year = di.groupby("district")["year"].shift(-1)
    di = di[(nxt_year - di["year"] == 1)].dropna(subset=["chg", "next"])
    return di


def companies():
    r = gpd.read_file("output/rcn_lokale_warszawa.gpkg",
                      columns=["TRAN_KUPUJACY", "LOK_FUNKCJA", "DOK_DATA", "TRAN_RODZAJ_RYNKU"])
    r = r[r["LOK_FUNKCJA"] == "mieszkalna"].copy()
    r["year"] = pd.to_numeric(r["DOK_DATA"].str[:4], errors="coerce")
    r["company"] = r["TRAN_KUPUJACY"].eq("osobaPrawna")
    by_year = r.groupby("year").agg(share=("company", "mean"), n=("company", "size"))
    by_year = by_year[(by_year["n"] > 2000) & (by_year.index >= 2006)]
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    rr = r[r["year"].between(2021, 2025)].to_crs(2180)
    j = gpd.sjoin(rr[["company", "geometry"]], d[["district", "geometry"]], predicate="within", how="inner")
    by_dist = j.groupby("district")["company"].mean().sort_values()
    return 100 * by_year["share"], 100 * by_dist


def main():
    m = seasonality()
    di = momentum()
    cy, cd = companies()
    r = np.corrcoef(di["chg"], di["next"])[0, 1]
    slope = np.polyfit(di["chg"], di["next"], 1)[0]
    print(m.round(3).to_string())
    print(f"momentum: {len(di)} district-years, corr {r:+.2f}, slope {slope:+.2f}")
    print(cy.round(1).to_string(), cd.round(1).to_string(), sep="\n")

    fig = page("Market rhythms",
               "When flats sell dearer, whether hot districts stay hot, and who besides people buys",
               "Seasonality: resales 2021–25 in buildings with ≥ 4 sales; each sale vs the other sales in the same "
               "building, with the quarterly city index removed; month of the deed (agreed earlier). Momentum: district "
               "indices relative to the city, consecutive years; part of the reversal is noise (a year measured too high "
               "looks like a fall the next). Companies: buyer type in RCN (osobaPrawna). "
               + style.SOURCE)
    a1 = fig.add_axes([0.10, 0.60, 0.86, 0.20])
    x = np.arange(12)
    a1.bar(x, m["pct"].values, color=[style.BLUE if v >= 0 else style.ORANGE for v in m["pct"].values], width=0.65)
    a1.errorbar(x, m["pct"].values, yerr=m["ci"].values, fmt="none", ecolor=style.INK_2, elinewidth=0.8)
    a1.axhline(0, color=style.AXIS, lw=1)
    a1.set_xticks(x)
    a1.set_xticklabels(MONTHS)
    a1.yaxis.set_major_locator(plt.MultipleLocator(0.5))
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    style.hgrid_only(a1)
    a1.set_title("Price by month of sale vs the yearly average (same building)", fontsize=11)

    a2 = fig.add_axes([0.10, 0.10, 0.38, 0.38])
    a2.scatter(100 * di["chg"], 100 * di["next"], s=22, color=style.BLUE, alpha=0.5, edgecolor="none")
    xs = np.linspace(100 * di["chg"].min(), 100 * di["chg"].max(), 10)
    a2.plot(xs, slope * xs + 100 * np.polyfit(di["chg"], di["next"], 1)[1], color=style.INK_2, lw=1.4,
            ls=(0, (4, 3)))
    a2.axhline(0, color=style.AXIS, lw=1)
    a2.axvline(0, color=style.AXIS, lw=1)
    style.pct(a2)
    style.pct(a2, "x")
    style.hgrid_only(a2)
    a2.set_xlabel("District vs city, this year", fontsize=9.5)
    a2.set_ylabel("District vs city, next year", fontsize=9.5)
    a2.set_title(f"Hot districts cool off (correlation {r:+.2f})", fontsize=10.5)

    a3 = fig.add_axes([0.60, 0.33, 0.36, 0.15])
    cy = cy.reindex(range(int(cy.index.min()), int(cy.index.max()) + 1))     # gap years break the line
    a3.plot(cy.index, cy.values, "-o", color=style.MAGENTA, lw=2, ms=4, mec=style.SURFACE)
    style.shade_gap(a3, label="")
    style.hgrid_only(a3)
    a3.set_ylim(0, None)
    a3.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a3.tick_params(labelsize=8.5)
    a3.set_title("Flats bought by companies", fontsize=10.5)
    a4 = fig.add_axes([0.72, 0.06, 0.24, 0.20])
    top = cd.tail(6)
    a4.barh(np.arange(len(top)), top.values, color=style.MAGENTA, height=0.6)
    a4.set_yticks(np.arange(len(top)))
    a4.set_yticklabels(top.index, fontsize=8.5)
    for yi, val in enumerate(top.values):
        a4.text(val + 0.1, yi, f"{val:.1f}%", va="center", fontsize=8, color=style.INK_2)
    a4.set_xlim(0, top.max() * 1.3)
    style.vgrid_only(a4)
    a4.tick_params(axis="x", labelsize=8)
    a4.set_title("Company buyers 2021–25", fontsize=9.5)
    worst = m["pct"].idxmin()
    verdict = "partly reverses" if slope < -0.1 else "fades" if slope < 0.15 else "persists"
    fig.text(0.03, 0.885, wrap(f"Seasons barely matter: every month is within about ±1 % of the yearly average "
                               f"({MONTHS[worst - 1]} lowest, {m.loc[worst, 'pct']:+.1f}%). A district that beats the city "
                               f"one year {verdict} the next (slope {slope:+.2f}). Companies buy about "
                               f"{cy.loc[2021:2025].mean():.0f}% of flats, most in the centre.", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "19_market_rhythms.png")


if __name__ == "__main__":
    main()
