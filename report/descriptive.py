"""
Plain descriptive slides for the report (what the data holds, no model): built from the cleaned flat sales
(price_index.py -> output/price_model.gpkg, layer transactions_adj) and the house sales (houses.py).

  r01_data.png        What's in the data: flat sales per year and per district, new vs resale
  r02_sizes.png       What sizes sell where: size categories by district (2021-26) and their share over time
  r03_new_resale.png  New vs resale: price per m² as paid, by year, and the size of flats each sells
  r99_methods.png     How to read these charts (text)

Output: output/report/src/
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, wrap

style.apply()
OUT = "output/report/src"
NEW, RESALE = style.BLUE, style.ORANGE        # new = blue, resale = orange (color-blind safe: simulated dE > 95)
SIZES = [(0, 25, "micro, under 25 m²"), (25, 35, "small, 25–35"), (35, 50, "compact, 35–50"),
         (50, 70, "medium, 50–70"), (70, 100, "spacious, 70–100"), (100, 1000, "large, 100+")]
from matplotlib.colors import LinearSegmentedColormap
_SIZE_RAMP = LinearSegmentedColormap.from_list("size", ["#f4eedd", "#f2c97d", "#e8914d", "#c9533f", "#8e2f55", "#4b1d5e"])
SIZE_COL = [_SIZE_RAMP(x) for x in (0.17, 0.32, 0.48, 0.64, 0.82, 1.0)]   # bigger flats = darker, as on the size map (2026-10-28)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}", dpi=300)
    plt.close(fig)
    print(f"Saved {OUT}/{name}")


def sales():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True)
    t = t[t["district"].notna() & t["year"].between(2006, 2026)].copy()
    t["size_cat"] = pd.cut(t["area"], [s[0] for s in SIZES] + [1000], right=False, labels=[s[2] for s in SIZES])
    return t


def strip(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def data_overview(t):
    from matplotlib.colors import to_rgb
    pale = lambda c: tuple(0.45 * np.array(to_rgb(c)) + 0.55)          # houses: the flats' colors, lighter
    per_year = t.groupby(["year", "market"]).size().unstack(fill_value=0)
    per_dist = t.groupby(["district", "market"]).size().unstack(fill_value=0)
    try:
        h = gpd.read_file("output/houses.gpkg", ignore_geometry=True)
    except Exception:
        h = pd.DataFrame(columns=["year", "market", "district"])
    hy = h.groupby(["year", "market"]).size().unstack(fill_value=0)
    hd = h.groupby(["district", "market"]).size().unstack(fill_value=0)
    tot = per_dist.sum(axis=1).add(hd.sum(axis=1), fill_value=0)
    per_dist = per_dist.reindex(tot.sort_values().index, fill_value=0)
    hd = hd.reindex(per_dist.index, fill_value=0)
    fig = page("What's in the data",
               f"Every flat and house sale in Warsaw's official price register, {t['year'].min()}–{t['year'].max()}",
               "Deeds in full ownership. New = sold by the developer (price agreed a year or two before the deed). "
               "Flats 2017–20 nearly empty, houses recorded from 2017; cooperative flats are often missing; 2026 is partial. "
               + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"{len(t):,} flat sales: {per_year.get('secondary', 0).sum():,} resales and "
                               f"{per_year.get('primary', 0).sum():,} from developers"
                               + (f", plus {len(h):,} house sales (recorded from 2017)" if len(h) else "")
                               + f". Most trade in {tot.idxmax()}, fewest in {tot.idxmin()}.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    layers = [("flat, resale", RESALE, "secondary", False), ("house, resale", pale(RESALE), "secondary", True),
              ("flat, new (developer)", NEW, "primary", False), ("house, new (developer)", pale(NEW), "primary", True)]

    a1 = fig.add_axes([0.10, 0.52, 0.86, 0.27])
    years = np.arange(2006, 2027)
    base = np.zeros(len(years))
    for lab, colr, mk, house in layers:
        v = (hy if house else per_year).reindex(years, fill_value=0).get(mk, pd.Series(0, index=years)).to_numpy() / 1000
        a1.bar(years, v, bottom=base, color=colr, width=0.75, label=lab)
        base += v
    style.shade_gap(a1)
    a1.set_xticks(range(2006, 2027, 2))
    a1.set_ylabel("thousand sales")
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    a1.set_title("Sales per year", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.20, 0.12, 0.74, 0.32])
    y = np.arange(len(per_dist))
    left = np.zeros(len(y))
    for lab, colr, mk, house in layers:
        v = (hd if house else per_dist).get(mk, pd.Series(0, index=per_dist.index)).to_numpy() / 1000
        a2.barh(y, v, left=left, color=colr, height=0.7)
        left += v
    a2.set_yticks(y)
    a2.set_yticklabels(per_dist.index, fontsize=8.5)
    a2.set_xlabel("thousand sales, 2006–2026")
    style.vgrid_only(a2)
    a2.set_title("Where they are", fontsize=10.5, loc="left")
    save(fig, "r01_data.png")


def sizes(t):
    recent = t[t["year"] >= 2021]
    share = pd.crosstab(recent["district"], recent["size_cat"], normalize="index") * 100
    share = share.loc[(share.iloc[:, 4] + share.iloc[:, 5]).sort_values().index]
    yr = pd.crosstab(t["year"], t["size_cat"], normalize="index") * 100
    yr = yr[t.groupby("year").size().reindex(yr.index) >= 300]
    small = recent["area"].lt(35).groupby(recent["district"]).mean() * 100
    fig = page("What sizes sell where? Small in Praga, big in Wawer",
               "Flats sold by size, by district (2021–26) and across the years",
               "RCN flat sales in full ownership, new and resale together. Size = usable floor area from the deed. Years with "
               "fewer than 300 sales left out (2017–20, Warsaw's register data gap). " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"Small flats (under 35 m²) are {small.max():.0f}% of sales in {small.idxmax()} but "
                               f"{small.min():.0f}% in {small.idxmin()}; flats of 70 m² or more are most common in "
                               f"{share.index[-1]} and {share.index[-2]}. Micro flats under 25 m² are rare: "
                               f"{(t['area'] < 25).mean() * 100:.0f}% of all sales.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    a1 = fig.add_axes([0.18, 0.41, 0.78, 0.34])
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for k, col in enumerate(share.columns):
        a1.barh(y, share[col], left=left, color=SIZE_COL[k], height=0.75, label=col, edgecolor=style.SURFACE, lw=0.4)
        left += share[col].to_numpy()
    a1.set_yticks(y)
    a1.set_yticklabels(share.index, fontsize=8.5)
    a1.set_xlim(0, 100)
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    a1.legend(loc="upper left", bbox_to_anchor=(0, 1.11), ncol=3, fontsize=8, frameon=False, handlelength=1)
    strip(a1)
    a1.set_title("Share of flats sold, 2021–26", fontsize=10.5, loc="left", pad=36)

    a2 = fig.add_axes([0.10, 0.08, 0.86, 0.25])
    years = yr.index.to_numpy()
    ends = {}
    for k, col in enumerate(yr.columns):
        s = yr[col].reindex(range(2006, 2027))
        a2.plot(s.index, s.values, "-o", color=SIZE_COL[k], lw=2, ms=3.5, mec=style.SURFACE)
        ends[k] = s.dropna().iloc[-1]
    placed = []                                       # end labels, pushed apart so none overlap
    for k in sorted(ends, key=ends.get):
        y_ = max(ends[k], placed[-1] + 2.2) if placed else ends[k]
        placed.append(y_)
        a2.text(2026.4, y_, yr.columns[k].split(",")[0], va="center", fontsize=8, color=SIZE_COL[k],
                fontweight="semibold")
    style.shade_gap(a2)
    a2.set_xlim(2005.5, 2028.5)
    a2.set_xticks(range(2006, 2027, 2))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style.hgrid_only(a2)
    a2.set_title("Share of flats sold, by year", fontsize=10.5, loc="left")
    save(fig, "r02_sizes.png")


def new_vs_resale(t):
    med = t.groupby(["year", "market"])["price_m2"].median().unstack()
    n = t.groupby(["year", "market"]).size().unstack()
    med = med.where(n >= 300)
    recent = t[t["year"] >= 2021]
    fig = page("New or resale?",
               "Price per m² as paid, and the size of flats each market sells",
               "Prices as paid (not inflation-adjusted), medians. New = sold by the developer: the deed is signed at completion, so its price "
               "was agreed a year or two earlier. Years with fewer than 300 sales of a kind left out. " + style.SOURCE)
    last = med.dropna().index.max()
    fig.text(0.03, 0.885, wrap(f"For years a resold flat cost more per m² in the register than a new one (2024: "
                               f"{med.loc[2024, 'secondary']:,.0f} vs {med.loc[2024, 'primary']:,.0f} PLN): developer "
                               "deeds carry prices agreed before completion, and new estates are more often on the "
                               f"edges. In {last} the two met ({med.loc[last, 'secondary']:,.0f} vs "
                               f"{med.loc[last, 'primary']:,.0f}).", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    a1 = fig.add_axes([0.11, 0.50, 0.85, 0.29])
    for mk, lab, colr in (("primary", "new (developer)", NEW), ("secondary", "resale", RESALE)):
        s = med[mk].reindex(range(2006, 2027))
        a1.plot(s.index, s.values, "-o", color=colr, lw=2.2, ms=4, mec=style.SURFACE, label=lab)
    style.shade_gap(a1)
    a1.set_xticks(range(2006, 2027, 2))
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K"))
    a1.set_ylabel("PLN per m²")
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False)
    a1.set_title("Median price per m², as paid", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.11, 0.12, 0.85, 0.28])
    from scipy.ndimage import gaussian_filter1d
    bins = np.arange(15, 125, 5)
    mid = bins[:-1] + 2.5
    for mk, lab, colr in (("primary", "new (developer)", NEW), ("secondary", "resale", RESALE)):
        a = recent.loc[recent["market"] == mk, "area"]
        h, _ = np.histogram(a, bins=bins)
        sm = gaussian_filter1d(100 * h / len(a), 1.0)
        a2.fill_between(mid, sm, color=colr, alpha=0.18, lw=0)
        a2.plot(mid, sm, color=colr, lw=2.4, label=f"{lab}, median {a.median():.0f} m²")
    a2.set_ylim(0, None)
    a2.set_xlabel("flat size, m² (smoothed; flats over 120 m² not shown)")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style.hgrid_only(a2)
    a2.legend(loc="upper right", fontsize=9, frameon=False)
    a2.set_title("Sizes sold, 2021–26 (share of each market's sales)", fontsize=10.5, loc="left")
    save(fig, "r03_new_resale.png")


def methods():
    fig = plt.figure(figsize=(8, 10))
    style.header(fig, "How to read these charts", "What the data is, what we compared, and what it can't tell", y=0.985)
    blocks = [
        ("The data", "Every notarial deed of a flat sale in Warsaw's official price register (GUGiK RCN), 2006–2026: about "
         "265,000 sales with price, size, floor and date. Maps, transit, noise and greenery from OpenStreetMap, BDOT10k, "
         "EU noise maps and Warsaw's timetables; building ages from GIS Expert and others."),
        ("The yardstick flat", "Flats differ, so most comparisons use one yardstick: a ~50 m² resold flat on the 2nd "
         "floor, priced in 2025, worked out for every building from all its sales. Warsaw's Big Mac index."),
        ("Reputation", "What a place sells for beyond what its location, transit, cafés, noise, greenery and building "
         "age predict. It also holds what the data can't see: condition, finish, views."),
        ("Associations, not causes", "Patterns in real sales, compared within the same district, not experiments: "
         "cafés follow money as much as they attract it. The closest thing to an experiment is the new metro stations."),
        ("What's missing", "Cooperative-right flats are often missing from the register, and Warsaw's register data for 2017–2020 is "
         "nearly empty. Houses are shown as sold, not modeled. Developer deeds carry prices agreed a year or two "
         "before completion; asking prices are not sale prices."),
        ("Other cities", "Same data, same method: Lublin and Gdańsk are mapped too."),
    ]
    y = 0.90
    for head, body in blocks:
        fig.text(0.06, y, head, fontsize=12, fontweight="bold", color=style.INK, va="top")
        lines = wrap(body, 96)
        fig.text(0.06, y - 0.027, lines, fontsize=10, color=style.INK_2, va="top", linespacing=1.4)
        y -= 0.048 + 0.0200 * (lines.count(chr(10)) + 1)
    style.footer(fig, "Analysis and charts: ceny_mieszkan project. " + style.SOURCE, y=0.012)
    save(fig, "r99_methods.png")


def main():
    t = sales()
    data_overview(t)
    sizes(t)
    new_vs_resale(t)
    methods()


if __name__ == "__main__":
    main()
