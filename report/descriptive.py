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
NEW, RESALE = "#c4521f", style.BLUE
SIZES = [(0, 25, "micro, under 25 m²"), (25, 35, "small, 25–35"), (35, 50, "compact, 35–50"),
         (50, 70, "medium, 50–70"), (70, 100, "spacious, 70–100"), (100, 1000, "large, 100+")]
SIZE_COL = ["#d7263d", "#f46036", "#f8b24a", "#c9d66b", "#4aa37b", "#1d5f8a"]


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}", dpi=150)
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
    per_year = t.groupby(["year", "market"]).size().unstack(fill_value=0)
    per_dist = t.groupby(["district", "market"]).size().unstack(fill_value=0)
    per_dist = per_dist.loc[per_dist.sum(axis=1).sort_values().index]
    try:
        n_houses = len(gpd.read_file("output/houses.gpkg", ignore_geometry=True))
    except Exception:
        n_houses = None
    fig = page("What's in the data",
               f"Every flat sale in Warsaw's official price register, {t['year'].min()}–{t['year'].max()}",
               "Notarial deeds of flats in full ownership, cleaned. New = sold by the developer (deed at completion, a year or two after the "
               "price was agreed). 2017–20 are nearly empty in the register; cooperative flats never enter it; 2026 is partial. " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"{len(t):,} flat sales: {per_year.get('secondary', 0).sum():,} resales and "
                               f"{per_year.get('primary', 0).sum():,} from developers"
                               + (f", plus {n_houses:,} house sales (recorded from 2017)" if n_houses else "")
                               + f". Most trade in {per_dist.sum(axis=1).idxmax()}, fewest in "
                               f"{per_dist.sum(axis=1).idxmin()}.", 100), fontsize=10.5, fontweight="semibold", va="top")
    a1 = fig.add_axes([0.10, 0.52, 0.86, 0.27])
    years = np.arange(2006, 2027)
    py = per_year.reindex(years, fill_value=0)
    a1.bar(years, py.get("secondary", 0) / 1000, color=RESALE, width=0.75, label="resale")
    a1.bar(years, py.get("primary", 0) / 1000, bottom=py.get("secondary", 0) / 1000, color=NEW, width=0.75,
           label="new (developer)")
    style.shade_gap(a1)
    a1.set_xticks(range(2006, 2027, 2))
    a1.set_ylabel("thousand sales")
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False)
    a1.set_title("Flat sales per year", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.20, 0.12, 0.74, 0.32])
    y = np.arange(len(per_dist))
    a2.barh(y, per_dist.get("secondary", 0) / 1000, color=RESALE, height=0.7)
    a2.barh(y, per_dist.get("primary", 0) / 1000, left=per_dist.get("secondary", 0) / 1000, color=NEW, height=0.7)
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
    fig = page("What sizes sell where",
               "Flats sold by size, by district (2021–26) and across the years",
               "RCN flat sales in full ownership, new and resale together. Size = usable floor area from the deed. Years with "
               "fewer than 300 sales left out (2017–20 register gap). " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"Small flats (under 35 m²) are {small.max():.0f} % of sales in {small.idxmax()} but "
                               f"{small.min():.0f} % in {small.idxmin()}; flats of 70 m² or more are most common in "
                               f"{share.index[-1]} and {share.index[-2]}. Micro flats under 25 m² are rare: "
                               f"{(t['area'] < 25).mean() * 100:.0f} % of all sales.", 100),
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
    bins = np.arange(15, 125, 5)
    for mk, lab, colr in (("primary", "new (developer)", NEW), ("secondary", "resale", RESALE)):
        a = recent.loc[recent["market"] == mk, "area"]
        h, _ = np.histogram(a, bins=bins)
        a2.step(bins[:-1], 100 * h / len(a), where="post", color=colr, lw=2, label=f"{lab}, median {a.median():.0f} m²")
    a2.set_xlabel("flat size, m² (5 m² steps; flats over 120 m² not shown)")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style.hgrid_only(a2)
    a2.legend(loc="upper right", fontsize=9, frameon=False)
    a2.set_title("Sizes sold, 2021–26 (share of each market's sales)", fontsize=10.5, loc="left")
    save(fig, "r03_new_resale.png")


def methods():
    fig = plt.figure(figsize=(8, 10))
    style.header(fig, "How to read these charts", "What the data is, what we compared, and what it can't tell", y=0.985)
    blocks = [
        ("The data", "Every notarial deed of a flat sale in Warsaw in the official price register (GUGiK RCN), 2006–2026: "
         "about 265,000 sales with price, size, floor and date. Locations, buildings, transit, noise and greenery come "
         "from OpenStreetMap, BDOT10k, the EU noise maps and Warsaw's timetables; building ages from GIS Expert's map "
         "of Warsaw's development and other sources."),
        ("The yardstick flat", "Flats differ, so most comparisons use one yardstick: a ~50 m² flat on the 2nd floor, "
         "resold (not new), priced in 2025. For each building we ask what that flat would sell for there, using every "
         "sale since 2006 and what we learned from flats in the same building about size, floor and the year of sale. "
         "Think of it as Warsaw's Big Mac index."),
        ("\"Within the same district\"", "When we say cafés or noise go with a price difference, we compare buildings "
         "against others in the same district, so that 'Mokotów is expensive' does not pass for 'cafés are expensive'."),
        ("Reputation", "What a place sells for beyond what its location, transport, amenities, noise and buildings "
         "predict. The model never sees district names. It also holds what the data can't see: condition, finish, "
         "views. The side of the river is not a feature: right-bank flats sell about 13 % below similar left-bank ones, "
         "but that gap is largely the right bank's reputation, so counting it would explain reputation away."),
        ("Houses", "House sales (RCN, from 2017) are shown as they are, not modelled: a house price includes land, and "
         "the register records a house's living area for about one deed in six. The rest is estimated from the "
         "building outline and floors (typical miss 14 % per house), fine for medians, not for a single house."),
        ("Associations, not causes", "These are patterns in real sales, not experiments. Cafés follow money as much as "
         "they attract it. Where we say more (metro openings), we compare the same buildings before and after."),
        ("What's missing", "Cooperative flats (spółdzielcze prawo) never enter the register: about one flat in five "
         "sits in a building with no recorded sale. 2017–2020 is nearly empty. Developer deeds carry prices agreed a "
         "year or two before completion. Asking prices are not sale prices."),
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
