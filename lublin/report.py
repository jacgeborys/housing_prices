"""
Lublin report: plain descriptive slides in the Warsaw deck's style (style.py, summary_charts.page), no reputation model.
Run from the lublin/ folder after build_prices.py, fetch_building_age.py and maps.py:  python report.py

    output/price_model.gpkg (transactions_adj, building_value), output/analysis/*.csv, output/building_age.gpkg,
    data/rcn_lokale_lublin.gpkg (flat ids for flips), data/buildings.gpkg (coverage)
 -> output/report/src/r01_data ... r99_methods.png, output/report/NN_*.png, output/report/lublin_flats.pdf

Slides: 01/02 maps (maps.py), r01 data, r02 coverage, r03 prices over time (nominal, real, vs Warsaw), r04 districts,
r05 sizes, r06 new vs resale, r07 building age, r08 flippers, r99 methods.
"""
import os
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import geopandas as gpd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from PIL import Image  # noqa: E402

import style  # noqa: E402
from summary_charts import page, wrap  # noqa: E402

style.apply()
SRC = "output/report/src"
OUT = "output/report"
NEW, RESALE = "#c4521f", style.BLUE
SOURCE = "Source: GUGiK Rejestr Cen Nieruchomości (RCN), BDOT10k, GIS Expert."
SIZES = [(0, 25, "micro, under 25 m²"), (25, 35, "small, 25–35"), (35, 50, "compact, 35–50"),
         (50, 70, "medium, 50–70"), (70, 100, "spacious, 70–100"), (100, 1000, "large, 100+")]
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
_RAMP = LinearSegmentedColormap.from_list("size", ["#f4eedd", "#f2c97d", "#e8914d", "#c9533f", "#8e2f55", "#4b1d5e"])
SIZE_COL = [_RAMP(x) for x in (0.17, 0.32, 0.48, 0.64, 0.82, 1.0)]
SLIDES = [("summary", "01_price_ridgelines"), ("summary", "02_price_points"), ("src", "r01_data"),
          ("src", "r02_coverage"), ("src", "r03_prices"), ("src", "r04_districts"), ("src", "r05_sizes"),
          ("src", "r06_new_resale"), ("src", "r07_age"), ("src", "r08_flippers"), ("src", "r99_methods")]


def save(fig, name):
    os.makedirs(SRC, exist_ok=True)
    fig.savefig(f"{SRC}/{name}", dpi=300)
    plt.close(fig)
    print(f"Saved {SRC}/{name}")


def headline(fig, text):
    fig.text(0.03, 0.885, wrap(text, 100), fontsize=10.5, fontweight="semibold", va="top")


def sales():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t["date"] = pd.to_datetime(t["date"])
    t["size_cat"] = pd.cut(t["area"], [s[0] for s in SIZES] + [1000], right=False, labels=[s[2] for s in SIZES])
    return t


# r01 ------------------------------------------------------------------------------------------
def data_overview(t):
    q = t.groupby([t["date"].dt.to_period("Q"), "market"]).size().unstack(fill_value=0)
    per_dist = t.groupby(["district", "market"]).size().unstack(fill_value=0)
    per_dist = per_dist.loc[per_dist.sum(axis=1).sort_values().index]
    small = per_dist[per_dist.sum(axis=1) < 200]
    per_dist = per_dist[per_dist.sum(axis=1) >= 200]
    full = t[t["year"].between(2017, 2025)]
    new_share = 100 * (full["market"] == "primary").groupby(full["year"]).mean()
    fig = page("What's in the data",
               f"Every flat sale in Lublin's official price register, Sept 2016 – {t['date'].max():%b %Y}",
               "Deeds of flats in full ownership, inside the city. New = sold by the developer (price agreed a year or two "
               "before the deed). Mid-2026 on is incomplete (deeds arrive months late). " + str(len(small)) + " districts with < 200 sales "
               "not listed. Source: GUGiK RCN.")
    top = per_dist.sum(axis=1)
    headline(fig, f"{len(t):,} flat sales: {(t['market'] == 'secondary').sum():,} resales and "
                  f"{(t['market'] == 'primary').sum():,} from developers ({new_share.min():.0f}–{new_share.max():.0f} % a year). "
                  f"Most trade in {top.index[-1]} and {top.index[-2]}; 2022 dipped by a quarter as mortgage rates rose.")
    a1 = fig.add_axes([0.10, 0.55, 0.86, 0.25])
    x = np.arange(len(q))
    a1.bar(x, q.get("secondary", 0), color=RESALE, width=0.8, label="resale")
    a1.bar(x, q.get("primary", 0), bottom=q.get("secondary", 0), color=NEW, width=0.8, label="new (developer)")
    yrs = [i for i, p in enumerate(q.index) if p.quarter == 1]
    a1.set_xticks(yrs)
    a1.set_xticklabels([q.index[i].year for i in yrs])
    a1.set_ylabel("sales per quarter")
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    a1.set_title("Flat sales per quarter", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.24, 0.10, 0.70, 0.38])
    y = np.arange(len(per_dist))
    a2.barh(y, per_dist.get("secondary", 0) / 1000, color=RESALE, height=0.7)
    a2.barh(y, per_dist.get("primary", 0) / 1000, left=per_dist.get("secondary", 0) / 1000, color=NEW, height=0.7)
    a2.set_yticks(y)
    a2.set_yticklabels(per_dist.index, fontsize=8.5)
    a2.set_xlabel("thousand sales, 2016–2026")
    style.vgrid_only(a2)
    a2.set_title("Where they are", fontsize=10.5, loc="left")
    save(fig, "r01_data.png")


# r02 ------------------------------------------------------------------------------------------
def coverage(t):
    """Floor area of blocks of flats (BDOT footprint x storeys) in buildings with at least one recorded sale."""
    b = gpd.read_file("data/buildings.gpkg")
    b = b[b["kind"].eq("budynek wielorodzinny")].copy()
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    c = gpd.sjoin(gpd.GeoDataFrame(geometry=b.geometry.representative_point(), crs=2180), d[["district", "geometry"]],
                  predicate="within")
    b["district"] = c[~c.index.duplicated()]["district"]
    b["floor_m2"] = b.area * b["storeys"].fillna(1).clip(1, 20)
    n_sales = t.groupby("osm_id").size()
    b["sales"] = b["osm_id"].map(n_sales).fillna(0)
    b["seen"] = b["sales"] > 0
    by = b.groupby("district").apply(lambda g: pd.Series({
        "share": 100 * g.loc[g["seen"], "floor_m2"].sum() / g["floor_m2"].sum(), "floor": g["floor_m2"].sum()}))
    by = by[by["floor"] > 50_000].sort_values("share")
    city = 100 * b.loc[b["seen"], "floor_m2"].sum() / b["floor_m2"].sum()
    raw = gpd.read_file("data/rcn_lokale_lublin.gpkg", ignore_geometry=True,
                        columns=["LOK_FUNKCJA", "NIER_UDZIAL", "LOK_CENA_BRUTTO", "NIER_CENA_BRUTTO"])
    flats = raw[raw["LOK_FUNKCJA"] == "mieszkalna"]
    lok_empty = 100 * pd.to_numeric(flats["LOK_CENA_BRUTTO"], errors="coerce").isna().mean()
    steps = [("Records in the register box", len(raw)),
             ("of which flats (not garages, shops)", len(flats)),
             ("whole flats (not a share of one)", None),
             ("inside the city, cleaned", len(t)),
             ("matched to a building", int(t["osm_id"].notna().sum()))]
    share = flats["NIER_UDZIAL"].fillna("").astype(str).str.extract(r"^(\d+)/(\d+)$").astype(float)
    steps[2] = (steps[2][0], int(len(flats) - (share[0].notna() & (share[0] != share[1])).sum()))

    fig = page("What the register can't see",
               "From raw records to usable sales, and how much of the city's flat stock ever shows up",
               "Floor area = BDOT10k outline x storeys, blocks of flats only ('budynek wielorodzinny'); seen = at least one recorded "
               "flat sale 2016-26. Districts with < 50,000 m² of blocks left off. " + SOURCE)
    headline(fig, f"The register sees {city:.0f} % of the floor area of Lublin's blocks of flats; the rest is "
                  "cooperative flats (never registered) and blocks where nothing sold since 2016. Lublin also leaves the "
                  f"flat price field empty ({lok_empty:.1f} % of flats): the price is taken from the property field instead.")
    a1 = fig.add_axes([0.08, 0.58, 0.88, 0.22])
    labs = [s[0] for s in steps]
    vals = [s[1] for s in steps]
    y = np.arange(len(steps))[::-1]
    a1.barh(y, vals, color=[style.CONTEXT, style.CONTEXT, style.CONTEXT, RESALE, RESALE], height=0.62)
    for yi, v, lab in zip(y, vals, labs):
        a1.text(1000, yi, f"{lab}", va="center", fontsize=9.5, color=style.INK)
        a1.text(v + 800, yi, f"{v:,}", va="center", fontsize=9.5, color=style.INK_2)
    a1.set_yticks([])
    a1.set_xlim(0, max(vals) * 1.15)
    a1.set_xticks([])
    for sp in ("left", "bottom", "top", "right"):
        a1.spines[sp].set_visible(False)
    a1.set_title("From records to sales", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.26, 0.07, 0.68, 0.43])
    y = np.arange(len(by))
    a2.barh(y, by["share"], color=RESALE, height=0.68)
    a2.barh(y, 100 - by["share"], left=by["share"], color=style.CONTEXT, height=0.68)
    for yi, s in enumerate(by["share"]):
        a2.text(s - 1.5, yi, f"{s:.0f} %", va="center", ha="right", fontsize=8, color="white")
    a2.set_yticks(y)
    a2.set_yticklabels(by.index, fontsize=8.5)
    a2.set_xlim(0, 100)
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a2.set_title("Share of blocks' floor area with a recorded sale (grey: never seen)", fontsize=10.5, loc="left")
    save(fig, "r02_coverage.png")


# r03 ------------------------------------------------------------------------------------------
def prices_over_time():
    q = pd.read_csv("output/analysis/price_index_quarterly.csv")
    q["year"] = q["quarter"].str[:4].astype(int)
    q["t"] = q["year"] + (q["quarter"].str[-1].astype(int) - 0.5) / 4
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    level = np.average(bv["value_2025"], weights=bv["n"])
    cpi = style.cpi_level(2025)
    q["nominal"] = level * np.exp(q["gamma"])
    q["real"] = q["nominal"] / q["year"].map(cpi)
    q = q[q["estimated"].astype(bool)]
    w = pd.read_csv(os.path.join(ROOT, "output/analysis/price_index_quarterly.csv"))
    w["year"] = w["quarter"].str[:4].astype(int)
    w["t"] = w["year"] + (w["quarter"].str[-1].astype(int) - 0.5) / 4
    w = w[w["t"] >= q["t"].min()].copy()
    w.loc[~w["estimated"].astype(bool), "gamma"] = np.nan
    first = lambda s, col: s.loc[s["year"] == 2017, col].mean()      # noqa: E731
    l_n = 100 * np.exp(q["gamma"] - q.loc[q["quarter"] == "2016Q4", "gamma"].iloc[0])
    w_n = 100 * np.exp(w["gamma"] - w.loc[w["quarter"] == "2016Q4", "gamma"].iloc[0])     # Warsaw: no sales 2017-20
    n17, n25 = first(q, "nominal"), q.loc[q["year"] == 2025, "nominal"].mean()
    r17, r25 = first(q, "real"), q.loc[q["year"] == 2025, "real"].mean()
    w25 = w_n[w["year"] == 2025].mean()

    fig = page(f"Lublin flats cost {n25 / n17:.1f}× what they did in 2017,\n{r25 / r17:.2f}× after inflation",
               "", "Same kind of flat each quarter (hedonic model, within-building comparisons). Real = GUS CPI, 2025 PLN. "
                   "Warsaw: same model, its register has almost no sales 2017-20. Source: GUGiK RCN.")
    for x, big, small in ((0.05, f"×{n25 / n17:.1f}", "price as paid\n2017 → 2025"),
                          (0.36, f"×{r25 / r17:.2f}", "after inflation\n(today's money), 2017 → 2025"),
                          (0.70, f"{style.fmt_int(n25)}", "PLN per m², typical\nflat, 2025")):
        fig.text(x, 0.885, big, fontsize=30, fontweight="bold", color=style.INK, va="top")
        fig.text(x, 0.835, small, fontsize=10.5, color=style.INK_2, va="top")
    ax = fig.add_axes([0.13, 0.43, 0.80, 0.33])
    for col, colr, lab, lw in (("nominal", style.BLUE, "Price as paid", 1.8), ("real", style.ORANGE, "In 2025 money", 3.0)):
        ax.plot(q["t"], q[col], color=colr, lw=lw)
        last = q.iloc[-1]
        ax.annotate(f"{lab}: {style.fmt_int(last[col])}", (last["t"], last[col]), xytext=(-4, 10 if col == "nominal" else -16),
                    textcoords="offset points", ha="right", fontsize=9.5, color=colr, fontweight="semibold")
    style.pln(ax)
    style.hgrid_only(ax)
    ax.set_ylim(0, None)
    ax.set_xlim(2016.6, 2026.8)
    ax.set_ylabel("PLN per m², typical flat")
    ax.set_title("Typical Lublin flat, per m²", fontsize=10.5, loc="left")

    a2 = fig.add_axes([0.13, 0.10, 0.80, 0.25])
    a2.plot(q["t"], l_n, color=style.ORANGE, lw=2.6, label="Lublin")
    a2.plot(w["t"], w_n, color=style.INK_2, lw=1.6, ls=(0, (3, 2)), label="Warsaw")
    a2.axhline(100, color=style.AXIS, lw=1)
    a2.legend(loc="upper left", fontsize=9, frameon=False)
    style.hgrid_only(a2)
    a2.set_xlim(2016.6, 2026.8)
    a2.set_title(f"Same flat, end of 2016 = 100, as paid: Lublin {l_n[q['year'] == 2025].mean():.0f}, Warsaw {w25:.0f} in 2025",
                 fontsize=10.5, loc="left")
    save(fig, "r03_prices.png")


# r04 ------------------------------------------------------------------------------------------
def districts():
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    val = bv.groupby("district").agg(value=("value_2025", "median"), n=("n", "sum"), b=("osm_id", "size"))
    val = val[val["b"] >= 15]
    di = pd.read_csv("output/analysis/district_index_annual.csv")
    base = di[di["year"].between(2017, 2018)].groupby("district").apply(
        lambda g: np.average(g["index_2025"], weights=g["n"]))
    grow = (100 / base - 1) * 100
    val["growth"] = grow
    val = val.sort_values("value")
    city = bv["value_2025"].median()                      # same measure as the bars (median over buildings)
    g = val.dropna(subset=["growth"])
    fig = page("Where flats cost most, and where they rose most",
               "Typical flat per district in 2025, and its price change since 2017-18 (same flats, as paid)",
               "Typical flat = ~50 m², 2 rooms, floor 2-3, resale, priced in 2025 (median over buildings, >= 15 buildings). "
               "Change: district index from repeat comparisons within buildings, 2017-18 to 2025. " + SOURCE)
    headline(fig, f"A typical flat costs {style.fmt_int(val['value'].iloc[-1])} PLN/m² in {val.index[-1]} and "
                  f"{style.fmt_int(val['value'].iloc[0])} in {val.index[0]} (median building city-wide {style.fmt_int(city)}). Since 2017-18 prices "
                  f"rose most in {g['growth'].idxmax()} (+{g['growth'].max():.0f} %) and least in {g['growth'].idxmin()} "
                  f"(+{g['growth'].min():.0f} %); most districts roughly doubled.")
    a1 = fig.add_axes([0.24, 0.08, 0.34, 0.70])
    y = np.arange(len(val))
    a1.barh(y, val["value"], color=style.PRICE_CMAP((val["value"] - 6500) / 6000), height=0.7)
    a1.axvline(city, color=style.INK_2, lw=1, ls=(0, (3, 2)))
    for yi, v in enumerate(val["value"]):
        a1.text(v + 100, yi, f"{v / 1000:.1f}K", va="center", fontsize=8, color=style.INK_2)
    a1.set_yticks(y)
    a1.set_yticklabels(val.index, fontsize=8.5)
    a1.set_xlim(0, val["value"].max() * 1.18)
    a1.xaxis.set_major_locator(plt.MultipleLocator(4000))
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    style.vgrid_only(a1)
    a1.set_title("PLN per m², 2025", fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.63, 0.08, 0.33, 0.70])
    gv = val["growth"]
    a2.barh(y, gv.fillna(0), color=[style.ORANGE if np.isfinite(v) else "none" for v in gv], height=0.7)
    for yi, v in enumerate(gv):
        a2.text((v if np.isfinite(v) else 0) + 2, yi, f"+{v:.0f}%" if np.isfinite(v) else "too few", va="center",
                fontsize=8, color=style.INK_2)
    a2.set_yticks([])
    a2.set_xlim(0, np.nanmax(gv) * 1.25)
    a2.xaxis.set_major_locator(plt.MultipleLocator(50))
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    style.vgrid_only(a2)
    a2.set_title("Change since 2017-18", fontsize=10.5, loc="left")
    save(fig, "r04_districts.png")


# r05 ------------------------------------------------------------------------------------------
def sizes(t):
    recent = t[t["year"] >= 2021]
    share = pd.crosstab(recent["district"], recent["size_cat"], normalize="index") * 100
    share = share[recent["district"].value_counts().reindex(share.index) >= 300]
    share = share.loc[(share.iloc[:, 4] + share.iloc[:, 5]).sort_values().index]
    yr = pd.crosstab(t["year"], t["size_cat"], normalize="index") * 100
    yr = yr[t.groupby("year").size().reindex(yr.index) >= 1500]
    med = t[t["year"].between(2017, 2025)].groupby(["year", "market"])["area"].median().unstack()
    small = recent["area"].lt(35).groupby(recent["district"]).mean().reindex(share.index) * 100
    fig = page("What sizes sell where",
               "Flats sold by size, by district (2021–26) and across the years",
               "RCN flat sales in full ownership, new and resale together. Size = usable floor area from the deed. Districts with "
               "fewer than 300 sales 2021-26 and the partial 2016 left off. " + SOURCE)
    headline(fig, f"Small flats (under 35 m²) are {small.max():.0f} % of sales in {small.idxmax()} but {small.min():.0f} % in "
                  f"{small.idxmin()}. New flats got smaller: a median {med.loc[2017, 'primary']:.0f} m² in 2017, "
                  f"{med.loc[2025, 'primary']:.0f} m² in 2025 (resales {med.loc[2025, 'secondary']:.0f} m²).")
    a1 = fig.add_axes([0.22, 0.41, 0.74, 0.34])
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
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a1.set_title("Share of flats sold, 2021–26", fontsize=10.5, loc="left", pad=36)
    a2 = fig.add_axes([0.10, 0.08, 0.80, 0.25])
    ends = {}
    for k, col in enumerate(yr.columns):
        s = yr[col]
        a2.plot(s.index, s.values, "-o", color=SIZE_COL[k], lw=2, ms=3.5, mec=style.SURFACE)
        ends[k] = s.iloc[-1]
    placed = []
    for k in sorted(ends, key=ends.get):
        y_ = max(ends[k], placed[-1] + 2.4) if placed else ends[k]
        placed.append(y_)
        a2.text(yr.index.max() + 0.3, y_, yr.columns[k].split(",")[0], va="center", fontsize=8, color=SIZE_COL[k],
                fontweight="semibold")
    a2.set_xlim(yr.index.min() - 0.5, yr.index.max() + 1.6)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style.hgrid_only(a2)
    a2.set_title("Share of flats sold, by year", fontsize=10.5, loc="left")
    save(fig, "r05_sizes.png")


# r06 ------------------------------------------------------------------------------------------
def new_vs_resale(t):
    t = t[t["year"].between(2017, 2026)]
    med = t.groupby(["year", "market"])["price_m2"].median().unstack()
    recent = t[t["year"] >= 2021]
    from scipy.ndimage import gaussian_filter1d
    fig = page("New or resale?",
               "Price per m² as paid, and the size of flats each market sells",
               "Prices as paid (not inflation-adjusted), medians. New = sold by the developer: the deed is signed at completion, so its "
               "price was agreed a year or two earlier. 2016 (from September) left off. " + SOURCE)
    dearer = [y for y in med.index if med.loc[y, "secondary"] > med.loc[y, "primary"]]
    headline(fig, f"Resales cost more per m² than new flats in {min(dearer)}–{max(dearer)} (2021: {med.loc[2021, 'secondary']:,.0f} vs "
                  f"{med.loc[2021, 'primary']:,.0f} PLN): developer deeds carry prices agreed before completion. In 2025 new flats "
                  f"overtook ({med.loc[2025, 'primary']:,.0f} vs {med.loc[2025, 'secondary']:,.0f}). Both sell the same sizes.")
    a1 = fig.add_axes([0.11, 0.50, 0.85, 0.29])
    for mk, lab, colr in (("primary", "new (developer)", NEW), ("secondary", "resale", RESALE)):
        a1.plot(med.index, med[mk], "-o", color=colr, lw=2.2, ms=4, mec=style.SURFACE, label=lab)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x / 1000:.0f}K"))
    a1.set_ylabel("PLN per m²")
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False)
    a1.set_title("Median price per m², as paid", fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.11, 0.12, 0.85, 0.28])
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
    save(fig, "r06_new_resale.png")


# r07 ------------------------------------------------------------------------------------------
def age(t):
    a = gpd.read_file("output/building_age.gpkg", ignore_geometry=True)[["osm_id", "decade_end"]]
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    bv = bv.merge(a, on="osm_id", how="left")
    groups = [(1900, 1940, "pre-war"), (1941, 1960, "1941–60"), (1961, 1970, "1961–70"), (1971, 1980, "1971–80"),
              (1981, 1990, "1981–90"), (1991, 2000, "1991–2000"), (2001, 2010, "2001–10"), (2011, 2020, "2011–20")]
    def grp(dec):
        for lo, hi, lab in groups:
            if pd.notna(dec) and dec > 0 and lo <= max(dec, 1900) <= hi:
                return lab
        return None
    bv["era"] = bv["decade_end"].map(grp)
    dated = bv["era"].notna()
    order = [g[2] for g in groups]
    s = bv[dated].groupby("era").agg(value=("value_2025", "median"), q1=("value_2025", lambda x: x.quantile(.25)),
                                     q3=("value_2025", lambda x: x.quantile(.75)), b=("osm_id", "size"),
                                     n=("n", "sum")).reindex(order)
    # same district: each building vs its district median, then median by era
    bv["rel"] = 100 * (bv["value_2025"] / bv.groupby("district")["value_2025"].transform("median") - 1)
    rel = bv[dated].groupby("era")["rel"].median().reindex(order)
    fig = page("Only the newest buildings cost more",
               "Typical flat by when the building went up",
               f"Typical flat = ~50 m², 2 rooms, floor 2-3, resale, 2025, per building (median, bars: middle half). Decade: GIS Expert, "
               f"'Rozwój zabudowy Lublina' (to 2020); {int(dated.sum()):,} of {len(bv):,} priced buildings dated. " + SOURCE)
    old = s.loc[order[:6], "value"]
    headline(fig, f"A typical flat in a building from before 2000 costs about the same whatever the decade "
                  f"({old.min() / 1000:.1f}–{old.max() / 1000:.1f}K PLN/m²); 2001-10 buildings {style.fmt_int(s.loc['2001–10', 'value'])}, "
                  f"2011-20 {style.fmt_int(s.loc['2011–20', 'value'])}. Against the same district: "
                  f"{rel.loc['2001–10']:+.0f} % and {rel.loc['2011–20']:+.0f} %, older ones {rel[order[:6]].min():+.0f} to {rel[order[:6]].max():+.0f} %.")
    a1 = fig.add_axes([0.10, 0.45, 0.86, 0.33])
    x = np.arange(len(order))
    a1.bar(x, s["value"], color=style.PRICE_CMAP((s["value"] - 6500) / 6000), width=0.68)
    a1.vlines(x, s["q1"], s["q3"], color=style.INK_2, lw=1.2)
    for xi, v, b in zip(x, s["value"], s["b"]):
        a1.text(xi, 300, f"{int(b)}\nbldgs", ha="center", fontsize=7.5, color="white")
        a1.text(xi + 0.08, v + 150, f"{v / 1000:.1f}K", fontsize=8.5, color=style.INK)
    a1.set_xticks(x)
    a1.set_xticklabels(order, fontsize=9)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}K"))
    style.hgrid_only(a1)
    a1.set_title("PLN per m², typical flat, 2025", fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.10, 0.09, 0.86, 0.26])
    a2.bar(x, rel, color=[style.ORANGE if v > 0 else style.BLUE for v in rel], width=0.68)
    a2.axhline(0, color=style.AXIS, lw=1)
    for xi, v in zip(x, rel):
        a2.text(xi, v + (0.6 if v >= 0 else -0.6), f"{v:+.0f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    a2.set_xticks(x)
    a2.set_xticklabels(order, fontsize=9)
    a2.yaxis.set_major_locator(plt.MultipleLocator(2))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    style.hgrid_only(a2)
    a2.set_ylim(min(rel.min() * 1.4, -5), max(rel.max() * 1.4, 5))
    a2.set_title("Against the median building of the same district", fontsize=10.5, loc="left")
    save(fig, "r07_age.png")


# r08 ------------------------------------------------------------------------------------------
def flips(t):
    """Flip = the same flat (land-register id, filled for every Lublin record) resold 14-365 days later."""
    r = gpd.read_file("data/rcn_lokale_lublin.gpkg", ignore_geometry=True,
                      columns=["LOK_ID_LOKALU", "LOK_ADRES", "DOK_DATA", "NIER_CENA_BRUTTO", "LOK_POW_UZYT", "LOK_FUNKCJA"])
    r = r[(r["LOK_FUNKCJA"] == "mieszkalna") & r["LOK_ID_LOKALU"].notna()]
    r = pd.DataFrame({"date": pd.to_datetime(r["DOK_DATA"].str[:10], errors="coerce"), "lok_adres": r["LOK_ADRES"],
                      "price": pd.to_numeric(r["NIER_CENA_BRUTTO"], errors="coerce").round(0),
                      "area": pd.to_numeric(r["LOK_POW_UZYT"], errors="coerce").round(1), "lok_id": r["LOK_ID_LOKALU"]})
    r = r.drop_duplicates(["date", "lok_adres", "price", "area"], keep=False)
    k = t.assign(price=t["price"].round(0), area=t["area"].round(1))
    m = k.merge(r, on=["date", "lok_adres", "price", "area"], how="left")
    print(f"flat id attached to {m['lok_id'].notna().mean():.0%} of sales")
    m = gpd.GeoDataFrame(m.dropna(subset=["lok_id"]), geometry="geometry", crs=2180).sort_values(["lok_id", "date"])
    nxt = m.groupby("lok_id").shift(-1)
    p = m.assign(date2=nxt["date"], price2=nxt["price"], adj2=nxt["price_m2_2025"], market2=nxt["market"]).dropna(subset=["date2"])
    p = p[p["market2"] == "secondary"]
    p["days"] = (p["date2"] - p["date"]).dt.days
    p["markup"] = 100 * (p["price2"] / p["price"] - 1)
    p["markup_net"] = 100 * (p["adj2"] / p["price_m2_2025"] - 1)
    p = p[p["markup"].between(-50, 200)]
    f = p[p["days"].between(14, 365)].copy()
    f["year2"] = f["date2"].dt.year
    new_resold = f[f["market"] == "primary"]          # bought from the developer: price agreed a year or two before the deed
    f = f[f["market"] == "secondary"]
    sec = t[t["market"] == "secondary"]
    by = pd.DataFrame({"flips": f.groupby("year2").size(), "new": new_resold.groupby("year2").size(),
                       "resales": sec.groupby("year").size()}).fillna(0)
    by = by.loc[2017:2025]
    by["share"] = 100 * by["flips"] / by["resales"]
    by["share_new"] = 100 * by["new"] / by["resales"]
    rec = f[f["year2"].between(2021, 2025)]
    rec_s = sec[sec["year"].between(2021, 2025)]
    dist = pd.DataFrame({"flips": rec.groupby("district").size(), "resales": rec_s.groupby("district").size()}).fillna(0)
    dist = dist[dist["resales"] >= 300]
    dist["share"] = 100 * dist["flips"] / dist["resales"]
    dist = dist.sort_values("share")
    print(by.round(1).to_string())
    os.makedirs("output/analysis", exist_ok=True)
    pd.concat([f, new_resold]).drop(columns="geometry").to_csv("output/analysis/flips.csv", index=False)
    rec_new = new_resold[new_resold["year2"].between(2021, 2025)]
    med, med_net = rec["markup"].median(), rec["markup_net"].median()
    fig = page("The flippers",
               "Flats bought on the resale market and sold again within a year: how often, for how much, and where",
               "One flat = its land-register id. Flip = bought resale, sold again 14-365 days later; net = beyond the city-wide index. "
               "Districts with >= 300 resales 2021-25. " + SOURCE)
    headline(fig, f"2021–25: {len(rec):,} flats were resold within a year of purchase ({100 * len(rec) / len(rec_s):.0f} in 100 "
                  f"resales), at a median markup of {med:+.0f} % as paid, {med_net:+.0f} % beyond the market, before "
                  f"renovation, tax and fees. Another {len(rec_new):,} flats bought new were resold within a year (median "
                  f"{rec_new['markup'].median():+.0f} %), partly because the developer's price was agreed years before.")
    a1 = fig.add_axes([0.10, 0.50, 0.38, 0.28])
    a1.plot(by.index, by["share"], "-o", color="#c4521f", lw=2, ms=5, mec=style.SURFACE)
    a1.plot(by.index, by["share_new"], "-o", color=style.MUTED, lw=1.5, ms=4, mec=style.SURFACE)
    a1.text(by.index[-1] + 0.2, by["share"].iloc[-1], "bought\nresale", fontsize=8, color="#c4521f", va="center")
    a1.text(by.index[-1] + 0.2, by["share_new"].iloc[-1], "bought\nnew", fontsize=8, color=style.MUTED, va="center")
    a1.set_xlim(by.index[0] - 0.5, by.index[-1] + 1.8)
    style.hgrid_only(a1)
    a1.set_ylim(0, None)
    a1.set_title("Flips per 100 resales, by year", fontsize=10.5)
    a1.tick_params(labelsize=8.5)
    a2 = fig.add_axes([0.58, 0.50, 0.38, 0.28])
    bins = np.arange(-30, 81, 5)
    a2.hist(rec["markup"].clip(-30, 80), bins=bins, color="#f08a4b", edgecolor=style.SURFACE, lw=1)
    a2.axvline(0, color=style.AXIS, lw=1)
    a2.axvline(med, color=style.INK, lw=1.4, ls=(0, (4, 3)))
    a2.set_ylim(0, a2.get_ylim()[1] * 1.14)
    a2.text(med + 1.5, a2.get_ylim()[1] * 0.99, f"median {med:+.0f}%", fontsize=9, color=style.INK, va="top")
    a2.set_xticks([-20, 0, 20, 40, 60, 80])
    a2.set_xticklabels(["−20%", "0%", "+20%", "+40%", "+60%", "+80%+"])
    a2.set_yticks([])
    style.hgrid_only(a2)
    a2.set_title("Markup on resale, as paid (2021–25)", fontsize=10.5)
    a2.tick_params(labelsize=8.5)
    a3 = fig.add_axes([0.26, 0.07, 0.68, 0.34])
    y = np.arange(len(dist))
    a3.barh(y, dist["share"], color="#c4521f", height=0.65)
    for yi, s in enumerate(dist["share"]):
        a3.text(s + 0.1, yi, f"{s:.1f}", va="center", fontsize=8.5, color=style.INK_2)
    a3.set_yticks(y)
    a3.set_yticklabels(dist.index, fontsize=8.5)
    style.vgrid_only(a3)
    a3.set_title("Flips per 100 resales by district, 2021–25", fontsize=10.5, loc="left")
    save(fig, "r08_flippers.png")


# r99 ------------------------------------------------------------------------------------------
def methods():
    fig = plt.figure(figsize=(8, 10))
    style.header(fig, "How to read these charts", "What the data is, what we compared, and what it can't tell", y=0.985)
    blocks = [
        ("The data", "Every notarial deed of a flat sale in Lublin in the official price register (GUGiK RCN), from "
         "September 2016: about 45,000 sales with price, size, floor and date. Lublin records the price in the property "
         "field, not the flat field, so that is used. Buildings from BDOT10k; building ages from GIS Expert's map of "
         "Lublin's development (decades to 2020)."),
        ("The yardstick flat", "Flats differ, so prices are compared on one yardstick: a ~50 m² flat on the 2nd floor, "
         "resold, priced in 2025. For each building we ask what that flat would sell for there, from every sale since 2016 "
         "and what flats in the same building tell us about size, floor and the year of sale. Same model as for Warsaw."),
        ("Real prices", "Prices as paid are deflated with GUS's annual consumer price index to 2025 złoty."),
        ("No reputation model", "Unlike the Warsaw report, there is no model of what location and buildings predict, so no "
         "'cheaper than expected' map. Differences shown are plain comparisons (by district, decade, size)."),
        ("Associations, not causes", "These are patterns in real sales, not experiments: an old block is cheaper also for "
         "its location, its flats' condition and its lack of lifts."),
        ("What's missing", "Cooperative flats (spółdzielcze prawo) never enter the register; nothing before September 2016; "
         "deeds reach the register months late, so mid-2026 on is incomplete. Developer deeds carry prices agreed a year "
         "or two before completion. The register does not record condition or renovation."),
    ]
    y = 0.90
    for head, body in blocks:
        fig.text(0.06, y, head, fontsize=12, fontweight="bold", color=style.INK, va="top")
        lines = wrap(body, 96)
        fig.text(0.06, y - 0.027, lines, fontsize=10, color=style.INK_2, va="top", linespacing=1.4)
        y -= 0.048 + 0.0200 * (lines.count(chr(10)) + 1)
    style.footer(fig, "Analysis and charts: ceny_mieszkan project. " + SOURCE, y=0.012)
    save(fig, "r99_methods.png")


def assemble():
    folders = {"summary": "output/summary", "src": SRC}
    for f in os.listdir(OUT):
        if f[:2].isdigit() and f[2] == "_" and f.endswith(".png"):
            os.remove(os.path.join(OUT, f))
    pdf_path = f"{OUT}/lublin_flats.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, (d, stem) in enumerate(SLIDES, 1):
            src = f"{folders[d]}/{stem}.png"
            shutil.copyfile(src, f"{OUT}/{i:02d}_{stem}.png")
            fig = plt.figure(figsize=(8, 10), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(Image.open(src).convert("RGB"), interpolation="antialiased")
            ax.axis("off")
            pdf.savefig(fig, dpi=300 if stem == "02_price_points" else 150)
            plt.close(fig)
    print(f"{len(SLIDES)} slides -> {OUT}, {pdf_path}")


def main():
    t = sales()
    data_overview(t)
    coverage(t)
    prices_over_time()
    districts()
    sizes(t)
    new_vs_resale(t)
    age(t)
    flips(t)
    methods()
    assemble()


if __name__ == "__main__":
    main()
