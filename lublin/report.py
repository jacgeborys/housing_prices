"""
Lublin report: plain descriptive slides in the Warsaw deck's style (style.py, summary_charts.page), no reputation model,
in English or Polish (common.T). Run from the lublin/ folder after build_prices.py, fetch_building_age.py, houses.py,
fetch_context.py, parse_offers.py:

    python report.py [--lang pl] [--all]      --all also reruns maps.py, houses.py, context.py, offers.py in that language

    output/price_model.gpkg (transactions_adj, building_value), output/analysis/*.csv, output/building_age.gpkg,
    data/rcn_lokale_lublin.gpkg (flat ids for flips), data/buildings.gpkg (coverage)
 -> output/report/src[_pl]/r01 ... r99 (.png), output/report[/pl]/NN_*.png, lublin_flats.pdf / lublin_mieszkania.pdf

Deck order (as Warsaw's): 01/02 maps; r01 data, r02 coverage; r05 sizes, r06 new vs resale; r03 prices, r04 districts;
r09/r10 houses; r07 age, r11 cafés, r12 noise, r13 trees, r14 transit; r08 flippers, r15 offers; r99 methods.
"""
import os
import shutil
import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image  # noqa: E402

from common import ROOT, LANG, T, fmt, headline, page, pct, save, set_lang, source, src_dir, style, wrap  # noqa: E402

NEW, RESALE = "#c4521f", style.BLUE
_RAMP = LinearSegmentedColormap.from_list("size", ["#f4eedd", "#f2c97d", "#e8914d", "#c9533f", "#8e2f55", "#4b1d5e"])
SIZE_COL = [_RAMP(x) for x in (0.17, 0.32, 0.48, 0.64, 0.82, 1.0)]
SLIDES = ["01_price_ridgelines", "02_price_points", "r01_data", "r02_coverage", "r05_sizes", "r06_new_resale",
          "r03_prices", "r04_districts", "r09_houses", "r10_house_or_flat", "r07_age", "r11_cafes", "r12_noise",
          "r13_trees", "r14_transit", "r08_flippers", "r15_offers", "r99_methods"]


def sizes_def():
    return [(0, 25, T("micro, under 25 m²", "mikro, poniżej 25 m²")), (25, 35, T("small, 25–35", "małe, 25–35")),
            (35, 50, T("compact, 35–50", "kompaktowe, 35–50")), (50, 70, T("medium, 50–70", "średnie, 50–70")),
            (70, 100, T("spacious, 70–100", "przestronne, 70–100")), (100, 1000, T("large, 100+", "duże, 100+"))]


def k(v):
    return T(f"{v / 1000:.1f}K", f"{v / 1000:.1f} tys.".replace(".", ","))


def sales():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t["date"] = pd.to_datetime(t["date"])
    s = sizes_def()
    t["size_cat"] = pd.cut(t["area"], [x[0] for x in s] + [1000], right=False, labels=[x[2] for x in s])
    return t


def market_labels():
    return T("resale", "rynek wtórny"), T("new (developer)", "nowe (deweloper)")


# r01 ------------------------------------------------------------------------------------------
def data_overview(t):
    q = t.groupby([t["date"].dt.to_period("Q"), "market"]).size().unstack(fill_value=0)
    per_dist = t.groupby(["district", "market"]).size().unstack(fill_value=0)
    per_dist = per_dist.loc[per_dist.sum(axis=1).sort_values().index]
    small = per_dist[per_dist.sum(axis=1) < 200]
    per_dist = per_dist[per_dist.sum(axis=1) >= 200]
    full = t[t["year"].between(2017, 2025)]
    new_share = 100 * (full["market"] == "primary").groupby(full["year"]).mean()
    res, new = market_labels()
    pl_months = ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"]
    last = t["date"].max()
    fig = page(T("What's in the data", "Co jest w danych"),
               T(f"Every flat sale in Lublin's official price register, Sept 2016 – {last:%b %Y}",
                 f"Każda sprzedaż mieszkania w Lublinie w rejestrze cen, wrzesień 2016 – {pl_months[last.month - 1]} {last.year}"),
               T("Deeds of flats in full ownership, inside the city. New = sold by the developer (price agreed a year or two before "
                 f"the deed). Mid-2026 on is incomplete (deeds arrive months late). {len(small)} districts with < 200 sales not listed. ",
                 "Akty notarialne mieszkań na własność, w granicach miasta. Nowe = od dewelopera (cena ustalona rok-dwa przed aktem). "
                 f"Od połowy 2026 dane niepełne (akty spływają z opóźnieniem). {len(small)} dzielnic z < 200 sprzedażami pominięto. ")
               + source())
    top = per_dist.sum(axis=1)
    headline(fig, T(f"{fmt(len(t))} flat sales: {fmt((t['market'] == 'secondary').sum())} resales and "
                    f"{fmt((t['market'] == 'primary').sum())} from developers ({new_share.min():.0f}–{new_share.max():.0f} % a year). "
                    f"Most trade in {top.index[-1]} and {top.index[-2]}; 2022 dipped by a quarter as mortgage rates rose.",
                    f"{fmt(len(t))} sprzedaży mieszkań: {fmt((t['market'] == 'secondary').sum())} z rynku wtórnego i "
                    f"{fmt((t['market'] == 'primary').sum())} od deweloperów ({new_share.min():.0f}–{new_share.max():.0f} % rocznie). "
                    f"Najwięcej na {top.index[-1].replace('Węglin Południowy', 'Węglinie Południowym')} i "
                    f"{top.index[-2].replace('Wrotków', 'Wrotkowie')}; w 2022 obroty spadły o jedną czwartą, gdy wzrosły stopy kredytów."))
    a1 = fig.add_axes([0.10, 0.55, 0.86, 0.25])
    x = np.arange(len(q))
    a1.bar(x, q.get("secondary", 0), color=RESALE, width=0.8, label=res)
    a1.bar(x, q.get("primary", 0), bottom=q.get("secondary", 0), color=NEW, width=0.8, label=new)
    yrs = [i for i, p in enumerate(q.index) if p.quarter == 1]
    a1.set_xticks(yrs)
    a1.set_xticklabels([q.index[i].year for i in yrs])
    a1.set_ylabel(T("sales per quarter", "sprzedaży na kwartał"))
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False, ncol=2)
    a1.set_title(T("Flat sales per quarter", "Sprzedaż mieszkań na kwartał"), fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.24, 0.10, 0.70, 0.38])
    y = np.arange(len(per_dist))
    a2.barh(y, per_dist.get("secondary", 0) / 1000, color=RESALE, height=0.7)
    a2.barh(y, per_dist.get("primary", 0) / 1000, left=per_dist.get("secondary", 0) / 1000, color=NEW, height=0.7)
    a2.set_yticks(y)
    a2.set_yticklabels(per_dist.index, fontsize=8.5)
    a2.set_xlabel(T("thousand sales, 2016–2026", "tys. sprzedaży, 2016–2026"))
    style.vgrid_only(a2)
    a2.set_title(T("Where they are", "Gdzie"), fontsize=10.5, loc="left")
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
    b["seen"] = b["osm_id"].isin(t["osm_id"].dropna().astype(int))
    by = b.groupby("district").apply(lambda g: pd.Series({
        "share": 100 * g.loc[g["seen"], "floor_m2"].sum() / g["floor_m2"].sum(), "floor": g["floor_m2"].sum()}))
    by = by[by["floor"] > 50_000].sort_values("share")
    city = 100 * b.loc[b["seen"], "floor_m2"].sum() / b["floor_m2"].sum()
    raw = gpd.read_file("data/rcn_lokale_lublin.gpkg", ignore_geometry=True,
                        columns=["LOK_FUNKCJA", "NIER_UDZIAL", "LOK_CENA_BRUTTO", "NIER_CENA_BRUTTO"])
    flats = raw[raw["LOK_FUNKCJA"] == "mieszkalna"]
    lok_empty = 100 * pd.to_numeric(flats["LOK_CENA_BRUTTO"], errors="coerce").isna().mean()
    share = flats["NIER_UDZIAL"].fillna("").astype(str).str.extract(r"^(\d+)/(\d+)$").astype(float)
    steps = [(T("Records in the register box", "Rekordy w obszarze pobrania"), len(raw)),
             (T("of which flats (not garages, shops)", "w tym mieszkania (bez garaży, lokali usługowych)"), len(flats)),
             (T("whole flats (not a share of one)", "całe mieszkania (nie udziały)"),
              int(len(flats) - (share[0].notna() & (share[0] != share[1])).sum())),
             (T("inside the city, cleaned", "w granicach miasta, po czyszczeniu"), len(t)),
             (T("matched to a building", "przypisane do budynku"), int(t["osm_id"].notna().sum()))]
    fig = page(T("What the register can't see", "Czego rejestr nie widzi"),
               T("From raw records to usable sales, and how much of the city's flat stock ever shows up",
                 "Od surowych rekordów do sprzedaży i jaka część zasobu mieszkań w ogóle się pojawia"),
               T("Floor area = BDOT10k outline x storeys, blocks of flats only ('budynek wielorodzinny'); seen = at least one recorded "
                 "flat sale 2016-26. Districts with < 50,000 m² of blocks left off. ",
                 "Powierzchnia = obrys BDOT10k x kondygnacje, tylko budynki wielorodzinne; widziany = co najmniej jedna sprzedaż "
                 "mieszkania 2016-26. Pominięto dzielnice z < 50 000 m² bloków. ") + source())
    headline(fig, T(f"The register sees {city:.0f} % of the floor area of Lublin's blocks of flats; the rest is cooperative flats "
                    "(never registered) and blocks where nothing sold since 2016. Lublin also leaves the flat price field empty "
                    f"({lok_empty:.1f} % of flats): the price is taken from the property field instead.",
                    f"Rejestr widzi {city:.0f} % powierzchni lubelskich bloków; reszta to mieszkania spółdzielcze (nie trafiają do "
                    "rejestru) i bloki, w których od 2016 nic się nie sprzedało. Lublin zostawia też puste pole ceny lokalu "
                    f"({lok_empty:.1f} % mieszkań): cenę bierzemy z pola ceny nieruchomości.".replace(".", ",", 1)))
    a1 = fig.add_axes([0.08, 0.58, 0.88, 0.22])
    vals = [s[1] for s in steps]
    y = np.arange(len(steps))[::-1]
    a1.barh(y, vals, color=[style.CONTEXT] * 3 + [RESALE] * 2, height=0.62)
    for yi, (lab, v) in zip(y, steps):
        a1.text(1000, yi, lab, va="center", fontsize=9.5, color=style.INK)
        a1.text(v + 800, yi, fmt(v), va="center", fontsize=9.5, color=style.INK_2)
    a1.set_yticks([])
    a1.set_xlim(0, max(vals) * 1.15)
    a1.set_xticks([])
    for sp in ("left", "bottom", "top", "right"):
        a1.spines[sp].set_visible(False)
    a1.set_title(T("From records to sales", "Od rekordów do sprzedaży"), fontsize=10.5, loc="left")
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
    a2.set_title(T("Share of blocks' floor area with a recorded sale (grey: never seen)",
                   "Udział powierzchni bloków z zarejestrowaną sprzedażą (szare: nigdy)"), fontsize=10.5, loc="left")
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
    l_n = 100 * np.exp(q["gamma"] - q.loc[q["quarter"] == "2016Q4", "gamma"].iloc[0])
    w_n = 100 * np.exp(w["gamma"] - w.loc[w["quarter"] == "2016Q4", "gamma"].iloc[0])     # Warsaw: no sales 2017-20
    first = lambda s, col: s.loc[s["year"] == 2017, col].mean()      # noqa: E731
    n17, n25 = first(q, "nominal"), q.loc[q["year"] == 2025, "nominal"].mean()
    r17, r25 = first(q, "real"), q.loc[q["year"] == 2025, "real"].mean()
    w25 = w_n[w["year"] == 2025].mean()
    dec = lambda v, p: T(f"{v:.{p}f}", f"{v:.{p}f}".replace(".", ","))  # noqa: E731
    fig = page(T(f"Lublin flats cost {n25 / n17:.1f}× what they did in 2017,\n{r25 / r17:.2f}× after inflation",
                 f"Mieszkania w Lublinie kosztują {dec(n25 / n17, 1)}× tyle co w 2017,\npo inflacji {dec(r25 / r17, 2)}×"),
               "", T("Same kind of flat each quarter (hedonic model, within-building comparisons). Real = GUS CPI, 2025 PLN. "
                     "Warsaw: same model, its register has almost no sales 2017-20. Source: GUGiK RCN.",
                     "To samo mieszkanie w każdym kwartale (model hedoniczny, porównania w obrębie budynku). Realnie = CPI GUS, zł 2025. "
                     "Warszawa: ten sam model, jej rejestr prawie nie ma sprzedaży 2017-20. Źródło: GUGiK RCN."))
    for x, big, small in ((0.05, f"×{dec(n25 / n17, 1)}", T("price as paid\n2017 → 2025", "cena zapłacona\n2017 → 2025")),
                          (0.36, f"×{dec(r25 / r17, 2)}", T("after inflation\n(today's money), 2017 → 2025",
                                                            "po inflacji\n(dzisiejsze złote), 2017 → 2025")),
                          (0.70, fmt(n25), T("PLN per m², typical\nflat, 2025", "zł za m², typowe\nmieszkanie, 2025"))):
        fig.text(x, 0.885, big, fontsize=30, fontweight="bold", color=style.INK, va="top")
        fig.text(x, 0.835, small, fontsize=10.5, color=style.INK_2, va="top")
    ax = fig.add_axes([0.13, 0.43, 0.80, 0.33])
    for col, colr, lab, lw in (("nominal", style.BLUE, T("Price as paid", "Cena zapłacona"), 1.8),
                               ("real", style.ORANGE, T("In 2025 money", "W złotych z 2025"), 3.0)):
        ax.plot(q["t"], q[col], color=colr, lw=lw)
        last = q.iloc[-1]
        ax.annotate(f"{lab}: {fmt(last[col])}", (last["t"], last[col]), xytext=(-4, 10 if col == "nominal" else -16),
                    textcoords="offset points", ha="right", fontsize=9.5, color=colr, fontweight="semibold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
    style.hgrid_only(ax)
    ax.set_ylim(0, None)
    ax.set_xlim(2016.6, 2026.8)
    ax.set_ylabel(T("PLN per m², typical flat", "zł za m², typowe mieszkanie"))
    ax.set_title(T("Typical Lublin flat, per m²", "Typowe mieszkanie w Lublinie, za m²"), fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.13, 0.10, 0.80, 0.25])
    a2.plot(q["t"], l_n, color=style.ORANGE, lw=2.6, label="Lublin")
    a2.plot(w["t"], w_n, color=style.INK_2, lw=1.6, ls=(0, (3, 2)), label=T("Warsaw", "Warszawa"))
    a2.axhline(100, color=style.AXIS, lw=1)
    a2.legend(loc="upper left", fontsize=9, frameon=False)
    style.hgrid_only(a2)
    a2.set_xlim(2016.6, 2026.8)
    a2.set_title(T(f"Same flat, end of 2016 = 100, as paid: Lublin {l_n[q['year'] == 2025].mean():.0f}, Warsaw {w25:.0f} in 2025",
                   f"To samo mieszkanie, koniec 2016 = 100, ceny zapłacone: Lublin {l_n[q['year'] == 2025].mean():.0f}, "
                   f"Warszawa {w25:.0f} w 2025"), fontsize=10.5, loc="left")
    save(fig, "r03_prices.png")


# r04 ------------------------------------------------------------------------------------------
def districts():
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    val = bv.groupby("district").agg(value=("value_2025", "median"), n=("n", "sum"), b=("osm_id", "size"))
    val = val[val["b"] >= 15]
    di = pd.read_csv("output/analysis/district_index_annual.csv")
    base = di[di["year"].between(2017, 2018)].groupby("district").apply(
        lambda g: np.average(g["index_2025"], weights=g["n"]))
    val["growth"] = (100 / base - 1) * 100
    val = val.sort_values("value")
    city = bv["value_2025"].median()                      # same measure as the bars (median over buildings)
    g = val.dropna(subset=["growth"])
    fig = page(T("Where flats cost most, and where they rose most", "Gdzie mieszkania są najdroższe, a gdzie najbardziej podrożały"),
               T("Typical flat per district in 2025, and its price change since 2017-18 (same flats, as paid)",
                 "Typowe mieszkanie w dzielnicy w 2025 i zmiana jego ceny od 2017-18 (te same mieszkania, ceny zapłacone)"),
               T("Typical flat = ~50 m², 2 rooms, floor 2-3, resale, priced in 2025 (median over buildings, >= 15 buildings). "
                 "Change: district index from comparisons within buildings, 2017-18 to 2025. ",
                 "Typowe mieszkanie = ok. 50 m², 2 pokoje, piętro 1-2, rynek wtórny, ceny 2025 (mediana budynków, >= 15 budynków). "
                 "Zmiana: indeks dzielnicy z porównań w obrębie budynków, 2017-18 do 2025. ") + source())
    headline(fig, T(f"A typical flat costs {fmt(val['value'].iloc[-1])} PLN/m² in {val.index[-1]} and {fmt(val['value'].iloc[0])} "
                    f"in {val.index[0]} (median building city-wide {fmt(city)}). Since 2017-18 prices rose most in "
                    f"{g['growth'].idxmax()} (+{g['growth'].max():.0f} %) and least in {g['growth'].idxmin()} "
                    f"(+{g['growth'].min():.0f} %); most districts roughly doubled.",
                    f"Typowe mieszkanie kosztuje {fmt(val['value'].iloc[-1])} zł/m² ({val.index[-1]}) i {fmt(val['value'].iloc[0])} "
                    f"zł/m² ({val.index[0]}); mediana budynków w mieście {fmt(city)}. Od 2017-18 najbardziej podrożał "
                    f"{g['growth'].idxmax()} (+{g['growth'].max():.0f} %), najmniej {g['growth'].idxmin()} "
                    f"(+{g['growth'].min():.0f} %); większość dzielnic mniej więcej podwoiła ceny."))
    a1 = fig.add_axes([0.24, 0.08, 0.34, 0.70])
    y = np.arange(len(val))
    a1.barh(y, val["value"], color=style.PRICE_CMAP((val["value"] - 6500) / 6000), height=0.7)
    a1.axvline(city, color=style.INK_2, lw=1, ls=(0, (3, 2)))
    for yi, v in enumerate(val["value"]):
        a1.text(v + 100, yi, k(v), va="center", fontsize=8, color=style.INK_2)
    a1.set_yticks(y)
    a1.set_yticklabels(val.index, fontsize=8.5)
    a1.set_xlim(0, val["value"].max() * 1.22)
    a1.xaxis.set_major_locator(plt.MultipleLocator(4000))
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: T(f"{v / 1000:.0f}K", f"{v / 1000:.0f} tys.")))
    style.vgrid_only(a1)
    a1.set_title(T("PLN per m², 2025", "zł za m², 2025"), fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.63, 0.08, 0.33, 0.70])
    gv = val["growth"]
    a2.barh(y, gv.fillna(0), color=[style.ORANGE if np.isfinite(v) else "none" for v in gv], height=0.7)
    for yi, v in enumerate(gv):
        a2.text((v if np.isfinite(v) else 0) + 2, yi, f"+{v:.0f}%" if np.isfinite(v) else T("too few", "za mało"),
                va="center", fontsize=8, color=style.INK_2)
    a2.set_yticks([])
    a2.set_xlim(0, np.nanmax(gv) * 1.25)
    a2.xaxis.set_major_locator(plt.MultipleLocator(50))
    a2.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"+{v:.0f}%"))
    style.vgrid_only(a2)
    a2.set_title(T("Change since 2017-18", "Zmiana od 2017-18"), fontsize=10.5, loc="left")
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
    fig = page(T("What sizes sell where", "Jakie metraże gdzie się sprzedają"),
               T("Flats sold by size, by district (2021–26) and across the years",
                 "Sprzedane mieszkania wg metrażu, w dzielnicach (2021–26) i w kolejnych latach"),
               T("RCN flat sales in full ownership, new and resale together. Size = usable floor area from the deed. Districts with "
                 "fewer than 300 sales 2021-26 and the partial 2016 left off. ",
                 "Sprzedaże mieszkań na własność z RCN, nowe i wtórne razem. Metraż = powierzchnia użytkowa z aktu. Pominięto dzielnice "
                 "z < 300 sprzedażami 2021-26 i niepełny 2016. ") + source())
    headline(fig, T(f"Small flats (under 35 m²) are {small.max():.0f} % of sales in {small.idxmax()} but {small.min():.0f} % in "
                    f"{small.idxmin()}. New flats got smaller: a median {med.loc[2017, 'primary']:.0f} m² in 2017, "
                    f"{med.loc[2025, 'primary']:.0f} m² in 2025 (resales {med.loc[2025, 'secondary']:.0f} m²).",
                    f"Małe mieszkania (poniżej 35 m²) to {small.max():.0f} % sprzedaży w dzielnicy {small.idxmax()}, ale "
                    f"{small.min():.0f} % w dzielnicy {small.idxmin()}. Nowe mieszkania się zmniejszyły: mediana "
                    f"{med.loc[2017, 'primary']:.0f} m² w 2017, {med.loc[2025, 'primary']:.0f} m² w 2025 (wtórne "
                    f"{med.loc[2025, 'secondary']:.0f} m²)."))
    a1 = fig.add_axes([0.22, 0.41, 0.74, 0.34])
    left = np.zeros(len(share))
    y = np.arange(len(share))
    for kk, col in enumerate(share.columns):
        a1.barh(y, share[col], left=left, color=SIZE_COL[kk], height=0.75, label=col, edgecolor=style.SURFACE, lw=0.4)
        left += share[col].to_numpy()
    a1.set_yticks(y)
    a1.set_yticklabels(share.index, fontsize=8.5)
    a1.set_xlim(0, 100)
    a1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    a1.legend(loc="upper left", bbox_to_anchor=(0, 1.11), ncol=3, fontsize=8, frameon=False, handlelength=1)
    for sp in ("top", "right"):
        a1.spines[sp].set_visible(False)
    a1.set_title(T("Share of flats sold, 2021–26", "Udział w sprzedaży, 2021–26"), fontsize=10.5, loc="left", pad=36)
    a2 = fig.add_axes([0.10, 0.08, 0.80, 0.25])
    ends = {}
    for kk, col in enumerate(yr.columns):
        s = yr[col]
        a2.plot(s.index, s.values, "-o", color=SIZE_COL[kk], lw=2, ms=3.5, mec=style.SURFACE)
        ends[kk] = s.iloc[-1]
    placed = []
    for kk in sorted(ends, key=ends.get):
        y_ = max(ends[kk], placed[-1] + 2.4) if placed else ends[kk]
        placed.append(y_)
        a2.text(yr.index.max() + 0.3, y_, yr.columns[kk].split(",")[0], va="center", fontsize=8, color=SIZE_COL[kk],
                fontweight="semibold")
    a2.set_xlim(yr.index.min() - 0.5, yr.index.max() + 1.8)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style.hgrid_only(a2)
    a2.set_title(T("Share of flats sold, by year", "Udział w sprzedaży, wg lat"), fontsize=10.5, loc="left")
    save(fig, "r05_sizes.png")


# r06 ------------------------------------------------------------------------------------------
def new_vs_resale(t):
    t = t[t["year"].between(2017, 2026)]
    med = t.groupby(["year", "market"])["price_m2"].median().unstack()
    recent = t[t["year"] >= 2021]
    from scipy.ndimage import gaussian_filter1d
    res, new = market_labels()
    fig = page(T("New or resale?", "Nowe czy z drugiej ręki?"),
               T("Price per m² as paid, and the size of flats each market sells",
                 "Cena za m² zapłacona i metraże sprzedawane na obu rynkach"),
               T("Prices as paid (not inflation-adjusted), medians. New = sold by the developer: the deed is signed at completion, so its "
                 "price was agreed a year or two earlier. 2016 (from September) left off. ",
                 "Ceny zapłacone (bez korekty o inflację), mediany. Nowe = od dewelopera: akt podpisuje się po odbiorze, więc cenę ustalono "
                 "rok-dwa wcześniej. Pominięto 2016 (od września). ") + source())
    dearer = [yy for yy in med.index if med.loc[yy, "secondary"] > med.loc[yy, "primary"]]
    headline(fig, T(f"Resales cost more per m² than new flats in {min(dearer)}–{max(dearer)} (2021: {fmt(med.loc[2021, 'secondary'])} vs "
                    f"{fmt(med.loc[2021, 'primary'])} PLN): developer deeds carry prices agreed before completion. In 2025 new flats "
                    f"overtook ({fmt(med.loc[2025, 'primary'])} vs {fmt(med.loc[2025, 'secondary'])}). Both sell the same sizes.",
                    f"Mieszkania z drugiej ręki były droższe za m² od nowych w latach {min(dearer)}–{max(dearer)} (2021: "
                    f"{fmt(med.loc[2021, 'secondary'])} wobec {fmt(med.loc[2021, 'primary'])} zł): akty deweloperskie niosą ceny "
                    f"sprzed odbioru. W 2025 nowe wyprzedziły ({fmt(med.loc[2025, 'primary'])} wobec {fmt(med.loc[2025, 'secondary'])}). "
                    "Oba rynki sprzedają te same metraże."))
    a1 = fig.add_axes([0.11, 0.50, 0.85, 0.29])
    for mk, lab, colr in (("primary", new, NEW), ("secondary", res, RESALE)):
        a1.plot(med.index, med[mk], "-o", color=colr, lw=2.2, ms=4, mec=style.SURFACE, label=lab)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: T(f"{x / 1000:.0f}K", f"{x / 1000:.0f} tys.")))
    a1.set_ylabel(T("PLN per m²", "zł za m²"))
    style.hgrid_only(a1)
    a1.legend(loc="upper left", fontsize=9, frameon=False)
    a1.set_title(T("Median price per m², as paid", "Mediana ceny za m², zapłacona"), fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.11, 0.12, 0.85, 0.28])
    bins = np.arange(15, 125, 5)
    mid = bins[:-1] + 2.5
    for mk, lab, colr in (("primary", new, NEW), ("secondary", res, RESALE)):
        a = recent.loc[recent["market"] == mk, "area"]
        h, _ = np.histogram(a, bins=bins)
        sm = gaussian_filter1d(100 * h / len(a), 1.0)
        a2.fill_between(mid, sm, color=colr, alpha=0.18, lw=0)
        a2.plot(mid, sm, color=colr, lw=2.4, label=f"{lab}, {T('median', 'mediana')} {a.median():.0f} m²")
    a2.set_ylim(0, None)
    a2.set_xlabel(T("flat size, m² (smoothed; flats over 120 m² not shown)", "metraż, m² (wygładzone; bez mieszkań > 120 m²)"))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    style.hgrid_only(a2)
    a2.legend(loc="upper right", fontsize=9, frameon=False)
    a2.set_title(T("Sizes sold, 2021–26 (share of each market's sales)", "Sprzedane metraże, 2021–26 (udział w każdym rynku)"),
                 fontsize=10.5, loc="left")
    save(fig, "r06_new_resale.png")


# r07 ------------------------------------------------------------------------------------------
def age():
    a = gpd.read_file("output/building_age.gpkg", ignore_geometry=True)[["osm_id", "decade_end"]]
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True).merge(a, on="osm_id", how="left")
    groups = [(1900, 1940, T("pre-war", "przedwojenne")), (1941, 1960, "1941–60"), (1961, 1970, "1961–70"),
              (1971, 1980, "1971–80"), (1981, 1990, "1981–90"), (1991, 2000, "1991–2000"), (2001, 2010, "2001–10"),
              (2011, 2020, "2011–20")]

    def grp(dec):
        for lo, hi, lab in groups:
            if pd.notna(dec) and dec > 0 and lo <= max(dec, 1900) <= hi:
                return lab
        return None
    bv["era"] = bv["decade_end"].map(grp)
    dated = bv["era"].notna()
    order = [g[2] for g in groups]
    s = bv[dated].groupby("era").agg(value=("value_2025", "median"), q1=("value_2025", lambda x: x.quantile(.25)),
                                     q3=("value_2025", lambda x: x.quantile(.75)), b=("osm_id", "size")).reindex(order)
    bv["rel"] = 100 * (bv["value_2025"] / bv.groupby("district")["value_2025"].transform("median") - 1)
    rel = bv[dated].groupby("era")["rel"].median().reindex(order)
    old = s.loc[order[:6], "value"]
    fig = page(T("Only the newest buildings cost more", "Drożej tylko w najnowszych budynkach"),
               T("Typical flat by when the building went up", "Typowe mieszkanie wg czasu budowy budynku"),
               T(f"Typical flat = ~50 m², 2 rooms, floor 2-3, resale, 2025, per building (median, bars: middle half). Decade: GIS Expert, "
                 f"'Rozwój zabudowy Lublina' (to 2020); {fmt(dated.sum())} of {fmt(len(bv))} priced buildings dated. ",
                 f"Typowe mieszkanie = ok. 50 m², 2 pokoje, piętro 1-2, rynek wtórny, 2025, na budynek (mediana, kreski: środkowa połowa). "
                 f"Dekada: GIS Expert, 'Rozwój zabudowy Lublina' (do 2020); datowane {fmt(dated.sum())} z {fmt(len(bv))} budynków. ")
               + source(", GIS Expert"))
    headline(fig, T(f"A typical flat in a building from before 2000 costs about the same whatever the decade "
                    f"({old.min() / 1000:.1f}–{old.max() / 1000:.1f}K PLN/m²); 2001-10 buildings {fmt(s.loc['2001–10', 'value'])}, "
                    f"2011-20 {fmt(s.loc['2011–20', 'value'])}. Against the same district: {pct(rel.loc['2001–10'])} and "
                    f"{pct(rel.loc['2011–20'])}, older ones {pct(rel[order[:6]].min())} to {pct(rel[order[:6]].max())}.",
                    f"Typowe mieszkanie w budynku sprzed 2000 kosztuje mniej więcej tyle samo bez względu na dekadę "
                    f"({k(old.min())}–{k(old.max())} zł/m²); budynki z lat 2001-10 {fmt(s.loc['2001–10', 'value'])}, 2011-20 "
                    f"{fmt(s.loc['2011–20', 'value'])}. Wobec tej samej dzielnicy: {pct(rel.loc['2001–10'])} i "
                    f"{pct(rel.loc['2011–20'])}, starsze od {pct(rel[order[:6]].min())} do {pct(rel[order[:6]].max())}."))
    a1 = fig.add_axes([0.10, 0.45, 0.86, 0.33])
    x = np.arange(len(order))
    a1.bar(x, s["value"], color=style.PRICE_CMAP((s["value"] - 6500) / 6000), width=0.68)
    a1.vlines(x, s["q1"], s["q3"], color=style.INK_2, lw=1.2)
    for xi, v, b in zip(x, s["value"], s["b"]):
        a1.text(xi, 300, f"{int(b)}\n{T('bldgs', 'bud.')}", ha="center", fontsize=7.5, color="white")
        a1.text(xi + 0.08, v + 150, k(v), fontsize=8.5, color=style.INK)
    a1.set_xticks(x)
    a1.set_xticklabels(order, fontsize=9)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: T(f"{v / 1000:.0f}K", f"{v / 1000:.0f} tys.")))
    style.hgrid_only(a1)
    a1.set_title(T("PLN per m², typical flat, 2025", "zł za m², typowe mieszkanie, 2025"), fontsize=10.5, loc="left")
    a2 = fig.add_axes([0.10, 0.09, 0.86, 0.26])
    a2.bar(x, rel, color=[style.ORANGE if v > 0 else style.BLUE for v in rel], width=0.68)
    a2.axhline(0, color=style.AXIS, lw=1)
    for xi, v in zip(x, rel):
        a2.text(xi, v + (0.6 if v >= 0 else -0.6), pct(v), ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
    a2.set_xticks(x)
    a2.set_xticklabels(order, fontsize=9)
    a2.yaxis.set_major_locator(plt.MultipleLocator(2))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    style.hgrid_only(a2)
    a2.set_ylim(min(rel.min() * 1.4, -5), max(rel.max() * 1.4, 5))
    a2.set_title(T("Against the median building of the same district", "Wobec mediany budynków tej samej dzielnicy"),
                 fontsize=10.5, loc="left")
    save(fig, "r07_age.png")


# r08 ------------------------------------------------------------------------------------------
def flips(t):
    """Flip = the same flat (land-register id, filled for every Lublin record) bought resale, sold again 14-365 days later."""
    r = gpd.read_file("data/rcn_lokale_lublin.gpkg", ignore_geometry=True,
                      columns=["LOK_ID_LOKALU", "LOK_ADRES", "DOK_DATA", "NIER_CENA_BRUTTO", "LOK_POW_UZYT", "LOK_FUNKCJA"])
    r = r[(r["LOK_FUNKCJA"] == "mieszkalna") & r["LOK_ID_LOKALU"].notna()]
    r = pd.DataFrame({"date": pd.to_datetime(r["DOK_DATA"].str[:10], errors="coerce"), "lok_adres": r["LOK_ADRES"],
                      "price": pd.to_numeric(r["NIER_CENA_BRUTTO"], errors="coerce").round(0),
                      "area": pd.to_numeric(r["LOK_POW_UZYT"], errors="coerce").round(1), "lok_id": r["LOK_ID_LOKALU"]})
    r = r.drop_duplicates(["date", "lok_adres", "price", "area"], keep=False)
    kk = t.assign(price=t["price"].round(0), area=t["area"].round(1))
    m = kk.merge(r, on=["date", "lok_adres", "price", "area"], how="left")
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
                       "resales": sec.groupby("year").size()}).fillna(0).loc[2017:2025]
    by["share"] = 100 * by["flips"] / by["resales"]
    by["share_new"] = 100 * by["new"] / by["resales"]
    rec = f[f["year2"].between(2021, 2025)]
    rec_new = new_resold[new_resold["year2"].between(2021, 2025)]
    rec_s = sec[sec["year"].between(2021, 2025)]
    dist = pd.DataFrame({"flips": rec.groupby("district").size(), "resales": rec_s.groupby("district").size()}).fillna(0)
    dist = dist[dist["resales"] >= 300]
    dist["share"] = 100 * dist["flips"] / dist["resales"]
    dist = dist.sort_values("share")
    os.makedirs("output/analysis", exist_ok=True)
    pd.concat([f, new_resold]).drop(columns="geometry").to_csv("output/analysis/flips.csv", index=False)
    med, med_net = rec["markup"].median(), rec["markup_net"].median()
    fig = page(T("The flippers", "Flipperzy"),
               T("Flats bought on the resale market and sold again within a year: how often, for how much, and where",
                 "Mieszkania kupione z drugiej ręki i sprzedane w ciągu roku: jak często, z jakim zyskiem i gdzie"),
               T("One flat = its land-register id. Flip = bought resale, sold again 14-365 days later; net = beyond the city-wide index. "
                 "Districts with >= 300 resales 2021-25. ",
                 "Jedno mieszkanie = jego identyfikator z ewidencji. Flip = kupione z rynku wtórnego, sprzedane 14-365 dni później; "
                 "netto = ponad indeks miasta. Dzielnice z >= 300 sprzedażami wtórnymi 2021-25. ") + source())
    headline(fig, T(f"2021–25: {fmt(len(rec))} flats were resold within a year of purchase ({100 * len(rec) / len(rec_s):.0f} in 100 "
                    f"resales), at a median markup of {pct(med)} as paid, {pct(med_net)} beyond the market, before renovation, tax and "
                    f"fees. Another {fmt(len(rec_new))} flats bought new were resold within a year (median {pct(rec_new['markup'].median())}), "
                    "partly because the developer's price was agreed years before.",
                    f"2021–25: {fmt(len(rec))} mieszkań odsprzedano w ciągu roku od zakupu ({100 * len(rec) / len(rec_s):.0f} na 100 "
                    f"sprzedaży wtórnych), z medianą narzutu {pct(med)} w cenach zapłaconych, {pct(med_net)} ponad rynek, przed "
                    f"remontem, podatkiem i opłatami. Kolejne {fmt(len(rec_new))} mieszkań kupionych od dewelopera odsprzedano w ciągu "
                    f"roku (mediana {pct(rec_new['markup'].median())}), częściowo dlatego, że cenę deweloperską ustalono lata wcześniej."))
    a1 = fig.add_axes([0.10, 0.50, 0.38, 0.28])
    a1.plot(by.index, by["share"], "-o", color="#c4521f", lw=2, ms=5, mec=style.SURFACE)
    a1.plot(by.index, by["share_new"], "-o", color=style.MUTED, lw=1.5, ms=4, mec=style.SURFACE)
    a1.text(by.index[-1] + 0.2, by["share"].iloc[-1], T("bought\nresale", "kupione\nwtórne"), fontsize=8, color="#c4521f", va="center")
    a1.text(by.index[-1] + 0.2, by["share_new"].iloc[-1], T("bought\nnew", "kupione\nnowe"), fontsize=8, color=style.MUTED, va="center")
    a1.set_xlim(by.index[0] - 0.5, by.index[-1] + 1.8)
    style.hgrid_only(a1)
    a1.set_ylim(0, None)
    a1.set_title(T("Flips per 100 resales, by year", "Flipy na 100 sprzedaży wtórnych"), fontsize=10.5)
    a1.tick_params(labelsize=8.5)
    a2 = fig.add_axes([0.58, 0.50, 0.38, 0.28])
    a2.hist(rec["markup"].clip(-30, 80), bins=np.arange(-30, 81, 5), color="#f08a4b", edgecolor=style.SURFACE, lw=1)
    a2.axvline(0, color=style.AXIS, lw=1)
    a2.axvline(med, color=style.INK, lw=1.4, ls=(0, (4, 3)))
    a2.set_ylim(0, a2.get_ylim()[1] * 1.14)
    a2.text(med + 1.5, a2.get_ylim()[1] * 0.99, f"{T('median', 'mediana')} {pct(med)}", fontsize=9, color=style.INK, va="top")
    a2.set_xticks([-20, 0, 20, 40, 60, 80])
    a2.set_xticklabels(["−20%", "0%", "+20%", "+40%", "+60%", "+80%+"])
    a2.set_yticks([])
    style.hgrid_only(a2)
    a2.set_title(T("Markup on resale, as paid (2021–25)", "Narzut przy odsprzedaży (2021–25)"), fontsize=10.5)
    a2.tick_params(labelsize=8.5)
    a3 = fig.add_axes([0.26, 0.07, 0.68, 0.34])
    y = np.arange(len(dist))
    a3.barh(y, dist["share"], color="#c4521f", height=0.65)
    for yi, s in enumerate(dist["share"]):
        a3.text(s + 0.1, yi, T(f"{s:.1f}", f"{s:.1f}".replace(".", ",")), va="center", fontsize=8.5, color=style.INK_2)
    a3.set_yticks(y)
    a3.set_yticklabels(dist.index, fontsize=8.5)
    style.vgrid_only(a3)
    a3.set_title(T("Flips per 100 resales by district, 2021–25", "Flipy na 100 sprzedaży wtórnych wg dzielnic, 2021–25"),
                 fontsize=10.5, loc="left")
    save(fig, "r08_flippers.png")


# r99 ------------------------------------------------------------------------------------------
def methods():
    fig = plt.figure(figsize=(8, 10))
    style.header(fig, T("How to read these charts", "Jak czytać te wykresy"),
                 T("What the data is, what we compared, and what it can't tell", "Jakie to dane, co porównujemy i czego nie wiemy"),
                 y=0.985)
    blocks = [
        (T("The data", "Dane"),
         T("Every notarial deed of a flat sale in Lublin in the official price register (GUGiK RCN), from September 2016: about "
           "45,000 sales with price, size, floor and date. Lublin records the price in the property field, not the flat field, so "
           "that is used. Buildings from BDOT10k; building ages from GIS Expert's map of Lublin's development (decades to 2020); "
           "houses from the register's building layer; noise from the EU noise maps, trees from Copernicus, cafés from "
           "OpenStreetMap, transit from ZTM Lublin's timetable, flats on offer from developers' price lists (dane.gov.pl).",
           "Każdy akt notarialny sprzedaży mieszkania w Lublinie w Rejestrze Cen Nieruchomości (GUGiK RCN), od września 2016: "
           "ok. 45 000 sprzedaży z ceną, metrażem, piętrem i datą. Lublin wpisuje cenę w pole ceny nieruchomości, nie lokalu, więc "
           "tę bierzemy. Budynki z BDOT10k; wiek budynków z mapy GIS Expert 'Rozwój zabudowy Lublina' (dekady do 2020); domy "
           "z warstwy budynków rejestru; hałas z map akustycznych UE, drzewa z Copernicus, kawiarnie z OpenStreetMap, komunikacja "
           "z rozkładu ZTM Lublin, oferty z cenników deweloperów (dane.gov.pl).")),
        (T("The yardstick flat", "Mieszkanie-miarka"),
         T("Flats differ, so prices are compared on one yardstick: a ~50 m² flat on the 2nd floor, resold, priced in 2025. For each "
           "building we ask what that flat would sell for there, from every sale since 2016 and what flats in the same building tell "
           "us about size, floor and the year of sale. Same model as for Warsaw.",
           "Mieszkania się różnią, więc ceny porównujemy na jednej miarce: ok. 50 m², 1. piętro, z drugiej ręki, w cenach 2025. Dla "
           "każdego budynku pytamy, za ile sprzedałoby się tam takie mieszkanie, na podstawie wszystkich sprzedaży od 2016 i tego, "
           "co mieszkania w tym samym budynku mówią o metrażu, piętrze i roku sprzedaży. Ten sam model co dla Warszawy.")),
        (T("Real prices", "Ceny realne"),
         T("Prices as paid are deflated with GUS's annual consumer price index to 2025 złoty.",
           "Ceny zapłacone są przeliczone rocznym wskaźnikiem cen konsumpcyjnych GUS na złote z 2025.")),
        (T("\"Against the district\"", "\"Wobec dzielnicy\""),
         T("Unlike the Warsaw report, there is no model of what location and buildings predict. Where we compare cafés, noise, "
           "trees or transit, each building's typical flat is set against the median building of its own district, so that "
           "'Sławin is expensive' does not pass for 'trees are expensive'. Nothing else is held equal.",
           "W odróżnieniu od raportu warszawskiego nie ma tu modelu tego, co przewidują lokalizacja i budynki. Porównując kawiarnie, "
           "hałas, drzewa czy komunikację, zestawiamy typowe mieszkanie w budynku z medianą budynków jego dzielnicy, żeby "
           "'Sławin jest drogi' nie udawał 'drzewa są drogie'. Niczego innego nie wyrównujemy.")),
        (T("Associations, not causes", "Związki, nie przyczyny"),
         T("These are patterns in real sales, not experiments: an old block is cheaper also for its location, its flats' condition "
           "and its lack of lifts; cafés follow money as much as they attract it.",
           "To wzorce w prawdziwych sprzedażach, nie eksperymenty: stary blok jest tańszy także przez położenie, stan mieszkań i brak "
           "windy; kawiarnie idą za pieniędzmi tak samo, jak je przyciągają.")),
        (T("What's missing", "Czego brakuje"),
         T("Cooperative flats (spółdzielcze prawo) never enter the register; nothing before September 2016; deeds reach the register "
           "months late, so mid-2026 on is incomplete. Developer deeds carry prices agreed a year or two before completion. The "
           "register does not record condition or renovation. Asking prices are not sale prices.",
           "Mieszkania spółdzielcze nie trafiają do rejestru; nic sprzed września 2016; akty spływają z kilkumiesięcznym "
           "opóźnieniem, więc od połowy 2026 dane są niepełne. Akty deweloperskie niosą ceny ustalone rok-dwa przed odbiorem. "
           "Rejestr nie zna stanu ani remontu mieszkania. Ceny ofertowe to nie ceny transakcyjne.")),
    ]
    y = 0.90
    for head, body in blocks:
        fig.text(0.06, y, head, fontsize=12, fontweight="bold", color=style.INK, va="top")
        lines = wrap(body, 96)
        fig.text(0.06, y - 0.027, lines, fontsize=10, color=style.INK_2, va="top", linespacing=1.4)
        y -= 0.048 + 0.0200 * (lines.count(chr(10)) + 1)
    style.footer(fig, T("Analysis and charts: ceny_mieszkan project. ", "Analiza i wykresy: projekt ceny_mieszkan. ")
                 + source(", GIS Expert, EEA, Copernicus, OSM, ZTM Lublin, dane.gov.pl"), y=0.012)
    save(fig, "r99_methods.png")


def assemble():
    pl = LANG["code"] == "pl"
    out = "output/report/pl" if pl else "output/report"
    summary = "output/summary_pl" if pl else "output/summary"
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f[:2].isdigit() and f[2] == "_" and f.endswith(".png"):
            os.remove(os.path.join(out, f))
    pdf_path = f"{out}/{'lublin_mieszkania' if pl else 'lublin_flats'}.pdf"
    have = [s for s in SLIDES if os.path.exists(f"{summary if s[:2].isdigit() else src_dir()}/{s}.png")]
    missing = sorted(set(SLIDES) - set(have))
    if missing:
        print("missing (left out):", missing)
    with PdfPages(pdf_path) as pdf:
        for i, stem in enumerate(have, 1):
            src = f"{summary if stem[:2].isdigit() else src_dir()}/{stem}.png"
            shutil.copyfile(src, f"{out}/{i:02d}_{stem}.png")
            fig = plt.figure(figsize=(8, 10), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(Image.open(src).convert("RGB"), interpolation="antialiased")
            ax.axis("off")
            pdf.savefig(fig, dpi=300 if stem == "02_price_points" else 150)
            plt.close(fig)
    print(f"{len(have)} slides -> {out}, {pdf_path}")


def main():
    lang = "pl" if "pl" in sys.argv[1:] else "en"
    set_lang(lang)
    if "--all" in sys.argv:
        import subprocess
        for script in ("maps.py", "houses.py", "context.py", "offers.py"):
            if os.path.exists(script):
                subprocess.run([sys.executable, script] + (["--lang", "pl"] if lang == "pl" else []), check=True)
    t = sales()
    data_overview(t)
    coverage(t)
    prices_over_time()
    districts()
    sizes(t)
    new_vs_resale(t)
    age()
    flips(t)
    methods()
    assemble()


if __name__ == "__main__":
    main()
