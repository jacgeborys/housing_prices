"""
Lublin houses: single-house sales from the RCN building layer (data/rcn_budynki_lublin.gpkg), Warsaw's method (houses.py,
house_cells.py) with Lublin's codes. Run from lublin/:  python houses.py [--lang pl]

House sale = Lublin (TERYT 0663), built-up land, full ownership (share n/n), residential building, one building per deed,
free market or a developer's sale (Lublin codes those 'sprzedazBezprzetargowa'), price > 100K. Date from DOK_DATA.
Area: the deed's usable area (50-400 m², ~1 deed in 4) else BDOT10k outline x Warsaw's ratio by storeys.

-> output/houses.gpkg; slides r09_houses (300 m cells + district medians), r10_house_or_flat (curves)
"""
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

from common import T, basemap, districts, fmt, headline, page, save, set_lang, source, style

FLOOR_RATIO = {1: 1.2, 2: 1.52, 3: 1.97, 4: 2.4}
USABLE = (50, 400)
M2 = (2000, 30000)
CELL, MIN_IN_CELL = 300, 3
CMAP = style.PRICE_CMAP


def clean():
    b = gpd.read_file("data/rcn_budynki_lublin.gpkg")
    b = b[(b["TERYT"].str[:4] == "0663") & (b["NIER_RODZAJ"] == "nieruchomoscGruntowaZabudowana")]
    b = b[b.groupby("TRAN_LOKALNY_ID_IIP")["gid"].transform("size") == 1].copy()
    sh = b["NIER_UDZIAL"].fillna("1/1").str.extract(r"^(\d+)/(\d+)$").astype(float)
    full = sh[0].eq(sh[1])
    market_ok = b["TRAN_RODZAJ_TRANS"].eq("wolnyRynek") | (
        b["TRAN_RODZAJ_TRANS"].eq("sprzedazBezprzetargowa") & b["TRAN_RODZAJ_RYNKU"].eq("rynekPierwotny"))
    b["price"] = pd.to_numeric(b["NIER_CENA_BRUTTO"], errors="coerce")
    h = b[(b["BUD_RODZAJ"] == "mieszkalny") & full & market_ok & (b["price"] > 100_000)].copy()
    h["date"] = pd.to_datetime(h["DOK_DATA"].str[:10], errors="coerce")
    h["year"] = h["date"].dt.year
    h["usable_m2"] = pd.to_numeric(h["BUD_POW_UZYT"], errors="coerce").where(lambda s: s.between(*USABLE))
    h["plot_m2"] = pd.to_numeric(h["NIER_POW_GRUNTU"], errors="coerce")
    h["market"] = np.where(h["TRAN_RODZAJ_RYNKU"] == "rynekPierwotny", "primary", "secondary")
    h = gpd.GeoDataFrame(h[["date", "year", "price", "usable_m2", "plot_m2", "market", "BUD_ADRES"]],
                         geometry=h.geometry.to_crs(2180).representative_point().values, crs=2180)
    d = districts()
    h = gpd.sjoin(h, d[["district", "geometry"]], predicate="within").drop(columns="index_right")
    bd = gpd.read_file("data/buildings.gpkg")
    bd["outline_m2"] = bd.area
    j = gpd.sjoin(h, bd[["storeys", "outline_m2", "geometry"]], predicate="within", how="left")
    j = j[~j.index.duplicated()].drop(columns="index_right")
    est = (j["outline_m2"] * j["storeys"].fillna(2).clip(1, 4).map(FLOOR_RATIO)).where(lambda s: s.between(*USABLE))
    j["usable_all"] = j["usable_m2"].fillna(est)
    j["area_source"] = np.where(j["usable_m2"].notna(), "deed", np.where(est.notna(), "estimated", "none"))
    j["price_m2"] = j["price"] / j["usable_all"]
    j.to_file("output/houses.gpkg", layer="houses", driver="GPKG")
    print(f"{len(j):,} clean house sales; area {j['area_source'].value_counts().to_dict()}; by year "
          f"{j['year'].value_counts().sort_index().to_dict()}")
    return j


def dec1(v):
    """One decimal with a Polish decimal comma."""
    return f"{v:.1f}".replace(".", ",")


def year_index(h):
    X = pd.concat([pd.get_dummies(h["district"]), pd.get_dummies(h["year"].astype(int), prefix="y")], axis=1)
    X = X.drop(columns="y_2025").astype(float)
    beta = np.linalg.lstsq(X.to_numpy(), np.log(h["price_m2"].to_numpy()), rcond=None)[0]
    g = dict(zip(X.columns, beta))
    gy = {int(c[2:]): v for c, v in g.items() if c.startswith("y_")}
    gy[2025] = 0.0
    return gy


def cells_slide(h):
    h = h[h["price_m2"].between(*M2) & h["year"].between(2017, 2026)].copy()
    gy = year_index(h)
    h["adj"] = h["price_m2"] * np.exp(-h["year"].astype(int).map(gy))
    h["cx"], h["cy"] = (h.geometry.x // CELL).astype(int), (h.geometry.y // CELL).astype(int)
    cells = h.groupby(["cx", "cy"]).agg(n=("adj", "size"), val=("adj", "median"))
    good = cells[cells["n"] >= MIN_IN_CELL]
    in_good = h.set_index(["cx", "cy"]).index.isin(good.index)
    dist = h.groupby("district").agg(n=("adj", "size"), val=("adj", "median"))
    dist = dist[dist["n"] >= 30].sort_values("val")
    rise = 100 * (np.exp(-gy[2017]) - 1)
    norm = Normalize(5000, 11000)
    fig = page(T(f"A house m² costs {dist['val'].iloc[0] / 1000:.1f}K in {dist.index[0]} and "
                 f"{dist['val'].iloc[-1] / 1000:.1f}K in {dist.index[-1]}",
                 f"Metr domu: od {dec1(dist['val'].iloc[0] / 1000)} tys. zł ({dist.index[0]}) "
                 f"do {dec1(dist['val'].iloc[-1] / 1000)} tys. ({dist.index[-1]})"),
               T(f"Median price per m² of living space (plot included) of houses sold 2017–2026, 2025 prices, per {CELL} m cell",
                 f"Mediana ceny m² powierzchni domu (z działką), domy sprzedane 2017–2026, ceny 2025, kratki {CELL} m"),
               T(f"{len(h):,} house sales (RCN) in 2025 prices (house index net of district). Cells with >= {MIN_IN_CELL} sales coloured "
                 f"({in_good.mean():.0%} of sales). Area: deed's ({(h['area_source'] == 'deed').mean():.0%}) or outline x storeys. ",
                 f"{fmt(len(h))} sprzedaży domów (RCN), ceny 2025 (indeks domów w obrębie dzielnicy). Kolor: kratki z >= {MIN_IN_CELL} "
                 f"sprzedażami ({in_good.mean():.0%}). Powierzchnia z aktu ({(h['area_source'] == 'deed').mean():.0%}) lub obrys x kondygnacje. ")
               + source())
    headline(fig, T(f"Houses sell mostly on the edges: {', '.join(dist.sort_values('n').index[-3:][::-1])} lead. A house m², plot "
                    f"included, costs a median {fmt(h['adj'].median())} PLN in 2025 prices, below a flat's; house prices rose "
                    f"{rise:.0f} % from 2017 to 2025.",
                    f"Domy sprzedają się głównie na obrzeżach: najwięcej {', '.join(dist.sort_values('n').index[-3:][::-1])}. "
                    f"Metr domu z działką kosztuje w medianie {fmt(h['adj'].median())} zł (ceny 2025), mniej niż mieszkania; "
                    f"ceny domów wzrosły o {rise:.0f} % od 2017 do 2025."))
    d = districts()
    ax = basemap(fig, d, rect=(0.03, 0.36, 0.94, 0.47))
    rest = h[~in_good]
    ax.scatter(rest.geometry.x, rest.geometry.y, s=2.5, color="#a9a69c", lw=0, zorder=3)
    for (cx, cy), r in good.iterrows():
        ax.add_patch(Rectangle((cx * CELL, cy * CELL), CELL, CELL, fc=CMAP(norm(r["val"])), ec="none", zorder=4))
    cax = fig.add_axes([0.70, 0.37, 0.24, 0.011])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([5000, 7000, 9000, 11000])
    cb.set_ticklabels(T(["5K", "7K", "9K", "11K+"], ["5 tys.", "7 tys.", "9 tys.", "11 tys.+"]))
    cax.tick_params(labelsize=8.5, length=0)
    cax.set_title(T("PLN per m²", "zł za m²"), fontsize=8.5, loc="left")
    a = fig.add_axes([0.22, 0.075, 0.72, 0.23])
    yy = np.arange(len(dist))
    a.barh(yy, dist["val"], color=[CMAP(norm(x)) for x in dist["val"]], height=0.7)
    for i, (x, n) in enumerate(zip(dist["val"], dist["n"])):
        a.text(x, i, f" {T(f'{x / 1000:.1f}K', f'{x / 1000:.1f}'.replace('.', ',') + ' tys.')}  ({fmt(n)} {T('sales', 'sprz.')})", va="center", fontsize=8)
    a.set_yticks(yy)
    a.set_yticklabels(dist.index, fontsize=8)
    a.set_xlim(0, dist["val"].max() * 1.4)
    a.set_xticks([])
    for sp in ("top", "right", "bottom"):
        a.spines[sp].set_visible(False)
    a.set_title(T("Median by district (>= 30 sales)", "Mediana w dzielnicy (>= 30 sprzedaży)"), fontsize=10.5, loc="left")
    save(fig, "r09_houses.png")


def curves(ax, house, flat, ticks, labels, xlabel):
    from scipy.stats import gaussian_kde
    lo, hi = np.log10(ticks[0] * 0.8), np.log10(ticks[-1] * 1.25)
    x = np.linspace(lo, hi, 400)
    for vals, colr, lab in ((flat, "#9a968b", T("Resold flat", "Mieszkanie z rynku wtórnego")),
                            (house, "#2a6b35", T("House (with plot)", "Dom (z działką)"))):
        v = np.log10(vals.dropna())
        y = gaussian_kde(v, bw_method=0.2)(x)
        ax.fill_between(10 ** x, y, color=colr, alpha=0.45, lw=0)
        ax.plot(10 ** x, y, color=colr, lw=1.6, label=lab)
        m = np.median(vals.dropna())
        ax.axvline(m, color=colr, lw=1.2, ls=(0, (4, 3)))
        ax.text(m, y.max() * 1.07, labels(m), color=colr, ha="center", va="bottom", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.85))
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    ax.set_xticklabels([labels(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(10 ** lo, 10 ** hi)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_xlabel(xlabel, fontsize=9)


def house_or_flat(h):
    r = h[h["year"].between(2023, 2026)]
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True)
    f = t[t["year"].between(2023, 2026) & (t["market"] == "secondary")]
    hp, fp = r["price"], f["price"]
    hm2 = r["price_m2"].where(r["price_m2"].between(*M2))
    fm2 = f["price_m2"].where(f["price_m2"].between(*M2))
    tot_x, m2_x = hp.median() / fp.median(), hm2.median() / fm2.median()
    mil = lambda m: (f"{m / 1e6:.2g} M" if m >= 1e6 else f"{m / 1e6:.2f} M") if T(1, 0) else (  # noqa: E731
        f"{m / 1e6:.2g} mln".replace(".", ",") if m >= 1e6 else f"{m / 1000:.0f} tys.")
    k = lambda m: T(f"{m / 1000:.1f}K", f"{m / 1000:.1f} tys.".replace(".", ","))  # noqa: E731
    fig = page(T(f"A house costs {tot_x:.1f}x a flat, a square metre {abs(1 - m2_x):.0%} {'less' if m2_x < 1 else 'more'}",
                 f"Dom kosztuje {tot_x:.1f}x tyle co mieszkanie, metr {abs(1 - m2_x):.0%} {'mniej' if m2_x < 1 else 'więcej'}".replace(".", ",")),
               T("Houses and resold flats sold 2023–2026 in Lublin", "Domy i mieszkania z rynku wtórnego sprzedane w Lublinie 2023–2026"),
               T(f"Houses: {len(r):,} sales with plot; flats: {len(f):,} resales, same years. House area: the deed's "
                 f"({(r['area_source'] == 'deed').mean():.0%}) or estimated from outline and storeys. ",
                 f"Domy: {fmt(len(r))} sprzedaży z działką; mieszkania: {fmt(len(f))} z rynku wtórnego, te same lata. Powierzchnia domu "
                 f"z aktu ({(r['area_source'] == 'deed').mean():.0%}) lub z obrysu i kondygnacji. ") + source())
    headline(fig, T(f"In total a median house costs {mil(hp.median())} and a flat {mil(fp.median())}. Per m² it flips: "
                    f"{k(hm2.median())} for a house, plot included, {k(fm2.median())} for a flat.",
                    f"Mediana domu to {mil(hp.median())} zł, mieszkania {mil(fp.median())} zł. Za metr odwrotnie: "
                    f"{k(hm2.median())} zł za dom z działką, {k(fm2.median())} zł za mieszkanie."))
    a1 = fig.add_axes([0.06, 0.52, 0.90, 0.29])
    curves(a1, hp, fp, [0.2e6, 0.4e6, 0.8e6, 1.6e6, 3.2e6], mil, T("price paid, PLN", "cena zapłacona, zł"))
    a1.legend(loc="upper right", frameon=False, fontsize=9.5)
    a1.set_title(T("What you pay in total", "Ile płacisz w sumie"), fontsize=11.5, loc="left")
    a2 = fig.add_axes([0.06, 0.135, 0.90, 0.29])
    curves(a2, hm2, fm2, [3000, 5000, 8000, 12000, 18000], k, T("price per m² of living space, PLN", "cena za m² powierzchni, zł"))
    a2.set_title(T("What a square metre costs", "Ile kosztuje metr"), fontsize=11.5, loc="left")
    save(fig, "r10_house_or_flat.png")


def main(lang="en"):
    set_lang(lang)
    h = gpd.read_file("output/houses.gpkg") if lang == "pl" else clean()
    cells_slide(h)
    house_or_flat(h)


if __name__ == "__main__":
    main("pl" if "--lang" in sys.argv and sys.argv[-1] == "pl" else "en")
