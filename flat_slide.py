"""
Two pages for one Warsaw flat: what it is worth against the flats that sold around it, and how its spot compares with the
rest of the city. The Warsaw twin of lublin/flat_slide.py, run from the project root. The address is an argument, so no
address lives in the code or git:

    python flat_slide.py --street "<street>" --no <number> --flat <flat number> [--lang pl]
 -> output/report/flats[/pl]/flat_<street>_<no>_<flat>.png        (value)
    output/report/flats[/pl]/flat_<street>_<no>_<flat>_place.png  (the spot vs Warsaw, the flat's own sales)

The flat (area, rooms, floor, its own sales) is found in the register by its unit id (..._BUD.<flat>_LOK, recorded since
~2021); --m2 / --rooms / --floor override or stand in when the flat has no recorded sale. Building, decade and stories
come from the data (register address -> priced building, GIS Expert decade, BDOT10k floors).
Comparables: resales since 2023 within 500 m in buildings of the same decade, area +-10 m², price per m² in 2025 prices
(transactions_adj) times the index's latest level. As-is = not sold as a flip's second leg (middle half); renovated =
second legs of flips (resold within a year of a resale purchase, flips.gpkg) since 2021, same decade, area +-15 m².
The flat's own sales and same-sized flats in its building are left out of the comparables.
"""
import argparse
import os
import re
import unicodedata

import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

import style
from summary_charts import page, wrap

style.apply()
R = 500
NORM = Normalize(11000, 21000)
LANG = {"code": "en"}
PL_MONTHS = ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"]


def T(en, pl):
    return pl if LANG["code"] == "pl" else en


def fmt(v):
    s = f"{v:,.0f}"
    return s.replace(",", " ") if LANG["code"] == "pl" else s


def K(v, step=5):
    """Zloty amount in thousands, rounded to `step` K."""
    k = round(v / (1000 * step)) * step
    return T(f"{k:.0f}K", f"{k:.0f} tys.")


def KR(lo, hi, step=5):
    """A zloty range in thousands with the unit once: 660–775K / 660–775 tys. zł."""
    r = f"{round(lo / (1000 * step)) * step:.0f}–{round(hi / (1000 * step)) * step:.0f}"
    return T(f"{r}K", f"{r} tys. zł")


def month(d):
    return f"{PL_MONTHS[d.month - 1]} {d.year}" if LANG["code"] == "pl" else f"{d:%b %Y}"


def num(v, d=1):
    s = f"{v:.{d}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")                  # 28 m², not 28.00 m²; 47.5 stays
    return s.replace(".", ",") if LANG["code"] == "pl" else s


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()).strip("_")


def load(a):
    pat = f"UL:[^;]*{re.escape(a.street)};NR_PORZ:{re.escape(a.no)}$"
    r = gpd.read_file("output/rcn_lokale_warszawa.gpkg", where=f"LOK_ADRES LIKE '%{a.street.replace(chr(39), '')}%'")
    r = r[r["LOK_ADRES"].fillna("").str.contains(pat, case=False)].to_crs(2180)
    if r.empty:
        raise SystemExit(f"no register address '{a.street} {a.no}'")
    pt = r.geometry.iloc[0]
    mine = r[r["LOK_ID_LOKALU"].fillna("").str.contains(rf"\.{re.escape(a.flat)}_LOK$")].copy()
    mine["date"] = pd.to_datetime(mine["DOK_DATA"])
    mine = mine.sort_values("date")
    flat = {"m2": a.m2, "rooms": a.rooms, "floor": a.floor}
    if len(mine):
        last = mine.iloc[-1]
        flat = {"m2": a.m2 or float(last["LOK_POW_UZYT"]), "rooms": a.rooms or int(last["LOK_LICZBA_IZB"]),
                "floor": a.floor or int(last["LOK_NR_KOND"])}
    if None in flat.values():
        raise SystemExit("flat not found in the register: pass --m2, --rooms and --floor")

    bv = gpd.read_file("output/price_model.gpkg", layer="building_value")
    inside = bv[bv.contains(pt)]
    me = inside.iloc[0] if len(inside) else bv.loc[bv.distance(pt).idxmin()]
    age = gpd.read_file("output/building_age.gpkg", ignore_geometry=True)[["osm_id", "decade_end"]]
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj").merge(age, on="osm_id", how="left")
    t["date"] = pd.to_datetime(t["date"])
    f = gpd.read_file("output/flips.gpkg", layer="flips", ignore_geometry=True)
    f = f[f["market"] == "secondary"]
    key = f[["lok_adres", "date2", "area"]].rename(columns={"date2": "date"}).drop_duplicates().assign(flip=True)
    t = t.merge(key, on=["lok_adres", "date", "area"], how="left")
    t["flip"] = t["flip"].fillna(False).astype(bool)
    q = pd.read_csv("output/analysis/price_index_quarterly.csv")
    now = float(np.exp(q["gamma"].iloc[-3:].mean()))
    t["today"] = t["price_m2_2025"] * now
    t["dist"] = t.distance(pt)
    t["own_flat"] = (t["osm_id"] == me["osm_id"]) & (t["area"] - flat["m2"]).abs().lt(0.05)
    c = me.geometry.centroid
    bd = gpd.read_file("data/bdot_buildings.gpkg", bbox=(c.x - 900, c.y - 900, c.x + 900, c.y + 900))
    hit = bd[bd.contains(c)]
    own = t[(t["osm_id"] == me["osm_id"]) & (t["market"] == "secondary")]
    storeys = int(hit["n_floors"].iloc[0]) if len(hit) and hit["n_floors"].iloc[0] > 0 else int(own["floor"].max())
    drv = gpd.read_file("output/price_drivers.gpkg", layer="building_drivers", ignore_geometry=True)
    return dict(pt=pt, me=me, bv=bv, bd=bd, t=t, f=f, now=now, mine=mine, flat=flat, storeys=storeys, drv=drv, own=own)


def notes_block(fig, notes, x, y, width=48, size=9.2):
    for n in notes:
        txt = wrap(n, width)
        fig.text(x, y, "•", fontsize=11, color=style.ORANGE, va="top", fontweight="bold")
        fig.text(x + 0.02, y, txt, fontsize=size, color=style.INK, va="top", linespacing=1.35)
        y -= 0.020 * (txt.count("\n") + 1) + 0.011
    return y


def value_page(a, d, label, out):
    t, me, flat, now = d["t"], d["me"], d["flat"], d["now"]
    AREA, storeys = flat["m2"], d["storeys"]
    dec = d["drv"].loc[d["drv"]["osm_id"] == me["osm_id"], "decade_end"]
    dec_end = float(dec.iloc[0]) if len(dec) else float(t.loc[t["osm_id"] == me["osm_id"], "decade_end"].dropna().iloc[0])
    decade = f"{int(dec_end) - 9}–{int(dec_end)}" if dec_end > 1940 else T("pre-war", "przedwojenny")
    district = me["district"]
    near = t[(t["market"] == "secondary") & (t["dist"] < R) & (t["decade_end"] == dec_end) & ~t["own_flat"]]
    comp = near[(near["year"] >= 2023) & near["area"].between(AREA - 10, AREA + 10)]
    asis = comp[~comp["flip"]]
    reno_all = near[near["flip"] & (near["year"] >= 2021) & near["area"].between(AREA - 15, AREA + 15)]
    a_lo, a_mid, a_hi = asis["today"].quantile([.25, .5, .75]) * AREA
    r_lo, r_mid, r_hi = reno_all["today"].quantile([.25, .5, .75]) * AREA
    print(f"{district}, {storeys} stories, {decade}; as-is {len(asis)} sales, {a_lo:,.0f}-{a_hi:,.0f} (mid {a_mid:,.0f}); "
          f"renovated {len(reno_all)} flips, {r_lo:,.0f}-{r_hi:,.0f} (mid {r_mid:,.0f})")
    ground, top = flat["floor"] == 1, flat["floor"] >= storeys
    floor_en = "ground floor" if ground else ("top floor" if top else f"floor {flat['floor'] - 1}")
    floor_pl = "parter" if ground else ("ostatnie piętro" if top else f"{flat['floor'] - 1}. piętro")

    rooms_en = ("1 room" if flat["rooms"] == 1 else       # register 'izby' = rooms + a separate kitchen
                f"{flat['rooms'] - 1} room{'s' if flat['rooms'] > 2 else ''} + kitchen")
    title_en = (f"{label}: ~{K(a_mid)} as it is, ~{K(r_mid)} renovated" if a.public     # same numbers as the big figures
                else f"{label}: about {KR(a_lo, a_hi)} as it is, ~{K(r_mid)} renovated")
    fig = page(T(title_en,
                 f"{label}: {KR(a_lo, a_hi)} bez remontu, ok. {K(r_mid)} po"),
               T(f"{num(AREA, 2)} m², {rooms_en}, " + ("in " if a.public else f"{floor_en} of ") + f"{'an' if str(storeys).startswith(('8', '11', '18')) else 'a'} {storeys}-story {decade} block "
                 f"({district})",
                 f"{num(AREA, 2)} m², {flat['rooms']} izby, {floor_pl}, {storeys}-kondygnacyjny blok z lat {decade} ({district})"),
               T(f"Comparables: resales since 2023 within {R} m in {decade} buildings, {AREA - 10:.0f}–{AREA + 10:.0f} m², in today's "
                 f"prices (city index); renovated = resold within a year of purchase, since 2021, {AREA - 15:.0f}–{AREA + 15:.0f} m². "
                 "Condition is not in the register. Source: GUGiK RCN, BDOT10k, GIS Expert.",
                 f"Porównania: sprzedaże wtórne od 2023 do {R} m, budynki z lat {decade}, {AREA - 10:.0f}–{AREA + 10:.0f} m², w dzisiejszych "
                 f"cenach (indeks miasta); po remoncie = odsprzedane w ciągu roku od zakupu, od 2021, {AREA - 15:.0f}–{AREA + 15:.0f} m². "
                 "Rejestr nie zna stanu. Źródło: GUGiK RCN, BDOT10k, GIS Expert."))
    for x, big, small in ((0.05, K(a_mid), T(f"as it is: middle of {len(asis)}\nnearby resales, {num(AREA, 2)} m²",
                                             f"bez remontu: środek {len(asis)}\nsprzedaży w okolicy, {num(AREA, 2)} m²")),
                          (0.38, K(r_mid), T(f"renovated: middle of {len(reno_all)}\nflipped flats nearby",
                                             f"po remoncie: środek {len(reno_all)}\nodsprzedanych mieszkań w okolicy")),
                          (0.71, fmt(me["value_2025"]), T("PLN/m², typical flat in\nthis building (2025)",
                                                          "zł/m², typowe mieszkanie\nw tym budynku (2025)"))):
        fig.text(x, 0.885, big, fontsize=28, fontweight="bold", color=style.INK, va="top")
        fig.text(x, 0.838, small, fontsize=10, color=style.INK_2, va="top")

    # map: priced buildings around, coloured by typical-flat value --------------------------------
    ax = fig.add_axes([0.03, 0.43, 0.50, 0.32])
    cx, cy = d["pt"].x, d["pt"].y
    half = 650
    d["bd"].cx[cx - half:cx + half, cy - half:cy + half].plot(ax=ax, color="#e4e2da", lw=0, zorder=1)
    v = d["bv"].cx[cx - half:cx + half, cy - half:cy + half]
    v.plot(ax=ax, color=[style.PRICE_CMAP(NORM(x)) for x in v["value_2025"]], lw=0, zorder=2)
    gpd.GeoSeries([me.geometry]).plot(ax=ax, facecolor=style.INK, edgecolor=style.INK, lw=2.5, zorder=4)
    ax.add_patch(plt.Circle((cx, cy), R, fill=False, ls=(0, (4, 3)), color=style.INK_2, lw=0.9, zorder=3))
    ax.annotate(f"{a.street} {a.no}" if a.public else label, (cx, cy), xytext=(cx + 120, cy + 260), fontsize=9.5,
                fontweight="bold", zorder=5, arrowprops=dict(arrowstyle="-", color=style.INK, lw=0.8),
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])     # white halo: legible over buildings
    ax.text(cx, cy - R - 35, f"{R} m", ha="center", va="top", fontsize=8, color=style.INK_2)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(T("Typical flat per building, 2025", "Typowe mieszkanie w budynku, 2025"), fontsize=10.5, loc="left")
    cax = fig.add_axes([0.07, 0.415, 0.42, 0.010])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=style.PRICE_CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    ticks = [11000, 14000, 17000, 20000]
    cb.set_ticks(ticks)
    cb.set_ticklabels([T(f"{x // 1000}K", f"{x // 1000} tys.") for x in ticks])
    cax.tick_params(labelsize=8, length=0)

    # side notes ------------------------------------------------------------------------------
    notes = []
    mine = d["mine"]
    if len(mine) >= 2:
        p0, p1 = mine.iloc[-2], mine.iloc[-1]
        own = t[t["own_flat"] & (t["date"] == p1["date"])]
        today = float(own["today"].iloc[0]) * AREA if len(own) else np.nan
        gap = (p1["date"] - p0["date"]).days / 30.4
        notes.append(T(f"This flat sold twice: {K(float(p0['LOK_CENA_BRUTTO']))} in {month(p0['date'])}, then {K(float(p1['LOK_CENA_BRUTTO']))} "
                       f"{gap:.0f} months later (+{100 * (float(p1['LOK_CENA_BRUTTO']) / float(p0['LOK_CENA_BRUTTO']) - 1):.0f}%), the pattern "
                       f"of a renovation flip. {K(float(p1['LOK_CENA_BRUTTO']))} then is about {K(today)} in today's prices.",
                       f"To mieszkanie sprzedano dwa razy: za {K(float(p0['LOK_CENA_BRUTTO']))} zł ({month(p0['date'])}), potem za "
                       f"{K(float(p1['LOK_CENA_BRUTTO']))} zł {gap:.0f} mies. później (+{100 * (float(p1['LOK_CENA_BRUTTO']) / float(p0['LOK_CENA_BRUTTO']) - 1):.0f}%), "
                       f"jak po remoncie na sprzedaż. {K(float(p1['LOK_CENA_BRUTTO']))} zł wtedy to dziś ok. {K(today)} zł."))
    elif len(mine) == 1:
        p1 = mine.iloc[-1]
        notes.append(T(f"This flat last sold for {K(float(p1['LOK_CENA_BRUTTO']))} in {month(p1['date'])}.",
                       f"To mieszkanie sprzedano ostatnio za {K(float(p1['LOK_CENA_BRUTTO']))} zł ({month(p1['date'])})."))
    if ground or top:
        sel = (asis["floor"] == 1) if ground else (asis["floor"] >= storeys)
        m_in, m_out = asis.loc[sel, "today"].median(), asis.loc[~sel, "today"].median()
        g = 100 * (m_in / m_out - 1)
        word_en, word_pl = ("Ground floor", "Parter") if ground else ("Top floor", "Ostatnie piętro")
        notes.append(T(f"{word_en}: {'no clear discount here' if g > -3 else f'{g:.0f}% here'}. {sel.sum()} such resales nearby went for "
                       f"{fmt(m_in)} PLN/m², the other {(~sel).sum()} for {fmt(m_out)}.",
                       f"{word_pl}: {'tu bez wyraźnej zniżki' if g > -3 else f'tu {g:.0f}%'}. {sel.sum()} takich sprzedaży w okolicy "
                       f"szło po {fmt(m_in)} zł/m², pozostałe {(~sel).sum()} po {fmt(m_out)}."))
    bld = t[(t["osm_id"] == me["osm_id"]) & (t["market"] == "secondary") & ~t["own_flat"] & (t["year"] >= 2023)]
    if len(bld) and a.public:          # sample page: the range only, no single stranger's sale
        notes.append(T(f"Same building since 2023: {len(bld)} other resales, {fmt(bld['today'].min())}–"
                       f"{fmt(bld['today'].max())} PLN/m² in today's prices, middle {fmt(bld['today'].median())}.",
                       f"Ten sam budynek od 2023: {len(bld)} innych sprzedaży, {fmt(bld['today'].min())}–"
                       f"{fmt(bld['today'].max())} zł/m² w dzisiejszych cenach, środek {fmt(bld['today'].median())}."))
    elif len(bld):
        best = bld.loc[bld["today"].idxmax()]
        notes.append(T(f"Same building since 2023: {len(bld)} other resales, {fmt(bld['today'].min())}–{fmt(best['today'])} PLN/m² "
                       f"in today's prices; the top one, {num(best['area'])} m² for {K(best['price'])} ({month(best['date'])}).",
                       f"Ten sam budynek od 2023: {len(bld)} innych sprzedaży, {fmt(bld['today'].min())}–{fmt(best['today'])} zł/m² w "
                       f"dzisiejszych cenach; najdroższa: {num(best['area'])} m² za {K(best['price'])} zł ({month(best['date'])})."))
    new = t[(t["market"] == "primary") & (t["dist"] < R) & (t["date"] >= t["date"].max() - pd.DateOffset(years=1))]
    if len(new) >= 10:                  # new-build flats sold by developers in the last year of data
        sec = t[(t["market"] == "secondary") & (t["dist"] < R) & (t["date"] >= new["date"].min())]
        top = new["lok_adres"].str.extract(r"UL:(?:ulica )?([^;]+);NR_PORZ:(.+)$").agg(" ".join, axis=1).value_counts()
        where = ", ".join(top.index[:2])
        gap = 100 * (new["price_m2"].median() / sec["price_m2"].median() - 1)
        notes.append(T(f"New flats nearby: {len(new)} sold by developers in the last year ({where}), "
                       f"{fmt(new['price_m2'].median())} PLN/m², {gap:.0f}% above second-hand flats around.",
                       f"Nowe mieszkania w okolicy: {len(new)} sprzedanych przez deweloperów w ostatnim roku ({where}), "
                       f"{fmt(new['price_m2'].median())} zł/m², {gap:.0f}% drożej niż z drugiej ręki."))
    if not a.public:                   # seller advice: private pages only
      notes.append(T(f"The asking price to aim at: {KR(a_mid, a_hi)} for a well-kept flat, {KR(r_lo, r_hi)} if the renovation "
                   "still looks fresh.",
                   f"Cena do celowania: {KR(a_mid, a_hi)} za zadbane mieszkanie, {KR(r_lo, r_hi)}, jeśli remont "
                   "wciąż wygląda świeżo."))
    if a.public:                       # how prices moved here: the timeline's own median line, first vs last half-year
        pl = near[near["area"].between(AREA - 15, AREA + 15) & ~near["flip"]
                  & (near["year"] >= 2021)].set_index("date")["price_m2"]          # the timeline's years
        hy = pl.resample("2QS-JAN").agg(["median", "size"])
        hy = hy[hy["size"] >= 8]["median"]
        if len(hy) >= 2:
            notes.append(T(f"Second-hand flats like this one nearby went from {fmt(hy.iloc[0])} PLN/m² in "
                           f"{hy.index[0].year} to {fmt(hy.iloc[-1])} in {hy.index[-1].year}, as paid: "
                           f"{100 * (hy.iloc[-1] / hy.iloc[0] - 1):+.0f}%.",
                           f"Mieszkania z drugiej ręki jak to w okolicy: z {fmt(hy.iloc[0])} zł/m² w {hy.index[0].year} "
                           f"do {fmt(hy.iloc[-1])} w {hy.index[-1].year}, ceny transakcyjne: "
                           f"{100 * (hy.iloc[-1] / hy.iloc[0] - 1):+.0f}%."))
    notes_block(fig, notes, 0.57, 0.765)

    # timeline --------------------------------------------------------------------------------
    a2 = fig.add_axes([0.11, 0.09, 0.85, 0.27])
    pool = near[near["area"].between(AREA - 15, AREA + 15)]
    plain = pool[~pool["flip"]]
    a2.scatter(plain["date"], plain["price_m2"], s=9, color="#b9b6ac", lw=0,
               label=T(f"second-hand, {decade} blocks", f"z drugiej ręki, bloki z lat {decade}"))
    fl = pool[pool["flip"]]
    a2.scatter(fl["date"], fl["price_m2"], s=22, color=style.YELLOW, lw=0,
               label=T("second-hand, resold within a year", "z drugiej ręki, odsprzedane w ciągu roku"))
    prim = t[(t["market"] == "primary") & (t["dist"] < R) & t["area"].between(AREA - 15, AREA + 15)]
    a2.scatter(prim["date"], prim["price_m2"], s=14, color=style.AQUA, lw=0, alpha=0.7,
               label=T("new, from a developer (any block)", "nowe, od dewelopera (dowolny budynek)"))
    own_b = t[(t["osm_id"] == me["osm_id"]) & (t["market"] == "secondary") & ~t["own_flat"]]
    a2.scatter(own_b["date"], own_b["price_m2"], s=34, color=style.INK, lw=0, label=T("this building", "ten budynek"), zorder=5)
    if len(mine):
        mp = mine["LOK_CENA_BRUTTO"].astype(float) / AREA
        a2.scatter(mine["date"], mp, s=90, color=style.RED, lw=0, zorder=6, label=T("this flat", "to mieszkanie"))
        for dte, p, pr in zip(mine["date"], mp, mine["LOK_CENA_BRUTTO"].astype(float)):
            a2.annotate(K(pr), (dte, p), xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=8.5,
                        color=style.RED, fontweight="bold", zorder=7)
    qs = plain.set_index("date")["price_m2"].resample("2QS-JAN").agg(["median", "size"])   # half-years: quarters ran thin
    qs.index = qs.index + pd.DateOffset(months=3)                                          # plotted mid half-year
    m = qs["median"].where(qs["size"] >= 8)                                                # a thin last half-year is left out
    a2.plot(m.index, m.values, color=style.BLUE, lw=2, label=T("median, second-hand", "mediana z drugiej ręki"))
    end = pd.Timestamp("2026-10-01")
    a2.fill_between([pd.Timestamp("2026-01-01"), end], a_lo / AREA, a_hi / AREA, color=style.BLUE, alpha=0.18, lw=0)
    a2.text(pd.Timestamp("2026-01-15"), a_hi / AREA + 150, T("today, as is", "dziś, bez remontu"), fontsize=8,
            color=style.BLUE, va="bottom")
    a2.yaxis.set_major_locator(plt.MultipleLocator(2000))
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v_, _: T(f"{v_ / 1000:.0f}K", f"{v_ / 1000:.0f} tys.")))
    a2.set_ylabel(T("PLN per m², as paid", "zł za m², ceny zapłacone"))
    style.hgrid_only(a2)
    a2.set_xlim(pd.Timestamp("2020-10-01"), pd.Timestamp("2026-12-31"))
    a2.set_ylim(bottom=max(0, plain.loc[plain["date"] >= "2021-01-01", "price_m2"].quantile(0.005) - 1000),
                top=pd.concat([pool, prim]).loc[lambda x: x["date"] >= "2021-01-01", "price_m2"].quantile(0.995) + 4500)
    a2.legend(loc="upper left", fontsize=8.2, frameon=False, ncol=3)
    a2.set_title(T(f"Every flat sale within {R} m since 2021, {AREA - 15:.0f}–{AREA + 15:.0f} m²",
                   f"Wszystkie sprzedaże mieszkań do {R} m od 2021, {AREA - 15:.0f}–{AREA + 15:.0f} m²"),
                 fontsize=10.5, loc="left")
    path = f"{out}/flat_{slug(a.street)}_{slug(a.no)}_{slug(a.flat)}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")
    return dict(a_lo=a_lo, a_mid=a_mid, a_hi=a_hi, r_lo=r_lo, r_mid=r_mid, r_hi=r_hi, decade=decade)


def place_page(a, d, label, out, val):
    me, drv, flat, t = d["me"], d["drv"], d["flat"], d["t"]
    AREA = flat["m2"]
    row = drv[drv["osm_id"] == me["osm_id"]]
    if row.empty:
        print("building not in the price model (fewer than 3 sales): no place page")
        return
    b = row.iloc[0]
    district = b["district"]
    dist_med = drv.groupby("district")["resid_pct"].median()
    rep, rep_d = b["resid_pct"], dist_med[district]

    fig = page(T(f"{label.split('/')[0]}: flats here sell {abs(rep):.0f}% {'below' if rep < 0 else 'above'} similar buildings",
                 f"{label.split('/')[0]}: mieszkania tu idą {abs(rep):.0f}% {'poniżej' if rep < 0 else 'powyżej'} podobnych budynków"),
               T("How the spot compares with Warsaw's priced buildings, and what the flat's own sales say",
                 "Jak to miejsce wypada na tle Warszawy i co mówią sprzedaże samego mieszkania"),
               T(f"Rows: where the building falls among {len(drv):,} Warsaw buildings with >= 3 sales (left = fewer / less). 'Similar': same age, height, "
                 "metro walk, cafés, green, noise. Today's prices: city index. Source: GUGiK RCN, OSM, BDOT10k, GIS Expert.",
                 f"Wiersze: miejsce budynku wśród {fmt(len(drv))} budynków w Warszawie z >= 3 sprzedażami (lewo = mniej). 'Podobne': ten sam "
                 "wiek, wysokość, dojście do metra, kawiarnie, zieleń, hałas. Ceny dzisiejsze: indeks miasta. Źródło: GUGiK RCN, OSM, BDOT10k."))
    fig.text(0.03, 0.885, wrap(T(f"A building with the same age, height, walk to the metro, cafés, green and noise elsewhere in Warsaw "
                                 f"would go for about {fmt(b['pred_2025'])} PLN/m² for a typical flat; this one goes for {fmt(b['value_2025'])}. "
                                 f"That gap is reputation: {district} as a whole sits {abs(rep_d):.0f}% {'below' if rep_d < 0 else 'above'}.",
                                 f"Budynek w tym samym wieku, tej samej wysokości, z takim dojściem do metra, kawiarniami, zielenią i hałasem "
                                 f"gdzie indziej w Warszawie dawałby ok. {fmt(b['pred_2025'])} zł/m² za typowe mieszkanie; ten daje "
                                 f"{fmt(b['value_2025'])}. Ta różnica to reputacja: cała {district} jest {abs(rep_d):.0f}% "
                                 f"{'poniżej' if rep_d < 0 else 'powyżej'}."), 100),
             fontsize=10.5, fontweight="semibold", va="top")

    # strips: the building among all priced buildings ------------------------------------------
    cafes = 2 ** drv["walk7_food_drink"] - 1
    rows = [
        (T("Walk to the metro", "Dojście do metra"), drv["walk_metro_km"], b["walk_metro_km"], lambda v: f"{num(v)} km"),
        (T("Walk to a tram stop", "Dojście do tramwaju"), drv["walk_tram_km"], b["walk_tram_km"], lambda v: f"{num(v)} km"),
        (T("Distance to the center (PKiN)", "Odległość od centrum (PKiN)"), drv["centre_km"], b["centre_km"], lambda v: f"{num(v)} km"),
        (T("Cafés, restaurants, bars, 7-min walk", "Kawiarnie, restauracje, bary, 7 min"), cafes, 2 ** b["walk7_food_drink"] - 1,
         lambda v: f"{v:.0f}"),
        (T("Park or forest within 500 m", "Park lub las do 500 m"), drv["green_ha_500"], b["green_ha_500"], lambda v: f"{num(v)} ha"),
        (T("Tree cover within 100 m", "Pokrycie drzewami do 100 m"), drv["canopy_100"], b["canopy_100"], lambda v: f"{v:.0f}%"),
        (T("Road noise", "Hałas drogowy"), drv["noise_road_db"], b["noise_road_db"],
         lambda v: T("quiet", "cicho") if v <= 45 else f"{v:.0f} dB"),
        (T("Tram and train noise", "Hałas tramwajów i kolei"), drv["noise_rail_db"], b["noise_rail_db"],
         lambda v: T("quiet", "cicho") if v <= 45 else f"{v:.0f} dB"),
        (T("Typical flat, PLN/m² (2025)", "Typowe mieszkanie, zł/m² (2025)"), drv["value_2025"], b["value_2025"], fmt),
    ]
    ax = fig.add_axes([0.40, 0.44, 0.44, 0.36])
    in_d = drv["district"] == district
    for i, (lab, s, v, f_) in enumerate(rows):
        y = len(rows) - 1 - i
        pct = 100 * ((s < v).mean() + 0.5 * (s == v).mean())
        pd_ = 100 * ((s < s[in_d].median()).mean() + 0.5 * (s == s[in_d].median()).mean())
        ax.plot([0, 100], [y, y], color=style.GRID, lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([pd_], [y], s=40, color=style.MUTED, lw=0, zorder=2)
        ax.scatter([pct], [y], s=110, color=style.ORANGE, lw=0, zorder=3)
        fig.text(0.38, 0.44 + 0.36 * (y + 0.5) / len(rows), lab, ha="right", va="center", fontsize=9.5, color=style.INK)
        ax.text(104, y, f_(v), va="center", fontsize=9.5, fontweight="bold", color=style.INK)
    ax.set_xlim(-3, 100)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.axis("off")
    ax.text(0, -1.0, T("fewer / less", "mniej"), fontsize=8, color=style.INK_2, ha="left", va="top")
    ax.text(100, -1.0, T("more", "więcej"), fontsize=8, color=style.INK_2, ha="right", va="top")
    ax.text(50, -1.0, T("Warsaw median", "mediana Warszawy"), fontsize=8, color=style.INK_2, ha="center", va="top")
    ax.plot([50, 50], [-0.5, len(rows) - 0.5], color=style.AXIS, lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.scatter([], [], s=110, color=style.ORANGE, lw=0, label=T("this building", "ten budynek"))
    ax.scatter([], [], s=40, color=style.MUTED, lw=0, label=T(f"{district}, median building", f"{district}, mediana"))
    ax.legend(loc="lower left", bbox_to_anchor=(-0.05, 1.0), ncol=2, frameon=False, fontsize=9)
    fig.text(0.03, 0.80, T("The spot", "Miejsce"), fontsize=11, fontweight="semibold")

    # the flat's own sales, in today's prices ---------------------------------------------------
    mine = d["mine"]
    bars = []
    for _, p in mine.iterrows():
        sold = t[t["own_flat"] & (t["date"] == p["date"])]
        if len(sold):
            bars.append((T(f"Sold {month(p['date'])}", f"Sprzedaż {month(p['date'])}"), float(sold["today"].iloc[0]) * AREA,
                         float(p["LOK_CENA_BRUTTO"]), style.RED))
    bars += [(T("Today, as it is", "Dziś, bez remontu"), val["a_mid"], None, style.BLUE),
             (T("Today, renovated", "Dziś, po remoncie"), val["r_mid"], None, style.YELLOW)]
    a2 = fig.add_axes([0.25, 0.10, 0.62, 0.22])
    for i, (lab, v, paid, col) in enumerate(bars):
        y = len(bars) - 1 - i
        a2.barh(y, v, color=col, height=0.6, alpha=0.9)
        txt = (T(f"{K(paid)} paid = {K(v)} today", f"{K(paid)} zł wtedy = {K(v)} zł dziś") if paid
               else T(f"{K(v)}", f"{K(v)} zł"))
        if col == style.BLUE:
            r_ = f"{round(val['a_lo'] / 5000) * 5:.0f}–{round(val['a_hi'] / 5000) * 5:.0f}"
            txt = T(f"{K(v)}  (range {r_}K)", f"{K(v)} zł  (przedział {r_} tys.)")
        a2.text(max(v, val["a_hi"] if col == style.BLUE else 0) + 8000, y, txt, va="center", fontsize=9.5, fontweight="bold", color=style.INK)
        a2.text(-15000, y, lab, va="center", ha="right", fontsize=9.5, color=style.INK)
    lo, hi = val["a_lo"], val["a_hi"]
    yb = len(bars) - 1 - (len(bars) - 2)
    a2.plot([lo, hi], [yb, yb], color=style.INK, lw=1.2, zorder=4)
    a2.set_xlim(0, max(b_[1] for b_ in bars) * 1.45)
    a2.set_ylim(-0.6, len(bars) - 0.4)
    a2.axis("off")
    fig.text(0.03, 0.355, T(f"This flat's sales and today's range, in today's prices ({num(AREA, 2)} m²)",
                            f"Sprzedaże tego mieszkania i dzisiejszy przedział, w dzisiejszych cenach ({num(AREA, 2)} m²)"),
             fontsize=11, fontweight="semibold")
    fig.text(0.03, 0.335, T("Black line: middle half of nearby as-is resales.", "Czarna linia: środkowa połowa sprzedaży w okolicy bez remontu."),
             fontsize=8.5, color=style.INK_2)
    path = f"{out}/flat_{slug(a.street)}_{slug(a.no)}_{slug(a.flat)}_place.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", required=True, help="as in the register, e.g. the surname for 'ulica <Name Surname>'")
    ap.add_argument("--no", required=True)
    ap.add_argument("--flat", required=True)
    ap.add_argument("--m2", type=float)
    ap.add_argument("--rooms", type=int, help="izby as in the register (rooms + a separate kitchen)")
    ap.add_argument("--floor", type=int, help="RCN floor, ground floor = 1")
    ap.add_argument("--lang", default="en")
    ap.add_argument("--public", action="store_true", help="sample page for publishing: no single sale named, no seller advice")
    ap.add_argument("--label", help="title instead of 'street no/flat', e.g. for a sample flat with an invented number")
    a = ap.parse_args()
    LANG["code"] = a.lang
    d = load(a)
    label = a.label or f"{a.street} {a.no}/{a.flat}"
    out = "output/report/flats/pl" if a.lang == "pl" else "output/report/flats"
    os.makedirs(out, exist_ok=True)
    val = value_page(a, d, label, out)
    place_page(a, d, label, out, val)


if __name__ == "__main__":
    main()
