"""
One slide for one flat in Lublin: what it is worth against the flats that sold around it.
Run from lublin/ (the address is an argument, so no address lives in the code or git):

    python flat_slide.py --street "<street>" --no <number> --m2 <area> --rooms <izby> --floor <RCN floor, ground = 1> [--lang pl]
 -> output/report[/pl]/flat_<street>_<no>.png

Building, storeys, district and decade come from the data (register address -> BDOT10k building, GIS Expert decade).
Comparables: resales since 2023 within 500 m, in buildings with the same number of storeys, area +-10 m² around the flat,
price per m² in 2025 prices (transactions_adj) times the index's 2026 level. As-is = those not sold as a flip's second leg
(middle half); renovated = second legs of flips (bought resale, sold again within a year, analysis/flips.csv) since 2021
within 500 m, same storeys, area +-20 m². Top floor: shown as a note (top-floor resales nearby vs the rest), not adjusted.
"""
import argparse
import os
import re
import unicodedata

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from common import LANG, T, fmt, page, set_lang, style, wrap

R = 500
NORM = Normalize(6500, 11500)
PL_MONTHS = ["sty", "lut", "mar", "kwi", "maj", "cze", "lip", "sie", "wrz", "paź", "lis", "gru"]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", s.lower()).encode("ascii", "ignore").decode()).strip("_")


def load(street, no):
    r = gpd.read_file("data/rcn_lokale_lublin.gpkg", columns=["LOK_ADRES"]).to_crs(2180)
    hit = r[r["LOK_ADRES"].fillna("").str.contains(f"UL:{re.escape(street)};NR_PORZ:{re.escape(no)}$", case=False)]
    if hit.empty:
        raise SystemExit(f"no register address 'UL:{street};NR_PORZ:{no}'")
    me_pt = hit.geometry.iloc[0]
    b = gpd.read_file("data/buildings.gpkg")
    me = b[b.contains(me_pt)].iloc[0] if b.contains(me_pt).any() else b.loc[b.distance(me_pt).idxmin()]
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t = t.merge(b[["osm_id", "storeys"]], on="osm_id", how="left")
    t["date"] = pd.to_datetime(t["date"])
    q = pd.read_csv("output/analysis/price_index_quarterly.csv")
    now = float(np.exp(q["gamma"].iloc[-3:].mean()))
    f = pd.read_csv("output/analysis/flips.csv", parse_dates=["date", "date2"])
    f = f[f["market"] == "secondary"]
    key = f[["lok_adres", "date2", "area"]].rename(columns={"date2": "date"}).assign(flip=True)
    t = t.merge(key.drop_duplicates(), on=["lok_adres", "date", "area"], how="left")
    t["flip"] = t["flip"].fillna(False).astype(bool)
    t["dist"] = t.distance(me_pt)
    t["today"] = t["price_m2_2025"] * now
    return me_pt, me, b, t, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", required=True)
    ap.add_argument("--no", required=True)
    ap.add_argument("--m2", type=float, required=True)
    ap.add_argument("--rooms", type=int, required=True, help="izby as in the register (rooms + a separate kitchen)")
    ap.add_argument("--floor", type=int, required=True, help="RCN floor, ground floor = 1")
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    set_lang(a.lang)
    AREA = a.m2
    label = f"{a.street} {a.no}"
    me_pt, me, b, t, f = load(a.street, a.no)
    storeys = int(me["storeys"])
    top = a.floor >= storeys
    age = gpd.read_file("output/building_age.gpkg", ignore_geometry=True)
    dec = age.loc[age["osm_id"] == me["osm_id"], "decade_end"]
    decade = f"{int(dec.iloc[0]) - 9}–{int(dec.iloc[0])}" if len(dec) and dec.iloc[0] > 1900 else T("unknown decade", "nieznana dekada")
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value")
    mine = bv[bv["osm_id"] == me["osm_id"]]
    own_val, district = float(mine["value_2025"].iloc[0]), mine["district"].iloc[0]

    near = t[(t["market"] == "secondary") & (t["dist"] < R) & (t["storeys"] == storeys)]
    comp = near[(near["year"] >= 2023) & near["area"].between(AREA - 10, AREA + 10)]
    asis = comp.loc[~comp["flip"], "today"]
    reno_all = near[near["flip"] & (near["year"] >= 2021) & near["area"].between(AREA - 20, AREA + 20)]
    reno = reno_all["today"]
    twin = reno_all.iloc[(reno_all["area"] - AREA).abs().argsort()[:1]].iloc[0] if len(reno_all) else None
    a_lo, a_mid, a_hi = asis.quantile([.25, .5, .75]) * AREA
    r_lo, r_mid, r_hi = reno.quantile([.25, .5, .75]) * AREA
    own = t[t["osm_id"] == me["osm_id"]]
    same = own[(own["floor"] == a.floor) & (own["market"] == "secondary")]
    same = same.iloc[(same["area"] - AREA).abs().argsort()[:1]] if len(same) else same
    rec_s = t[(t["market"] == "secondary") & t["year"].between(2021, 2025)]
    rec_f = f[f["date2"].dt.year.between(2021, 2025)]
    flip_rate = 100 * (rec_f["district"] == district).sum() / (rec_s["district"] == district).sum()
    print(f"{district}, {storeys} storeys, {decade}; as-is {len(asis)} sales, {a_lo:,.0f}-{a_hi:,.0f} (mid {a_mid:,.0f}); "
          f"renovated {len(reno)} flips, {r_lo:,.0f}-{r_hi:,.0f} (mid {r_mid:,.0f})")

    tys = lambda v: T(f"{round(v / 5000) * 5:.0f}K", f"{round(v / 5000) * 5:.0f} tys.")  # noqa: E731   (rounded to 5K)
    r5 = lambda v: f"{round(v / 5000) * 5:.0f}"  # noqa: E731
    floor_en = T("top floor" if top else f"floor {a.floor} of {storeys}", "")
    floor_pl = "ostatnie piętro" if top else f"kondygnacja {a.floor} z {storeys}"
    fig = page(T(f"{label}: about {r5(a_lo)}–{r5(a_hi)}K as it is, more if renovated",
                 f"{label}: ok. {r5(a_lo)}–{r5(a_hi)} tys. zł bez remontu, więcej po"),
               T(f"{AREA:g} m², {a.rooms} rooms incl. kitchen, {floor_en} of a {decade} block ({district}), against the sales around it",
                 f"{AREA:g} m², {a.rooms} izby, {floor_pl}, blok z lat {decade} ({district}), na tle okolicy".replace(".", ",", 0)),
               T(f"Comparables: resales since 2023 within {R} m in {storeys}-storey blocks, {AREA - 10:g}–{AREA + 10:g} m², in today's prices "
                 f"(city index); renovated = flats resold within a year of purchase since 2021, {AREA - 20:g}–{AREA + 20:g} m². Condition is "
                 "not in the register. Source: GUGiK RCN, BDOT10k, GIS Expert.",
                 f"Porównania: sprzedaże wtórne od 2023 do {R} m, bloki {storeys}-kondygnacyjne, {AREA - 10:g}–{AREA + 10:g} m², w dzisiejszych "
                 f"cenach (indeks miasta); po remoncie = odsprzedane w ciągu roku od zakupu od 2021, {AREA - 20:g}–{AREA + 20:g} m². Rejestr nie "
                 "zna stanu. Źródło: GUGiK RCN, BDOT10k, GIS Expert."))
    for x, big, small in ((0.05, tys(a_mid), T(f"as it is: middle of {len(asis)}\nnearby sales, {AREA:g} m²",
                                               f"w obecnym stanie: środek {len(asis)}\nsprzedaży w okolicy, {AREA:g} m²")),
                          (0.38, tys(r_mid), T(f"renovated: middle of {len(reno)}\nflipped flats nearby",
                                               f"po remoncie: środek {len(reno)}\nodsprzedanych mieszkań w okolicy")),
                          (0.71, fmt(own_val), T("PLN/m², typical flat in\nthis building (2025)",
                                                 "zł/m², typowe mieszkanie\nw tym budynku (2025)"))):
        fig.text(x, 0.885, big, fontsize=28, fontweight="bold", color=style.INK, va="top")
        fig.text(x, 0.838, small, fontsize=10, color=style.INK_2, va="top")

    # map: priced buildings around, coloured by typical-flat value -------------------------------
    ax = fig.add_axes([0.03, 0.43, 0.50, 0.32])
    cx, cy = me_pt.x, me_pt.y
    half = 650
    b.cx[cx - half:cx + half, cy - half:cy + half].plot(ax=ax, color="#e4e2da", lw=0, zorder=1)
    v = bv.cx[cx - half:cx + half, cy - half:cy + half]
    v.plot(ax=ax, color=[style.PRICE_CMAP(NORM(x)) for x in v["value_2025"]], lw=0, zorder=2)
    gpd.GeoSeries([me.geometry]).plot(ax=ax, facecolor=style.INK, edgecolor=style.INK, lw=2.5, zorder=4)
    ax.add_patch(plt.Circle((cx, cy), R, fill=False, ls=(0, (4, 3)), color=style.INK_2, lw=0.9, zorder=3))
    ax.annotate(label, (cx, cy), xytext=(cx + 120, cy + 260), fontsize=9.5, fontweight="bold", zorder=5,
                arrowprops=dict(arrowstyle="-", color=style.INK, lw=0.8))
    ax.text(cx, cy - R - 35, f"{R} m", ha="center", va="top", fontsize=8, color=style.INK_2)
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(T("Typical flat per building, 2025", "Typowe mieszkanie w budynku, 2025"), fontsize=10.5, loc="left")
    cax = fig.add_axes([0.07, 0.415, 0.42, 0.010])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=NORM, cmap=style.PRICE_CMAP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([7000, 8500, 10000, 11500])
    cb.set_ticklabels([T(f"{x / 1000:.1f}K", f"{x / 1000:.1f}".replace(".", ",") + " tys.") for x in (7000, 8500, 10000, 11500)])
    cax.tick_params(labelsize=8, length=0)

    # side notes ------------------------------------------------------------------------------
    notes = []
    if top:
        top_med = comp.loc[~comp["flip"] & (comp["floor"] >= storeys), "today"].median()
        low_med = comp.loc[~comp["flip"] & (comp["floor"] < storeys), "today"].median()
        gap = 100 * (top_med / low_med - 1)
        notes.append(T(f"Top floor: {'no penalty here' if gap > -2 else f'{gap:.0f} % here'}. Top-floor resales nearby went for "
                       f"{fmt(top_med)} PLN/m², the others for {fmt(low_med)}.",
                       f"Ostatnie piętro: {'tu bez zniżki' if gap > -2 else f'tu {gap:.0f} %'}. Ostatnie piętra w okolicy szły po "
                       f"{fmt(top_med)} zł/m², pozostałe po {fmt(low_med)}."))
    if len(same):
        s = same.iloc[0]
        notes.append(T(f"Same building, same floor: {s['area']:g} m² sold for {s['price'] / 1000:.0f}K in {s['date']:%b %Y}, about "
                       f"{tys(s['today'] * AREA)} for {AREA:g} m² in today's prices.",
                       f"Ten sam budynek, to samo piętro: {str(s['area']).replace('.', ',')} m² sprzedano w {PL_MONTHS[s['date'].month - 1]} {s['date'].year} "
                       f"za {s['price'] / 1000:.0f} tys., dziś ok. {tys(s['today'] * AREA)} zł za {AREA:g} m²."))
    if twin is not None:
        notes.append(T(f"Renovated, a similar size: a {twin['area']:g} m² flat {int(twin['dist'])} m away was resold for {twin['price'] / 1000:.0f}K "
                       f"in {twin['date']:%b %Y}, months after its purchase. {district}: {flip_rate:.0f} flips in 100 resales (2021–25).",
                       f"Po remoncie, podobny metraż: {str(twin['area']).replace('.', ',')} m² {int(twin['dist'])} m stąd odsprzedano za "
                       f"{twin['price'] / 1000:.0f} tys. zł "
                       f"({twin['date'].month:02d}.{twin['date'].year}), kilka miesięcy po zakupie. {district}: {flip_rate:.0f} flipów na 100 "
                       "sprzedaży (2021–25)."))
    notes.append(T(f"The asking price to aim at: {r5(a_mid)}–{r5(a_hi)}K for a well-kept flat, {tys(r_lo)}+ only if it is freshly renovated.",
                   f"Cena do celowania: {r5(a_mid)}–{r5(a_hi)} tys. zł za zadbane mieszkanie, {tys(r_lo)}+ tylko po świeżym remoncie."))
    y = 0.75
    for n in notes:
        txt = wrap(n, 46)
        fig.text(0.57, y, "•", fontsize=11, color=style.ORANGE, va="top", fontweight="bold")
        fig.text(0.59, y, txt, fontsize=9.5, color=style.INK, va="top", linespacing=1.35)
        y -= 0.022 * (txt.count("\n") + 1) + 0.014

    # timeline --------------------------------------------------------------------------------
    a2 = fig.add_axes([0.11, 0.09, 0.85, 0.27])
    plain = near[near["area"].between(AREA - 15, AREA + 15) & ~near["flip"]]
    a2.scatter(plain["date"], plain["price_m2"], s=10, color="#b9b6ac", lw=0,
               label=T(f"resales nearby ({AREA - 15:g}–{AREA + 15:g} m²)", f"sprzedaże w okolicy ({AREA - 15:g}–{AREA + 15:g} m²)"))
    fl = near[near["flip"]]
    a2.scatter(fl["date"], fl["price_m2"], s=26, color=style.ORANGE, lw=0, label=T("flipped flats nearby", "odsprzedane mieszkania w okolicy"))
    a2.scatter(own["date"], own["price_m2"], s=46, color=style.INK, marker="D", lw=0, label=T("this building", "ten budynek"), zorder=5)
    m = plain.set_index("date")["price_m2"].resample("QS").median().rolling(3, min_periods=2, center=True).median()
    a2.plot(m.index, m.values, color=style.BLUE, lw=2, label=T("median nearby", "mediana w okolicy"))
    end = pd.Timestamp("2026-10-01")
    a2.fill_between([pd.Timestamp("2026-01-01"), end], a_lo / AREA, a_hi / AREA, color=style.BLUE, alpha=0.18, lw=0)
    a2.text(end, a_hi / AREA, T(" today,\n as is", " dziś,\n stan obecny"), fontsize=8, color=style.BLUE, va="top")
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: T(f"{v / 1000:.0f}K", f"{v / 1000:.0f} tys.")))
    a2.set_ylabel(T("PLN per m², as paid", "zł za m², ceny zapłacone"))
    style.hgrid_only(a2)
    a2.set_xlim(pd.Timestamp("2016-09-01"), pd.Timestamp("2027-06-01"))
    a2.legend(loc="upper left", fontsize=8.5, frameon=False, ncol=2)
    a2.set_title(T(f"Every resale within {R} m in {storeys}-storey blocks since 2016",
                   f"Wszystkie sprzedaże wtórne do {R} m w blokach {storeys}-kondygnacyjnych od 2016"), fontsize=10.5, loc="left")
    out = "output/report/pl" if LANG["code"] == "pl" else "output/report"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/flat_{slug(a.street)}_{slug(a.no)}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
