"""
How many flats sell, quarter by quarter: Lublin as a whole (second-hand vs new from a developer) and round one address
(its district and a 500 m circle, second-hand). Answers "is it harder to sell because fewer flats change hands?".
Run from lublin/ (the address is an argument, so no address lives in the code or git):

    python volume_slide.py --street "<street>" --no <number> [--lang pl]
 -> output/report[/pl]/volume_<street>_<no>.png

Counts are deeds in transactions_adj (one flat = one sale). The last quarter of data is incomplete (deeds reach the
register weeks to months late) and is drawn hatched; the headline compares January-June of the latest two years.
"""
import argparse
import os
import re

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from common import LANG, T, fmt, page, set_lang, style, wrap
from flat_slide import slug

R = 500
START = pd.Period("2018Q1")
NEW = style.AQUA          # new from a developer, as on the Warsaw flat page
SEC = "#9a978d"           # second-hand
NEAR = style.INK


def bars(ax, s, color, partial, width=0.8, label=None, zorder=2):
    x = range(len(s))
    for i, (q, v) in zip(x, s.items()):
        last = q == partial
        ax.bar(i, v, width=width, color=color, alpha=0.35 if last else 1, lw=0, zorder=zorder,
               hatch="////" if last else None, label=label if i == 0 else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--street", required=True)
    ap.add_argument("--no", required=True)
    ap.add_argument("--lang", default="en")
    a = ap.parse_args()
    set_lang(a.lang)
    r = gpd.read_file("data/rcn_lokale_lublin.gpkg", columns=["LOK_ADRES"]).to_crs(2180)
    hit = r[r["LOK_ADRES"].fillna("").str.contains(f"UL:[^;]*{re.escape(a.street)};NR_PORZ:{re.escape(a.no)}$", case=False)]
    if hit.empty:
        raise SystemExit(f"no register address '{a.street} {a.no}'")
    pt = hit.geometry.iloc[0]
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t["date"] = pd.to_datetime(t["date"])
    t = t[t["date"].dt.to_period("Q") >= START]
    t["q"] = t["date"].dt.to_period("Q")
    t["dist"] = t.distance(pt)
    district = t.loc[t["dist"].idxmin(), "district"]
    last_day = t["date"].max()
    partial = last_day.to_period("Q")
    qs = pd.period_range(START, partial, freq="Q")
    sec = t[t["market"] == "secondary"]
    city_sec = sec.groupby("q").size().reindex(qs, fill_value=0)
    city_new = t[t["market"] == "primary"].groupby("q").size().reindex(qs, fill_value=0)
    dist_sec = sec[sec["district"] == district].groupby("q").size().reindex(qs, fill_value=0)
    near_sec = sec[sec["dist"] < R].groupby("q").size().reindex(qs, fill_value=0)

    y1 = last_day.year if last_day.month > 6 else last_day.year - 1      # latest year with a complete January-June
    h1 = lambda d, y: ((d["date"].dt.year == y) & (d["date"].dt.month <= 6)).sum()  # noqa: E731
    chg = lambda d: 100 * (h1(d, y1) / h1(d, y1 - 1) - 1)  # noqa: E731
    c_city, c_dist, c_near = chg(sec), chg(sec[sec["district"] == district]), chg(sec[sec["dist"] < R])
    c_new = chg(t[t["market"] == "primary"])
    print(f"Jan-Jun {y1} vs {y1 - 1}: second-hand city {c_city:+.0f} %, {district} {c_dist:+.0f} %, {R} m {c_near:+.0f} %, "
          f"new city {c_new:+.0f} %")

    def pct(v):
        s = f"{v:+.0f} %"
        return s.replace("-", "−")

    more = c_city > -5
    fig = page(T("Second-hand flats are not selling less" if more else "Fewer second-hand flats change hands",
                 "Mieszkań z drugiej ręki nie sprzedaje się mniej" if more else "Mieszkań z drugiej ręki sprzedaje się mniej"),
               T(f"Flat sales per quarter: Lublin, and round {a.street} {a.no} ({district})",
                 f"Sprzedaże mieszkań w kwartale: Lublin i okolica {a.street} {a.no} ({district})"),
               T(f"Deeds in the price register, one flat = one sale. Hatched: the register so far ({last_day:%d %b %Y}); deeds arrive "
                 "late, the quarter will fill up. Source: GUGiK RCN.",
                 f"Akty w rejestrze cen, jedno mieszkanie = jedna sprzedaż. Kreskowane: dotychczasowe dane (do "
                 f"{last_day:%d.%m.%Y}); akty spływają z opóźnieniem, kwartał się jeszcze zapełni. Źródło: GUGiK RCN."))
    near = sec[sec["dist"] < R]
    n0, n1 = h1(near, y1 - 1), h1(near, y1)
    peak = city_new[(city_new.index.year >= y1 - 3) & (city_new.index.year <= y1 - 2)].mean()
    now_new = city_new[(city_new.index.year == y1) & (city_new.index.quarter <= 2)].mean()
    fig.text(0.03, 0.885, wrap(T(
        f"January–June {y1} against the same months of {y1 - 1}: second-hand sales in Lublin {pct(c_city)}, in {district} "
        f"{pct(c_dist)}, within {R} m {n0} → {n1} (few sales, within normal swings). New flats from developers {pct(c_new)}, "
        f"still {abs(100 * (now_new / peak - 1)):.0f} % below their {y1 - 3}–{str(y1 - 2)[2:]} level.",
        f"Styczeń–czerwiec {y1} wobec tych samych miesięcy {y1 - 1}: sprzedaże z drugiej ręki w Lublinie {pct(c_city)}, "
        f"{district} {pct(c_dist)}, do {R} m {n0} → {n1} (mało transakcji, zwykłe wahania). Nowe od deweloperów {pct(c_new)}, "
        f"wciąż o {abs(100 * (now_new / peak - 1)):.0f} % poniżej poziomu z lat {y1 - 3}–{str(y1 - 2)[2:]}."), 100),
        fontsize=10.5, fontweight="semibold", va="top")

    ticks = [i for i, q in enumerate(qs) if q.quarter == 1]

    # panel 1: Lublin -------------------------------------------------------------------------
    ax = fig.add_axes([0.09, 0.50, 0.87, 0.30])
    bars(ax, city_sec, SEC, partial, label=T("second-hand", "z drugiej ręki"))
    full = [i for i, q in enumerate(qs) if q != partial]
    ax.plot(full, city_new.iloc[full].values, color=NEW, lw=2.2, marker="o", ms=4, zorder=3,
            label=T("new, from a developer", "nowe, od dewelopera"))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(qs[i].year) for i in ticks])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt(v)))
    style.hgrid_only(ax)
    ax.set_xlim(-0.7, len(qs) - 0.3)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ax.set_title(T("Lublin: flats sold per quarter", "Lublin: mieszkania sprzedane w kwartale"), fontsize=10.5, loc="left")
    ax.set_ylim(0, max(city_sec.max(), city_new.max()) * 1.18)

    # panel 2: the district and 500 m --------------------------------------------------------
    a2 = fig.add_axes([0.09, 0.09, 0.87, 0.30])
    bars(a2, dist_sec, "#cfccc2", partial, label=T(f"{district}, whole district", f"{district}, cała dzielnica"))
    bars(a2, near_sec, NEAR, partial, width=0.5, label=T(f"within {R} m", f"do {R} m"), zorder=3)
    a2.set_xticks(ticks)
    a2.set_xticklabels([str(qs[i].year) for i in ticks])
    style.hgrid_only(a2)
    a2.set_xlim(-0.7, len(qs) - 0.3)
    a2.set_ylim(0, dist_sec.max() * 1.25)
    a2.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    a2.set_title(T(f"Round {a.street} {a.no}: second-hand flats sold per quarter",
                   f"Okolica {a.street} {a.no}: mieszkania z drugiej ręki sprzedane w kwartale"), fontsize=10.5, loc="left")

    out = "output/report/pl" if LANG["code"] == "pl" else "output/report"
    os.makedirs(out, exist_ok=True)
    path = f"{out}/volume_{slug(a.street)}_{slug(a.no)}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
