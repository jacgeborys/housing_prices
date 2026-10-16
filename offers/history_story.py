"""
Asking prices over a year (developer price lists, monthly snapshots Sep 2025 - Oct 2026; parse_price_history.py),
told in plain terms:

  1  what happened to each flat's asking price while it was listed: unchanged / pricier / cheaper
  2  the asking price of the same flats month by month, vs September 2025 (y axis +-5 %, so small moves look small)
  3  how long a newly listed flat stays for sale (share still listed after 3, 6, 9, 12 months); a flat leaves when
     it disappears while its developer keeps publishing (sold or withdrawn); Kaplan-Meier, flats listed before
     October 2025 left out because their start is unknown
  4  the investments with the biggest cuts and raises (median per flat, >= 10 flats followed)

Output: output/summary/27_asking_prices_over_time.png
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import style  # noqa: E402
from summary_charts import page, save, wrap  # noqa: E402

style.apply()
START = "2025-09"
COL = {"unchanged": "#c9c7bf", "pricier": "#c4521f", "cheaper": style.BLUE}
MONTH_PL = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def short_dev(s):
    s = re.sub(r"\s+(sp\.|spółka|s\.a\.|sa\b|sp z o\.o\.).*$", "", str(s), flags=re.I).strip()
    return s.title() if s.isupper() else s


def main():
    h = pd.read_csv(os.path.join(HERE, "data", "history_flats.csv"), low_memory=False)
    h = h[h["month"] >= START].copy()
    months = sorted(h["month"].unique())
    mi = {m: i for i, m in enumerate(months)}
    h["t"] = h["month"].map(mi)
    h = h.sort_values(["flat", "t"])
    h["lp"] = np.log(h["price_m2"])

    # 1 per flat: first vs last asking price (followed >= 3 months)
    g = h.groupby("flat").agg(first=("price_m2", "first"), last=("price_m2", "last"), n=("t", "nunique"),
                              developer=("developer", "first"), street=("inv_street", "first"))
    g = g[g["n"] >= 3]
    g["chg"] = 100 * (g["last"] / g["first"] - 1)
    g["kind"] = np.select([g["chg"] > 0.5, g["chg"] < -0.5], ["pricier", "cheaper"], "unchanged")
    share = g["kind"].value_counts(normalize=True).reindex(["unchanged", "pricier", "cheaper"]).fillna(0)
    typ = g.groupby("kind")["chg"].median()

    # 2 same flats month by month
    nxt = h.groupby("flat").shift(-1)
    step = h.assign(t2=nxt["t"], lp2=nxt["lp"]).dropna(subset=["t2"])
    step = step[step["t2"] == step["t"] + 1]
    chg = step.assign(d=step["lp2"] - step["lp"]).groupby("t2")["d"].mean().reindex(range(1, len(months))).fillna(0)
    idx = 100 * (np.exp(np.r_[0, chg.cumsum().to_numpy()]) - 1)

    # 3 time on the list
    dev_last = h.groupby("developer")["t"].max()
    g2 = h.groupby("flat").agg(t0=("t", "first"), t1=("t", "last"), developer=("developer", "first"))
    g2 = g2[g2["t0"] > 0]
    g2["dur"] = g2["t1"] - g2["t0"] + 1
    g2["event"] = (g2["t1"] < g2["developer"].map(dev_last)).astype(int)
    surv, s = {}, 1.0
    for k in range(1, int(g2["dur"].max()) + 1):
        at_risk = (g2["dur"] >= k).sum()
        ev = ((g2["dur"] == k) & (g2["event"] == 1)).sum()
        s *= 1 - ev / at_risk if at_risk else 1
        surv[k] = s
    marks = [m for m in (3, 6, 9, 12) if m in surv]

    # 4 investments
    g["inv"] = g["developer"].map(short_dev) + ", " + g["street"].astype(str).str.replace(r"^(ul\.|ulica)\s*", "",
                                                                                          regex=True)
    inv = g.groupby("inv")["chg"].agg(["median", "size"])
    inv = inv[(inv["size"] >= 10) & ~inv.index.str.contains(r", (X|nan|None)$")].sort_values("median")
    cuts, raises = inv.head(5), inv.tail(5)

    fig = page("Developers barely move their prices",
               "A year of developers' published price lists for new flats in Warsaw",
               f"Developers' price lists on dane.gov.pl (2025 price-transparency law), one snapshot per month, {months[0]} "
               f"to {months[-1]}; about a third of Warsaw developers keep their history. A flat counts if it was listed "
               "in at least 3 snapshots. 'For sale' ends when a flat disappears while its developer keeps publishing "
               "(sold or withdrawn). " + style.SOURCE)
    # 1 stacked bar
    a1 = fig.add_axes([0.05, 0.72, 0.90, 0.07])
    left = 0
    for k in ("unchanged", "pricier", "cheaper"):
        w = share[k] * 100
        a1.barh(0, w, left=left, color=COL[k], height=0.8, edgecolor=style.SURFACE, lw=2)
        txt = {"unchanged": f"{w:.0f}%  price never changed",
               "pricier": f"{w:.0f}%  got pricier\n(typically {typ.get('pricier', 0):+.0f}%)",
               "cheaper": f"{w:.0f}%  cheaper\n({typ.get('cheaper', 0):+.0f}%)"}[k]
        if w > 15:
            a1.text(left + w / 2, 0, txt, ha="center", va="center", fontsize=10,
                    color="white" if k != "unchanged" else style.INK, fontweight="semibold")
        else:                                             # narrow segment: label underneath
            a1.text(left + w, -0.55, txt, ha="right", va="top", fontsize=9.5, color=COL[k],
                    fontweight="semibold")
        left += w
    a1.set_xlim(0, 100)
    a1.axis("off")
    a1.set_title(f"What happened to the asking price of {style.fmt_int(len(g))} flats while they were for sale",
                 fontsize=11, loc="left")

    # 2 same flats over time
    a2 = fig.add_axes([0.09, 0.40, 0.38, 0.22])
    x = np.arange(len(months))
    a2.plot(x, idx, "-o", color="#c4521f", lw=2.4, ms=5, mec=style.SURFACE)
    a2.axhline(0, color=style.AXIS, lw=1)
    a2.set_ylim(-5, 5)
    a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    a2.set_xticks(x[::3])
    a2.set_xticklabels([f"{MONTH_PL[int(m[5:]) - 1]} {m[2:4]}" for m in months][::3], fontsize=9)
    style.hgrid_only(a2)
    a2.annotate(f"{idx[-1]:+.1f}%", (x[-1], idx[-1]), xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=10, fontweight="semibold", color="#c4521f")
    a2.set_title("Asking price of the same flats,\nvs September 2025", fontsize=10.5, loc="left")

    # 3 time for sale
    a3 = fig.add_axes([0.59, 0.40, 0.37, 0.22])
    vals = [100 * surv[m] for m in marks]
    a3.bar(range(len(marks)), vals, color=style.BLUE, width=0.6)
    for i, v in enumerate(vals):
        a3.text(i, v + 2, f"{v:.0f}%", ha="center", fontsize=10, fontweight="semibold", color=style.INK)
    a3.set_xticks(range(len(marks)))
    a3.set_xticklabels([f"{m} months" for m in marks], fontsize=9.5)
    a3.set_ylim(0, 110)
    a3.set_yticks([])
    style.hgrid_only(a3)
    a3.set_title("New flats still for sale after…", fontsize=10.5, loc="left")

    # 4 investments
    a4 = fig.add_axes([0.40, 0.10, 0.56, 0.22])
    sel = pd.concat([cuts, raises])
    yy = np.arange(len(sel))
    a4.barh(yy, sel["median"], color=[COL["cheaper"] if v < 0 else COL["pricier"] for v in sel["median"]],
            height=0.65)
    for yi, v in zip(yy, sel["median"]):
        a4.text(v + (0.4 if v >= 0 else -0.4), yi, f"{v:+.0f}%", va="center", ha="left" if v >= 0 else "right",
                fontsize=8.5, color=style.INK)
    a4.axvline(0, color=style.INK_2, lw=1)
    a4.set_yticks(yy)
    a4.set_yticklabels(sel.index, fontsize=8.5)
    lim = max(abs(sel["median"]).max() * 1.3, 5)
    a4.set_xlim(-lim, lim)
    a4.set_xticks([])
    a4.grid(False)
    for sp in ("left", "bottom"):
        a4.spines[sp].set_visible(False)
    a4.set_title("Projects with the biggest price cuts and raises (typical flat)", fontsize=10.5, loc="left")

    fig.text(0.03, 0.885, wrap(f"Over a year, the same new flats ended up {idx[-1]:+.1f}% pricier on average: "
                               f"{share['unchanged']:.0%} never changed price. Flats sell slowly: a year after "
                               f"being listed, {surv.get(12, surv[max(surv)]):.0%} are still for sale.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "27_asking_prices_over_time.png")
    print(share.round(3).to_dict(), typ.round(1).to_dict(), {m: round(surv[m], 2) for m in marks})


if __name__ == "__main__":
    main()
