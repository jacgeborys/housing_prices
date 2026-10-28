"""
Summary charts: the story of the project in five portrait (4:5) figures, ready for a LinkedIn carousel.
Reads saved outputs only (no model is re-run):

  01_prices_over_time.png   constant-quality price per m², nominal vs real (price_index_quarterly.csv)
  02_districts.png          typical flat in 2025 by district (building_value)
  03_what_drives_price.png  within-district effects (price_drivers_coefficients.csv)
  (04_transit.png is made by transit_modes.py)
  23_blind_spot.png         flats in buildings with no / few recorded sales (coverage_by_district.csv)

-> output/summary/
"""
import os
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

style.apply()

OUT = "output/summary"
FIG = (8, 10)            # 4:5 portrait
DPI = 300                # -> 2400 x 3000 px (was 150 / 1200 x 1500 until 2026-10-28: viewers do not enlarge small images, so only the 300 dpi fig. 32 filled the screen)
REF_YEAR = 2025


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}"
    for attempt in range(6):            # Windows: an image viewer holding the file makes the write fail briefly
        try:
            fig.savefig(path, dpi=DPI)
            break
        except OSError:
            if attempt == 5:
                raise
            time.sleep(3)
    plt.close(fig)
    print(f"Saved {path}")


def page(title, subtitle, footer):
    fig = plt.figure(figsize=FIG)
    style.header(fig, title, subtitle, y=0.985)
    style.footer(fig, wrap(footer.replace("\n", " "), 138), y=0.012)
    return fig


def wrap(text, width):
    import textwrap
    return "\n".join(textwrap.wrap(text, width))


# 1 --------------------------------------------------------------------------------------------
def prices_over_time():
    q = pd.read_csv("output/analysis/price_index_quarterly.csv")
    q["year"] = q["quarter"].str[:4].astype(int)
    q["t"] = q["year"] + (q["quarter"].str[-1].astype(int) - 0.5) / 4
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    level = np.average(bv["value_2025"], weights=bv["n"])
    cpi = style.cpi_level(REF_YEAR)
    q["nominal"] = level * np.exp(q["gamma"])
    q["real"] = q["nominal"] / q["year"].map(cpi)
    est = q["estimated"].astype(bool)

    n13 = q.loc[q["year"] == 2013, "nominal"].mean()
    n25 = q.loc[q["year"] == REF_YEAR, "nominal"].mean()
    r08 = q.loc[est & (q["year"] <= 2012), "real"].max()
    r25 = q.loc[q["year"] == REF_YEAR, "real"].mean()

    fig = page("Warsaw flats: 2.2× dearer than in 2013,\nbut no dearer than in 2008",
               "", "Constant-quality index: the same kind of flat each quarter (hedonic model, 258K sales, "
                   "within-building comparisons).\nReal = deflated with GUS CPI to 2025 PLN. Dashed: RCN gap "
                   "2017–20, interpolated. " + style.SOURCE)
    ax = fig.add_axes([0.13, 0.17, 0.80, 0.62])
    for col, colr, lab in (("nominal", style.BLUE, "Price as paid (nominal)"),
                           ("real", style.ORANGE, "In today's money (real, 2025 PLN)")):
        ax.plot(q["t"], q[col].where(est), color=colr, lw=2.4, label=lab)
        ax.annotate(lab, (2009.5, q.loc[q["year"] == 2010, col].mean()), xytext=(0, 12 if col == "real" else -18),
                    textcoords="offset points", fontsize=10, color=colr, fontweight="semibold")
        gap = ~est & (q["year"] >= 2016)
        ax.plot(q["t"], q[col].where(gap | gap.shift(1, fill_value=False) | gap.shift(-1, fill_value=False)),
                color=colr, lw=1.4, ls=(0, (2, 2)), alpha=0.7)
    style.shade_gap(ax)
    style.pln(ax)
    style.hgrid_only(ax)
    ax.set_ylabel("PLN per m², typical flat")
    ax.set_xlim(2006, 2026.6)
    ax.set_ylim(0, None)
    ax.set_xticks(range(2006, 2027, 4))
    ax.legend(loc="lower right")
    last = q[est].iloc[-1]
    pk = q["real"].where(est & (q["year"] <= 2012)).idxmax()
    ax.annotate(f"2008 peak {style.fmt_int(r08)}", (q.loc[pk, "t"], r08), xytext=(12, 2),
                textcoords="offset points", fontsize=9.5, color=style.INK_2, va="bottom")
    for col in ("nominal", "real"):
        ax.annotate(style.fmt_int(last[col]), (last["t"], last[col]), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=10, color=style.INK_2)

    # hero numbers
    for x, big, small in ((0.05, f"×{n25 / n13:.1f}", "nominal price\n2013 → 2025"),
                          (0.40, "≈ 2008", f"in today's money: {100 * (r25 / r08 - 1):+.0f}%\nvs the 2008 peak"),
                          (0.72, style.fmt_int(n25), "PLN/m², typical\nflat in 2025")):
        fig.text(x, 0.885, big, fontsize=30, fontweight="bold", color=style.INK, va="top")
        fig.text(x, 0.835, small, fontsize=10.5, color=style.INK_2, va="top")
    fig.text(0.05, 0.105, wrap("Raw medians mix cheap old flats with new ones and studios with family flats. "
                               "The index compares like with like: the same buildings, floors and sizes over time.",
                               95), fontsize=10.5, color=style.INK_2, va="top")
    save(fig, "01_prices_over_time.png")


# 2 --------------------------------------------------------------------------------------------
def districts():
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    adj = bv.dropna(subset=["district"]).groupby("district").apply(
        lambda s: np.exp(np.average(np.log(s["value_2025"]), weights=s["n"])))
    d = pd.DataFrame({"adj": adj}).dropna().sort_values("adj")
    city = np.exp(np.average(np.log(bv["value_2025"]), weights=bv["n"]))

    fig = page("Where a typical flat costs most",
               "Same kind of flat in every district, priced in 2025",
               "Typical flat = hedonic value of the buildings in the district (sales-weighted), with date, "
               "floor, size and new-build effects removed, so districts compare like with like. "
               + style.SOURCE)
    ax = fig.add_axes([0.25, 0.09, 0.66, 0.77])
    y = np.arange(len(d))
    ax.barh(y, d["adj"], height=0.68, color=style.BLUE)
    for yi, v in zip(y, d["adj"]):
        ax.text(v + 250, yi, style.fmt_int(v), va="center", fontsize=10, color=style.INK_2, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc=style.SURFACE, ec="none"))
    ax.axvline(city, color=style.MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(city, -0.9, f" city average {style.fmt_int(city)}", fontsize=9, color=style.MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(d.index, fontsize=10.5)
    ax.set_xlim(0, d["adj"].max() * 1.13)
    style.pln(ax, "x")
    style.vgrid_only(ax)
    ax.set_xlabel("PLN per m²")
    ax.set_ylim(-1.3, len(d) - 0.4)
    save(fig, "02_districts.png")


# 3 --------------------------------------------------------------------------------------------
PLAIN = [  # feature, plain-language label, sign flip (effect of the "worse" direction reads naturally)
    ("pct_primary", "New building (sold by the developer)"),
    ("age_2001_10", "Built 2001–2010 (vs a 1961–90 prefab block)"),
    ("age_1941_60", "Built 1941–1960 (vs a 1961–90 prefab block)"),
    ("canopy_100", "10 points more tree cover within 100 m"),
    ("walk7_food_drink", "Twice as many cafés, restaurants and bars within a 7-min walk"),
    ("walk7_zabka", "Twice as many Żabka shops within a 7-min walk"),
    ("green_ha_500", "10 ha more park or forest within 500 m"),
    ("log_footprint", "Twice the building footprint"),
    ("log2_trips", "Twice as many bus and tram departures"),
    ("centre_km", "Each km farther from the centre"),
    ("height_m", "Building 10 m taller"),
    ("walk_metro_km", "Each km farther to walk to the metro"),
    ("noise_rail_db", "Tram & train noise 10 dB louder"),
    ("noise_road_db", "Road noise 10 dB louder"),
]


def drivers():
    c = pd.read_csv("output/analysis/price_drivers_coefficients.csv").set_index("feature")
    rows = [(lab, *c.loc[f, ["pct_district", "pct_district_lo", "pct_district_hi"]]) for f, lab in PLAIN]
    d = pd.DataFrame(rows, columns=["label", "pct", "lo", "hi"]).sort_values("pct")
    gv = gpd.read_file("output/price_drivers.gpkg", ignore_geometry=True, columns=["osm_id"])
    r2 = pd.read_csv("output/analysis/price_drivers_scores.csv", index_col=0)["r2"]

    fig = page("What moves the price of a Warsaw flat",
               "Effect of each feature, comparing buildings within the same district",
               f"{len(gv):,} buildings with ≥ 3 flat sales; typical flat (date, floor, size removed). Regression with district effects, "
               f"bars = 95 % CI; the features explain {r2['gbm']:.0%} of price differences out of sample. " + style.SOURCE)
    ax = fig.add_axes([0.47, 0.13, 0.46, 0.74])
    lim = 8.0
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        colr = style.BLUE if r["pct"] >= 0 else style.ORANGE
        shown = np.clip(r["pct"], -lim, lim)
        ax.barh(yi, shown, height=0.62, color=colr)
        if abs(r["pct"]) <= lim:
            ax.plot([r["lo"], min(r["hi"], lim)], [yi, yi], color=style.INK, lw=1.2)
        if abs(r["pct"]) > lim:     # off scale: value inside the bar
            ax.text(shown - 0.3, yi, f"{r['pct']:+.0f}%  (off scale)", va="center", ha="right", fontsize=10.5,
                    color="white", fontweight="bold")
            continue
        ax.text(shown + (0.25 if r["pct"] >= 0 else -0.25), yi + 0.33, f"{r['pct']:+.1f}%", va="center",
                ha="left" if r["pct"] >= 0 else "right", fontsize=10, color=style.INK, fontweight="semibold")
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap(t, 30) for t in d["label"]], fontsize=10.5)
    ax.set_xlim(-lim, lim)
    style.pct(ax, "x")
    style.vgrid_only(ax)
    ax.set_xlabel("Change in price per m²")
    save(fig, "03_what_drives_price.png")


# 5 --------------------------------------------------------------------------------------------
def blind_spot():
    s = pd.read_csv("output/analysis/coverage_by_district.csv").set_index("district")
    s = s.sort_values("share_none")
    tot = s[["none", "thin", "ok"]].sum()
    none_share = tot["none"] / tot.sum()

    fig = page("The price register's blind spot",
               wrap("Cooperative-right sales (spółdzielcze własnościowe prawo) never enter RCN, "
                    "and rented city blocks rarely sell", 80),
               "Flats estimated from building footprint × storeys (BDOT10k / OSM). Few sales = under 25 % of the city's usual sales per flat. "
               "Only flats in full ownership are registered. " + style.SOURCE)
    fig.text(0.05, 0.885, f"1 in {round(1 / none_share)}", fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.835, "flats sits in a building with\nno recorded sale at all", fontsize=10.5,
             color=style.INK_2, va="top")
    fig.text(0.47, 0.885, f"{tot['thin'] / tot.sum():.0%}", fontsize=30, fontweight="bold", va="top")
    fig.text(0.47, 0.835, "more are in buildings with\nonly a handful of sales", fontsize=10.5,
             color=style.INK_2, va="top")
    ax = fig.add_axes([0.27, 0.10, 0.66, 0.66])
    y = np.arange(len(s))
    left = np.zeros(len(s))
    for col, colr, lab in (("share_none", "#104281", "No recorded sale"),
                           ("share_thin", "#3987e5", "Few sales"),
                           ("share_ok", "#d6e4f5", "Covered")):
        ax.barh(y, s[col] * 100, left=left, height=0.7, color=colr, edgecolor=style.SURFACE, lw=1.5, label=lab)
        left += s[col].values * 100
    for yi, v in zip(y, s["share_none"] * 100):
        ax.text(v / 2, yi, f"{v:.0f}%", ha="center", va="center", fontsize=9, color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(s.index, fontsize=10.5)
    ax.set_xlim(0, 100)
    style.pct(ax, "x", signed=False)
    style.vgrid_only(ax)
    ax.set_xlabel("Share of the district's flats")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.07), ncol=3, fontsize=9.5)
    save(fig, "23_blind_spot.png")


def main():
    prices_over_time()
    districts()
    drivers()
    blind_spot()


if __name__ == "__main__":
    main()
