"""
Shared bits of the Lublin slides: language switch, page helpers, the light basemap, typical-flat values with districts.

Language: every visible string goes through T(en, pl). The language is set once per run with set_lang("pl") (report.py
--lang pl); slides then land in output/report/src_pl/ instead of output/report/src/.
"""
import os
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import geopandas as gpd  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import style  # noqa: E402
from summary_charts import page, wrap  # noqa: E402,F401

style.apply()
LANG = {"code": "en"}
LABELS = ["Śródmieście", "Czuby Północne", "Rury", "Sławin", "Czechów Północny",
          "Kalinowszczyzna", "Tatary", "Bronowice", "Wrotków", "Węglin Południowy", "Felin", "Ponikwoda", "Dziesiąta",
          "Zemborzyce", "Hajdów-Zadębie", "Abramowice", "Głusk", "Szerokie"]
SHORT = {"Czuby Północne": "Czuby", "Czechów Północny": "Czechów", "Węglin Południowy": "Węglin",
         "Hajdów-Zadębie": "Hajdów"}


def set_lang(code):
    LANG["code"] = code


def T(en, pl):
    return pl if LANG["code"] == "pl" else en


def src_dir():
    return "output/report/src_pl" if LANG["code"] == "pl" else "output/report/src"


def source(extra=""):
    return T("Source: GUGiK RCN (price register), BDOT10k", "Źródło: GUGiK RCN (rejestr cen), BDOT10k") + extra + "."


def save(fig, name):
    os.makedirs(src_dir(), exist_ok=True)
    fig.savefig(f"{src_dir()}/{name}", dpi=300)
    plt.close(fig)
    print(f"Saved {src_dir()}/{name}")


def headline(fig, text, y=0.885):
    fig.text(0.03, y, wrap(text, 100), fontsize=10.5, fontweight="semibold", va="top")


def fmt(v):
    """Thousands with a thin space in Polish, a comma in English."""
    s = f"{v:,.0f}"
    return s.replace(",", "\u202f") if LANG["code"] == "pl" else s


def pct(v, signed=True):
    if round(v) == 0:
        return "0 %"
    s = f"{v:+.0f}" if signed else f"{v:.0f}"
    return s.replace("-", "−") + " %"


def districts():
    return gpd.read_file("data/districts.gpkg").to_crs(2180)


def water(city):
    w = gpd.read_file("D:/QGIS/osm_basemap/lublin/water.gpkg").to_crs(2180)
    w = w[w["water"].isin(["river", "lake", "reservoir"]) & (w.area > 2e4)]
    return w.clip(city.buffer(500))


def basemap(fig, d, rect=(0.03, 0.40, 0.94, 0.45), labels=True, fill="#ffffff"):
    """City districts on a light page, water in pale blue; returns the axes (EPSG:2180)."""
    ax = fig.add_axes(rect)
    city = d.unary_union
    gpd.GeoSeries([city], crs=2180).plot(ax=ax, color=fill, lw=0, zorder=0)
    water(city).plot(ax=ax, color="#cfdcec", lw=0, zorder=1)
    d.boundary.plot(ax=ax, color="#55534e", lw=0.5, zorder=6)
    x0, y0, x1, y1 = city.bounds
    ax.set_xlim(x0 - 300, x1 + 300)
    ax.set_ylim(y0 - 300, y1 + 300)
    ax.set_aspect("equal")
    ax.axis("off")
    if labels:
        reps = d.set_index("district").geometry.representative_point()
        for k in LABELS:
            if k in reps.index:
                ax.text(reps[k].x, reps[k].y, SHORT.get(k, k), fontsize=7.5, ha="center", va="center", zorder=20,
                        color=style.INK, path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])
    return ax


def building_values():
    """Typical-flat value per building with its district, and the value against the district's median building."""
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value")
    bv["rel"] = 100 * (bv["value_2025"] / bv.groupby("district")["value_2025"].transform("median") - 1)
    return bv


def rel_bars(ax, groups, rel, labels, xlabel, n=None):
    """Bars of the median 'against the district' by group (descriptive, no model)."""
    x = np.arange(len(labels))
    med = [rel[groups == g].median() if (groups == g).sum() >= 15 else np.nan for g in labels]
    cnt = [int((groups == g).sum()) for g in labels]
    ax.bar(x, np.nan_to_num(med), color=[style.BLUE if (v or 0) >= 0 else style.ORANGE for v in med], width=0.62)
    ax.axhline(0, color=style.AXIS, lw=1)
    for xi, v, c in zip(x, med, cnt):
        if np.isfinite(v):
            ax.text(xi, v + (0.35 if v >= 0 else -0.35), pct(v), ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=9.5)
        ax.text(xi, 0, "", fontsize=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n({c:,} {T('bldgs', 'bud.')})".replace(",", "\u202f" if LANG["code"] == "pl" else ",")
                        for l, c in zip(labels, cnt)], fontsize=8.5)
    lim = np.nanmax(np.abs(med)) * 1.5 if np.isfinite(np.nanmax(np.abs(med))) else 5
    ax.set_ylim(-max(lim, 4), max(lim, 4))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: pct(v)))
    style.hgrid_only(ax)
    ax.set_xlabel(xlabel, fontsize=9.5)
    return med
