"""
Shared visual style for all charts and maps.

One palette, one typeface, one layout grammar:
  - left-aligned title + a one-line takeaway subtitle (header())
  - a source/method footnote (footer())
  - hairline solid gridlines, no top/right spines
  - price magnitude = PRICE_CMAP (pale gold -> magenta -> plum), used everywhere
  - change / polarity = DIV_CMAP (blue <-> red, neutral gray midpoint)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --- Ink & surfaces ---------------------------------------------------------
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
CONTEXT = "#d9d7cf"      # background buildings / non-highlighted series
CONTEXT_DARK = "#2c2c2a"

# --- Categorical (fixed order, never cycled) --------------------------------
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CAT

# --- Price magnitude ---------------------------------------------------------
# A trimmed slice of magma: every step stays mid-light, so it reads on both the
# light chart surface and the dark 3D ground. Cheap = pale gold, dear = plum.
PRICE_CMAP = ListedColormap(plt.cm.magma_r(np.linspace(0.10, 0.72, 256)), name="price")

# --- Polarity / change -------------------------------------------------------
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f4b9b3", "#e34948", "#8f1d1d"])

CPI_NOTE = "Real prices deflated with GUS annual CPI (2025 PLN)."
SOURCE = "Source: GUGiK Rejestr Cen Nieruchomości (RCN), OSM, BDOT10k."


def apply():
    """Set global rcParams. Call once at import time of any plotting script."""
    mpl.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10.5,
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.linestyle": "-",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "text.color": INK,
    })


def header(fig, title, subtitle=None, y=0.995):
    """Left-aligned figure title with a takeaway subtitle underneath."""
    fig.text(0.01, y, title, fontsize=17, fontweight="bold", color=INK, va="top", ha="left")
    if subtitle:
        gap = 0.36 / fig.get_figheight()      # constant ~0.36 in, whatever the figure size
        fig.text(0.01, y - gap, subtitle, fontsize=11.5, color=INK_2, va="top", ha="left")


def footer(fig, text=SOURCE, y=0.005):
    n = text.count("\n") + 1
    if n > 2:                       # footnotes: two lines at most (2026-10-28), the rest belongs on the methods page
        print(f"LONG FOOTNOTE ({n} lines): {text[:70]!r}")
    fig.text(0.01, y, text, fontsize=8.5, color=MUTED, va="bottom", ha="left")


def pln(ax, axis="y"):
    """Thousands-separated PLN tick labels."""
    f = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def pct(ax, axis="y", signed=True):
    f = mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%" if signed else f"{v:.0f}%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(f)


def hgrid_only(ax):
    ax.grid(axis="x", visible=False)


def vgrid_only(ax):
    ax.grid(axis="y", visible=False)


def fmt_int(v):
    return f"{v:,.0f}".replace(",", " ")


def shade_gap(ax, lo=2016.5, hi=2020.5, label="RCN gap\n2017–20"):
    ax.axvspan(lo, hi, color=GRID, alpha=0.55, lw=0, zorder=0)
    ax.text((lo + hi) / 2, 0.97, label, transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=8.5, color=MUTED, style="italic")


def colorbar(fig, mappable, ax, label, fmt="pln", **kw):
    kw.setdefault("fraction", 0.03)
    kw.setdefault("pad", 0.01)
    kw.setdefault("shrink", 0.55)
    cb = fig.colorbar(mappable, ax=ax, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=9, labelcolor=INK_2)
    cb.set_label(label, color=INK_2, fontsize=10)
    if fmt == "pln":
        cb.ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    elif fmt == "pct":
        cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    return cb


# Polish annual CPI, previous year = 100 (GUS "średnioroczny wskaźnik cen").
# 2025 ≈ 3.6 per GUS monthly prints (2.8–4.9% y/y through the year);
# 2026 is a year-to-date estimate — only ~5K transactions fall in 2026 anyway.
CPI_YOY = {
    2006: 1.0, 2007: 2.5, 2008: 4.2, 2009: 3.5, 2010: 2.6, 2011: 4.3, 2012: 3.7,
    2013: 0.9, 2014: 0.0, 2015: -0.9, 2016: -0.6, 2017: 2.0, 2018: 1.6, 2019: 2.3,
    2020: 3.4, 2021: 5.1, 2022: 14.4, 2023: 11.4, 2024: 3.6, 2025: 3.6, 2026: 2.8,
}


def cpi_level(base_year=2025):
    """Annual CPI level with base_year = 1.0."""
    years = sorted(CPI_YOY)
    lvl, acc = {}, 1.0
    for y in years:
        acc *= 1 + CPI_YOY[y] / 100
        lvl[y] = acc
    b = lvl[base_year]
    return {y: v / b for y, v in lvl.items()}
