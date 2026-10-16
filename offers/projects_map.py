"""
Full-page map of every new-build project on sale in Warsaw (offers/data/investments.gpkg from offers_story.py):
circle size = flats on offer, colour = median asking price per m²; the N_LABEL largest projects are named in two
callout columns at the map's sides (west / east of the centre), sorted top to bottom so the leader lines never cross.

Output: output/summary/28_projects_map.png
"""
import os
import re
import sys

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.dont_write_bytecode = True
import style  # noqa: E402
from summary_charts import page, save, wrap  # noqa: E402
from summary_maps import load  # noqa: E402

style.apply()
N_LABEL = 20
CMAP = ListedColormap(["#fde4cf", "#f9b98a", "#f08a4b", "#d4602a", "#a33f14", "#6e2508"])
BOUNDS = [0, 13000, 15000, 17000, 20000, 25000, 1e6]


def short(dev, street):
    d = re.sub(r"\s+(sp\.|spółka|s\.a\.|sa\b|sp z o\.o\.).*$", "", str(dev), flags=re.I).strip()
    d = d.title() if d.isupper() else d
    st = re.sub(r"^(ul\.|ulica|al\.)\s*", "", str(street)).split("/")[0].split(",")[0].strip()
    d = " ".join(d.split()[:2]) if len(" ".join(d.split()[:2])) <= 15 else d.split()[0]
    return f"{d} · {st if len(st) <= 15 else st[:14] + '.'}"


def main():
    v, d, w = load()
    inv = gpd.read_file(os.path.join(HERE, "data", "investments.gpkg")).to_crs(2180)
    # one project filed under two street spellings (same developer, flats and asking price) counts once
    inv = inv.sort_values("flats", ascending=False)
    inv = inv.assign(_ask=inv["ask"].round(-1)).drop_duplicates(["developer", "flats", "_ask"]).drop(columns="_ask")

    fig = page("Every new-build project on sale in Warsaw",
               f"{len(inv)} projects and {style.fmt_int(inv['flats'].sum())} flats from the developers' own price lists",
               "Developers' price lists on dane.gov.pl (2025 price-transparency law), latest price per flat; one circle per "
               "project (developer + address), size = flats on offer, colour = median asking price per m². ~85 % of "
               "Warsaw files readable; projects placed via the city's address points (a third at street level). "
               + style.SOURCE)
    ax = fig.add_axes([0.22, 0.13, 0.56, 0.70])           # map in the middle, label columns either side
    d.plot(ax=ax, color=style.SURFACE, edgecolor="none")
    w.plot(ax=ax, color="#cfdcec", linewidth=0, zorder=1)
    d.boundary.plot(ax=ax, color=style.AXIS, linewidth=0.6, zorder=2)
    for _, r in d.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, r["district"], fontsize=7, color=style.MUTED, ha="center", va="center", zorder=2)
    ax.scatter(inv.geometry.x, inv.geometry.y, s=inv["flats"].clip(upper=800) * 1.2 + 8, c=inv["ask"], cmap=CMAP,
               norm=BoundaryNorm(BOUNDS, CMAP.N), edgecolor=style.SURFACE, lw=0.7, alpha=0.92, zorder=4)
    x0, y0, x1, y1 = d.total_bounds
    ax.set_xlim(x0 - 300, x1 + 300)
    ax.set_ylim(y0 - 300, y1 + 300)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # callout columns in figure coordinates, outside the map; half the labels per side, top to bottom
    top = inv.head(N_LABEL)
    cx = np.median(top.geometry.x)
    ax.apply_aspect()                                      # final map position before converting coordinates
    to_fig = (ax.transData + fig.transFigure.inverted()).transform
    for side, sub in (("west", top[top.geometry.x < cx]), ("east", top[top.geometry.x >= cx])):
        sub = sub.assign(_y=sub.geometry.y).sort_values("_y", ascending=False)
        ys = np.linspace(0.76, 0.20, max(len(sub), 2))[:len(sub)]
        lx = 0.205 if side == "west" else 0.795
        for (_, r), ly in zip(sub.iterrows(), ys):
            px, py = to_fig((r.geometry.x, r.geometry.y))
            fig.add_artist(plt.Line2D([px, lx], [py, ly], transform=fig.transFigure, color=style.MUTED, lw=0.5,
                                      zorder=3))
            fig.text(lx + (-0.004 if side == "west" else 0.004), ly,
                     f"{short(r['developer'], r['street'])}\n{int(r['flats'])} flats · {r['ask'] / 1000:.1f}k PLN/m²",
                     ha="right" if side == "west" else "left", va="center", fontsize=7, color=style.INK,
                     linespacing=1.1)
    from matplotlib.patches import Patch
    h1 = [Patch(color=CMAP(i), label=l) for i, l in enumerate(["< 13k", "13–15k", "15–17k", "17–20k", "20–25k", "25k+"])]
    fig.legend(handles=h1, loc="lower left", bbox_to_anchor=(0.03, 0.06), ncol=6, fontsize=8, frameon=False,
               title="Asking price, PLN/m²", title_fontsize=8.5, columnspacing=1.0, handlelength=1.4)
    h2 = [ax.scatter([], [], s=n * 1.2 + 8, facecolor="#f08a4b", edgecolor=style.SURFACE, label=f"{n}")
          for n in (50, 200, 500)]
    fig.legend(handles=h2, loc="lower right", bbox_to_anchor=(0.97, 0.06), ncol=3, fontsize=8, frameon=False,
               title="Flats on offer", title_fontsize=8.5, columnspacing=1.6)
    big = inv.head(5)
    names = ", ".join(short(a, b).split(" · ")[1] for a, b in zip(big["developer"], big["street"]))
    fig.text(0.03, 0.885, wrap(f"Supply sits mostly on the edges, in Ursus, Białołęka and Bemowo; the five largest "
                               f"projects ({names}) hold {big['flats'].sum() / inv['flats'].sum():.0%} of all flats on "
                               "offer. The priciest projects cluster around Rondo Daszyńskiego.", 105),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "28_projects_map.png")
    print(inv.head(N_LABEL)[["developer", "street", "flats", "ask", "district"]].to_string())


if __name__ == "__main__":
    main()
