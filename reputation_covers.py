"""
Two candidate covers for the second part ("Warsaw, beyond price"), siblings of the ridgeline opening map. Both show
reputation: resid_pct from price_drivers.py (typical-flat price vs the out-of-fold prediction from location, transit,
amenities, noise, green and building; buildings with >= 3 sales), averaged over neighbors within SMOOTH_M (Gaussian
weights) so streets read as areas rather than single-building noise.

  spikes()  every building a thin spike: up and warm above prediction, down and cold below; drawn north to south
  glow()    every building a soft glowing dot on black, warm above prediction, cold below ("night lights")

Output: output/summary/43_reputation_spikes.png, 44_reputation_glow.png (dark, like 31)
"""
import geopandas as gpd
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree

import style
from price_ridgelines import BG, INK, LABELS, MUTED
from reputation_ridgelines import LIM, RAMP
from summary_charts import FIG, save

SMOOTH_M = 250                # Gaussian sigma of the neighbor average
SPIKE_M_PER_PCT = 55          # spike length per 1 % above / below prediction
SPIKE_HALF_W = 14             # half width of a spike's base, m (thin: needles, not blocks)


def load():
    b = gpd.read_file("output/price_drivers.gpkg", layer="building_drivers").dropna(subset=["resid_pct"])
    pts = b.geometry.representative_point()
    xy = np.c_[pts.x, pts.y]
    r = np.log1p(np.clip(b["resid_pct"].to_numpy(float), -40, 40) / 100)
    w = np.minimum(b["n"].to_numpy(float), 20) ** 0.5
    tree = cKDTree(xy)
    sm = np.empty(len(b))
    for i, nb in enumerate(tree.query_ball_point(xy, 3 * SMOOTH_M)):
        d2 = ((xy[nb] - xy[i]) ** 2).sum(1)
        k = w[nb] * np.exp(-d2 / (2 * SMOOTH_M ** 2))
        sm[i] = (k * r[nb]).sum() / k.sum()
    b["rep"] = 100 * np.expm1(sm)
    b["x"], b["y"] = xy[:, 0], xy[:, 1]
    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    w_ = gpd.read_file("D:/QGIS/mapy_warszawy_misc/data/osm/water.gpkg").to_crs(2180)
    river = w_[(w_["water"] == "river") & (w_.area > 2e5)].buffer(0).unary_union.intersection(d.unary_union)
    return b, d, river


def frame(fig, d, title, subtitle, foot, n):
    reps = d.set_index("district").geometry.representative_point()
    style.byline(fig, y=0.975, color=MUTED, fontsize=10, x=0.95)
    fig.text(0.05, 0.955, title, color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, subtitle, color=MUTED, fontsize=12.5, va="top", linespacing=1.4)
    cax = fig.add_axes([0.66, 0.075, 0.28, 0.012])
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=Normalize(-LIM, LIM), cmap=RAMP), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.set_ticks([-10, 0, 10])
    cb.set_ticklabels(["−10%", "as predicted", "+10%"])
    cax.tick_params(colors=MUTED, labelsize=9, length=0)
    cax.set_title("vs prediction", color=MUTED, fontsize=9, loc="left")
    fig.text(0.05, 0.022, f"Typical-flat price vs an out-of-sample prediction from location, transit, amenities, noise, green "
                          f"and building, {n:,} buildings with ≥ 3 sales,\naveraged over neighbors within ~{SMOOTH_M} m. "
                          f"{foot} Source: GUGiK RCN, OSM, BDOT10k.", color=MUTED, fontsize=7.5, va="bottom")
    return reps


def base_axes(fig, d, river):
    ax = fig.add_axes([0.03, 0.07, 0.94, 0.80])
    ax.set_facecolor(BG)
    d.boundary.plot(ax=ax, color="#262838", lw=0.6, zorder=1)
    gpd.GeoSeries([river]).plot(ax=ax, color="#161a2a", lw=0, zorder=0)
    x0, y0, x1, y1 = d.total_bounds
    ax.set_xlim(x0 - 500, x1 + 500)
    ax.set_ylim(y0 - 300, y1 + 2600)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def labels(ax, reps, dy=2000):
    for name, (dx, ddy) in LABELS.items():
        pt = reps[name]
        ax.text(pt.x + dx, pt.y + ddy + dy, name, color=INK, fontsize=8.5, ha="center", zorder=10_000,
                fontweight="semibold", path_effects=[pe.withStroke(linewidth=2.5, foreground=BG)])


def spikes(b, d, river):
    fig = plt.figure(figsize=FIG, facecolor=BG)
    ax = base_axes(fig, d, river)
    s = b.sort_values("y", ascending=False)                 # north first: southern spikes are drawn on top
    h = s["rep"].to_numpy() * SPIKE_M_PER_PCT
    x, y = s["x"].to_numpy(), s["y"].to_numpy()
    tri = np.stack([np.c_[x - SPIKE_HALF_W, y], np.c_[x, y + h], np.c_[x + SPIKE_HALF_W, y]], axis=1)
    norm = Normalize(-LIM, LIM)
    a = np.clip(0.18 + 0.7 * np.abs(s["rep"].to_numpy()) / LIM, 0.18, 0.85)
    cols = RAMP(norm(s["rep"].to_numpy()))
    cols[:, 3] = a
    ax.add_collection(PolyCollection(tri, facecolors=cols, edgecolors="none", zorder=3))
    reps = frame(fig, d, "Warsaw, beyond price",
                 "Every building with flat sales is a spike: up where flats sell for more than their\n"
                 "location and building predict, down where they sell for less. The gap is the address.",
                 f"Spike length {SPIKE_M_PER_PCT} m per 1 %.", len(b))
    labels(ax, reps)
    mpl.rcParams["savefig.facecolor"] = BG
    save(fig, "43_reputation_spikes.png")


def glow(b, d, river):
    fig = plt.figure(figsize=FIG, facecolor=BG)
    ax = base_axes(fig, d, river)
    norm = Normalize(-LIM, LIM)
    s = b.iloc[np.argsort(np.abs(b["rep"].to_numpy()))]     # the strongest on top
    cols = RAMP(norm(s["rep"].to_numpy()))
    size = 0.4 + 0.5 * np.minimum(s["n"].to_numpy(float), 40) ** 0.5
    for scale, alpha in ((40, 0.012), (12, 0.035), (4, 0.12), (1, 1.0)):   # wide faint halo -> bright core
        c = cols.copy()
        c[:, 3] = alpha
        ax.scatter(s["x"], s["y"], s=size * scale, c=c, lw=0, zorder=3)
    reps = frame(fig, d, "Warsaw, beyond price",
                 "Every building with flat sales, lit warm where flats sell for more than their location\n"
                 "and building predict, cold where they sell for less. The glow is the address.",
                 "Bigger dot = more sales.", len(b))
    labels(ax, reps)
    mpl.rcParams["savefig.facecolor"] = BG
    save(fig, "44_reputation_glow.png")


def main():
    b, d, river = load()
    spikes(b, d, river)
    glow(b, d, river)


if __name__ == "__main__":
    main()
