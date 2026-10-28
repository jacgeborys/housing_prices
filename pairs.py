"""
Look-alike pairs: buildings of the same decade, height, footprint, distance to the metro and market type
(resale-dominated), 1-8 km apart, and what their typical flats sell for.

Matching (no model price involved): same construction decade, height within 3 m, footprint within 25 %, walk to
the metro within 300 m, road noise within 6 dB, both resale-dominated (< 20 % developer sales), >= 8 priced
sales each. Every building is paired with its single closest look-alike in a different district-or-far-away spot
(>= 1.5 km). Headline = the median gap over ALL pairs, so the examples are not cherry-picked; examples are the
largest gaps among well-sold, street-named pairs.

Output: output/analysis/pairs.csv, output/summary/41_pairs.png
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load, outlines

style.apply()
MIN_SALES, MIN_KM, MAX_KM = 8, 1.5, 8


def match(v):
    v = v[(v["decade_end"] > 0) & (v["n"] >= MIN_SALES) & (v["pct_primary"] < 0.2)
          & (v["height_m"] > 3) & v["addr_street"].notna()].copy()
    v["x"], v["y"] = v.geometry.centroid.x, v.geometry.centroid.y
    rows = []
    for dec, g in v.groupby("decade_end"):
        if len(g) < 5:
            continue
        feat = np.c_[g["height_m"] / 3, g["log_footprint"] / 0.22, g["walk_metro_km"] / 0.3, g["noise_road_db"] / 6]
        tr = cKDTree(feat)
        gi = g.reset_index(drop=True)
        for i in range(len(gi)):
            for j in tr.query_ball_point(feat[i], 1.0):
                if j <= i:
                    continue
                km = np.hypot(gi.x[i] - gi.x[j], gi.y[i] - gi.y[j]) / 1000
                if MIN_KM <= km <= MAX_KM:
                    a, b = (i, j) if gi.value_2025[i] >= gi.value_2025[j] else (j, i)
                    rows.append(dict(decade=dec, hi=gi.osm_id[a], lo=gi.osm_id[b], km=km,
                                     gap=100 * (gi.value_2025[a] / gi.value_2025[b] - 1),
                                     dist=np.linalg.norm(feat[i] - feat[j])))
    p = pd.DataFrame(rows)
    return v.set_index("osm_id"), p


PICKS = [("Wiktorska", "86", "Strzelecka", "7/9"), ("Gwiaździsta", "7C", "Górczewska", "228C"),
         ("Księdza Jerzego Popiełuszki", "13", "Wolska", "78/80"), ("Inflancka", "5", "Samarytanka", "3")]
DEAR, CHEAP = "#8e2f55", "#e8914d"


def short(st):
    for a, b in (("Księdza Jerzego ", "ks. J. "), ("Generała Tadeusza ", "gen. T. "), ("Jana Karola ", "J. K. ")):
        st = st.replace(a, b)
    return st


def decade_label(dec):
    return "after 2020" if dec >= 2030 else f"{int(dec) - 9}–{int(dec)}" if dec > 1900 else "pre-1901"


def main():
    v, d, w = load()
    b, p = match(v)
    best = p.sort_values("dist").drop_duplicates("hi").drop_duplicates("lo")
    best.to_csv("output/analysis/pairs.csv", index=False)
    med, q3 = best["gap"].median(), best["gap"].quantile(0.75)
    bi = b.reset_index()
    key = lambda st, no: bi[(bi.addr_street == st) & (bi.addr_housenumber == no)].iloc[0]
    ex = []
    for hs, hn, ls, ln in PICKS:
        a, c = key(hs, hn), key(ls, ln)
        ex.append((a, c, 100 * (a.value_2025 / c.value_2025 - 1), float(np.hypot(a.x - c.x, a.y - c.y)) / 1000))
    fig = page("Same building, different postcode",
               "Look-alike blocks: same decade, same height and footprint, same walk to the metro",
               f"{len(best)} pairs of resale-dominated buildings (≥ {MIN_SALES} priced sales each), 1.5–8 km apart, matched on decade, "
               "height, footprint, walk to the metro and road noise. Typical-flat price, 2025 PLN/m². " + style.SOURCE)
    a0 = ex[0]
    fig.text(0.03, 0.885, wrap(f"Take two blocks from the same decade, equally tall, equally far from the metro. The typical pair "
                               f"differs by {med:.0f}% per m²; one pair in four by {q3:.0f}% or more. A {decade_label(a0[0].decade_end)} block on "
                               f"{a0[0].addr_street} sells for {a0[2]:.0f}% more than its twin on {a0[1].addr_street}.", 105),
             fontsize=10.5, fontweight="semibold", va="top")
    ax = basemap(fig, d, w, box=[0.02, 0.385, 0.96, 0.43])
    outlines(ax, d, labels=False)
    for k, (a, c, gap, km) in enumerate(ex):
        ax.plot([a.x, c.x], [a.y, c.y], color=style.INK_2, lw=1.2, zorder=4)
        ax.scatter([a.x], [a.y], s=230, color=DEAR, zorder=6)
        ax.scatter([c.x], [c.y], s=230, color=CHEAP, zorder=6)
        for pt in (a, c):
            ax.text(pt.x, pt.y, "ABCD"[k], color="white", fontsize=8.5, fontweight="bold", ha="center", va="center", zorder=7)
    for k, (a, c, gap, km) in enumerate(ex):
        col, row = k % 2, k // 2
        x0, y0 = 0.04 + col * 0.50, 0.255 - row * 0.165
        fig.text(x0, y0 + 0.095, f"{'ABCD'[k]}   {short(a.addr_street)} {a.addr_housenumber}  vs  {short(c.addr_street)} {c.addr_housenumber}", fontsize=8.8, fontweight="semibold", va="bottom")
        fig.text(x0, y0 + 0.078, f"{a.district} vs {c.district} · both {decade_label(a.decade_end)}, ~{a.height_m:.0f} m tall, {km:.0f} km apart", fontsize=8.5,
                 color=style.INK_2, va="bottom")
        bx = fig.add_axes([x0 + 0.01, y0 - 0.005, 0.40, 0.07])
        bx.barh([1, 0], [a.value_2025, c.value_2025], color=[DEAR, CHEAP], height=0.7)
        for yy, val in ((1, a.value_2025), (0, c.value_2025)):
            bx.text(val - 300, yy, f"{style.fmt_int(val)}", color="white", fontsize=9, fontweight="bold", ha="right", va="center")
        bx.text(a.value_2025 + 300, 1, f"+{gap:.0f}%", fontsize=10.5, fontweight="bold", color=DEAR, va="center")
        bx.set_xlim(0, a.value_2025 * 1.22)
        bx.set_axis_off()
    save(fig, "41_pairs.png")
    for a, c, gap, km in ex:
        print(f"{a.addr_street} {a.addr_housenumber} {a.value_2025:.0f} vs {c.addr_street} {c.addr_housenumber} {c.value_2025:.0f}: +{gap:.0f}%, {km:.1f} km, n {a.n}/{c.n}")
    print(f"median gap {med:.1f}%, q3 {q3:.1f}%, {len(best)} pairs")


if __name__ == "__main__":
    main()
