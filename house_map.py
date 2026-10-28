"""
Where houses sell and where flats sell: every house sale (RCN budynki layer, houses.py -> output/houses.gpkg) as a dot
over every building with a flat sale, plus sales per year for both with the register's blind spots marked.
Plain description, no model: counts and places only.

-> output/summary/38_house_map.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import style
from summary_charts import page, save, wrap
from summary_maps import MAP_BOX, basemap, load

style.apply()

HOUSE = "#2a6b35"
FLAT = "#b9b7ad"


def main():
    v, d, w = load()
    h = gpd.read_file("output/houses.gpkg")
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True)
    b = gpd.read_file("output/price_model.gpkg", layer="building_value").drop_duplicates("osm_id")
    pts = b.geometry.representative_point()

    n_flat, n_house = len(t), len(h)
    ratio = n_flat / n_house
    fig = page(f"One house sale for every {ratio:.0f} flat sales",
               "Houses ring the city; flats fill the middle. Every house sale 2017–2026, and every building with a flat sale",
               f"Houses: {n_house:,} single-family house sales (RCN, recorded from 2017). Flats: {n_flat:,} sales 2006–2026; grey = buildings "
               "with a recorded sale. The register's flat records for 2017–2020 are nearly empty. " + style.SOURCE)

    ax = basemap(fig, d, w)
    ax.scatter(pts.x, pts.y, s=1.1, color=FLAT, lw=0, zorder=3)
    ax.scatter(h.geometry.x, h.geometry.y, s=3.2, color=HOUSE, alpha=0.55, lw=0, zorder=4)
    ax.scatter([], [], s=22, color=FLAT, label=f"building with a flat sale ({len(b):,})")
    ax.scatter([], [], s=22, color=HOUSE, label=f"one house sale ({n_house:,})")
    ax.set_position([0.03, 0.37, 0.94, 0.47])
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=9, handletextpad=0.3)

    top = h.groupby("district").size().sort_values(ascending=False)
    share = top.head(3).sum() / n_house
    fig.text(0.04, 0.855, wrap(f"{top.index[0]} alone has {top.iloc[0]:,} house sales, {top.index[1]} {top.iloc[1]:,}, "
                               f"{top.index[2]} {top.iloc[2]:,}: together {share:.0%} of all. Śródmieście has "
                               f"{int(top.get('Śródmieście', 0))}.", 95), fontsize=10.5, fontweight="semibold", va="top")

    years = np.arange(2006, 2027)
    fy = t.groupby("year").size().reindex(years, fill_value=0)
    hy = h.groupby("year").size().reindex(years, fill_value=0)
    x0, wd, hh = 0.10, 0.86, 0.085
    for k, (series, colr, lab, gap_lo, gap_hi, gap_lab) in enumerate((
            (fy / 1000, FLAT, "Flat sales per year (thousands)", 2016.5, 2020.5, "register gap\n2017–20"),
            (hy, HOUSE, "House sales per year", 2005.5, 2016.5, "not recorded\nbefore 2017"))):
        a = fig.add_axes([x0, 0.26 - k * 0.135, wd, hh])
        a.bar(series.index, series.values, color=colr, width=0.8, zorder=2)
        a.axvspan(gap_lo, gap_hi, color=style.GRID, alpha=0.55, lw=0, zorder=0)
        a.text((gap_lo + gap_hi) / 2, 0.93, gap_lab, transform=a.get_xaxis_transform(), ha="center", va="top",
               fontsize=8.5, color=style.MUTED, style="italic")
        a.set_xlim(2005.4, 2026.6)
        a.set_xticks([2006, 2010, 2014, 2018, 2022, 2026] if k == 1 else [])
        a.set_title(lab, fontsize=9.5, loc="left")
        style.hgrid_only(a)
    fig.axes[-2].set_xticks([])
    save(fig, "38_house_map.png")


if __name__ == "__main__":
    main()
