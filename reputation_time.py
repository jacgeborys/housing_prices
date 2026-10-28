"""
Is reputation changing? Within-building price change 2010-14 -> 2021-25, relative to the city.

For every building with >= MIN_SALES resales in both periods and no developer (primary) sale at all
(so it existed before the data starts and nobody compares a new flat with a ten-year-old one), take the
median of log(price_m2_2025) in each period. price_m2_2025 already removes the city-wide index, so the
difference is how much the building gained or lost against Warsaw as a whole. Same building, so quality
is held fixed (apart from renovation and ageing, which hit all old buildings alike).

Outputs:
  output/reputation_time.gpkg                   layer building_change, layer msi_change
  output/summary/14_reputation_over_time.png    hex map of relative change + catch-up scatter by neighbourhood
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load, outlines

style.apply()

EARLY, LATE = (2010, 2014), (2021, 2025)
MIN_SALES = 3
MIN_BUILDINGS = 8            # per neighbourhood in the scatter
OUT = "output/reputation_time.gpkg"


def building_change():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                      columns=["osm_id", "year", "market", "price_m2_2025"])
    t = t.dropna(subset=["osm_id"])
    primary = t.loc[t["market"] == "primary", "osm_id"].unique()
    t = t[(t["market"] == "secondary") & ~t["osm_id"].isin(primary)]
    t["lp"] = np.log(t["price_m2_2025"])
    per = np.select([t["year"].between(*EARLY), t["year"].between(*LATE)], ["early", "late"], "")
    t = t[per != ""].assign(per=per[per != ""])
    g = t.groupby(["osm_id", "per"])["lp"].agg(["median", "size"]).unstack()
    ok = (g[("size", "early")] >= MIN_SALES) & (g[("size", "late")] >= MIN_SALES)
    g = g[ok]
    ch = pd.DataFrame({"osm_id": g.index, "change_pct": 100 * (np.exp(g[("median", "late")] - g[("median", "early")]) - 1),
                       "n_early": g[("size", "early")], "n_late": g[("size", "late")]}).reset_index(drop=True)
    b = gpd.read_file("output/price_model.gpkg", layer="building_value")[["osm_id", "district", "geometry"]]
    ch = b.merge(ch, on="osm_id")
    print(f"{len(ch):,} buildings with >= {MIN_SALES} resales in both {EARLY} and {LATE}; "
          f"median change vs city {ch['change_pct'].median():+.1f}%")
    return ch


def main():
    ch = building_change()
    msi = gpd.read_file("data/msi.gpkg")
    pts = gpd.GeoDataFrame(ch[["osm_id"]], geometry=ch.geometry.representative_point(), crs=2180)
    ch["msi"] = gpd.sjoin(pts, msi[["name", "geometry"]], predicate="within", how="left") \
        .groupby(level=0)["name"].first().reindex(ch.index).to_numpy()

    v, d, w = load()                          # price_drivers.gpkg: today's residual (reputation)
    vp = gpd.GeoDataFrame(v[["resid_pct"]], geometry=v.geometry.representative_point(), crs=2180)
    v["msi"] = gpd.sjoin(vp, msi[["name", "geometry"]], predicate="within", how="left") \
        .groupby(level=0)["name"].first().reindex(v.index).to_numpy()
    rep = v.groupby("msi")["resid_pct"].median()
    area = ch.groupby("msi").agg(change=("change_pct", "median"), n=("change_pct", "size"),
                                 district=("district", "first"))
    area["reputation"] = rep.reindex(area.index)
    area = area[area["n"] >= MIN_BUILDINGS].dropna()
    r = np.corrcoef(area["reputation"], area["change"])[0, 1]
    slope = np.polyfit(area["reputation"], area["change"], 1)[0]
    print(f"{len(area)} neighbourhoods with >= {MIN_BUILDINGS} buildings; corr(reputation, change) = {r:.2f}, "
          f"slope {slope:.2f}")
    ch.to_file(OUT, layer="building_change", driver="GPKG")
    msi.merge(area, left_on="name", right_index=True).to_file(OUT, layer="msi_change", driver="GPKG")

    fig = page("The centre is pulling away",
               f"Same buildings, {EARLY[0]}–{str(EARLY[1])[2:]} vs {LATE[0]}–{str(LATE[1])[2:]}: "
               "price change relative to Warsaw as a whole",
               f"Resales only, buildings that existed before 2006; prices net of the city-wide index (0 % = moved with the city). Map: 600 m "
               f"hexagons. Scatter: MSI neighbourhoods with ≥ {MIN_BUILDINGS} buildings; reputation = fig. 06. " + style.SOURCE)
    ax = basemap(fig, d, w)
    ax.set_position([0.03, 0.47, 0.94, 0.40])
    p = ch.geometry.representative_point()
    norm = TwoSlopeNorm(vmin=-20, vcenter=0, vmax=20)
    hb = ax.hexbin(p.x, p.y, C=ch["change_pct"].clip(-50, 50), gridsize=(48, 42), reduce_C_function=np.median,
                   mincnt=3, cmap=style.DIV_CMAP, norm=norm, linewidths=0.3, edgecolors=style.SURFACE, zorder=3,
                   extent=(*ax.get_xlim(), *ax.get_ylim()))
    outlines(ax, d)
    cax = fig.add_axes([0.80, 0.69, 0.02, 0.13])
    cb = fig.colorbar(hb, cax=cax, ticks=[-20, -10, 0, 10, 20])
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8.5, length=0)
    cb.ax.set_yticklabels(["−20%", "−10%", "with\nthe city", "+10%", "+20%"])

    a2 = fig.add_axes([0.14, 0.12, 0.80, 0.28])
    a2.scatter(area["reputation"], area["change"], s=np.sqrt(area["n"]) * 9, color=style.BLUE, alpha=0.55,
               edgecolor=style.SURFACE, lw=0.8, zorder=3)
    xs = np.linspace(area["reputation"].min(), area["reputation"].max(), 10)
    a2.plot(xs, np.polyval(np.polyfit(area["reputation"], area["change"], 1), xs), color=style.INK_2, lw=1.4,
            ls=(0, (4, 3)))
    a2.axhline(0, color=style.AXIS, lw=1)
    a2.axvline(0, color=style.AXIS, lw=1)
    lab = pd.concat([area.nlargest(4, "change"), area.nsmallest(3, "change"),
                     area.nsmallest(3, "reputation"), area.nlargest(3, "reputation")])
    lab = lab[~lab.index.duplicated()]
    for name, row in lab.iterrows():
        a2.annotate(name, (row["reputation"], row["change"]), xytext=(5, 3), textcoords="offset points",
                    fontsize=8, color=style.INK)
    a2.set_xlim(area["reputation"].min() - 3, area["reputation"].max() + 9)
    style.pct(a2)
    style.pct(a2, "x")
    style.hgrid_only(a2)
    a2.set_xlabel("Reputation today: median price vs model prediction")
    a2.set_ylabel("Change vs city, 2010–14 → 2021–25")
    verdict = "low-reputation areas are catching up" if slope < -0.15 else \
        "high-reputation areas are pulling away" if slope > 0.15 else "no clear catch-up either way"
    a2.set_title(f"Neighbourhoods: {verdict} (correlation {r:+.2f}); the two Pragas gain most", fontsize=11)
    gd = ch.groupby("district")["change_pct"].agg(["median", "size"])
    gain = gd.loc[gd["size"] >= 25, "median"].sort_values()
    fig.text(0.03, 0.885, wrap(f"Biggest gains against the city: {gain.index[-1]} ({gain.iloc[-1]:+.0f}%), "
                               f"{gain.index[-2]} ({gain.iloc[-2]:+.0f}%); biggest losses: {gain.index[0]} "
                               f"({gain.iloc[0]:+.0f}%), {gain.index[1]} ({gain.iloc[1]:+.0f}%).", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    save(fig, "14_reputation_over_time.png")
    print(gain.round(1).to_string())


if __name__ == "__main__":
    main()
