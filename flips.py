"""
Flippers: the same flat sold again within a year.

A flat = address + floor + usable area (0.1 m²) in output/price_model.gpkg / transactions_adj (RCN has no
reliable flat id). Consecutive sales of one flat form a pair; a flip is a pair 14-365 days apart (shorter gaps
are usually duplicate or corrected deeds). Markup: nominal (price ratio) and net of the market (ratio of
price_m2_2025, which removes the city-wide quality-adjusted index over the holding period).

Outputs:
  output/flips.gpkg                    layer flips (every flip with both legs), layer msi_flips
  output/analysis/flips_by_year.csv
  output/summary/16_flippers.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from summary_charts import page, save, wrap
from summary_maps import basemap, load, outlines

style.apply()

MIN_DAYS, MAX_DAYS = 14, 365
OUT = "output/flips.gpkg"


def flat_ids(t):
    """Attach RCN's land-register flat id (LOK_ID_LOKALU, filled for every sale since 2018) by matching
    deed date, address, price, area and floor to the raw register."""
    r = gpd.read_file("output/rcn_lokale_warszawa.gpkg", ignore_geometry=True,
                      columns=["LOK_ID_LOKALU", "LOK_ADRES", "DOK_DATA", "LOK_CENA_BRUTTO", "LOK_POW_UZYT",
                               "LOK_NR_KOND", "LOK_FUNKCJA"])
    r = r[(r["LOK_FUNKCJA"] == "mieszkalna") & r["LOK_ID_LOKALU"].notna()]
    r = pd.DataFrame({"date": pd.to_datetime(r["DOK_DATA"].str[:10], errors="coerce"), "lok_adres": r["LOK_ADRES"],
                      "price": pd.to_numeric(r["LOK_CENA_BRUTTO"], errors="coerce").round(0),
                      "area": pd.to_numeric(r["LOK_POW_UZYT"], errors="coerce").round(1),
                      "lok_id": r["LOK_ID_LOKALU"]})
    r = r.drop_duplicates(["date", "lok_adres", "price", "area"], keep=False)
    k = t.assign(price=t["price"].round(0), area=t["area"].round(1))
    m = k.merge(r, on=["date", "lok_adres", "price", "area"], how="left")
    print(f"flat id attached to {m['lok_id'].notna().mean():.0%} of sales "
          f"({m.loc[m['date'].dt.year >= 2018, 'lok_id'].notna().mean():.0%} since 2018)")
    return m["lok_id"].to_numpy()


def pairs():
    t = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    t = t.dropna(subset=["lok_adres", "area"]).copy().reset_index(drop=True)
    t["date"] = pd.to_datetime(t["date"])
    t["lok_id"] = flat_ids(t)
    approx = t["lok_adres"].str.upper().str.strip() + "|" + t["floor"].fillna(-99).astype(int).astype(str)         + "|" + t["area"].round(1).astype(str)
    t["exact"] = t["lok_id"].notna()
    t["unit"] = np.where(t["exact"], "ID|" + t["lok_id"].astype(str), "AP|" + approx)
    all_sales = t
    n_unit = t.groupby("unit")["price"].transform("size")
    t = t[n_unit.between(2, 10)].sort_values(["unit", "date"])          # >10: a shared address, not one flat
    nxt = t.groupby("unit").shift(-1)
    p = t.assign(date2=nxt["date"], price2=nxt["price"], adj2=nxt["price_m2_2025"], market2=nxt["market"],
                 exact2=nxt["exact"])
    p = p.dropna(subset=["date2"])
    p["exact_pair"] = p["exact"] & p["exact2"].astype(bool)
    # a developer sells each flat once; approximate keys can't tell identical flats apart -> resale pairs only
    p = p[(p["market2"] == "secondary") & (p["exact_pair"] | (p["market"] == "secondary"))]
    p["days"] = (p["date2"] - p["date"]).dt.days
    p["markup"] = 100 * (p["price2"] / p["price"] - 1)
    p["markup_net"] = 100 * (p["adj2"] / p["price_m2_2025"] - 1)
    p = p[p["markup"].between(-50, 200)]
    print(f"{len(p):,} consecutive sale pairs ({p['exact_pair'].mean():.0%} matched by flat id)")
    return all_sales, p     # all sales (denominators), flat pairs


def main():
    t, p = pairs()
    f = p[p["days"].between(MIN_DAYS, MAX_DAYS)].copy()
    f["first_leg"] = np.where(f["market"] == "primary", "bought new", "bought resale")
    f["year2"] = f["date2"].dt.year
    print(f"flips (resold within {MAX_DAYS} days): {len(f):,}; median markup {f['markup'].median():+.1f}% nominal, "
          f"{f['markup_net'].median():+.1f}% net of the market; median holding {f['days'].median():.0f} days")
    first = f["market"].value_counts().to_dict()
    print(f"first leg bought: {first}")

    # do flips hold their value? the flat's next sale after the flip (>= 6 months later), net of the market
    nxt = p[["unit", "date", "markup_net", "days"]].rename(
        columns={"date": "date2", "markup_net": "after_net", "days": "days_after"})
    held = f.merge(nxt, on=["unit", "date2"], how="inner")
    held = held[(held["days_after"] >= 180) & held["after_net"].between(-60, 150)]
    print(f"{len(held):,} flipped flats sold again >= 6 months later: flipper's markup net "
          f"{held['markup_net'].median():+.1f}%, next sale vs flipper's price net {held['after_net'].median():+.1f}%")

    sec = t[t["market"] == "secondary"]
    by_year = pd.DataFrame({"flips": f.groupby("year2").size(),
                            "resales": sec.groupby(sec["date"].dt.year).size(),
                            "markup": f.groupby("year2")["markup"].median(),
                            "markup_net": f.groupby("year2")["markup_net"].median()})
    by_year["share"] = 100 * by_year["flips"] / by_year["resales"]
    by_year.round(2).to_csv("output/analysis/flips_by_year.csv")
    print(by_year.round(1).to_string())

    # where: flips (second legs 2021-25) per resale, by hexagon and MSI neighbourhood
    recent_f = f[f["year2"].between(2021, 2025)]
    recent_s = sec[sec["date"].dt.year.between(2021, 2025)]
    msi = gpd.read_file("data/msi.gpkg")

    def to_msi(g):
        j = gpd.sjoin(gpd.GeoDataFrame(geometry=g.geometry.values, crs=2180), msi[["name", "geometry"]],
                      predicate="within", how="left")
        return j[~j.index.duplicated()]["name"].to_numpy()
    area = pd.DataFrame({"flips": pd.Series(to_msi(recent_f)).value_counts(),
                         "resales": pd.Series(to_msi(recent_s)).value_counts()}).fillna(0)
    area = area[area["resales"] >= 150]
    area["share"] = 100 * area["flips"] / area["resales"]
    rt = gpd.read_file("output/reputation_time.gpkg", layer="msi_change", ignore_geometry=True).set_index("name")
    both = area.join(rt[["change", "reputation"]], how="inner")
    r = np.corrcoef(both["share"], both["change"])[0, 1] if len(both) > 5 else np.nan
    print(f"{len(area)} neighbourhoods with >= 150 resales 2021-25; corr(flip share, gain vs city) = {r:.2f} "
          f"over {len(both)} areas")
    print(area.sort_values("share", ascending=False).head(10).round(1).to_string())

    f.drop(columns=["unit", "exact", "exact2"]).to_file(OUT, layer="flips", driver="GPKG")
    msi.merge(area, left_on="name", right_index=True).to_file(OUT, layer="msi_flips", driver="GPKG")

    # figure ------------------------------------------------------------------------------------
    v, d, w = load()
    n_r, med, med_net = len(recent_f), recent_f["markup"].median(), recent_f["markup_net"].median()
    fig = page("The flippers",
               "Flats resold within a year of being bought: where, how often, and for how much",
               f"RCN flat sales; one flat = the land-register flat id (all sales since 2018), before 2018 the same address, "
               f"floor and area (resale→resale pairs only). Flip = resold {MIN_DAYS}–{MAX_DAYS} "
               "days after the previous sale. Markup net of market = change in price per m² after removing the "
               "city-wide quality-adjusted index over the holding period. Map: flips per 100 resales, 2021–25, "
               "600 m hexagons with ≥ 40 resales. " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"2021–25: {n_r:,} flips, {100 * n_r / len(recent_s):.1f} per 100 resales. "
                               f"Median markup {med:+.0f}% ({med_net:+.0f}% beyond what the market did), "
                               f"median holding {recent_f['days'].median():.0f} days. The markup sticks: flipped flats that sold again "
                               f"later went for {held['after_net'].median():+.0f}% vs the flipper's price, net of the "
                               f"market ({len(held)} flats). Flipping is busiest where "
                               f"prices gained most on the city since 2010–14 (correlation {r:+.2f} across "
                               f"{len(both)} neighbourhoods).", 95),
             fontsize=10.5, fontweight="semibold", va="top")
    ax = basemap(fig, d, w)
    ax.set_position([0.03, 0.46, 0.60, 0.36])
    kw = dict(gridsize=(40, 35), extent=(*ax.get_xlim(), *ax.get_ylim()))
    hf = ax.hexbin(recent_f.geometry.x, recent_f.geometry.y, **kw)
    hs = ax.hexbin(recent_s.geometry.x, recent_s.geometry.y, **kw)
    cf, cs = hf.get_array(), hs.get_array()
    share = np.where(cs >= 40, 100 * cf / np.maximum(cs, 1), np.nan)
    hf.remove()
    hs.set_array(np.ma.masked_invalid(share))
    hs.set_cmap(plt.cm.colors.ListedColormap(["#fde4cf", "#f9b98a", "#f08a4b", "#c4521f", "#7a2a08"]))
    hs.set_norm(plt.cm.colors.BoundaryNorm([0, 2, 4, 6, 9, 100], 5))
    hs.set_edgecolor(style.SURFACE)
    hs.set_linewidth(0.3)
    hs.set_zorder(3)
    outlines(ax, d)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(
        ["#fde4cf", "#f9b98a", "#f08a4b", "#c4521f", "#7a2a08"], ["< 2", "2–4", "4–6", "6–9", "9+"])],
        title="Flips per 100 resales", loc="upper right", fontsize=8.5, title_fontsize=9, frameon=True,
        framealpha=0.9, edgecolor="none")

    top = area.sort_values("share", ascending=False).head(10).iloc[::-1]
    a3 = fig.add_axes([0.80, 0.50, 0.17, 0.33])
    a3.barh(np.arange(len(top)), top["share"], color="#c4521f", height=0.65)
    a3.set_yticks(np.arange(len(top)))
    a3.set_yticklabels(top.index, fontsize=8.5)
    for yi, s in enumerate(top["share"]):
        a3.text(s + 0.2, yi, f"{s:.1f}", va="center", fontsize=8.5, color=style.INK_2)
    a3.set_xlim(0, top["share"].max() * 1.3)
    style.vgrid_only(a3)
    a3.tick_params(axis="x", labelsize=8)
    a3.set_title("Top neighbourhoods", fontsize=10)

    a1 = fig.add_axes([0.10, 0.12, 0.38, 0.27])
    yy = by_year[(by_year.index >= 2006) & (by_year["resales"] > 2000)]["share"].reindex(range(2006, 2027))
    a1.plot(yy.index, yy.values, "-o", color="#c4521f", lw=2, ms=5, mec=style.SURFACE)   # NaN years break the line
    style.shade_gap(a1)
    style.hgrid_only(a1)
    a1.set_ylim(0, None)
    a1.set_title("Flips per 100 resales, by year", fontsize=10.5)
    a1.tick_params(labelsize=8.5)

    a2 = fig.add_axes([0.58, 0.12, 0.38, 0.27])
    bins = np.arange(-30, 81, 5)
    a2.hist(recent_f["markup"].clip(-30, 80), bins=bins, color="#f08a4b", edgecolor=style.SURFACE, lw=1,
            label="Nominal")
    a2.hist(recent_f["markup_net"].clip(-30, 80), bins=bins, histtype="step", color=style.INK, lw=1.6,
            label="Net of the market")
    a2.axvline(0, color=style.AXIS, lw=1)
    a2.set_xticks([-20, 0, 20, 40, 60, 80])
    a2.set_xticklabels(["−20%", "0%", "+20%", "+40%", "+60%", "+80%+"])
    style.hgrid_only(a2)
    a2.legend(fontsize=8.5, loc="upper right")
    a2.set_title("Markup on resale, 2021–25 (flips)", fontsize=10.5)
    a2.tick_params(labelsize=8.5)
    save(fig, "16_flippers.png")


if __name__ == "__main__":
    main()
