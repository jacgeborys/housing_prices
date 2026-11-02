"""
Forward test ("would 2014 have known?"): reputation and price drivers measured on 2010-14 resales, checked
against the same buildings' resales in 2021-25.

Sample: buildings with >= MIN_RES resales in both periods (older stock: buildings finished after 2014 are not in it).
Building level per period = median resale price per m² in 2025 PLN (city index removed), so the change is
relative to Warsaw as a whole.
  reputation  out-of-fold gradient boosting residual of the period level on the static features of
              price_drivers.py (spatial 5-fold on 2 km blocks), separately per period; averaged per MSI
              neighbourhood (>= MIN_NB buildings) where sampling noise averages out
  persistence corr(reputation 2010-14, reputation 2021-25) across neighbourhoods
  catch-up    corr(reputation 2010-14, change 2010-14 -> 2021-25)
  drivers     OLS of the change on the features + district fixed effects, SEs clustered by 2 km block;
              new metro stations (central M2, opened 2015; M2 extensions 2019-22) within NEW_STATION_M

Outputs: output/analysis/forward_test_neighbourhoods.csv, forward_test_drivers.csv, summary/30_forward_test.png
(reputation then vs now) and summary/30b_what_moved.png (what moved the same buildings)
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import style
from price_drivers import ols
from summary_charts import page, save, wrap

style.apply()
MIN_RES, MIN_NB, NEW_STATION_M = 4, 15, 800
PERIODS = {"1": (2010, 2014), "2": (2021, 2025)}
STATIONS = "D:/QGIS/mapy_warszawy_misc/data/osm/train_stations.gpkg"
M2_CENTRAL = ["Rondo Daszyńskiego", "Rondo ONZ", "Nowy Świat-Uniwersytet", "Centrum Nauki Kopernik",
              "Stadion Narodowy", "Dworzec Wileński"]
M2_EXT = ["Płocka", "Młynów", "Księcia Janusza", "Ulrychów", "Bemowo", "Szwedzka", "Targówek Mieszkaniowy",
          "Trocka", "Zacisze", "Kondratowicza", "Bródno"]
STATIC = ["walk_metro_km", "walk_rail_km", "walk_tram_km", "walk_bus_km", "walk_park_km", "walk_school_km",
          "noise_road_db", "noise_rail_db", "heritage", "centre_km", "river_km", "green_ha_500", "height_m",
          "log_footprint", "walk7_food_drink", "walk7_grocery", "canopy_100",
          "age_prewar", "age_1941_60", "age_1991_00", "age_2001_10", "age_2011_20"]
SCALE = {"noise_road_db": 10, "noise_rail_db": 10, "canopy_100": 10, "green_ha_500": 10, "height_m": 10}
SHOW = [("age_prewar", "Built before 1941"), ("age_1941_60", "Built 1941–60"), ("age_1991_00", "Built 1991–2000"),
        ("age_2001_10", "Built 2001–10"), ("m2_ext", "New M2 station ≤ 800 m (2019–22)"),
        ("m2_central", "New central M2 station ≤ 800 m (2015)"), ("heritage", "Listed heritage building"),
        ("walk7_food_drink", "Twice as many cafés & restaurants"), ("canopy_100", "+10 pp tree cover within 100 m"),
        ("green_ha_500", "+10 ha park/forest within 500 m"), ("walk_tram_km", "1 km farther to a tram stop"),
        ("walk_metro_km", "1 km farther to the metro")]
# neighbourhood labels per panel: name -> (dx, dy points, ha)
LABELS_PERSIST = {"Stary Żoliborz": (-5, 4, "right"), "Wyględów": (-5, 4, "right"), "Kabaty": (-5, 4, "right"),
                  "Śródmieście Południowe": (5, -9, "left"), "Nowa Praga": (-6, -12, "right"),
                  "Bródno-Podgrodzie": (-5, 4, "right"), "Grodzisk": (5, -3, "left")}
LABELS_CHANGE = {"Stary Żoliborz": (0, 7, "center"),
                 "Nowa Praga": (6, -12, "left"), "Bródno-Podgrodzie": (-5, 5, "right"), "Młynów": (6, -10, "left"),
                 "Błonia Wilanowskie": (5, 2, "left"), "Wyględów": (5, 2, "left"), "Kabaty": (5, -3, "left"),
                 "Grodzisk": (5, 2, "left"), "Skorosze": (-5, -10, "right")}


def buildings():
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                       columns=["year", "osm_id", "market", "price_m2_2025"])
    tx = tx[(tx["market"] == "secondary") & tx["osm_id"].notna()]
    tx["lp"] = np.log(tx["price_m2_2025"])
    per = [tx[tx["year"].between(*r)].groupby("osm_id")["lp"].agg(["median", "size"]).add_suffix(k)
           for k, r in PERIODS.items()]
    b = per[0].join(per[1], how="inner")
    b = b[(b["size1"] >= MIN_RES) & (b["size2"] >= MIN_RES)]
    v = gpd.read_file("output/price_drivers.gpkg", layer="building_drivers").set_index("osm_id").join(b, how="inner")
    v["d"] = v["median2"] - v["median1"]
    pts = v.geometry.representative_point()
    st = gpd.read_file(STATIONS).to_crs(2180)
    st = st[st["station"] == "subway"].dissolve("name").reset_index()
    for col, names in (("m2_central", M2_CENTRAL), ("m2_ext", M2_EXT)):
        s = st[st["name"].isin(names)].geometry.centroid
        assert len(s) == len(names), set(names) - set(st["name"])
        v[col] = (np.min([pts.distance(p).to_numpy() for p in s], axis=0) < NEW_STATION_M).astype(float)
    v["block"] = (pts.x // 2000).astype(int) * 10000 + (pts.y // 2000).astype(int)
    v["pt"] = pts
    print(f"{len(v):,} buildings with >= {MIN_RES} resales in 2010-14 and 2021-25; "
          f"{int(v['m2_ext'].sum())} near an M2 extension station, {int(v['m2_central'].sum())} near central M2")
    return v


def oof_resid(y, X, blocks, k=5):
    ub = np.unique(blocks)
    fold = pd.Series(np.random.default_rng(0).permutation(len(ub)) % k, index=ub)[blocks].to_numpy()
    pred = np.zeros(len(y))
    for f in range(k):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=30, random_state=0)
        pred[fold == f] = m.fit(X[fold != f], y[fold != f]).predict(X[fold == f])
    return y - pred


def neighbourhoods(v):
    X = v[STATIC].to_numpy(float)
    for k in PERIODS:
        v[f"rep{k}"] = oof_resid(v[f"median{k}"].to_numpy(), X, v["block"].to_numpy())
    msi = gpd.read_file("data/msi.gpkg").to_crs(2180)
    j = gpd.sjoin(gpd.GeoDataFrame(v[["rep1", "rep2", "d", "district"]], geometry=v["pt"].values, crs=2180),
                  msi[["name", "geometry"]], predicate="within")
    n = j.groupby("name").agg(rep1=("rep1", "mean"), rep2=("rep2", "mean"), d=("d", "mean"),
                              buildings=("d", "size"), district=("district", "first"))
    n = n[n["buildings"] >= MIN_NB]
    for c in ("rep1", "rep2", "d"):
        n[f"{c}_pct"] = 100 * (np.exp(n[c]) - 1)
    r_persist, r_catch = n["rep1"].corr(n["rep2"]), n["rep1"].corr(n["d"])
    print(f"{len(n)} neighbourhoods: persistence r = {r_persist:+.2f}, catch-up r = {r_catch:+.2f}")
    n.round(2).to_csv("output/analysis/forward_test_neighbourhoods.csv")
    return n, r_persist, r_catch


def drivers(v):
    feats = STATIC + ["m2_central", "m2_ext"]
    X = v[feats].to_numpy(float) / np.array([SCALE.get(f, 1) for f in feats])
    D = pd.get_dummies(v["district"], drop_first=True).to_numpy(float)
    b, s, r2 = ols(np.column_stack([X, D]), v["d"].to_numpy(), v["block"].to_numpy())
    out = pd.DataFrame({"feature": feats, "pct": 100 * (np.exp(b[:len(feats)]) - 1),
                        "lo": 100 * (np.exp(b[:len(feats)] - 1.96 * s[:len(feats)]) - 1),
                        "hi": 100 * (np.exp(b[:len(feats)] + 1.96 * s[:len(feats)]) - 1)})
    print(f"change model R² {r2:.2f}\n" + out.round(1).to_string(index=False))
    out.round(2).to_csv("output/analysis/forward_test_drivers.csv", index=False)
    return out.set_index("feature")


def plot(n, r_persist, r_catch, dr, nb_total):
    """Two slides (2026-11-02): 30_forward_test (reputation lasts, and is priced in) and 30b_what_moved (what moved
    the same buildings against Warsaw)."""
    zo = n.loc["Stary Żoliborz"]
    fig = page("Can you get rich buying the next Żoliborz?",
               "Probably not: a place's reputation lasts for years, and it's already in the price",
               f"{len(n)} neighbourhoods, the same {nb_total:,} buildings in both periods (≥ {MIN_RES} resales each). Reputation = "
               "price vs what location, transit, amenities, noise and building age predict. " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"Stary Żoliborz sold {zo['rep1_pct']:+.0f} % above its prediction in 2010–14 and "
                               f"{zo['rep2_pct']:+.0f} % in 2021–25: reputation barely moves (correlation {r_persist:+.2f}). But "
                               f"the cheap-for-their-reputation places did not rise faster, nor the dear ones slower "
                               f"(correlation {r_catch:+.2f}). The market already knows.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    lim = (-28, 32)
    for k, (ax_pos, ycol, ytitle, title, labels) in enumerate((
            ([0.12, 0.47, 0.80, 0.31], "rep2_pct", "Reputation 2021–25", "Then and now: the same places stay dear or cheap",
             LABELS_PERSIST),
            ([0.12, 0.08, 0.80, 0.29], "d_pct", "Price change vs Warsaw",
             "Reputation in 2014 vs what prices did next: no pattern", LABELS_CHANGE))):
        ax = fig.add_axes(ax_pos)
        ax.axhline(0, color=style.AXIS, lw=1)
        ax.axvline(0, color=style.AXIS, lw=1)
        if k == 0:
            ax.plot(lim, lim, color=style.MUTED, lw=1, ls="--")
            ax.text(24, 16, "same as\nin 2014", fontsize=8.5, color=style.MUTED, ha="left", va="top")
        ax.scatter(n["rep1_pct"], n[ycol], s=14 + n["buildings"] / 2, color=style.BLUE, alpha=0.75, lw=0)
        for nm, (dx, dy, ha) in labels.items():
            if nm in n.index:
                ax.annotate(nm, (n.loc[nm, "rep1_pct"], n.loc[nm, ycol]), xytext=(dx, dy), textcoords="offset points",
                            fontsize=8, color=style.INK_2, ha=ha)
        ax.set_xlim(*lim)
        ax.set_ylim(*(lim if k == 0 else (-13, 13)))
        ax.set_xlabel("Reputation 2010–14")
        ax.set_ylabel(ytitle)
        style.pct(ax, "x")
        style.pct(ax, "y")
        ax.set_title(title, fontsize=11.5, loc="left")
    save(fig, "30_forward_test.png")

    # what moved the same buildings ----------------------------------------------------------------
    rows = dr.loc[[f for f, _ in SHOW]][::-1]
    pre, ext, d90 = dr.loc["age_prewar", "pct"], dr.loc["m2_ext", "pct"], dr.loc[["age_1991_00", "age_2001_10"], "pct"].mean()
    fig = page("What made old buildings gain?",
               "Pre-war blocks and blocks near a new metro station gained; the 1990s–2000s ones lost their shine",
               f"The same {nb_total:,} buildings, 2010–14 vs 2021–25, against Warsaw as a whole, compared within their district; lines = 95 % "
               "range, grey = no clear effect. New station = within 800 m. " + style.SOURCE)
    fig.text(0.03, 0.885, wrap(f"Between 2010–14 and 2021–25, against the city: pre-war blocks {pre:+.1f} %, blocks near a new M2 "
                               f"station {ext:+.1f} %, blocks from 1991–2010 {d90:+.1f} %. Cafés, trees and parks: nothing measurable. "
                               "The metro extension is the closest thing here to an experiment.", 100),
             fontsize=10.5, fontweight="semibold", va="top")
    ax = fig.add_axes([0.42, 0.10, 0.52, 0.68])
    y = np.arange(len(rows))
    sig = (rows["lo"] > 0) | (rows["hi"] < 0)
    col = np.where(~sig, style.CONTEXT, np.where(rows["pct"] > 0, style.BLUE, style.ORANGE))
    ax.barh(y, rows["pct"], color=col, height=0.62)
    ax.errorbar(rows["pct"], y, xerr=[rows["pct"] - rows["lo"], rows["hi"] - rows["pct"]], fmt="none",
                ecolor=style.INK_2, elinewidth=1, capsize=0)
    for yi, (p, h, l) in enumerate(zip(rows["pct"], rows["hi"], rows["lo"])):
        ax.text(max(h, 0) + 0.4 if p >= 0 else min(l, 0) - 0.4, yi, f"{p:+.1f} %", va="center",
                ha="left" if p >= 0 else "right", fontsize=9.5)
    ax.set_yticks(y)
    ax.set_yticklabels([dict(SHOW)[f] for f in rows.index], fontsize=10)
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.set_xlim(-11, 13)
    ax.set_xticks(range(-10, 11, 5))
    style.pct(ax, "x")
    style.vgrid_only(ax)
    ax.set_xlabel("Price change vs Warsaw, 2010–14 → 2021–25")
    save(fig, "30b_what_moved.png")


def main():
    v = buildings()
    n, r_persist, r_catch = neighbourhoods(v)
    dr = drivers(v)
    plot(n, r_persist, r_catch, dr, len(v))


if __name__ == "__main__":
    main()
