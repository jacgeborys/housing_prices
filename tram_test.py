"""
Do trams count for more than buses? Per-building access by mode from TransitFrequency's 06_building_modes.py
(read only: walking minutes to the nearest stop of each mode, departures 06-22 within a 10-minute walk, each line and
direction once; tram sections opened since 2015).

1. Cross-section (output/price_drivers_modes.gpkg, price_drivers.py --transit modes): log typical-flat value 2025 on
   every non-transit feature of price_drivers.py (rail/tram noise at the facade included, so "near the tracks" is not
   "loud") + per mode either
     bands     walking minutes to the nearest tram / bus stop in bands; metro and rail walking minutes linear
     service   log2(1 + departures within 10 min) per mode
   with district or neighbourhood (MSI, 143 areas) fixed effects, OLS, SEs clustered by 1 km cell. Double ML
   (gradient boosting, ml_compare.py) for tram and bus departures, district as a control.
2. New tram lines (opened 2015, 2017, 2021): resales of the same buildings, city index removed (price_m2_2025),
   median 2-5 years before opening vs 1-5 years after; buildings with a new section's stop within 10 minutes
   vs the rest of their district, OLS with district x section and building-decade fixed effects.

Outputs: output/analysis/tram_test.csv, tram_test_new_lines.csv, tram_test.png
"""
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style
from ml_compare import K, SEED, cluster_slope, models, oof
from price_drivers import ols
from summary_charts import wrap
from summary_maps import FEATURES

style.apply()

MODES = "D:/QGIS/TransitFrequency/_data/warsaw/2026_10_04/buildings_modes.gpkg"   # read only
OUT = "output/analysis/tram_test"
OLD_TRANSIT = ["log2_trips", "walk_metro_km", "walk_rail_km", "walk_tram_km", "walk_bus_km"]
TRAM_BANDS = [(0, 3, "≤ 3 min (on the tram street)"), (3, 6, "3–6 min"), (6, 10, "6–10 min"), (10, 15, "10–15 min")]
BUS_BANDS = [(0, 2, "≤ 2 min"), (2, 4, "2–4 min"), (4, 6, "4–6 min")]          # bus reference: > 6 min
NEW_LINES = {20150214: "Bemowo (Powstańców Śląskich), 2015", 20170227: "Tarchomin – Nowodwory, 2017",
             20210904: "Nowodwory – Winnica, 2021"}
PRE, POST = (-5, -2), (1, 5)                                                     # years relative to opening


def load():
    v = gpd.read_file("output/price_drivers_modes.gpkg")
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True,
                       columns=["osm_id", "value_2025_raw"]).drop_duplicates("osm_id").set_index("osm_id")
    v["y"] = np.log(v["osm_id"].map(bv["value_2025_raw"]))
    m = gpd.read_file(MODES, ignore_geometry=True).drop_duplicates("osm_id").set_index("osm_id")
    for c in ("walk_metro", "walk_tram", "walk_bus", "walk_rail", "dep10_metro", "dep10_tram", "dep10_bus",
              "dep10_rail", "newtram10"):
        v[c] = v["osm_id"].map(m[c]).to_numpy()
    v = v[v["y"].notna() & v["walk_tram"].notna()].reset_index(drop=True)
    msi = gpd.read_file("data/msi.gpkg").to_crs(2180).rename(columns={"name": "msi_name"})
    pts = gpd.GeoDataFrame(geometry=v.geometry.representative_point(), crs=2180)
    j = gpd.sjoin(pts, msi[["msi_name", "geometry"]], predicate="within", how="left")
    v["msi"] = j[~j.index.duplicated()]["msi_name"].fillna(v["district"]).to_numpy()
    c = v.geometry.representative_point()
    v["cell"] = (c.x // 1000).astype(int) * 10000 + (c.y // 1000).astype(int)
    v["block"] = (c.x // 2000).astype(int) * 10000 + (c.y // 2000).astype(int)
    rng = np.random.default_rng(SEED)
    blocks = v["block"].unique()
    v["fold"] = v["block"].map(dict(zip(blocks, rng.permutation(len(blocks)) % K)))
    for lo, hi, lab in TRAM_BANDS:
        v[f"tram_{lo}_{hi}"] = ((v["walk_tram"] > lo) & (v["walk_tram"] <= hi)).astype(float)
    for lo, hi, lab in BUS_BANDS:
        v[f"bus_{lo}_{hi}"] = ((v["walk_bus"] > lo) & (v["walk_bus"] <= hi)).astype(float)
    v["metro_5min"] = v["walk_metro"] / 5
    v["rail_5min"] = v["walk_rail"] / 5
    for mo in ("metro", "tram", "bus", "rail"):
        v[f"log2_dep_{mo}"] = np.log2(1 + v[f"dep10_{mo}"])
    return v


SPECS = {
    "bands": [f"tram_{lo}_{hi}" for lo, hi, _ in TRAM_BANDS] + [f"bus_{lo}_{hi}" for lo, hi, _ in BUS_BANDS]
             + ["metro_5min", "rail_5min"],
    "service": ["log2_dep_tram", "log2_dep_bus", "log2_dep_metro", "log2_dep_rail"],
}
LABEL = {**{f"tram_{lo}_{hi}": f"Tram stop {lab}" for lo, hi, lab in TRAM_BANDS},
         **{f"bus_{lo}_{hi}": f"Bus stop {lab}" for lo, hi, lab in BUS_BANDS},
         "metro_5min": "Metro, per 5 min further", "rail_5min": "Rail station, per 5 min further",
         "log2_dep_tram": "Tram departures within 10 min (×2)", "log2_dep_bus": "Bus departures within 10 min (×2)",
         "log2_dep_metro": "Metro departures within 10 min (×2)",
         "log2_dep_rail": "Rail departures within 10 min (×2)"}


def cross_section(v):
    controls = [f for f in FEATURES if f not in OLD_TRANSIT]
    rows = []
    for spec, feats in SPECS.items():
        for fe in ("district", "msi"):
            X = np.column_stack([v[feats + controls].to_numpy(float), pd.get_dummies(v[fe], drop_first=True)
                                 .to_numpy(float)])
            b, s, r2 = ols(X, v["y"].to_numpy(), v["cell"].to_numpy())
            for i, f in enumerate(feats):
                rows.append({"spec": spec, "fe": fe, "feature": f, "label": LABEL[f],
                             "pct": 100 * (np.exp(b[i]) - 1), "lo": 100 * (np.exp(b[i] - 1.96 * s[i]) - 1),
                             "hi": 100 * (np.exp(b[i] + 1.96 * s[i]) - 1), "n": int(v[f].gt(0).sum()), "r2": r2})
    out = pd.DataFrame(rows)
    print(out.round(2).to_string(index=False))
    return out


def double_ml(v):
    controls = [f for f in FEATURES if f not in OLD_TRANSIT]
    make = models()["Gradient boosting (current)"][0]
    y, fold = v["y"].to_numpy(), v["fold"].to_numpy()
    rows = []
    for f in ("log2_dep_tram", "log2_dep_bus"):
        others = controls + [c for c in SPECS["service"] if c != f]
        X = np.column_stack([v[others].to_numpy(float), pd.get_dummies(v["district"]).to_numpy(float)])
        d = v[f].to_numpy(float)
        b, se = cluster_slope(y - oof(make, X, y, fold), d - oof(make, X, d, fold), v["cell"].to_numpy())
        rows.append({"spec": "service", "fe": "double ML", "feature": f, "label": LABEL[f],
                     "pct": 100 * (np.exp(b) - 1), "lo": 100 * (np.exp(b - 1.96 * se) - 1),
                     "hi": 100 * (np.exp(b + 1.96 * se) - 1)})
        print(f"  double ML {LABEL[f]}: {rows[-1]['pct']:+.2f} % ({rows[-1]['lo']:+.2f}..{rows[-1]['hi']:+.2f})")
    return pd.DataFrame(rows)


def new_lines(v):
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                       columns=["year", "osm_id", "market", "price_m2_2025"])
    tx = tx[(tx["market"] != "primary") & tx["osm_id"].isin(v["osm_id"])]
    tx["lp"] = np.log(tx["price_m2_2025"])
    b = v.set_index("osm_id")[["district", "decade_end", "newtram10", "cell"]]
    rows = []
    for opened, name in NEW_LINES.items():
        yr = opened // 10000
        pre = tx[tx["year"].between(yr + PRE[0], yr + PRE[1])].groupby("osm_id")["lp"].agg(["median", "size"])
        post = tx[tx["year"].between(yr + POST[0], yr + POST[1])].groupby("osm_id")["lp"].agg(["median", "size"])
        both = pre.join(post, lsuffix="_pre", rsuffix="_post", how="inner")
        both = both[(both["size_pre"] >= 2) & (both["size_post"] >= 2)].join(b)
        treated = both["newtram10"] == opened
        dist = both.loc[treated, "district"].unique()
        both = both[both["district"].isin(dist) & (treated | (both["newtram10"] == 0))]
        both["treated"] = (both["newtram10"] == opened).astype(float)
        both["change"] = both["median_post"] - both["median_pre"]
        both["section"] = name
        rows.append(both.reset_index())
        print(f"  {name}: {int(both['treated'].sum())} buildings in reach, {int((1 - both['treated']).sum())} "
              f"comparison in {', '.join(dist)}")
    d = pd.concat(rows, ignore_index=True)
    X = np.column_stack([d["treated"], pd.get_dummies(d["district"] + d["section"], drop_first=True).to_numpy(float),
                         pd.get_dummies(d["decade_end"].fillna(-1), drop_first=True).to_numpy(float)])
    bb, ss, _ = ols(X, d["change"].to_numpy(), d["cell"].to_numpy())
    res = {"pct": 100 * (np.exp(bb[0]) - 1), "lo": 100 * (np.exp(bb[0] - 1.96 * ss[0]) - 1),
           "hi": 100 * (np.exp(bb[0] + 1.96 * ss[0]) - 1), "treated": int(d["treated"].sum()),
           "control": int((1 - d["treated"]).sum())}
    print(f"  new tram line within 10 min, change vs rest of district: {res['pct']:+.1f} % "
          f"({res['lo']:+.1f}..{res['hi']:+.1f}), {res['treated']} vs {res['control']} buildings")
    per = d.groupby(["section", "treated"])["change"].mean().unstack()
    print((100 * (np.exp(per) - 1)).round(1).to_string())
    return d, res


def plot(cs, dml, nl, n):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.2, right=0.97, wspace=0.55)
    feats = SPECS["bands"][:len(TRAM_BANDS) + len(BUS_BANDS)]
    yy = np.arange(len(feats))[::-1]
    for off, fe, colr, lab in ((0.15, "district", style.BLUE, "Within the same district"),
                               (-0.15, "msi", style.ORANGE, "Within the same neighbourhood")):
        t = cs[(cs["spec"] == "bands") & (cs["fe"] == fe)].set_index("feature").loc[feats]
        a1.errorbar(t["pct"], yy + off, xerr=[t["pct"] - t["lo"], t["hi"] - t["pct"]], fmt="o", ms=7, color=colr,
                    ecolor=colr, elinewidth=2, capsize=0, label=lab)
    a1.set_yticks(yy)
    a1.set_yticklabels([LABEL[f] for f in feats])
    a1.axvline(0, color=style.AXIS, lw=1)
    a1.axhline(len(BUS_BANDS) - 0.5, color=style.GRID, lw=1)
    style.pct(a1, "x")
    style.vgrid_only(a1)
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, frameon=False)
    a1.set_title("Price vs walking time to the nearest stop, all else equal\n(tram vs > 15 min; bus vs > 6 min; 95 % CI)",
                 loc="left", fontsize=11.5)
    sv = ["log2_dep_tram", "log2_dep_bus", "log2_dep_metro", "log2_dep_rail"]
    rows = []
    for f in sv:
        t = cs[(cs["spec"] == "service") & (cs["fe"] == "district") & (cs["feature"] == f)].iloc[0]
        rows.append((LABEL[f].replace(" within 10 min", ""), "Within district (OLS)", t, style.BLUE))
        if f in set(dml["feature"]):
            rows.append(("", "Double ML", dml[dml["feature"] == f].iloc[0], style.CONTEXT_DARK))
    rows.append(("New tram line in reach,\nresale change vs district", "Before / after", nl, style.GREEN))
    yy = np.arange(len(rows))[::-1]
    for y_, (lab, kind, t, colr) in zip(yy, rows):
        a2.errorbar([t["pct"]], [y_], xerr=[[t["pct"] - t["lo"]], [t["hi"] - t["pct"]]], fmt="o", ms=7, color=colr,
                    ecolor=colr, elinewidth=2, capsize=0)
        if kind == "Before / after":
            a2.text(t["pct"], y_ + 0.3, f"{kind}: {t['treated']} buildings near a new line vs {t['control']}",
                    ha="center", va="bottom", fontsize=9, color=colr)
        else:
            a2.text(t["hi"] + 0.3, y_, kind, va="center", fontsize=9, color=colr)
    a2.set_yticks(yy)
    a2.set_yticklabels([r[0] for r in rows])
    a2.axvline(0, color=style.AXIS, lw=1)
    style.pct(a2, "x")
    style.vgrid_only(a2)
    a2.set_title("Twice the departures within a 10-minute walk, by mode;\nnew tram lines before and after",
                 loc="left", fontsize=11.5)
    style.header(fig, "Trams don't count for more than buses: near a stop or not, prices are the same",
                 f"{n:,} buildings; access by mode from walking isochrones (TransitFrequency), every other feature "
                 "held equal, tram and train noise at the façade included")
    style.footer(fig, wrap("Target: typical-flat value 2025 (log), buildings with ≥ 3 sales. Controls: every feature of "
                           "the price model except transit (age, centre, cafés and shops, parks, trees, noise, building "
                           "size …) + district or MSI neighbourhood; SEs clustered by 1 km cell. Departures 06–22 h, each "
                           "line and direction once. Double ML: gradient boosting, district as a control. New lines "
                           f"({', '.join(NEW_LINES.values())}): median resale price 2–5 years before vs 1–5 after "
                           "opening, city index removed, buildings with a new stop within a 10-minute walk vs the rest "
                           "of their district, building decade held equal. Associations. " + style.SOURCE, 230))
    fig.savefig(OUT + ".png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}.png")


def main():
    v = load()
    print(f"{len(v):,} buildings")
    cs = cross_section(v)
    dml = double_ml(v)
    pd.concat([cs, dml]).round(3).to_csv(OUT + ".csv", index=False)
    d, nl = new_lines(v)
    d.to_csv(OUT + "_new_lines.csv", index=False)
    plot(cs, dml, nl, len(v))


if __name__ == "__main__":
    main()
