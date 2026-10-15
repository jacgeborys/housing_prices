"""
What explains the price of a building? Stage 2 on top of the hedonic model (price_index.py).

Target: log(value_2025) = what a typical flat in the building would sell for in 2025 (date, floor,
size, new-build effects already removed), buildings with >= MIN_SALES sales.

Features (cheap batch, all from local files):
  transit   departures/day within a 5-min walk (D:/QGIS/TransitFrequency, read only), metro and
            train station distance, tram stops within 500 m
  location  distance to PKiN and to the Vistula, park/forest area within 500 m
  building  height, storeys, footprint, share of developer sales, first recorded sale (new-build proxy)

Models:
  1. OLS on log price (numpy, SEs clustered by 1 km cell), without and with district fixed effects:
     readable premiums ("+x % per doubling of departures").
  2. Gradient boosting (scikit-learn), spatial 5-fold CV on 2 km blocks: out-of-sample R² = how much
     these features explain; out-of-fold prediction -> residual "cheaper/dearer than expected".

Outputs:
  output/price_drivers.gpkg                         layer building_drivers
  output/analysis/price_drivers_coefficients.csv
  output/analysis/16_price_drivers.png
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

import style

style.apply()

MIN_SALES = 3
INDEX = "city"                      # --index district: re-base the target with district price indices
TRANSIT_SNAP_M = 25
WALK_AMEN = "output/walk_amenities.gpkg"   # walk_amenities.py: POIs within a 7-minute walk
TRANSIT = "D:/QGIS/TransitFrequency/_data/warsaw/2026_10_04/buildings_gates.gpkg"   # read only
OSM = "D:/QGIS/mapy_warszawy_misc/data/osm"
WALK = "output/walk_access.gpkg"    # walk_access.py: network walking distances
BDOT_RAW = "D:/QGIS/bdot_basemap/Warszawa/PL.PZGiK.330.1465__OT_BUBD_A.shp"   # read only (ZABYTEK)
NOISE = {s: f"data/noise_{s}_lden.tif" for s in ("road", "rail", "air")}   # fetch_noise.py
NOISE_DB = np.array([45, 57, 62, 67, 72, 77])   # EEA band -> dB (0 = below 55, taken as 45)
FACADE_M = 10                                   # noise = loudest band within this distance of the footprint
OUT_GPKG = "output/price_drivers.gpkg"
OUT_CSV = "output/analysis/price_drivers_coefficients.csv"
OUT_PNG = "output/analysis/16_price_drivers.png"

LABELS = {
    "log2_trips": "Bus/tram/train departures (×2)",
    "walk_metro_km": "Walk to metro (per km)",
    "train_km": "Distance to rail station (per km)",
    "walk_tram_km": "Walk to tram stop (per km)",
    "walk_park_km": "Walk to a park ≥ 2 ha (per km)",
    "walk_school_km": "Walk to a school (per km)",
    "noise_road_db": "Road noise at façade (per 10 dB)",
    "noise_rail_db": "Tram & train noise at façade (per 10 dB)",
    "noise_air_db": "Aircraft noise (per 10 dB)",
    "heritage": "Listed heritage building",
    "centre_km": "Distance to PKiN (per km)",
    "river_km": "Distance to the Vistula (per km)",
    "green_ha_500": "Park/forest ≤ 500 m (per 10 ha)",
    "height_m": "Building height (per 10 m)",
    "pct_primary": "Developer-sale share (0→1)",
    "new_build": "First sale 2015 or later",
    "log_footprint": "Footprint (×2)",
}
AMENITY_LABELS = {   # counts within a 7-minute walk (560 m on footpaths), modelled as log2(1 + n)
    "walk7_food_drink": "Cafés, restaurants, bars, 7-min walk (×2)",
    "walk7_grocery": "Grocery shops, 7-min walk (×2)",
    "walk7_zabka": "Żabka shops, 7-min walk (×2)",
    "walk7_schools": "Schools, kindergartens, 7-min walk (×2)",
    "walk7_health": "Pharmacies, clinics, 7-min walk (×2)",
    "walk7_sport_play": "Playgrounds, gyms, sports, 7-min walk (×2)",
}
SCALE = {"green_ha_500": 10, "height_m": 10, "noise_road_db": 10, "noise_rail_db": 10, "noise_air_db": 10}     # coefficients reported per this many units


def pkin_xy():
    from pyproj import Transformer
    return Transformer.from_crs("EPSG:4326", "EPSG:2180", always_xy=True).transform(21.0060, 52.2318)


def area_within(points, polys, r):
    """Area (ha) of polys within r m of each point (polygons clipped to the buffer)."""
    polys = polys[polys.geom_type.isin(["Polygon", "MultiPolygon"])]
    sidx = polys.sindex
    out = np.zeros(len(points))
    geoms = polys.geometry.values
    for i, p in enumerate(points):
        buf = p.buffer(r, 16)
        idx = sidx.query(buf, predicate="intersects")
        if len(idx):
            out[i] = sum(g.intersection(buf).area for g in geoms[idx]) / 1e4
    return out


def facade_noise(v):
    """Loudest EEA Lden band within FACADE_M of each footprint, as dB, for road / rail / air."""
    import rasterio
    from rasterio.mask import mask
    shapes = v.geometry.to_crs(3035).buffer(FACADE_M)
    out = {}
    for src_name, path in NOISE.items():
        vals = np.zeros(len(v), int)
        with rasterio.open(path) as src:
            for i, g in enumerate(shapes):
                try:
                    a, _ = mask(src, [g], crop=True, filled=True, nodata=0)
                    vals[i] = a.max()
                except ValueError:          # outside the raster
                    vals[i] = 0
        out[f"noise_{src_name}_db"] = NOISE_DB[vals]
        print(f"noise {src_name}: {(vals > 0).mean():.0%} of buildings >= 55 dB Lden")
    return pd.DataFrame(out, index=v.index)


def heritage(pts):
    """1 if the BDOT10k building under the point is a listed monument (ZABYTEK = 'Tak')."""
    b = gpd.read_file(BDOT_RAW).to_crs(2180)
    j = gpd.sjoin(gpd.GeoDataFrame(geometry=pts.values, crs=2180), b[["ZABYTEK", "geometry"]],
                  predicate="within", how="left")
    z = j[~j.index.duplicated()]["ZABYTEK"].eq("Tak").astype(float).to_numpy()
    print(f"heritage: {int(z.sum()):,} listed buildings")
    return z


def district_target(v):
    """value_2025 re-based with the district's own annual index instead of the city index:
    each sale's city deflator exp(-gamma_city(year)) is swapped for exp(-gamma_district(year)),
    averaged (geometric mean) over the building's sales."""
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj", ignore_geometry=True,
                       columns=["osm_id", "year", "district"])
    q = pd.read_csv("output/analysis/price_index_quarterly.csv")
    q["year"] = q["quarter"].str[:4].astype(int)
    city = q.groupby("year")["gamma"].mean()
    di = pd.read_csv("output/analysis/district_index_annual.csv").set_index(["district", "year"])["gamma"]
    g_d = pd.Series(list(zip(tx["district"], tx["year"]))).map(di).to_numpy()
    g_c = tx["year"].map(city).to_numpy()
    tx["lf"] = np.where(np.isfinite(g_d), g_c - g_d, 0.0)       # log factor district vs city
    f = tx.groupby("osm_id")["lf"].mean()
    return v["y"] + v["osm_id"].map(f).fillna(0).to_numpy()


def build_features():
    v = gpd.read_file("output/price_model.gpkg", layer="building_value")
    v = v[v["n"] >= MIN_SALES].drop_duplicates("osm_id").reset_index(drop=True)
    cov = gpd.read_file("output/coverage.gpkg", layer="building_coverage", ignore_geometry=True)
    v = v.merge(cov[["osm_id", "floors", "footprint_m2"]], on="osm_id", how="left")
    v["footprint_m2"] = v["footprint_m2"].fillna(v.geometry.area)
    pts = v.geometry.representative_point()
    xy = np.c_[pts.x, pts.y]

    # transit: own osm_id, else the transit footprint the building's point lies in
    t = gpd.read_file(TRANSIT).to_crs(2180)
    trips = v["osm_id"].map(t.drop_duplicates("osm_id").set_index("osm_id")["max_trips"])
    miss = trips.isna()
    j = gpd.sjoin_nearest(gpd.GeoDataFrame(geometry=v.geometry[miss], crs=2180), t[["max_trips", "geometry"]],
                          max_distance=TRANSIT_SNAP_M, how="left").groupby(level=0)["max_trips"].max()
    trips[miss] = j
    print(f"transit: {(~miss).sum():,} by osm_id, {j.notna().sum():,} by nearest footprint "
          f"(<= {TRANSIT_SNAP_M} m), {trips.isna().sum():,} unmatched (dropped)")
    v["trips"] = trips
    keep = trips.notna().to_numpy()
    v, pts = v[keep].reset_index(drop=True), pts[keep].reset_index(drop=True)
    xy = np.c_[pts.x, pts.y]
    v["log2_trips"] = np.log2(v["trips"] + 10)

    st = gpd.read_file(f"{OSM}/train_stations.gpkg").to_crs(2180)
    train = st[st["station"] != "subway"]
    v["train_km"] = cKDTree(np.c_[train.geometry.x, train.geometry.y]).query(xy)[0] / 1000
    w = gpd.read_file(WALK, ignore_geometry=True).drop_duplicates("osm_id").set_index("osm_id")
    for c in ("metro", "tram", "park", "school"):
        v[f"walk_{c}_km"] = v["osm_id"].map(w[f"walk_{c}_m"]).to_numpy() / 1000
    print(f"walking distances: {v['walk_metro_km'].notna().sum():,} buildings")

    v = v.join(facade_noise(v))
    v["heritage"] = heritage(pts)

    cx, cy = pkin_xy()
    v["centre_km"] = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) / 1000
    ww = gpd.read_file(f"{OSM}/waterways.gpkg").to_crs(2180)
    wisla = ww[ww["name"].fillna("").str.fullmatch("Wisła")].geometry.unary_union
    v["river_km"] = pts.distance(wisla).values / 1000

    green = pd.concat([gpd.read_file(f"{OSM}/{f}.gpkg").to_crs(2180)[["geometry"]]
                       for f in ("parks", "forests")], ignore_index=True)
    green = gpd.GeoDataFrame(geometry=green.geometry.buffer(0), crs=2180)
    print("green area within 500 m ...")
    v["green_ha_500"] = area_within(pts.values, green, 500)

    # everyday amenities within a 7-minute walk on the footpath network (walk_amenities.py)
    am = gpd.read_file(WALK_AMEN, ignore_geometry=True).drop_duplicates("osm_id").set_index("osm_id")
    for c in AMENITY_LABELS:
        v[c] = np.log2(1 + v["osm_id"].map(am[c]).fillna(0).to_numpy())
    print(f"walking amenities: {', '.join(f'{c} median {np.median(2 ** v[c] - 1):.0f}' for c in AMENITY_LABELS)}")

    v["height_m"] = v["height_m"].fillna(v["floors"] * 3).fillna(12)
    v["log_footprint"] = np.log2(v["footprint_m2"].clip(lower=50))
    v["pct_primary"] = v["pct_primary"].fillna(0) / 100        # stored as 0-100
    v["new_build"] = ((v["first_year"] >= 2015) & (v["pct_primary"] > 0.3)).astype(float)
    v["y"] = np.log(v["value_2025"])
    if INDEX == "district":
        v["y"] = district_target(v)
        print("target re-based with district indices")
    v["cell_1km"] = (xy[:, 0] // 1000).astype(int) * 10000 + (xy[:, 1] // 1000).astype(int)
    v["block_2km"] = (xy[:, 0] // 2000).astype(int) * 10000 + (xy[:, 1] // 2000).astype(int)
    return v


def ols(X, y, groups):
    """OLS with cluster-robust (CR1) standard errors."""
    X = np.column_stack([np.ones(len(X)), X])
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = np.unique(groups)
    for g in ug:
        m = groups == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    n, k, G = len(y), X.shape[1], len(ug)
    V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((n - 1) / (n - k))
    r2 = 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return beta[1:], np.sqrt(np.diag(V))[1:], r2


def linear_models(v, feats):
    rows = []
    X = v[feats].to_numpy(float) / np.array([SCALE.get(f, 1) for f in feats])
    y = v["y"].to_numpy()
    b0, s0, r0 = ols(X, y, v["cell_1km"].to_numpy())
    D = pd.get_dummies(v["district"], drop_first=True).to_numpy(float)
    b1, s1, r1 = ols(np.column_stack([X, D]), y, v["cell_1km"].to_numpy())
    for i, f in enumerate(feats):
        rows.append({"feature": f, "label": LABELS[f],
                     "pct": 100 * (np.exp(b0[i]) - 1), "pct_lo": 100 * (np.exp(b0[i] - 1.96 * s0[i]) - 1),
                     "pct_hi": 100 * (np.exp(b0[i] + 1.96 * s0[i]) - 1),
                     "pct_district": 100 * (np.exp(b1[i]) - 1),
                     "pct_district_lo": 100 * (np.exp(b1[i] - 1.96 * s1[i]) - 1),
                     "pct_district_hi": 100 * (np.exp(b1[i] + 1.96 * s1[i]) - 1)})
    _, _, r_d = ols(D, y, v["cell_1km"].to_numpy())
    print(f"OLS R²: features {r0:.3f}, district only {r_d:.3f}, features + district {r1:.3f}")
    return pd.DataFrame(rows), {"ols": r0, "district": r_d, "ols_district": r1}


def boosted(v, feats, k=5, seed=0):
    """Spatial k-fold on 2 km blocks: out-of-fold predictions."""
    rng = np.random.default_rng(seed)
    blocks = v["block_2km"].unique()
    fold_of = dict(zip(blocks, rng.permutation(len(blocks)) % k))
    fold = v["block_2km"].map(fold_of).to_numpy()
    X, y = v[feats].to_numpy(float), v["y"].to_numpy()
    pred = np.zeros(len(v))
    for f in range(k):
        tr, te = fold != f, fold == f
        m = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    full = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                         min_samples_leaf=40, l2_regularization=1.0, random_state=seed).fit(X, y)
    imp = permutation_importance(full, X, y, n_repeats=5, random_state=seed)
    importance = pd.Series(imp.importances_mean, index=feats).sort_values(ascending=False)
    print(f"Boosting, spatial 5-fold on 2 km blocks: out-of-sample R² {r2:.3f}, "
          f"residual SD {np.std(y - pred):.3f} log points")
    return pred, r2, importance


def plot(coef, importance, scores, v):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.35, 1]})
    fig.subplots_adjust(top=0.82, bottom=0.17, left=0.2, right=0.97, wspace=0.55)
    c = coef.set_index("feature").loc[[f for f in importance.index[::-1] if f != "pct_primary"]]
    dev = coef.set_index("feature").loc["pct_primary"]
    y = np.arange(len(c))
    for off, col, colr, lab in ((0.17, "pct", style.CONTEXT_DARK, "Whole city"),
                                (-0.17, "pct_district", style.BLUE, "Within the same district")):
        a1.errorbar(c[col], y + off, xerr=[c[col] - c[f"{col}_lo"], c[f"{col}_hi"] - c[col]],
                    fmt="o", ms=7, color=colr, ecolor=colr, elinewidth=2, capsize=0, label=lab)
    a1.axvline(0, color=style.AXIS, lw=1)
    a1.set_yticks(y)
    a1.set_yticklabels(c["label"])
    style.pct(a1, "x")
    style.vgrid_only(a1)
    a1.set_title("Price effect of each feature, others held equal (95 % CI)", fontsize=11.5)
    a1.text(0.0, -0.13, f"Not shown (off scale): building sold mostly by its developer (new) "
            f"{dev['pct_district']:+.0f} % within the district ({dev['pct']:+.0f} % city-wide)",
            transform=a1.transAxes, fontsize=9.5, color=style.INK_2)
    a1.legend(loc="upper right", ncol=2, borderaxespad=0.3, handletextpad=0.3, columnspacing=1.0)

    imp = importance[::-1] / importance.sum() * 100
    a2.barh(np.arange(len(imp)), imp.values, color=style.BLUE, height=0.65)
    a2.set_yticks(np.arange(len(imp)))
    a2.set_yticklabels([LABELS[f].split(" (")[0] for f in imp.index])
    style.pct(a2, "x", signed=False)
    style.vgrid_only(a2)
    a2.set_title("Share of the model's predictive power", fontsize=11.5)

    style.header(fig, "What makes a Warsaw flat expensive?",
                 f"Location, amenity and building features explain {scores['gbm']:.0%} of the price differences "
                 f"between buildings (out of sample, {len(v):,} buildings); district alone: {scores['district']:.0%}")
    style.footer(fig, "Target: hedonic value of a typical flat in the building, 2025 PLN/m² (date, floor, size and "
                      "new-build effects removed), buildings with ≥ 3 sales. Left: OLS on log price, SEs clustered by "
                      "1 km cell.\nRight: gradient boosting, permutation importance; R² from spatial 5-fold CV on 2 km "
                      "blocks. Transit: TransitFrequency (departures/day within a 5-min walk). " + style.SOURCE)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


def main():
    global INDEX, OUT_GPKG, OUT_CSV, OUT_PNG
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", choices=["city", "district"], default="city")
    INDEX = ap.parse_args().index
    if INDEX == "district":      # robustness run: separate outputs
        OUT_GPKG, OUT_CSV = OUT_GPKG.replace(".gpkg", "_district_index.gpkg"), OUT_CSV.replace(".csv", "_district_index.csv")
        OUT_PNG = OUT_PNG.replace(".png", "_district_index.png")
    v = build_features()
    feats = list(LABELS) + [f for f in AMENITY_LABELS if f in v]
    LABELS.update(AMENITY_LABELS)
    print(v[feats + ["value_2025"]].describe().T[["mean", "min", "50%", "max"]].round(2).to_string())
    coef, scores = linear_models(v, feats)
    pred, r2, importance = boosted(v, feats)
    scores["gbm"] = r2
    v["pred_2025"] = np.exp(pred)
    v["resid_pct"] = 100 * (np.exp(v["y"] - pred) - 1)
    coef.round(2).to_csv(OUT_CSV, index=False)
    print(coef.round(1).to_string(index=False))
    print("Importance:\n" + (importance / importance.sum()).round(3).to_string())
    keep = ["osm_id", "name", "addr_street", "addr_housenumber", "district", "n", "value_2025", "pred_2025",
            "resid_pct", "trips"] + feats + ["geometry"]
    v[keep].to_file(OUT_GPKG, layer="building_drivers", driver="GPKG")
    print(f"Saved {OUT_GPKG}, {OUT_CSV}")
    plot(coef, importance, scores, v)


if __name__ == "__main__":
    main()
