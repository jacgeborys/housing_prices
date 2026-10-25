"""
Which model? The same question (what a typical flat in a building costs, from its features) put to several model
families, all scored the same way, plus double machine learning for a few price tags. Learning / robustness
exercise; nothing downstream reads its outputs.

Data: output/price_drivers.gpkg (features, as price_drivers.py built them) + output/price_model.gpkg (unshrunk
building value and its sampling variance). Target log(value_2025_raw), as in price_drivers.py.

Scoring: spatial 5-fold cross-validation on 2 km blocks (a building is never predicted by a model that saw its
neighbours), R² weighted by precision 1 / (tau² + fe_var) so thinly sold buildings count less, and the median
absolute miss in %.

Models: district average only; linear regression (OLS) with and without district; lasso; k-nearest neighbours;
random forest; histogram gradient boosting (the one price_drivers.py uses); XGBoost; a small neural network (MLP);
and gradient boosting given the coordinates (it can then learn places, i.e. it absorbs the reputation).

Double ML (partially linear model, Chernozhukov et al. 2018): for a feature D, gradient boosting predicts the price
and D from every other feature + district (cross-fitted on the same blocks); the slope of price residual on D
residual is D's price tag with flexible controls, SE clustered by 1 km cell. Compared with the OLS tag.

Outputs: output/analysis/ml_compare.csv, ml_compare_dml.csv, ml_compare.png
"""
import time

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

import style
from summary_charts import wrap
from summary_maps import FEATURES

style.apply()

K, SEED = 5, 0
OUT = "output/analysis/ml_compare"
DML = {"canopy_100": ("Tree cover within 100 m", 10, "per 10 pp"),
       "age_2011_20": ("Built 2011-20 vs 1961-90", 1, ""),
       "walk_metro_km": ("Walk to metro", 1, "per km"),
       "walk7_food_drink": ("Cafés & restaurants, 7-min walk", 1, "per doubling"),
       "noise_road_db": ("Road noise at façade", 10, "per 10 dB")}


def load():
    v = gpd.read_file("output/price_drivers.gpkg")
    bv = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True,
                       columns=["osm_id", "value_2025_raw", "fe_var"]).drop_duplicates("osm_id").set_index("osm_id")
    v["y"] = np.log(v["osm_id"].map(bv["value_2025_raw"]))
    v["fe_var"] = v["osm_id"].map(bv["fe_var"])
    v = v[v["y"].notna() & v["fe_var"].notna()].reset_index(drop=True)
    c = v.geometry.representative_point()
    v["x"], v["yy"] = c.x.to_numpy(), c.y.to_numpy()
    v["block"] = (v["x"] // 2000).astype(int) * 10000 + (v["yy"] // 2000).astype(int)
    v["cell"] = (v["x"] // 1000).astype(int) * 10000 + (v["yy"] // 1000).astype(int)
    rng = np.random.default_rng(SEED)
    blocks = v["block"].unique()
    v["fold"] = v["block"].map(dict(zip(blocks, rng.permutation(len(blocks)) % K)))
    return v


def models():
    scaled = lambda m: make_pipeline(StandardScaler(), m)
    return {
        "Linear regression": (lambda: LinearRegression(), "feat"),
        "Linear regression + district": (lambda: LinearRegression(), "feat+district"),
        "Lasso (shrinks weak tags to 0)": (lambda: scaled(LassoCV(cv=5, random_state=SEED)), "feat+district"),
        "k-nearest neighbours (k = 25)": (lambda: scaled(KNeighborsRegressor(25, weights="distance")), "feat"),
        "Random forest": (lambda: RandomForestRegressor(400, min_samples_leaf=5, max_features=0.4, n_jobs=-1,
                                                        random_state=SEED), "feat"),
        "Gradient boosting (current)": (lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            random_state=SEED), "feat"),
        "XGBoost": (lambda: XGBRegressor(n_estimators=600, learning_rate=0.03, max_depth=6, subsample=0.8,
                                         colsample_bytree=0.7, min_child_weight=10, n_jobs=8,
                                         random_state=SEED), "feat"),
        "Neural network (MLP 64-32)": (lambda: scaled(MLPRegressor((64, 32), alpha=1e-3, early_stopping=True,
                                                                   max_iter=500, random_state=SEED)), "feat"),
        "Gradient boosting + coordinates": (lambda: HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0,
            random_state=SEED), "feat+xy"),
    }


def matrix(v, kind):
    X = v[FEATURES].to_numpy(float)
    if "district" in kind:
        X = np.column_stack([X, pd.get_dummies(v["district"]).to_numpy(float)])
    if "xy" in kind:
        X = np.column_stack([X, v[["x", "yy"]].to_numpy()])
    return X


def oof(make, X, y, fold):
    pred = np.zeros(len(y))
    for f in range(K):
        tr, te = fold != f, fold == f
        pred[te] = make().fit(X[tr], y[tr]).predict(X[te])
    return pred


def score(y, pred, w):
    r2 = 1 - (w * (y - pred) ** 2).sum() / (w * (y - np.average(y, weights=w)) ** 2).sum()
    return r2, 100 * np.median(np.abs(np.exp(y - pred) - 1))


def compare(v):
    y, fold = v["y"].to_numpy(), v["fold"].to_numpy()
    # precision weights: tau² = spread of buildings around a good prediction beyond their sampling noise
    base = oof(models()["Gradient boosting (current)"][0], matrix(v, "feat"), y, fold)
    tau2 = max(np.var(y - base) - v["fe_var"].mean(), 1e-4)
    w = 1 / (tau2 + v["fe_var"].to_numpy())
    rows = []
    pred = np.zeros(len(y))                                       # district average, fitted out of fold
    for f in range(K):
        tr = fold != f
        means = v[tr].groupby("district")["y"].mean()
        pred[fold == f] = v.loc[fold == f, "district"].map(means).fillna(y[tr].mean())
    rows.append(("District average only", *score(y, pred, w), 0.0))
    for name, (make, kind) in models().items():
        t = time.time()
        pred = oof(make, matrix(v, kind), y, fold)
        rows.append((name, *score(y, pred, w), time.time() - t))
        print(f"  {name:34s} R² {rows[-1][1]:.3f}  median miss {rows[-1][2]:.1f} %  {rows[-1][3]:.0f} s", flush=True)
    return pd.DataFrame(rows, columns=["model", "r2", "median_miss_pct", "seconds"])


def cluster_slope(ry, rd, groups):
    b = (rd * ry).sum() / (rd * rd).sum()
    e = ry - b * rd
    s = pd.Series(rd * e).groupby(groups).sum().to_numpy()
    G = len(s)
    se = np.sqrt((s ** 2).sum() * G / (G - 1)) / (rd * rd).sum()
    return b, se


def ols_tags(v):
    X = np.column_stack([np.ones(len(v)), matrix(v, "feat+district")])
    y = v["y"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ beta
    XtX = np.linalg.pinv(X.T @ X)
    S = pd.DataFrame(X * e[:, None]).groupby(v["cell"].to_numpy()).sum().to_numpy()
    G = S.shape[0]
    se = np.sqrt(np.diag(XtX @ (S.T @ S) @ XtX * G / (G - 1)))
    return {f: (beta[1 + i], se[1 + i]) for i, f in enumerate(FEATURES)}


def dml(v):
    y, fold, groups = v["y"].to_numpy(), v["fold"].to_numpy(), v["cell"].to_numpy()
    make = models()["Gradient boosting (current)"][0]
    ols = ols_tags(v)
    rows = []
    for f, (label, unit, per) in DML.items():
        others = [c for c in FEATURES if c != f]
        X = np.column_stack([v[others].to_numpy(float), pd.get_dummies(v["district"]).to_numpy(float)])
        d = v[f].to_numpy(float)
        ry, rd = y - oof(make, X, y, fold), d - oof(make, X, d, fold)
        b, se = cluster_slope(ry, rd, groups)
        for method, (bb, ss) in (("Linear regression", ols[f]), ("Double ML", (b, se))):
            rows.append({"feature": f, "label": label, "per": per, "method": method,
                         "pct": 100 * (np.exp(unit * bb) - 1), "lo": 100 * (np.exp(unit * (bb - 1.96 * ss)) - 1),
                         "hi": 100 * (np.exp(unit * (bb + 1.96 * ss)) - 1)})
        print(f"  {label}: OLS {rows[-2]['pct']:+.1f} % ({rows[-2]['lo']:+.1f}..{rows[-2]['hi']:+.1f}), "
              f"DML {rows[-1]['pct']:+.1f} % ({rows[-1]['lo']:+.1f}..{rows[-1]['hi']:+.1f})", flush=True)
    return pd.DataFrame(rows)


def plot(cmp, tags, n):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw={"width_ratios": [1, 1.1]})
    fig.subplots_adjust(top=0.8, bottom=0.16, left=0.2, right=0.97, wspace=0.75)
    c = cmp.sort_values("r2")
    colr = [style.CONTEXT_DARK if "coordinates" in m else style.MUTED if "District" in m else style.BLUE
            for m in c["model"]]
    a1.barh(c["model"], c["r2"], color=colr, height=0.65)
    for i, (r2, miss) in enumerate(zip(c["r2"], c["median_miss_pct"])):
        a1.text(r2 + 0.01, i, f"{r2:.2f}  (typical miss {miss:.0f} %)", va="center", fontsize=9.5)
    a1.set_xlim(0, 1)
    a1.set_title("How much of the price differences each model explains\n(buildings it has never seen, 2 km away)",
                 fontsize=11.5, loc="left")
    style.vgrid_only(a1)
    labels = list(DML)
    yy = np.arange(len(labels))
    for off, method, colr in ((0.15, "Linear regression", style.CONTEXT_DARK), (-0.15, "Double ML", style.ORANGE)):
        t = tags[tags["method"] == method].set_index("feature").loc[labels]
        a2.errorbar(t["pct"], yy + off, xerr=[t["pct"] - t["lo"], t["hi"] - t["pct"]], fmt="o", ms=7, color=colr,
                    ecolor=colr, elinewidth=2, capsize=0, label=method)
    t = tags.drop_duplicates("feature").set_index("feature").loc[labels]
    a2.set_yticks(yy)
    a2.set_yticklabels([f"{r.label}\n{r.per}" if isinstance(r.per, str) and r.per else r.label for r in t.itertuples()])
    a2.axvline(0, color=style.AXIS, lw=1)
    a2.set_xscale("symlog", linthresh=5)
    a2.set_xticks([-10, -5, 0, 5, 10, 20, 40])
    style.pct(a2, "x")
    style.vgrid_only(a2)
    a2.legend(loc="lower right")
    a2.set_title("Price tags, within the same district (95 % CI):\nstraight-line corrections vs flexible ML ones",
                 fontsize=11.5, loc="left")
    style.header(fig, "Which model? The choice changes little",
                 f"{n:,} buildings, same features, same out-of-sample test. Without the district, flexible ML beats a "
                 "straight line by 4-6 points; given the district, the straight line does as well")
    style.footer(fig, wrap("Target: typical-flat value 2025 (log), buildings with ≥ 3 sales. R² precision-weighted, spatial "
                      "5-fold CV on 2 km blocks; typical miss = median |error| in %. Double ML: gradient boosting "
                      "predicts price and the feature from all other features + district (cross-fitted), slope of "
                      "residual on residual, SEs clustered by 1 km cell. Associations, not causal effects. "
                      + style.SOURCE, 230))
    fig.savefig(OUT + ".png", dpi=150)
    plt.close(fig)
    print(f"Saved {OUT}.png")


def main():
    v = load()
    print(f"{len(v):,} buildings, {len(FEATURES)} features")
    cmp = compare(v)
    cmp.round(4).to_csv(OUT + ".csv", index=False)
    print("Double ML:")
    tags = dml(v)
    tags.round(2).to_csv(OUT + "_dml.csv", index=False)
    plot(cmp, tags, len(v))


if __name__ == "__main__":
    main()
