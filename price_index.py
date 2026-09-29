"""
Quality-adjusted (hedonic) price model for Warsaw apartments.

Raw medians mix *when* and *what* was sold: a district that mostly traded in
2012 looks cheap, one full of new-build studios looks dear. This script fits

    log(price/m²) = building FE + quarter FE
                    + area bin + rooms + (floor bin × building height class)
                    + top-floor × height class + primary-market × era  + ε

by within-building demeaning (Frisch–Waugh–Lovell), with standard errors
clustered by building. Transactions that don't sit in a building fall back to a
100 m grid cell as their "location" group.

From that one model we get:
  * a quarterly price index for a constant-quality flat (nominal + real)
  * per-district annual indices (same controls, district-specific time path)
  * the within-building floor premium, top-floor effect and new-build premium
  * every transaction re-priced to 2025 PLN (time adjustment only)
  * a shrunk building value: "a typical 2025 flat in this building costs X/m²"

Outputs:
  output/price_model.gpkg        layers: transactions_adj, building_value
  output/analysis/price_index_quarterly.csv
  output/analysis/district_index_annual.csv
  output/analysis/model_coefficients.csv
  output/analysis/12-15_*.png
"""

import os
import pickle
import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

warnings.filterwarnings("ignore")

OUTPUT_DIR = os.path.join("output", "analysis")
CACHE = os.path.join("output", "cache", "model_input.pkl")
MODEL_GPKG = os.path.join("output", "price_model.gpkg")
REF_YEAR = 2025
MIN_Q_OBS = 300          # quarters thinner than this are not estimated (RCN gap)

AREA_BINS = [0, 25, 30, 35, 40, 45, 60, 75, 90, 120, 1000]
AREA_LABELS = ["<25", "25-30", "30-35", "35-40", "40-45", "45-60", "60-75", "75-90", "90-120", "120+"]
FLOOR_BINS = [-10, 1, 3, 5, 8, 12, 17, 100]
FLOOR_LABELS = ["ground", "2-3", "4-5", "6-8", "9-12", "13-17", "18+"]
HCLASS_LABELS = ["low-rise (≤5 fl.)", "mid-rise (6-10)", "high-rise (11+)"]
ERAS = [(2006, 2010), (2011, 2016), (2020, 2023), (2024, 2026)]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_input(force=False):
    if os.path.exists(CACHE) and not force:
        print(f"Loading cached model input: {CACHE}")
        with open(CACHE, "rb") as f:
            return pickle.load(f)

    from analyze_rcn import load_data
    df = load_data()
    df = df[df["prop_type"] == "apartment"].copy()

    df["date"] = pd.to_datetime(df["dok_data"].str[:10], errors="coerce")
    df = df[df["date"].between("2006-01-01", "2026-12-31")]
    df = df[df["price_m2"].between(2000, 80000) & df["area"].between(12, 300)]
    df = df[df["rooms"].between(1, 8) | df["rooms"].isna()]
    df.loc[~df["floor"].between(0, 60), "floor"] = np.nan

    # The same flat can appear in several RCN record versions
    n0 = len(df)
    df = df.drop_duplicates(subset=["dok_oznaczenie", "lok_adres", "area", "price"])
    print(f"  Dropped {n0 - len(df):,} duplicate records")

    # Building assignment: within footprint, else nearest within 10 m
    print("  Assigning transactions to OSM buildings...")
    bld = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all")[
        ["osm_id", "height_m", "geometry"]]
    pts = df[["geometry"]].copy()
    j = gpd.sjoin(pts, bld, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    miss = j["osm_id"].isna()
    if miss.any():
        near = gpd.sjoin_nearest(pts[miss], bld, how="left", max_distance=10)
        near = near[~near.index.duplicated(keep="first")]
        j.loc[near.index, ["osm_id", "height_m"]] = near[["osm_id", "height_m"]].values
    df["osm_id"] = j["osm_id"]
    df["bld_height"] = j["height_m"]
    print(f"  In a building: {df['osm_id'].notna().mean():.1%}")

    df["x"] = df.geometry.x
    df["y"] = df.geometry.y
    keep = ["date", "year", "district", "osm_id", "bld_height", "x", "y", "price", "price_m2",
            "area", "rooms", "floor", "market", "lok_adres"]
    out = pd.DataFrame(df[keep]).reset_index(drop=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def prepare(df):
    df = df.copy()
    df["q"] = df["date"].dt.to_period("Q")
    df["year"] = df["date"].dt.year
    df["logp"] = np.log(df["price_m2"])

    # Location group: building, else 100 m cell
    cell = ("c" + (df["x"] // 100).astype(int).astype(str) + "_" + (df["y"] // 100).astype(int).astype(str))
    df["grp"] = np.where(df["osm_id"].notna(), "b" + df["osm_id"].astype("Int64").astype(str), cell)

    # Storeys: take the larger of the height-based estimate and the highest sold floor
    max_floor = df.groupby("grp")["floor"].transform("max")
    storeys = np.fmax(np.round(df["bld_height"].fillna(0) / 3.0), max_floor.fillna(0))
    df["storeys"] = storeys
    df["hclass"] = pd.cut(storeys, [-1, 5, 10, 999], labels=HCLASS_LABELS)
    df["area_bin"] = pd.cut(df["area"], AREA_BINS, labels=AREA_LABELS)
    df["floor_bin"] = pd.cut(df["floor"].fillna(-5), FLOOR_BINS, labels=FLOOR_LABELS)
    df.loc[df["floor"].isna(), "floor_bin"] = np.nan
    df["rooms_c"] = df["rooms"].clip(upper=5).fillna(0).astype(int).astype(str)
    # "Top floor" only meaningful where we've seen several floors trade
    n_floors_seen = df.groupby("grp")["floor"].transform("nunique")
    df["top"] = ((df["floor"] == max_floor) & (max_floor >= 3) & (n_floors_seen >= 3)).astype(float)
    df["primary"] = (df["market"] == "primary").astype(float)
    df["era"] = pd.cut(df["year"], [e[0] - 1 for e in ERAS] + [ERAS[-1][1]],
                       labels=[f"{a}-{b}" for a, b in ERAS])
    return df


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def design(df, quarters, ref_q):
    """Dummy design matrix. Reference cells: ref quarter, 45-60 m², 2 rooms,
    floor 2-3 in a mid-rise, secondary market."""
    cols = {}
    for q in quarters:
        if q != ref_q:
            cols[f"q:{q}"] = (df["q"] == q).values
    for a in AREA_LABELS:
        if a != "45-60":
            cols[f"area:{a}"] = (df["area_bin"] == a).values
    for r in ["1", "3", "4", "5", "0"]:
        cols[f"rooms:{r}"] = (df["rooms_c"] == r).values
    for h in HCLASS_LABELS:
        for fb in FLOOR_LABELS:
            if h == HCLASS_LABELS[1] and fb == "2-3":
                continue
            cols[f"floor:{h}|{fb}"] = ((df["hclass"] == h) & (df["floor_bin"] == fb)).values
        cols[f"top:{h}"] = ((df["hclass"] == h).values & (df["top"].values == 1))
    cols["floor:unknown"] = df["floor_bin"].isna().values
    for e in ERAS:
        lab = f"{e[0]}-{e[1]}"
        cols[f"primary:{lab}"] = ((df["era"] == lab).values & (df["primary"].values == 1))
    X = pd.DataFrame(cols, index=df.index).astype(np.float64)
    return X.loc[:, X.std() > 0]


def fit(df):
    q_counts = df["q"].value_counts()
    quarters = sorted(q_counts[q_counts >= MIN_Q_OBS].index)
    est = df[df["q"].isin(quarters)].copy()
    ref_q = pd.Period(f"{REF_YEAR}Q1", "Q")
    X = design(est, quarters, ref_q)
    y = est["logp"]
    g = est["grp"]

    print(f"  Estimation sample: {len(est):,} sales, {g.nunique():,} location groups, "
          f"{X.shape[1]} regressors, {len(quarters)} quarters")

    size = g.map(g.value_counts())
    multi = (size >= 2).values
    Xm, ym, gm = X[multi], y[multi], g[multi]
    Xd = Xm - Xm.groupby(gm.values).transform("mean")
    yd = ym - ym.groupby(gm.values).transform("mean")

    A, b = Xd.values, yd.values
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ beta

    # Cluster-robust (by location group) covariance
    XtX_inv = np.linalg.pinv(A.T @ A)
    S = pd.DataFrame(A * resid[:, None]).groupby(gm.values).sum().values
    G = S.shape[0]
    cov = XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1)
    se = np.sqrt(np.diag(cov))
    coef = pd.DataFrame({"beta": beta, "se": se}, index=X.columns)
    r2 = 1 - resid.var() / b.var()
    print(f"  Within-R²: {r2:.3f}   residual sd: {resid.std():.3f}")

    # Location fixed effects for all groups (singletons included)
    fe = (y - X.values @ beta).groupby(g.values).mean()
    n_g = g.value_counts()
    smear = np.mean(np.exp(resid))
    return {"coef": coef, "beta": beta, "cols": list(X.columns), "fe": fe, "n_g": n_g,
            "quarters": quarters, "ref_q": ref_q, "smear": smear, "sample": est}


def quarter_index(model, df):
    """Quarterly log-index γ_q (ref quarter = 0), interpolated across thin quarters."""
    coef = model["coef"]["beta"]
    se = model["coef"]["se"]
    all_q = pd.period_range(df["q"].min(), df["q"].max(), freq="Q")
    g = pd.Series(np.nan, index=all_q)
    s = pd.Series(np.nan, index=all_q)
    for q in model["quarters"]:
        key = f"q:{q}"
        g[q] = 0.0 if q == model["ref_q"] else coef.get(key, np.nan)
        s[q] = 0.0 if q == model["ref_q"] else se.get(key, np.nan)
    estimated = g.notna()
    g_interp = pd.Series(g.values, index=np.arange(len(g))).interpolate(limit_direction="both")
    out = pd.DataFrame({"gamma": g_interp.values, "se": s.values, "estimated": estimated.values},
                       index=all_q)
    # Rebase: 2025 average = 0
    base = out.loc[[q for q in all_q if q.year == REF_YEAR], "gamma"].mean()
    out["gamma"] -= base
    return out


def district_indices(df, model):
    """Annual index per district: remove the fitted controls, then a
    within-building regression on year dummies inside each district."""
    cols, beta = model["cols"], model["beta"]
    ctrl = [i for i, c in enumerate(cols) if not c.startswith("q:")]
    est = df[df["q"].isin(model["quarters"])].copy()
    X = design(est, model["quarters"], model["ref_q"]).reindex(columns=cols, fill_value=0.0)
    est["r"] = est["logp"] - X.values[:, ctrl] @ beta[ctrl]

    rows = []
    for dist, d in est[est["district"].notna()].groupby("district"):
        years = sorted(y for y, n in d["year"].value_counts().items() if n >= 80)
        if REF_YEAR not in years:
            continue
        d = d[d["year"].isin(years)]
        d = d[d["grp"].map(d["grp"].value_counts()) >= 2]
        Y = pd.DataFrame({y: (d["year"] == y).astype(float) for y in years if y != REF_YEAR},
                         index=d.index)
        Yd = Y - Y.groupby(d["grp"].values).transform("mean")
        rd = d["r"] - d["r"].groupby(d["grp"].values).transform("mean")
        b, *_ = np.linalg.lstsq(Yd.values, rd.values, rcond=None)
        for y in years:
            g = 0.0 if y == REF_YEAR else b[list(Y.columns).index(y)]
            rows.append({"district": dist, "year": y, "gamma": g,
                         "n": int((d["year"] == y).sum())})
    di = pd.DataFrame(rows)
    di["index_2025"] = 100 * np.exp(di["gamma"])
    return di


def building_values(df, model, qidx):
    """Shrunk location effect -> 'typical flat (45-60 m², 2 rooms, floor 2-3,
    mid-rise, resale) in 2025' price/m², per location group."""
    fe, n = model["fe"], model["n_g"]
    info = df.groupby("grp").agg(district=("district", "first"), osm_id=("osm_id", "first"),
                                 x=("x", "median"), y=("y", "median"))
    info["fe"] = fe
    info["n"] = n
    info = info[info["fe"].notna()].copy()   # groups seen only in thin (gap) quarters
    # Empirical-Bayes shrink toward the district mean (κ = noise var / signal var)
    resid_var = np.log(model["smear"]) * 2          # ≈ σ² from the smearing factor
    dist_mean = info.groupby("district")["fe"].transform(lambda s: np.average(s, weights=info.loc[s.index, "n"]))
    dist_mean = dist_mean.fillna(info["fe"].mean())
    signal_var = info.groupby("district")["fe"].transform("var").fillna(info["fe"].var())
    kappa = resid_var / signal_var.clip(lower=1e-4)
    w = info["n"] / (info["n"] + kappa)
    info["fe_shrunk"] = dist_mean + w * (info["fe"] - dist_mean)
    # Reference quarter in the design is 2025Q1 but qidx is rebased to the
    # 2025 average: shift by the 2025Q1 offset so values are 2025-average levels.
    offset = -qidx.loc[model["ref_q"], "gamma"]
    info["value_2025"] = np.exp(info["fe_shrunk"] - offset) * model["smear"]
    return info


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def _q_to_float(q_index):
    return np.array([p.year + (p.quarter - 0.5) / 4 for p in q_index])


def plot_index(df, qidx, bv):
    cpi = style.cpi_level(REF_YEAR)
    level = np.average(bv["value_2025"], weights=bv["n"])  # city-average typical flat, 2025
    qi = qidx.copy()
    qi["t"] = _q_to_float(qi.index)
    qi["nominal"] = level * np.exp(qi["gamma"])
    qi["real"] = qi["nominal"] / np.array([cpi[p.year] for p in qi.index])
    raw = df.groupby("q")["price_m2"].median().reindex(qi.index)
    nobs = df.groupby("q").size().reindex(qi.index).fillna(0)
    raw[nobs < MIN_Q_OBS] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(17, 7.2), gridspec_kw={"width_ratios": [1.55, 1]})
    fig.subplots_adjust(top=0.84, bottom=0.12, wspace=0.18, left=0.05, right=0.98)

    ax = axes[0]
    est = qi["estimated"]
    for col, color, lab in [("nominal", style.BLUE, "Constant-quality flat — nominal"),
                            ("real", style.ORANGE, "Constant-quality flat — real (2025 PLN)")]:
        s = qi[col].where(est)
        ax.plot(qi["t"], s, color=color, lw=2.2, label=lab)
        ax.plot(qi["t"], qi[col].where(~est | est.shift(1, fill_value=False) & ~est.shift(-1, fill_value=True)),
                color=color, lw=1.2, ls=(0, (2, 2)), alpha=0.6)
    ax.plot(qi["t"], raw, color=style.MUTED, lw=1.2, label="Raw quarterly median (mixes quality)")
    style.shade_gap(ax)
    style.pln(ax)
    ax.set_ylabel("PLN per m²")
    ax.set_xlim(2006, qi["t"].max() + 0.3)
    ax.set_xticks(range(2006, 2027, 2))
    ax.legend(loc="lower left", bbox_to_anchor=(0.01, 0.36))
    ax.set_title("Price of the same kind of flat, quarter by quarter")
    last = qi[est].iloc[-1]
    for col, color in [("nominal", style.BLUE), ("real", style.ORANGE)]:
        ax.annotate(style.fmt_int(last[col]), (last["t"], last[col]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9.5, color=style.INK_2)
    peak_real = qi[est & (qi.index.year <= 2012)]["real"]
    pk = peak_real.idxmax()
    ax.annotate(f"2008 peak in today's money\n{style.fmt_int(peak_real.max())} PLN/m²",
                (qi.loc[pk, "t"], peak_real.max()), xytext=(2010.3, 17500), textcoords="data",
                fontsize=9, color=style.INK_2,
                arrowprops=dict(arrowstyle="-", color=style.MUTED, lw=0.8))

    # Annual growth of the adjusted index; gap bridged as one CAGR bar
    ax = axes[1]
    yr = qi[est].groupby(qi[est].index.year)[["nominal", "real"]].mean()
    yr = yr[yr.index <= REF_YEAR]
    years = list(yr.index)
    rows = []
    for a, b in zip(years[:-1], years[1:]):
        span = b - a
        for col in ["nominal", "real"]:
            rows.append({"year": b, "span": span, "kind": col,
                         "g": 100 * ((yr.loc[b, col] / yr.loc[a, col]) ** (1 / span) - 1)})
    gr = pd.DataFrame(rows)
    w = 0.38
    for k, (kind, color) in enumerate([("nominal", style.BLUE), ("real", style.ORANGE)]):
        s = gr[gr["kind"] == kind]
        xs = s["year"] + (k - 0.5) * w
        ax.bar(xs, s["g"], width=w * 0.92, color=color, label=kind.capitalize(), zorder=2)
    for _, r in gr[(gr["kind"] == "nominal") & (gr["span"] > 1)].iterrows():
        ax.annotate(f"{int(r['year'] - r['span'])}→{int(r['year'])}\nannualised",
                    (r["year"], max(r["g"], 0)), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=8.5, color=style.MUTED)
    ax.axhline(0, color=style.AXIS, lw=1)
    style.pct(ax)
    ax.set_title("Annual change, constant-quality index")
    ax.set_xticks(range(2007, 2026, 3))
    ax.legend(loc="upper left")
    style.vgrid_only(ax)

    real_since = 100 * (qi[est & (qi.index.year == REF_YEAR)]["real"].mean() / peak_real.max() - 1)
    nom_mult = yr.loc[REF_YEAR, "nominal"] / yr.loc[2013, "nominal"]
    vs_peak = "only just back to" if abs(real_since) < 3 else f"{real_since:+.0f}% vs"
    style.header(fig, "Warsaw apartment prices, quality-adjusted",
                 f"A like-for-like flat costs {nom_mult:.1f}× its 2013 price in nominal terms, "
                 f"but after inflation it is {vs_peak} the 2008 bubble peak.")
    style.footer(fig, style.SOURCE + " Hedonic index with building fixed effects; " + style.CPI_NOTE
                 + " Dashed = interpolated across thin quarters.")
    path = os.path.join(OUTPUT_DIR, "12_price_index.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  {path}")
    return qi


def plot_districts(df, di, bv, qidx):
    """Slopegraph raw vs adjusted rank + small-multiple district indices."""
    raw = df[df["district"].notna()].groupby("district")["price_m2"].median()
    adj = bv[bv["district"].notna()].groupby("district").apply(
        lambda s: np.exp(np.average(np.log(s["value_2025"]), weights=s["n"])))
    comp = pd.DataFrame({"raw": raw, "adj": adj}).dropna()
    comp["r_raw"] = comp["raw"].rank(ascending=False)
    comp["r_adj"] = comp["adj"].rank(ascending=False)

    fig = plt.figure(figsize=(19, 11))
    gs = fig.add_gridspec(3, 8, left=0.03, right=0.985, top=0.86, bottom=0.07,
                          wspace=0.28, hspace=0.42, width_ratios=[1.25, 1.25, .2, 1, 1, 1, 1, 1])
    ax = fig.add_subplot(gs[:, 0:2])
    n = len(comp)
    for d, r in comp.iterrows():
        moved = abs(r["r_raw"] - r["r_adj"]) >= 3
        c = style.MAGENTA if (r["r_adj"] < r["r_raw"] and moved) else \
            style.BLUE if moved else style.AXIS
        ax.plot([0, 1], [r["r_raw"], r["r_adj"]], color=c, lw=2.2 if moved else 1.2, zorder=2 if moved else 1)
        ax.scatter([0, 1], [r["r_raw"], r["r_adj"]], s=28, color=c, zorder=3, edgecolor=style.SURFACE, lw=1.5)
        ax.text(-0.04, r["r_raw"], f"{d}  {style.fmt_int(r['raw'])}", ha="right", va="center",
                fontsize=9.5, color=style.INK if moved else style.INK_2)
        ax.text(1.04, r["r_adj"], f"{style.fmt_int(r['adj'])}  {d}", ha="left", va="center",
                fontsize=9.5, color=style.INK if moved else style.INK_2,
                fontweight="semibold" if moved else "normal")
    ax.set_xlim(-0.9, 1.9)
    ax.set_ylim(n + 0.7, 0.3)
    ax.axis("off")
    ax.text(0, 0.0, "Raw median\n2006–26, all flats", ha="center", va="bottom", fontsize=10, color=style.INK_2)
    ax.text(1, 0.0, "Typical flat\nin 2025 (model)", ha="center", va="bottom", fontsize=10, color=style.INK_2)
    ax.set_title("Ranking by price per m²", pad=34)
    ax.text(0.5, n + 0.6, "pink = ranks higher once adjusted, blue = lower", ha="center",
            va="top", fontsize=9, color=style.MUTED)

    city = qidx.copy()
    city = city[city["estimated"]].groupby(city[city["estimated"]].index.year)["gamma"].mean()
    city = 100 * np.exp(city - city.get(REF_YEAR, 0))
    order = comp.sort_values("adj", ascending=False).index
    right = gs[:, 3:].subgridspec(3, 6, wspace=0.12, hspace=0.55)
    first = None
    for i, d in enumerate(order[:18]):
        axd = fig.add_subplot(right[i // 6, i % 6], sharey=first)
        first = first or axd
        axd.plot(city.index, city.values, color=style.CONTEXT, lw=1.6)
        s = di[di["district"] == d].sort_values("year")
        for seg in _split_gap(s):
            axd.plot(seg["year"], seg["index_2025"], color=style.BLUE, lw=1.9)
        g16 = s.loc[s["year"] == 2016, "index_2025"]
        chg = f"×{100 / g16.iloc[0]:.1f} since 2016" if len(g16) else ""
        axd.set_title(d, fontsize=10, pad=3)
        if s.empty:
            axd.text(0.5, 0.5, "too few repeat-building\nsales for an index", transform=axd.transAxes,
                     ha="center", va="center", fontsize=8.5, color=style.MUTED)
        axd.text(0.03, 0.93, chg, transform=axd.transAxes, fontsize=8.5, color=style.INK_2, va="top")
        axd.set_xlim(2006, 2026)
        axd.set_xticks([2010, 2016, 2024])
        axd.set_xticklabels(["'10", "'16", "'24"], fontsize=8)
        axd.tick_params(labelsize=8)
        if i % 6:
            plt.setp(axd.get_yticklabels(), visible=False)
    fig.text(0.305, 0.885, "Constant-quality index by district (2025 = 100), gray = city", fontsize=12,
             fontweight="semibold")

    movers = comp.assign(d=comp["r_raw"] - comp["r_adj"]).sort_values("d")
    up, down = movers.index[-1], movers.index[0]
    style.header(fig, "Which districts are really expensive?",
                 f"Pooling 20 years of sales flatters districts that sell a lot now. Adjusted for date and "
                 f"flat mix, {up} climbs {int(movers.loc[up, 'd'])} places, {down} drops {int(-movers.loc[down, 'd'])}.")
    style.footer(fig, style.SOURCE + " Typical flat = 45–60 m², 2 rooms, floor 2–3 of a mid-rise, resale.")
    path = os.path.join(OUTPUT_DIR, "13_districts_adjusted.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  {path}")
    return comp


def _split_gap(s):
    s = s.reset_index(drop=True)
    breaks = np.where(np.diff(s["year"].values) > 1)[0]
    return np.split(s, breaks + 1) if len(s) else []


def plot_floor_premium(model, df):
    coef = model["coef"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8), gridspec_kw={"width_ratios": [1.6, 1]})
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.05, right=0.98, wspace=0.2)
    ax = axes[0]
    colors = [style.AQUA, style.BLUE, style.VIOLET]
    for k, (h, color) in enumerate(zip(HCLASS_LABELS, colors)):
        xs, ys, lo, hi = [], [], [], []
        for i, fb in enumerate(FLOOR_LABELS):
            key = f"floor:{h}|{fb}"
            if h == HCLASS_LABELS[1] and fb == "2-3":
                b, s = 0.0, 0.0
            elif key in coef.index:
                b, s = coef.loc[key, "beta"], coef.loc[key, "se"]
            else:
                continue
            n = ((df["hclass"] == h) & (df["floor_bin"] == fb)).sum()
            if n < 150:
                continue
            xs.append(i + (k - 1) * 0.08)
            ys.append(100 * (np.exp(b) - 1))
            lo.append(100 * (np.exp(b - 1.96 * s) - 1))
            hi.append(100 * (np.exp(b + 1.96 * s) - 1))
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(xs, ys, "-o", color=color, ms=6, label=h, mec=style.SURFACE, mew=1.5)
        ax.annotate(h.split(" ")[0], (xs[-1], ys[-1]), xytext=[(0, 12), (8, 0), (8, 0)][k],
                    textcoords="offset points", va="bottom" if k == 0 else "center",
                    ha="center" if k == 0 else "left", color=style.INK_2, fontsize=9.5)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(range(len(FLOOR_LABELS)))
    ax.set_xticklabels(FLOOR_LABELS)
    ax.set_xlabel("Floor (RCN numbering, ground = 1)")
    style.pct(ax)
    ax.set_ylabel("Price/m² vs floor 2–3 of a mid-rise")
    ax.legend(loc="upper left")
    ax.set_title("Floor premium within the same building (95% CI)")

    ax = axes[1]
    rows = []
    for h in HCLASS_LABELS:
        key = f"top:{h}"
        if key in coef.index:
            b, s = coef.loc[key, "beta"], coef.loc[key, "se"]
            rows.append((h, 100 * (np.exp(b) - 1), 100 * 1.96 * s))
    for i, (h, v, e) in enumerate(rows):
        ax.barh(i, v, color=colors[i], height=0.55, zorder=2)
        ax.errorbar(v, i, xerr=e, color=style.INK_2, capsize=4, lw=1.2, zorder=3)
        ax.text(max(v, 0) + e + 0.15, i, f"{v:+.1f}%", va="center", ha="left",
                fontsize=10, color=style.INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.axvline(0, color=style.AXIS, lw=1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:+.1f}%"))
    ax.set_xlim(-4, 4)
    style.hgrid_only(ax)
    ax.invert_yaxis()
    ax.set_title("Extra effect of being the top floor (95% CI)")

    top_tower = 100 * (np.exp(coef.loc[f"floor:{HCLASS_LABELS[2]}|18+", "beta"]) - 1)
    ground = coef.loc[f"floor:{HCLASS_LABELS[1]}|ground", "beta"] if f"floor:{HCLASS_LABELS[1]}|ground" in coef.index else 0
    style.header(fig, "Does height pay?",
                 f"Comparing flats in the same building and quarter: ground floors sell "
                 f"{100 * (1 - np.exp(ground)):.0f}% below floor 2–3 of a mid-rise, tower flats above floor 17 fetch "
                 f"~{top_tower:.0f}% more — and being the top floor makes no measurable difference.")
    style.footer(fig, style.SOURCE + " Coefficients from the hedonic model; SEs clustered by building.")
    path = os.path.join(OUTPUT_DIR, "14_floor_premium.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  {path}")


def plot_size_and_newbuild(model, df):
    coef = model["coef"]
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.6))
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.05, right=0.98, wspace=0.18)

    ax = axes[0]
    vals, lo, hi = [], [], []
    for a in AREA_LABELS:
        b, s = (0.0, 0.0) if a == "45-60" else (coef.loc[f"area:{a}", "beta"], coef.loc[f"area:{a}", "se"])
        vals.append(100 * (np.exp(b) - 1))
        lo.append(100 * (np.exp(b - 1.96 * s) - 1))
        hi.append(100 * (np.exp(b + 1.96 * s) - 1))
    x = np.arange(len(AREA_LABELS))
    colors = [style.MAGENTA if v > 0.5 else style.BLUE if v < -0.5 else style.AXIS for v in vals]
    ax.bar(x, vals, color=colors, width=0.62, zorder=2)
    ax.errorbar(x, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none",
                color=style.INK_2, capsize=3, lw=1)
    for xi, v, h_, l_ in zip(x, vals, hi, lo):
        lab = "ref." if AREA_LABELS[xi] == "45-60" else f"{v:+.0f}%"
        ax.text(xi, (h_ + 0.6) if v >= 0 else (l_ - 0.6), lab, ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=style.INK_2)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(AREA_LABELS)
    ax.set_xlabel("Usable area (m²)")
    style.pct(ax)
    style.vgrid_only(ax)
    ax.set_title("Price/m² by flat size vs a 45–60 m² flat (same building & quarter)")

    ax = axes[1]
    labs, v2, e2 = [], [], []
    for a, b in ERAS:
        key = f"primary:{a}-{b}"
        if key in coef.index:
            labs.append(f"{a}–{str(b)[2:]}")
            v2.append(100 * (np.exp(coef.loc[key, "beta"]) - 1))
            e2.append(100 * 1.96 * coef.loc[key, "se"])
    ax.bar(range(len(labs)), v2, color=style.AQUA, width=0.55, zorder=2)
    ax.errorbar(range(len(labs)), v2, yerr=e2, fmt="none", color=style.INK_2, capsize=4, lw=1)
    for i, v in enumerate(v2):
        ax.text(i, v + (e2[i] + 0.5) * (1 if v >= 0 else -1), f"{v:+.1f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=10)
    ax.axhline(0, color=style.AXIS, lw=1)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs)
    style.pct(ax)
    style.vgrid_only(ax)
    ax.set_ylim(min(v2) - max(e2) - 5, 2)
    ax.set_title("Developer (primary) vs resale, same building & deed quarter")

    small = vals[0]
    style.header(fig, "Studios carry a premium; developer deeds record a discount",
                 f"Per m², a sub-25 m² studio costs {small:+.0f}% vs a 45–60 m² flat in the same building. Primary sales "
                 f"record {-min(v2):.0f}% less than resales — the deed price was fixed at the preliminary contract, "
                 f"often 1–2 years earlier, for an unfinished flat.")
    style.footer(fig, style.SOURCE + " Hedonic model coefficients, 95% CI clustered by building. "
                 "RCN dates primary sales at the notarial deed, not the reservation.")
    path = os.path.join(OUTPUT_DIR, "15_size_and_newbuild.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_outputs(df, qidx, di, bv, model):
    qi = qidx.copy()
    qi.index = qi.index.astype(str)
    qi.to_csv(os.path.join(OUTPUT_DIR, "price_index_quarterly.csv"), index_label="quarter")
    di.to_csv(os.path.join(OUTPUT_DIR, "district_index_annual.csv"), index=False)
    model["coef"].to_csv(os.path.join(OUTPUT_DIR, "model_coefficients.csv"), index_label="term")

    # Per-transaction time adjustment to 2025 PLN
    gamma_q = qidx["gamma"].reindex(df["q"]).values
    cpi = style.cpi_level(REF_YEAR)
    df = df.copy()
    df["price_m2_2025"] = df["price_m2"] * np.exp(-gamma_q)
    df["price_m2_real"] = df["price_m2"] / df["year"].map(cpi)
    txn = gpd.GeoDataFrame(
        df[["date", "year", "district", "osm_id", "floor", "area", "rooms", "market", "lok_adres",
            "price", "price_m2", "price_m2_2025", "price_m2_real"]].assign(date=df["date"].dt.strftime("%Y-%m-%d")),
        geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:2180")
    if os.path.exists(MODEL_GPKG):
        os.remove(MODEL_GPKG)
    txn.to_file(MODEL_GPKG, layer="transactions_adj", driver="GPKG")

    # Building polygons with model value
    bld = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all")[
        ["osm_id", "height_m", "name", "addr_street", "addr_housenumber", "geometry"]]
    b = bv[bv["osm_id"].notna()].copy()
    b["osm_id"] = b["osm_id"].astype("int64")
    agg = df[df["osm_id"].notna()].groupby("osm_id").agg(
        median_price_m2=("price_m2", "median"), median_price_m2_2025=("price_m2_2025", "median"),
        first_year=("year", "min"), last_year=("year", "max"),
        pct_primary=("market", lambda s: 100 * (s == "primary").mean()))
    agg.index = agg.index.astype("int64")
    b = b.set_index("osm_id").join(agg)
    out = bld.merge(b[["district", "n", "value_2025", "median_price_m2", "median_price_m2_2025",
                       "first_year", "last_year", "pct_primary"]],
                    left_on="osm_id", right_index=True, how="inner")
    out.to_file(MODEL_GPKG, layer="building_value", driver="GPKG")
    print(f"  {MODEL_GPKG}: transactions_adj ({len(txn):,}), building_value ({len(out):,})")


def main():
    style.apply()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Loading data...")
    df = prepare(load_input())
    print(f"  Apartments: {len(df):,}")

    print("\nFitting hedonic model...")
    model = fit(df)
    qidx = quarter_index(model, df)
    print("  District indices...")
    di = district_indices(df, model)
    bv = building_values(df, model, qidx)

    print("\nCharts...")
    plot_index(df, qidx, bv)
    plot_districts(df, di, bv, qidx)
    plot_floor_premium(model, df)
    plot_size_and_newbuild(model, df)

    print("\nSaving...")
    save_outputs(df, qidx, di, bv, model)
    print("Done.")


if __name__ == "__main__":
    main()
