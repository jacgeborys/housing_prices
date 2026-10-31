"""
Lublin: RCN flat sales -> building values, with Warsaw's hedonic model (price_index.py imported, not edited).
Run from the lublin/ folder:  python build_prices.py

    lublin/data/rcn_lokale_lublin.gpkg  (fetch_rcn.py --bbox-name lublin)  +  BDOT10k buildings (fetch_buildings.py)
 -> lublin/output/price_model.gpkg      layers transactions_adj, building_value (n, value_2025, value_2025_raw, fe_var)

Differences from Warsaw: Lublin's register leaves the flat price (lok_cena_brutto) empty in 99 % of records and puts it
in the property price (nier_cena_brutto), so that is used; records start in 2016; eras of the new-build term
redefined; BDOT10k storeys x 3 m stand in for the height.
"""
import os
import sys
import warnings

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import price_index as pi  # noqa: E402
from rcn_clean import residential_clean  # noqa: E402

warnings.filterwarnings("ignore")
pi.ERAS = [(2016, 2019), (2020, 2022), (2023, 2026)]
pi.MIN_Q_OBS = 150
BUILDINGS = "data/buildings.gpkg"      # fetch_buildings.py: BDOT10k, whole city


def load():
    rcn = gpd.read_file("data/rcn_lokale_lublin.gpkg")
    rcn.columns = [c.lower() for c in rcn.columns]
    for col in ["lok_cena_brutto", "nier_cena_brutto", "lok_pow_uzyt", "lok_nr_kond", "lok_liczba_izb"]:
        rcn[col] = pd.to_numeric(rcn[col], errors="coerce")
    rcn["lok_cena_brutto"] = rcn["lok_cena_brutto"].fillna(rcn["nier_cena_brutto"])
    res = residential_clean(rcn)
    res = res[~res["nier_prawo"].isin(["wlasnoscBudynkuWrazZPrawemZwiazanym", "wlasnoscNieruchomosciGruntowej"])].copy()
    res["price"] = res["lok_cena_brutto"]
    res["area"] = res["lok_pow_uzyt"]
    res["price_m2"] = res["price"] / res["area"]
    res["rooms"], res["floor"] = res["lok_liczba_izb"], res["lok_nr_kond"]
    res["market"] = res["tran_rodzaj_rynku"].map({"rynekPierwotny": "primary", "rynekWtorny": "secondary"})
    res["date"] = pd.to_datetime(res["dok_data"].str[:10], errors="coerce")
    res = res[res["date"].between("2016-01-01", "2026-12-31")]
    res["year"] = res["date"].dt.year
    res = res[res["price_m2"].between(2000, 60000) & res["area"].between(12, 300)]
    res = res[res["rooms"].between(1, 8) | res["rooms"].isna()]
    res.loc[~res["floor"].between(0, 60), "floor"] = np.nan
    n0 = len(res)
    res = res.drop_duplicates(subset=["dok_oznaczenie", "lok_adres", "area", "price"])
    print(f"  dropped {n0 - len(res):,} duplicate records")

    d = gpd.read_file("data/districts.gpkg").to_crs(2180)
    j = gpd.sjoin(res[["geometry"]], d[["district", "geometry"]], how="left", predicate="within")
    res["district"] = j[~j.index.duplicated()]["district"]
    res = res[res["district"].notna()].copy()
    print(f"  flat sales inside Lublin: {len(res):,}")

    bld = gpd.read_file(BUILDINGS)
    height = bld.set_index("osm_id")["storeys"].mul(3.0)             # BDOT storeys -> metres, the model's height class
    bld = bld[["osm_id", "residential", "geometry"]]
    pts = res[["geometry"]]
    j = gpd.sjoin(pts, bld[["osm_id", "geometry"]], how="left", predicate="within")
    j = j[~j.index.duplicated()]
    miss = j["osm_id"].isna()
    near = gpd.sjoin_nearest(pts[miss], bld[bld["residential"]], how="left", max_distance=10)
    near = near[~near.index.duplicated()]
    j.loc[near.index, "osm_id"] = near["osm_id"].values
    res["osm_id"] = j["osm_id"]
    print(f"  in a building: {res['osm_id'].notna().mean():.1%}")
    res["bld_height"] = res["osm_id"].map(height)
    res["x"], res["y"] = res.geometry.x, res.geometry.y
    keep = ["date", "year", "district", "osm_id", "bld_height", "x", "y", "price", "price_m2", "area", "rooms",
            "floor", "market", "lok_adres"]
    return pd.DataFrame(res[keep]).reset_index(drop=True), bld


def main():
    df, bld = load()
    df = pi.prepare(df)
    print(f"  apartments: {len(df):,}; per year:\n{df['year'].value_counts().sort_index().to_string()}")
    model = pi.fit(df)
    df = pi.trim_outliers(df, model)
    model = pi.fit(df)
    qidx = pi.quarter_index(model, df)
    bv = pi.building_values(df, model, qidx)
    q = qidx["gamma"]
    print("price index (2025 = 1):", {str(k): round(float(np.exp(v)), 2) for k, v in q.groupby(q.index.year).mean().items()})

    os.makedirs("output", exist_ok=True)
    gm = "output/price_model.gpkg"
    if os.path.exists(gm):
        os.remove(gm)
    gamma_q = qidx["gamma"].reindex(df["q"]).values
    df["price_m2_2025"] = df["price_m2"] * np.exp(-gamma_q)
    txn = gpd.GeoDataFrame(df[["date", "year", "district", "osm_id", "floor", "area", "rooms", "market", "price", "price_m2",
                               "price_m2_2025"]].assign(date=df["date"].dt.strftime("%Y-%m-%d")),
                           geometry=gpd.points_from_xy(df["x"], df["y"]), crs=2180)
    txn.to_file(gm, layer="transactions_adj", driver="GPKG")
    b = bv[bv["osm_id"].notna()].copy()
    b["osm_id"] = b["osm_id"].astype("int64")
    out = bld.merge(b[["osm_id", "district", "n", "value_2025", "value_2025_raw", "fe_var"]], on="osm_id", how="inner")
    out.to_file(gm, layer="building_value", driver="GPKG")
    print(f"  {gm}: transactions_adj {len(txn):,}, building_value {len(out):,}; value_2025 quantiles "
          f"{out['value_2025'].quantile([.05, .25, .5, .75, .95]).round(0).to_dict()}")


if __name__ == "__main__":
    main()
