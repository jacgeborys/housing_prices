"""
Data coverage audit: where does RCN see the flats that exist?

Universe = multi-family residential buildings: BDOT10k "3+ flats" or OSM building=apartments
(or any footprint with >= 3 recorded flat sales). Flats per building are estimated from the
gross floor area (footprint x storeys), calibrated on new builds whose developer sales are all
in RCN (each flat sold once on the primary market). Coverage = recorded sales / estimated flats,
compared with the city median.

Why holes exist: RCN only registers full ownership of a flat (wlasnoscLokalu...). Sales of
cooperative rights (spoldzielcze wlasnosciowe prawo do lokalu) are absent from the register,
so large cooperative estates (Ursynow, Bielany, Targowek...) show few or no sales.

Outputs:
  output/coverage.gpkg                       layer building_coverage
  output/analysis/coverage_by_district.csv
  output/analysis/map_09_coverage.png
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from shapely.geometry import box

import style

style.apply()

OUT_GPKG = "output/coverage.gpkg"
OUT_CSV = "output/analysis/coverage_by_district.csv"
OUT_PNG = "output/analysis/map_09_coverage.png"

STOREY_M = 3.0
THIN_SHARE = 0.25        # below 25 % of the city's median sales-per-flat = "thin"
# Ordinal blue ramp (DIV_CMAP's cool half): more saturated = bigger hole
COL = {"ok": "#b7d3f6", "thin": "#3987e5", "none": "#104281"}
LABEL = {"ok": "Covered", "thin": "Thin (< 25 % of city norm)", "none": "No sales recorded"}
GRID_M = 500
MIN_CELL_FLATS = 100
GRID_BINS = [0, 10, 25, 50, 75, 100.01]
GRID_COL = ["#e6eef8", "#b7d3f6", "#6aa5ee", "#2a6fc9", "#104281"]   # one hue, light -> dark


def load():
    osm = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="buildings_all")
    bdot = gpd.read_file("data/bdot_buildings.gpkg")[["building_type", "n_floors", "geometry"]]
    tx = gpd.read_file("output/price_model.gpkg", layer="transactions_adj")
    districts = gpd.read_file("data/districts.gpkg")
    return osm, bdot.to_crs(osm.crs), tx, districts.to_crs(osm.crs)


def attach_bdot(osm, bdot):
    """BDOT function + storeys for each OSM footprint (BDOT polygon holding its inner point)."""
    pts = gpd.GeoDataFrame({"osm_id": osm["osm_id"].values},
                           geometry=osm.geometry.representative_point().values, crs=osm.crs)
    j = gpd.sjoin(pts, bdot, how="left", predicate="within")
    j = j[~j.index.duplicated()]
    osm = osm.copy()
    osm["bdot_type"] = j["building_type"].values
    osm["bdot_floors"] = j["n_floors"].values
    return osm


def main():
    osm, bdot, tx, districts = load()
    print(f"OSM {len(osm):,}  BDOT {len(bdot):,}  flat sales {len(tx):,}")
    osm = attach_bdot(osm, bdot)

    tx = tx[tx["osm_id"].notna()].copy()
    tx["osm_id"] = tx["osm_id"].astype("int64")
    tx["primary"] = tx["market"].eq("primary")
    per_b = tx.groupby("osm_id").agg(n_sales=("price", "size"), n_primary=("primary", "sum"),
                                     first_year=("year", "min"))
    osm = osm.merge(per_b, left_on="osm_id", right_index=True, how="left")
    osm[["n_sales", "n_primary"]] = osm[["n_sales", "n_primary"]].fillna(0)

    osm["footprint_m2"] = osm.geometry.area
    floors = osm["bdot_floors"].where(osm["bdot_floors"] > 0,
                                      np.round(osm["height_m"] / STOREY_M))
    osm["floors"] = floors.fillna(1).clip(lower=1)
    osm["gfa_m2"] = osm["footprint_m2"] * osm["floors"]

    multi = (osm["bdot_type"].eq("budynkiOTrzechIWiecejMieszkaniach")
             | osm["building"].eq("apartments") | (osm["n_sales"] >= 3))
    uni = osm[multi & (osm["footprint_m2"] >= 100) & (osm["floors"] >= 2)].copy()
    # District of each building (by inner point)
    pts = gpd.GeoDataFrame(geometry=uni.geometry.representative_point(), crs=uni.crs)
    dcol = "district"
    d = gpd.sjoin(pts, districts[[dcol, "geometry"]], how="left", predicate="within")
    uni["district"] = d[~d.index.duplicated()][dcol].values
    uni = uni[uni["district"].notna()].copy()     # RCN extract = Warsaw (TERYT 1465) only

    print(f"Multi-family universe: {len(uni):,} buildings "
          f"(BDOT 3+ flats {uni['bdot_type'].eq('budynkiOTrzechIWiecejMieszkaniach').sum():,}, "
          f"OSM apartments {uni['building'].eq('apartments').sum():,})")

    # Calibrate flats per m2 of GFA on new builds fully inside the data:
    # first sale 2013+, mostly developer sales -> primary sales ~ all flats.
    cal = uni[(uni["first_year"] >= 2013) & (uni["n_primary"] >= 20)
              & (uni["n_primary"] / uni["n_sales"] >= 0.6)]
    flats_per_m2 = (cal["n_primary"] / cal["gfa_m2"]).median()
    print(f"Calibration on {len(cal):,} new builds: {flats_per_m2 * 1000:.1f} flats per 1000 m2 GFA "
          f"(~{1 / flats_per_m2:.0f} m2 GFA per flat)")
    uni["est_flats"] = (uni["gfa_m2"] * flats_per_m2).clip(lower=3).round()

    uni["sales_per_flat"] = uni["n_sales"] / uni["est_flats"]
    norm = uni.loc[uni["n_sales"] > 0, "sales_per_flat"].median()
    uni["coverage_rel"] = uni["sales_per_flat"] / norm
    uni["coverage"] = np.select(
        [uni["n_sales"] == 0, uni["coverage_rel"] < THIN_SHARE], ["none", "thin"], "ok")
    print(f"City norm: {norm:.2f} recorded sales per flat (2006-2026)")

    # --- District summary ---------------------------------------------------
    s = (uni.groupby(["district", "coverage"])["est_flats"].sum().unstack(fill_value=0))
    for c in COL:
        if c not in s:
            s[c] = 0
    s["est_flats"] = s[list(COL)].sum(axis=1)
    for c in COL:
        s[f"share_{c}"] = s[c] / s["est_flats"]
    unm = tx_unmatched_by_district(districts, dcol)
    s = s.join(unm).fillna({"unmatched_sales": 0})
    s = s.sort_values("share_none", ascending=False)
    s.round(3).to_csv(OUT_CSV, encoding="utf-8")
    tot = s[list(COL)].sum()
    print(f"\nCity: {tot['none'] / tot.sum():.0%} of est. flats in buildings with no sales, "
          f"{tot['thin'] / tot.sum():.0%} thin")
    print(s[["est_flats", "share_none", "share_thin", "share_ok", "unmatched_sales"]]
          .round(2).to_string())

    keep = ["osm_id", "building", "bdot_type", "district", "floors", "footprint_m2", "gfa_m2",
            "est_flats", "n_sales", "n_primary", "first_year", "sales_per_flat", "coverage_rel",
            "coverage", "geometry"]
    uni[keep].to_file(OUT_GPKG, layer="building_coverage", driver="GPKG")
    print(f"\nSaved {OUT_GPKG}, {OUT_CSV}")

    plot(osm, uni, districts, s, dcol, norm)


def tx_unmatched_by_district(districts, dcol):
    """Sales RCN located but build_viz could not attach to a footprint (matching holes)."""
    t = gpd.read_file("output/warsaw_prices_viz.gpkg", layer="transactions",
                      columns=["match", "geometry"]).to_crs(districts.crs)
    t = t[t["match"].fillna("none").str.contains("none|unmatched", case=False)]
    j = gpd.sjoin(t, districts[[dcol, "geometry"]], how="inner", predicate="within")
    return j.groupby(dcol).size().rename("unmatched_sales")


def plot(osm, uni, districts, s, dcol, norm):
    fig = plt.figure(figsize=(16, 10.5))
    ax = fig.add_axes([0.0, 0.06, 0.60, 0.80])
    bx = fig.add_axes([0.70, 0.10, 0.28, 0.72])

    # 500 m grid: share of a cell's estimated flats that sit in buildings with no recorded sale
    pts = uni.geometry.representative_point()
    g = pd.DataFrame({"gx": (pts.x // GRID_M).astype(int), "gy": (pts.y // GRID_M).astype(int),
                      "flats": uni["est_flats"].values,
                      "none": uni["est_flats"].where(uni["coverage"] == "none", 0).values})
    g = g.groupby(["gx", "gy"]).sum().reset_index()
    g = g[g["flats"] >= MIN_CELL_FLATS]
    g["share"] = g["none"] / g["flats"] * 100
    g["cls"] = np.digitize(g["share"], GRID_BINS[1:-1])
    cells = gpd.GeoDataFrame(g, crs=uni.crs, geometry=[
        box(x * GRID_M, y * GRID_M, (x + 1) * GRID_M, (y + 1) * GRID_M)
        for x, y in zip(g["gx"], g["gy"])])
    osm.plot(ax=ax, color=style.CONTEXT, linewidth=0)
    for k, colr in enumerate(GRID_COL):
        sub = cells[cells["cls"] == k]
        if len(sub):
            sub.plot(ax=ax, color=colr, edgecolor=style.SURFACE, linewidth=0.6)
    districts.boundary.plot(ax=ax, color=style.INK_2, linewidth=0.6)
    for _, r in districts.iterrows():
        p = r.geometry.representative_point()
        ax.text(p.x, p.y, r[dcol], fontsize=8, color=style.INK, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=style.SURFACE, ec="none", alpha=0.75))
    ax.set_axis_off()
    ax.set_aspect("equal")
    labels = [f"{a:.0f}–{b:.0f}%" for a, b in zip(GRID_BINS[:-1], GRID_BINS[1:])]
    ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(GRID_COL[::-1], labels[::-1])]
              + [Patch(color=style.CONTEXT, label="Buildings (context)")],
              loc="upper right", fontsize=10, title_fontsize=10,
              title=f"Flats with no recorded sale\n(500 m cells, ≥ {MIN_CELL_FLATS} flats)")

    # Stacked share of estimated flats per district, 2 px surface gaps between segments
    s = s.sort_values("share_none")
    y = np.arange(len(s))
    left = np.zeros(len(s))
    for c in ("none", "thin", "ok"):
        v = s[f"share_{c}"].values * 100
        bx.barh(y, v, left=left, height=0.72, color=COL[c], edgecolor=style.SURFACE,
                linewidth=1.5, label=LABEL[c])
        left += v
    for yi, v in zip(y, s["share_none"].values * 100):
        bx.text(-1.5, yi, f"{v:.0f}%", ha="right", va="center", fontsize=9, color=style.INK_2)
    bx.set_yticks(y)
    bx.set_yticklabels(s.index, fontsize=9.5)
    bx.tick_params(axis="y", pad=34)
    bx.set_xlim(0, 100)
    bx.set_xlabel("Share of estimated flats in the district")
    style.pct(bx, "x", signed=False)
    style.vgrid_only(bx)
    bx.set_title("Flats in buildings with no recorded sales", fontsize=11.5)

    style.header(fig, "Where the price register goes blind",
                 "One flat in five sits in a building with no recorded sale. RCN skips cooperative-right "
                 "sales (spółdzielcze własnościowe prawo); city-owned and rental blocks rarely sell")
    style.footer(fig, f"Flats estimated from footprint × storeys (BDOT10k / OSM), calibrated on new "
                      f"builds. 'Thin' = under {THIN_SHARE:.0%} of the city median "
                      f"({norm:.2f} sales per flat, 2006–2026). " + style.SOURCE)
    fig.savefig(OUT_PNG, dpi=150, facecolor=style.PAGE)
    plt.close(fig)
    print(f"Saved {OUT_PNG}")


if __name__ == "__main__":
    main()
