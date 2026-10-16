"""
Morning-peak accessibility with richer outputs (run with access/.venv after travel_times.py).

Same routing as travel_times.py (walk + transit, Wed 7.10.2026 07-09, median over the window, 250 m origin cells),
but per origin cell it keeps:
  grav_<purpose>_<h>   gravity access sum(weight * 0.5 ** (t / h)) for purposes jobs / office / shops / leisure
                       and half-lives h = 5, 10, 15, 20, 30, 45 min (the model picks the best h)
  n_<purpose>_<t>      cumulative opportunities within t = 15, 20, 30, 45 min
  t_cl<k>              minutes to the nearest cell of office cluster k (clusters found from the data:
                       neighbouring 250 m cells with >= 4000 office job equivalents, total >= 10k: Daszyńskiego,
                       ONZ, Służewiec, Central Station)
Office jobs = commercial-office floor area / 18 m² (SimCity category "budynki biurowe", build_inputs.py).

Output: access/output/gravity_am_peak.csv (per building) and access/output/office_clusters.csv
"""
import datetime as dt
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
from travel_times import CELL, DATA, OUT, SCENARIOS, network  # noqa: E402  (sets JAVA_HOME / memory first)

HALF_LIVES = [5, 10, 15, 20, 30, 45]
THRESHOLDS = [15, 20, 30, 45]
CLUSTER_MIN_JOBS = 4000     # splits the central continuum into Daszyńskiego, ONZ and Central Station
CLUSTER_MIN_TOTAL = 10000
CLUSTER_MIN_CELLS = 2       # a single dense cell is one office park, not a centre
CHUNK = 500


def office_layer(dest):
    """Commercial-office job equivalents per destination cell (build_inputs.py, SimCity category "budynki biurowe")."""
    return pd.Series(dest["office"].to_numpy(), index=dest["id"].to_numpy())


def clusters(dest, office):
    """Connected groups of office-dense cells (8-neighbourhood)."""
    d = dest.to_crs(2180)
    gx = ((d.geometry.x - 125) // 250).round().astype(int).to_numpy()
    gy = ((d.geometry.y - 125) // 250).round().astype(int).to_numpy()
    dense = {(x, y): i for x, y, i, o in zip(gx, gy, dest["id"], office.reindex(dest["id"]).to_numpy())
             if o >= CLUSTER_MIN_JOBS}
    label, k = {}, 0
    for cell in dense:
        if cell in label:
            continue
        stack = [cell]
        label[cell] = k
        while stack:
            x, y = stack.pop()
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n = (x + dx, y + dy)
                    if n in dense and n not in label:
                        label[n] = k
                        stack.append(n)
        k += 1
    cl = pd.DataFrame([(dense[c], lab) for c, lab in label.items()], columns=["id", "cluster"])
    cl["office"] = cl["id"].map(office)
    tot = cl.groupby("cluster")["office"].sum()
    ncell = cl.groupby("cluster").size()
    keep = tot[(tot >= CLUSTER_MIN_TOTAL) & (ncell >= CLUSTER_MIN_CELLS)].sort_values(ascending=False)
    cl = cl[cl["cluster"].isin(keep.index)].copy()
    cl["cluster"] = cl["cluster"].map({c: i for i, c in enumerate(keep.index)})
    xy = d.set_index("id").loc[cl["id"]]
    cl["x"], cl["y"] = xy.geometry.x.to_numpy(), xy.geometry.y.to_numpy()
    summ = cl.groupby("cluster").apply(lambda g: pd.Series({
        "office_jobs": g["office"].sum(), "cells": len(g),
        "x": np.average(g["x"], weights=g["office"]), "y": np.average(g["y"], weights=g["office"])}))
    return cl[["id", "cluster"]], summ


def main():
    import r5py
    b = gpd.read_file(os.path.join(DATA, "origins.gpkg")).to_crs(2180)
    b["cell"] = (b.geometry.x // CELL).astype(int) * 100000 + (b.geometry.y // CELL).astype(int)
    xy = b.assign(x=b.geometry.x, y=b.geometry.y).groupby("cell")[["x", "y"]].median()
    cells = gpd.GeoDataFrame({"id": xy.index.to_numpy()}, geometry=gpd.points_from_xy(xy["x"], xy["y"]),
                             crs=2180).to_crs(4326)
    dest = gpd.read_file(os.path.join(DATA, "destinations.gpkg"))
    office = office_layer(dest)
    cl, summ = clusters(dest, office)
    os.makedirs(OUT, exist_ok=True)
    summ.to_csv(os.path.join(OUT, "office_clusters.csv"))
    print(f"{len(summ)} office clusters; largest: {summ['office_jobs'].round(-3).head(8).to_dict()}", flush=True)
    w = pd.DataFrame({"jobs": dest.set_index("id")["jobs"], "office": office,
                      "shops": dest.set_index("id")["shop_gfa"] / 40, "leisure": dest.set_index("id")["leisure"]})
    cl_of = cl.set_index("id")["cluster"]
    dep, win = SCENARIOS["am_peak"]
    net = network()
    partial = os.path.join(OUT, "gravity_am_peak_cells_partial.csv")
    rows = [pd.read_csv(partial, index_col=0)] if os.path.exists(partial) else []
    done = set(rows[0].index) if rows else set()
    todo = cells[~cells["id"].isin(done)]
    for c0 in range(0, len(todo), CHUNK):
        o = todo.iloc[c0:c0 + CHUNK]
        tt = pd.DataFrame(r5py.TravelTimeMatrix(
            net, origins=o, destinations=dest, departure=dep, departure_time_window=win,
            transport_modes=[r5py.TransportMode.TRANSIT, r5py.TransportMode.WALK],
            max_time=dt.timedelta(minutes=60), percentiles=[50])).dropna(subset=["travel_time"])
        t = tt["travel_time"].to_numpy(float)
        out = {}
        for p in w.columns:
            wt = tt["to_id"].map(w[p]).fillna(0).to_numpy()
            for h in HALF_LIVES:
                out[f"grav_{p}_{h}"] = pd.Series(wt * 0.5 ** (t / h)).groupby(tt["from_id"].to_numpy()).sum()
            for lim in THRESHOLDS:
                out[f"n_{p}_{lim}"] = pd.Series(wt * (t <= lim)).groupby(tt["from_id"].to_numpy()).sum()
        r = pd.DataFrame(out)
        tc = tt.assign(cluster=tt["to_id"].map(cl_of)).dropna(subset=["cluster"])
        tmin = tc.groupby(["from_id", "cluster"])["travel_time"].min().unstack()
        tmin.columns = [f"t_cl{int(c)}" for c in tmin.columns]
        rows.append(r.join(tmin))
        pd.concat(rows).to_csv(partial)
        print(f"  {len(done) + min(c0 + CHUNK, len(todo)):,}/{len(cells):,} cells", flush=True)
    res = pd.concat(rows).reindex(b["cell"].to_numpy())
    res.index = b["id"].to_numpy()
    res.index.name = "osm_id"
    res.to_csv(os.path.join(OUT, "gravity_am_peak.csv"))
    print(f"Saved gravity_am_peak.csv: {res.shape[1]} columns for {len(res):,} buildings")


if __name__ == "__main__":
    main()
