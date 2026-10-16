"""
Door-to-door public-transport accessibility for every valued building (run with access/.venv: r5py + Java 21).

Network: OSM (data/mazowieckie-latest.osm.pbf, Geofabrik) + Warsaw GTFS (data/gtfs_warsaw.zip, from
TransitFrequency). R5 routes walk + transit (all modes) and reports, for every origin-destination pair, the median
travel time over every departure minute of the time window (waiting included).

Buildings route from their 250 m cell (median position of the cell's buildings), then inherit its values.
Per building and scenario (output/access_<scenario>.parquet):
  jobs_30, jobs_45    job equivalents reachable within 30 / 45 min (data/destinations.gpkg)
  jobs_grav           gravity: sum of jobs * exp(-t / 15 min), t <= 60 min
  leisure_20          cafés, restaurants, bars, cinemas, theatres within 20 min
  t_<hub>             minutes to each named hub (nearest destination cell to the hub)

    access/.venv/Scripts/python.exe access/travel_times.py [--scenario am_peak]
"""
import argparse
import datetime as dt
import os
import sys

import geopandas as gpd
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
# portable Java 21 in access/jdk (no system install); r5py starts its JVM on import
_JDK = os.path.join(HERE, "jdk")
if os.path.isdir(_JDK):
    os.environ["JAVA_HOME"] = os.path.join(_JDK, sorted(os.listdir(_JDK))[0])
if "--max-memory" not in sys.argv:
    sys.argv += ["--max-memory", "8G"]
OUT = os.path.join(HERE, "output")
SCENARIOS = {   # name: (departure, window)
    "am_peak": (dt.datetime(2026, 10, 7, 7, 0), dt.timedelta(hours=2)),
    "midday": (dt.datetime(2026, 10, 7, 11, 0), dt.timedelta(hours=2)),
    "evening": (dt.datetime(2026, 10, 7, 19, 0), dt.timedelta(hours=2)),
    "saturday": (dt.datetime(2026, 10, 10, 11, 0), dt.timedelta(hours=2)),
}
HUBS = {   # name: lon, lat
    "centrum": (21.0060, 52.2318),          # Palace of Culture / Centrum
    "daszynskiego": (20.9831, 52.2306),     # Rondo Daszyńskiego office towers
    "sluzewiec": (20.9990, 52.1820),        # Służewiec offices ("Mordor")
    "centralna": (21.0030, 52.2290),        # Warszawa Centralna
    "arkadia": (20.9845, 52.2564),          # Arkadia
    "uw": (21.0200, 52.2400),               # University of Warsaw main campus
}
MAX_MIN = 60
CHUNK = 500
CELL = 250      # buildings route from their 250 m cell (median building position): 3.3k origins, not 20.7k


def network():
    import r5py
    return r5py.TransportNetwork(os.path.join(DATA, "mazowieckie-latest.osm.pbf"),
                                 [os.path.join(DATA, "gtfs_warsaw.zip")])


def run(name, net, cells, dest, orig):
    import r5py
    dep, win = SCENARIOS[name]
    hub_ids = {}
    d2180 = dest.to_crs(2180)
    for h, (lon, lat) in HUBS.items():
        p = gpd.GeoSeries.from_xy([lon], [lat], crs=4326).to_crs(2180).iloc[0]
        hub_ids[h] = int(dest["id"].iloc[np.argmin(d2180.distance(p).to_numpy())])
    jobs = dest.set_index("id")["jobs"]
    leis = dest.set_index("id")["leisure"]
    rows = []
    for c0 in range(0, len(cells), CHUNK):
        o = cells.iloc[c0:c0 + CHUNK]
        tt = r5py.TravelTimeMatrix(net, origins=o, destinations=dest, departure=dep,
                                   departure_time_window=win, transport_modes=[r5py.TransportMode.TRANSIT,
                                                                                r5py.TransportMode.WALK],
                                   max_time=dt.timedelta(minutes=MAX_MIN), percentiles=[50])
        tt = pd.DataFrame(tt).dropna(subset=["travel_time"])
        tt["jobs"] = tt["to_id"].map(jobs)
        tt["leisure"] = tt["to_id"].map(leis)
        g = tt.groupby("from_id")
        r = pd.DataFrame({
            "jobs_30": g.apply(lambda x: x.loc[x["travel_time"] <= 30, "jobs"].sum()),
            "jobs_45": g.apply(lambda x: x.loc[x["travel_time"] <= 45, "jobs"].sum()),
            "jobs_grav": g.apply(lambda x: (x["jobs"] * np.exp(-x["travel_time"] / 15)).sum()),
            "leisure_20": g.apply(lambda x: x.loc[x["travel_time"] <= 20, "leisure"].sum()),
        })
        for h, hid in hub_ids.items():
            r[f"t_{h}"] = tt[tt["to_id"] == hid].set_index("from_id")["travel_time"]
        rows.append(r)
        print(f"  {name}: {min(c0 + CHUNK, len(cells)):,}/{len(cells):,} cells", flush=True)
    res = pd.concat(rows)
    res = res.reindex(orig["cell"].to_numpy())
    res.index = orig["osm_id"].to_numpy()
    res.index.name = "osm_id"
    os.makedirs(OUT, exist_ok=True)
    res.to_parquet(os.path.join(OUT, f"access_{name}.parquet"))
    print(f"Saved {name}: median jobs within 30 min {res['jobs_30'].median():,.0f}, "
          f"median minutes to Centrum {res['t_centrum'].median():.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=list(SCENARIOS) + ["all"], default="all")
    args, _ = ap.parse_known_args()      # --max-memory is for r5py
    b = gpd.read_file(os.path.join(DATA, "origins.gpkg")).to_crs(2180)
    b["cell"] = (b.geometry.x // CELL).astype(int) * 100000 + (b.geometry.y // CELL).astype(int)
    xy = b.assign(x=b.geometry.x, y=b.geometry.y).groupby("cell")[["x", "y"]].median()
    cells = gpd.GeoDataFrame({"id": xy.index.to_numpy()}, geometry=gpd.points_from_xy(xy["x"], xy["y"]),
                             crs=2180).to_crs(4326)
    orig = pd.DataFrame({"osm_id": b["id"].to_numpy(), "cell": b["cell"].to_numpy()})
    dest = gpd.read_file(os.path.join(DATA, "destinations.gpkg"))
    print(f"{len(orig):,} buildings in {len(cells):,} origin cells x {len(dest):,} destinations", flush=True)
    net = network()
    for name in (SCENARIOS if args.scenario == "all" else [args.scenario]):
        run(name, net, cells, dest, orig)


if __name__ == "__main__":
    sys.exit(main())
