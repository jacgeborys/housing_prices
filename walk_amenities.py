"""
Everyday amenities within a real walk: for every priced building, count the points of interest reachable
within WALK_M metres along the footpath/street graph (TransitFrequency's Warsaw walking network, read only,
loaded via walk_access.load_graph_min) -> output/walk_amenities.gpkg.

Categories (data/osm_pois.gpkg, fetch_pois.py):
  food_drink   cafe, restaurant, bar, pub, fast_food
  zabka        convenience stores named Żabka
  grocery      supermarket, convenience (incl. Żabka), greengrocer, bakery, butcher
  schools      school, kindergarten
  health       pharmacy, doctors, clinic
  sport_play   playground, fitness_centre, sports_centre

Method: POIs snap to their nearest graph node (counts per node); from each building's node a Dijkstra
bounded at WALK_M collects the reachable nodes and sums their counts. A pure-Python heap search over the
CSR arrays: each search touches only the few thousand nodes within reach.
"""
import heapq
import sys

import geopandas as gpd
import numpy as np

sys.dont_write_bytecode = True
from walk_access import MainTree, load_graph_min  # noqa: E402

WALK_M = 560            # 7 minutes at 4.8 km/h
SNAP_MAX_M = 60         # POIs farther than this from any graph node are dropped
OUT = "output/walk_amenities.gpkg"

CATS = {
    "food_drink": lambda p: p["amenity"].isin(["cafe", "restaurant", "bar", "pub", "fast_food"]),
    "zabka": lambda p: p["name"].fillna("").str.contains("Żabka|Zabka", case=False) & p["shop"].notna(),
    "grocery": lambda p: p["shop"].isin(["supermarket", "convenience", "greengrocer", "bakery", "butcher"]),
    "schools": lambda p: p["amenity"].isin(["school", "kindergarten"]),
    "health": lambda p: p["amenity"].isin(["pharmacy", "doctors", "clinic"]),
    "sport_play": lambda p: p["leisure"].isin(["playground", "fitness_centre", "sports_centre"]),
}


def bounded_reach(indptr, indices, data, src, limit):
    """Nodes reachable from src within `limit` (Dijkstra on CSR arrays)."""
    dist = {src: 0.0}
    heap = [(0.0, src)]
    done = []
    while heap:
        d, u = heapq.heappop(heap)
        if d > dist.get(u, np.inf):
            continue
        done.append(u)
        for k in range(indptr[u], indptr[u + 1]):
            nd = d + data[k]
            if nd <= limit:
                v = indices[k]
                if nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
    return done


def main():
    print("loading walking graph ...", flush=True)
    m, xy = load_graph_min()
    tree = MainTree(m, xy)
    indptr, indices, data = m.indptr.tolist(), m.indices.tolist(), m.data.tolist()

    p = gpd.read_file("data/osm_pois.gpkg")
    d, node = tree.query(np.c_[p.geometry.x, p.geometry.y])
    p["node"], p["snap"] = node, d
    p = p[p["snap"] <= SNAP_MAX_M]
    counts = {}
    for c, rule in CATS.items():
        sel = p[rule(p)]
        counts[c] = np.bincount(sel["node"], minlength=len(xy))
        print(f"  {c}: {len(sel):,} POIs")

    b = gpd.read_file("output/price_drivers.gpkg")[["osm_id", "geometry"]]
    bp = b.geometry.representative_point()
    snap, bnode = tree.query(np.c_[bp.x, bp.y])
    out = {c: np.zeros(len(b), int) for c in CATS}
    cache = {}
    for i, (n0, s0) in enumerate(zip(bnode, snap)):
        key = (n0, round(s0 / 20))                      # buildings on the same node share a search
        if key not in cache:
            reach = bounded_reach(indptr, indices, data, int(n0), max(WALK_M - s0, 0))
            cache[key] = {c: int(counts[c][reach].sum()) for c in CATS}
        for c in CATS:
            out[c][i] = cache[key][c]
        if i % 2000 == 0:
            print(f"  {i:,}/{len(b):,}", flush=True)
    for c in CATS:
        b[f"walk7_{c}"] = out[c]
    b.to_file(OUT, layer="walk_amenities", driver="GPKG")
    print(b[[f"walk7_{c}" for c in CATS]].describe().round(1).T[["mean", "50%", "max"]])
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
