"""
Walking distances (metres along the OSM footpath/street graph) from every priced building to the
nearest metro station, park (>= 2 ha), school and tram stop -> output/walk_access.gpkg.

The graph is TransitFrequency's Warsaw walking network (D:/QGIS/TransitFrequency/network/warsaw/
walking_network.pkl, networkx MultiDiGraph, 1.23 M nodes), loaded read-only. It is turned into a
scipy sparse matrix (undirected, min edge length) and searched once per target type with a
multi-source Dijkstra, so the cost is four graph searches, not one per building.

Buildings and targets snap to their nearest graph node (straight-line snap distance is added).
Parks: every graph node within 15 m of a park/forest polygon of >= 2 ha counts as an entrance.
"""
import pickle
import sys

import geopandas as gpd
import numpy as np
from pyproj import Transformer
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

sys.dont_write_bytecode = True
GRAPH = "D:/QGIS/TransitFrequency/network/warsaw/walking_network.pkl"   # read only
OSM = "D:/QGIS/mapy_warszawy_misc/data/osm"
OUT = "output/walk_access.gpkg"
PARK_MIN_HA = 2
PARK_SNAP_M = 15
MAX_M = 5000          # searches stop here (beyond: reported as 5000)


def min_csr(u, v, w, n):
    """CSR with the minimum length over parallel edges."""
    order = np.lexsort((w, v, u))
    u, v, w = u[order], v[order], w[order]
    first = np.r_[True, (u[1:] != u[:-1]) | (v[1:] != v[:-1])]
    return coo_matrix((w[first], (u[first], v[first])), shape=(n, n)).tocsr()


def load_graph_min():
    g = pickle.load(open(GRAPH, "rb"))
    nodes = list(g.nodes(data=True))
    index = {nid: i for i, (nid, _) in enumerate(nodes)}
    lon = np.array([d["x"] for _, d in nodes])
    lat = np.array([d["y"] for _, d in nodes])
    E = g.number_of_edges()
    u = np.empty(E, np.int64); v = np.empty(E, np.int64); w = np.empty(E, float)
    for k, (a, b, d) in enumerate(g.edges(data=True)):
        u[k], v[k], w[k] = index[a], index[b], d.get("length", 1.0)
    del g, nodes, index
    n = len(lon)
    m = min_csr(np.r_[u, v], np.r_[v, u], np.r_[w, w], n)
    x, y = Transformer.from_crs(4326, 2180, always_xy=True).transform(lon, lat)
    return m, np.c_[x, y]


class MainTree:
    """Nearest-node lookup restricted to the largest connected component of the walking graph, so a
    building never snaps to an isolated courtyard path or footway fragment (which reaches nothing)."""

    def __init__(self, m, xy):
        from scipy.sparse.csgraph import connected_components
        _, lab = connected_components(m, directed=False)
        self.nodes = np.flatnonzero(lab == np.bincount(lab).argmax())
        self.tree = cKDTree(xy[self.nodes])
        print(f"  main component: {len(self.nodes):,} of {len(xy):,} nodes")

    def query(self, pts):
        d, i = self.tree.query(pts)
        return d, self.nodes[i]


def distances(m, src_nodes, src_offset):
    """Multi-source Dijkstra; returns per-node distance to the nearest source (+ its snap offset)."""
    # a virtual super-source is avoided: scipy's min_only=True does multi-source directly
    d, _, src = dijkstra(m, directed=False, indices=src_nodes, min_only=True, limit=MAX_M,
                         return_predecessors=True)
    off = np.zeros(m.shape[0])
    lookup = dict(zip(src_nodes, src_offset))
    ok = src >= 0
    off[ok] = [lookup[s] for s in src[ok]]
    return d + off


def main():
    print("loading walking graph ...", flush=True)
    m, xy = load_graph_min()
    tree = MainTree(m, xy)
    print(f"  {m.shape[0]:,} nodes, {m.nnz // 2:,} edges")

    b = gpd.read_file("output/price_model.gpkg", layer="building_value")
    b = b[["osm_id", "geometry"]].drop_duplicates("osm_id")
    bp = b.geometry.representative_point()
    b_snap, b_node = tree.query(np.c_[bp.x, bp.y])

    def snap_points(pts):
        d, n = tree.query(pts)
        best = {}
        for node, off in zip(n, d):              # several targets on one node: keep the closest
            best[node] = min(off, best.get(node, np.inf))
        return np.array(list(best.keys())), np.array(list(best.values()))

    st = gpd.read_file(f"{OSM}/train_stations.gpkg").to_crs(2180)
    metro = st[st["station"] == "subway"]
    trams = gpd.read_file(f"{OSM}/tram_stops.gpkg").to_crs(2180)
    pois = gpd.read_file("data/osm_pois.gpkg")
    schools = pois[pois["amenity"] == "school"]
    targets = {
        "walk_metro_m": np.c_[metro.geometry.x, metro.geometry.y],
        "walk_tram_m": np.c_[trams.geometry.x, trams.geometry.y],
        "walk_school_m": np.c_[schools.geometry.x, schools.geometry.y],
    }
    for name, pts in targets.items():
        nodes, off = snap_points(pts)
        d = distances(m, nodes, off)
        b[name] = np.minimum(d[b_node] + b_snap, MAX_M)
        print(f"  {name}: {len(pts):,} targets, median {np.median(b[name]):.0f} m", flush=True)

    green = gpd.GeoDataFrame(geometry=gpd.pd.concat(
        [gpd.read_file(f"{OSM}/{f}.gpkg").to_crs(2180).geometry for f in ("parks", "forests")]), crs=2180)
    green = green[green.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].buffer(0)
    big = gpd.GeoDataFrame(geometry=green, crs=2180).dissolve().explode(index_parts=False)
    big = big[big.area >= PARK_MIN_HA * 1e4]
    near = gpd.GeoDataFrame(geometry=gpd.points_from_xy(xy[:, 0], xy[:, 1]), crs=2180)
    hit = gpd.sjoin(near, gpd.GeoDataFrame(geometry=big.buffer(PARK_SNAP_M).values, crs=2180),
                    predicate="within", how="inner")
    park_nodes = np.intersect1d(np.unique(hit.index.values), tree.nodes)
    d = distances(m, park_nodes, np.zeros(len(park_nodes)))
    b["walk_park_m"] = np.minimum(d[b_node] + b_snap, MAX_M)
    print(f"  walk_park_m: {len(big):,} parks >= {PARK_MIN_HA} ha, {len(park_nodes):,} entrance nodes, "
          f"median {np.median(b['walk_park_m']):.0f} m")
    b["snap_m"] = b_snap
    b.to_file(OUT, layer="walk_access", driver="GPKG")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
