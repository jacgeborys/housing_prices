"""
3D floor-by-floor price scenes of central Warsaw.

Every storey of every building that has sales becomes its own slab, coloured
by the median *time-adjusted* price/m² (2025 PLN, from price_index.py) of the
flats sold on that floor. Unsold storeys stay as dark massing.

Floor geometry:
  * RCN numbers storeys from the ground: floor 1 = ground floor.
  * Each OSM building:part inherits the floor grid of the building it belongs
    to, so floors line up across parts and are counted from the ground, not
    from the part's own base.
  * Storey height = building height / max(height/3 m, highest floor sold).

Outputs:
  output/analysis/map_07_3d_center.png   static render (matplotlib)
  output/web/warsaw_3d.html              interactive deck.gl scene

Needs: output/warsaw_prices_viz.gpkg (build_viz.py) and
       output/price_model.gpkg (price_index.py)
"""

import json
import os
import re
import warnings

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import style

warnings.filterwarnings("ignore")

VIZ = "output/warsaw_prices_viz.gpkg"
MODEL = "output/price_model.gpkg"
PKIN = (636953, 486968)          # Palace of Culture, EPSG:2180
MIN_FLOOR_SALES = 1


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

def _address(lok_adres):
    if not isinstance(lok_adres, str):
        return None
    ul = re.search(r"UL:([^;]+)", lok_adres)
    nr = re.search(r"NR_PORZ:([^;]+)", lok_adres)
    if not ul:
        return None
    street = re.sub(r"^(ulica|aleja|aleje|plac|al\.)\s+", "", ul.group(1).strip(), flags=re.I)
    return f"{street} {nr.group(1).strip()}" if nr else street


def build_scene(center=PKIN, radius=900):
    cx, cy = center
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    print(f"Building scene: r = {radius} m")
    b3d = gpd.read_file(VIZ, layer="buildings_3d", bbox=bbox)
    parents = gpd.read_file(VIZ, layer="buildings_all", bbox=bbox)[["osm_id", "height_m", "name", "geometry"]]
    txn = gpd.read_file(MODEL, layer="transactions_adj", bbox=bbox)
    bval = gpd.read_file(MODEL, layer="building_value", bbox=bbox)[["osm_id", "value_2025", "n"]]

    d = np.hypot(b3d.geometry.centroid.x - cx, b3d.geometry.centroid.y - cy)
    b3d = b3d[(d < radius) & (b3d["height_m"] > 0)].reset_index(drop=True)
    b3d = b3d[b3d.geometry.type.isin(["Polygon", "MultiPolygon"])].reset_index(drop=True)

    # Parent building for each part (representative point inside a footprint)
    rp = b3d.copy()
    rp["geometry"] = b3d.geometry.representative_point()
    j = gpd.sjoin(rp[["geometry"]], parents, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    b3d["parent"] = j["osm_id"].fillna(b3d["osm_id"]).astype("int64")
    ph = parents.set_index("osm_id")["height_m"]
    b3d["parent_h"] = b3d["parent"].map(ph).fillna(b3d["height_m"])
    b3d["parent_h"] = np.fmax(b3d["parent_h"], b3d.groupby("parent")["height_m"].transform("max"))

    # Floor-level prices per parent building
    t = txn[txn["osm_id"].notna() & txn["floor"].notna()].copy()
    t["osm_id"] = t["osm_id"].astype("int64")
    t["floor"] = t["floor"].astype(int)
    fl = t.groupby(["osm_id", "floor"]).agg(
        p_adj=("price_m2_2025", "median"), p_nom=("price_m2", "median"), n=("price_m2", "size"),
        y0=("year", "min"), y1=("year", "max")).reset_index()
    fl = fl[fl["n"] >= MIN_FLOOR_SALES]
    fl_idx = {(r.osm_id, r.floor): r for r in fl.itertuples()}
    max_floor = t.groupby("osm_id")["floor"].max()
    addr = t.groupby("osm_id")["lok_adres"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else None).map(_address)

    # Building info table (one row per parent that appears in the scene)
    pinfo = pd.DataFrame({"parent": b3d["parent"].unique()})
    pinfo["height"] = pinfo["parent"].map(b3d.groupby("parent")["parent_h"].max())
    mf = pinfo["parent"].map(max_floor).fillna(0)
    pinfo["storeys"] = np.maximum(np.round(pinfo["height"] / 3.0), mf).clip(lower=1).astype(int)
    pinfo["floor_h"] = pinfo["height"] / pinfo["storeys"]
    pinfo["name"] = pinfo["parent"].map(parents.set_index("osm_id")["name"])
    pinfo["addr"] = pinfo["parent"].map(addr)
    bv = bval.dropna(subset=["osm_id"]).assign(osm_id=lambda x: x["osm_id"].astype("int64")).set_index("osm_id")
    pinfo["value_2025"] = pinfo["parent"].map(bv["value_2025"])
    pinfo["n_sales"] = pinfo["parent"].map(t.groupby("osm_id").size()).fillna(0).astype(int)
    pinfo = pinfo.set_index("parent")

    # Segments: priced storeys individually; consecutive unpriced storeys merged
    segs = []
    for i, row in b3d.iterrows():
        p = row["parent"]
        info = pinfo.loc[p]
        fh, n_st = info["floor_h"], int(info["storeys"])
        z_lo, z_hi = float(row["min_height_m"] or 0), float(row["height_m"])
        run = None
        for k in range(1, n_st + 1):
            a, b = max((k - 1) * fh, z_lo), min(k * fh, z_hi)
            if b - a < 0.3:
                continue
            f = fl_idx.get((p, k))
            if f is None:
                run = [a, b] if run is None else [run[0], b]
                continue
            if run:
                segs.append((i, p, run[0], run[1], 0, np.nan, np.nan, 0, 0, 0))
                run = None
            segs.append((i, p, a, b, k, f.p_adj, f.p_nom, f.n, f.y0, f.y1))
        if run:
            segs.append((i, p, run[0], run[1], 0, np.nan, np.nan, 0, 0, 0))
        # Part above the parent's floor grid (spires, roofs) as massing
        top = n_st * fh
        if z_hi > top + 0.3:
            segs.append((i, p, max(top, z_lo), z_hi, 0, np.nan, np.nan, 0, 0, 0))
    seg = pd.DataFrame(segs, columns=["fp", "parent", "z0", "z1", "floor", "p_adj", "p_nom", "n", "y0", "y1"])
    print(f"  {len(b3d):,} footprints, {len(seg):,} segments, "
          f"{(seg['floor'] > 0).sum():,} priced storeys in {seg.loc[seg['floor'] > 0, 'parent'].nunique():,} buildings")
    ground = load_ground(center, radius)
    return {"b3d": b3d, "seg": seg, "pinfo": pinfo, "center": center, "radius": radius, "ground": ground}


def price_norm(seg):
    p = seg.loc[seg["floor"] > 0, "p_adj"]
    lo, hi = np.percentile(p, [5, 95])
    return mcolors.Normalize(vmin=np.floor(lo / 1000) * 1000, vmax=np.ceil(hi / 1000) * 1000)


OSM_DIR = "D:/QGIS/osm_basemap/warsaw"
ROAD_W = {"motorway": 16, "trunk": 16, "primary": 13, "secondary": 11, "tertiary": 9,
          "motorway_link": 7, "trunk_link": 7, "primary_link": 7, "secondary_link": 6, "tertiary_link": 6,
          "residential": 6, "unclassified": 6, "living_street": 4, "service": 3.5}
RIVER_W = 300   # the Vistula is only a centreline in OSM here; ~300 m reads right at this scale


def load_ground(center, radius):
    """Roads, rails, green areas and the river from the canonical OSM basemap,
    clipped to the scene circle, in EPSG:2180."""
    from shapely.geometry import Point
    cx, cy = center
    circle = gpd.GeoSeries([Point(cx, cy).buffer(radius * 1.05)], crs=2180)
    bbox = tuple(circle.to_crs(4326).total_bounds)
    out = {}
    for key, fname in [("roads", "roads"), ("rail", "railways"), ("water", "waterways"),
                       ("parks", "parks"), ("forest", "forests"), ("grass", "grass")]:
        path = os.path.join(OSM_DIR, fname + ".gpkg")
        if not os.path.exists(path):
            continue
        g = gpd.read_file(path, bbox=bbox).to_crs(2180)
        g = g[g.geometry.notna()]
        g["geometry"] = g.geometry.intersection(circle.iloc[0])
        out[key] = g[~g.geometry.is_empty]
    if "roads" in out:
        r = out["roads"]
        r = r[r["highway"].isin(ROAD_W) & (r.get("tunnel", pd.Series(index=r.index)).fillna("no") == "no")]
        r = r[r.geom_type.isin(["LineString", "MultiLineString"])]
        out["roads"] = r.assign(width=r["highway"].map(ROAD_W))
    if "rail" in out:
        r = out["rail"]
        out["rail"] = r[r["railway"].isin(["rail", "tram", "light_rail"]) &
                        (r["tunnel"].fillna("no") == "no")]
    if "water" in out:
        w = out["water"]
        out["water"] = w[w["name"] == "Wisła"]
    green = [out.pop(k) for k in ("parks", "forest", "grass") if k in out]
    if green:
        gr = pd.concat([g[["geometry"]] for g in green])
        gr = gr[gr.geom_type.isin(["Polygon", "MultiPolygon"])]
        out["green"] = gpd.GeoDataFrame(gr, crs=2180)
    for k, v in out.items():
        print(f"  ground {k}: {len(v):,}")
    return out


# ---------------------------------------------------------------------------
# Static render
# ---------------------------------------------------------------------------

def render_static(scene, path, elev=36, azim=-58):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    b3d, seg = scene["b3d"], scene["seg"]
    cx, cy = scene["center"]
    r = scene["radius"]
    norm = price_norm(seg)
    cmap = style.PRICE_CMAP

    GROUND = "#121211"
    MASS = np.array(mcolors.to_rgb("#6b6a64"))
    # Light from the south-west, a little above the horizon
    light = np.array([-0.55, -0.65, 0.52])
    light /= np.linalg.norm(light)

    polys, cols = [], []
    rings = {}
    for i, g in b3d.geometry.simplify(0.6).items():
        parts = list(g.geoms) if g.geom_type == "MultiPolygon" else [g]
        rings[i] = [np.asarray(pp.exterior.coords) - (cx, cy) for pp in parts if len(pp.exterior.coords) >= 4]

    def shade(base, nrm, amb=0.34):
        return np.clip(base * (amb + (1 - amb) * max(0.0, float(nrm @ light))), 0, 1)

    for s in seg.itertuples():
        priced = s.floor > 0
        base = np.array(cmap(norm(s.p_adj))[:3]) if priced else MASS
        for ring in rings.get(s.fp, []):
            xs, ys = ring[:, 0], ring[:, 1]
            # Make sure ring is counter-clockwise so outward normals are right
            if np.sum((xs[1:] - xs[:-1]) * (ys[1:] + ys[:-1])) > 0:
                xs, ys = xs[::-1], ys[::-1]
            for k in range(len(xs) - 1):
                dx, dy = xs[k + 1] - xs[k], ys[k + 1] - ys[k]
                ln = np.hypot(dx, dy)
                if ln < 0.2:
                    continue
                nrm = np.array([dy / ln, -dx / ln, 0.0])
                polys.append([(xs[k], ys[k], s.z0), (xs[k + 1], ys[k + 1], s.z0),
                              (xs[k + 1], ys[k + 1], s.z1), (xs[k], ys[k], s.z1)])
                cols.append((*shade(base, nrm), 1.0))
            polys.append(list(zip(xs, ys, np.full(len(xs), s.z1))))
            cols.append((*shade(base, np.array([0, 0, 1.0]), amb=0.55), 1.0))
    print(f"  static render: {len(polys):,} faces")

    fig = plt.figure(figsize=(20, 14), facecolor=GROUND)
    ax = fig.add_axes([0.0, -0.1, 1.0, 1.08], projection="3d", facecolor=GROUND)
    ax.set_proj_type("persp", focal_length=0.45)
    coll = Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.3), linewidths=0.15,
                            zsort="average")
    ax.add_collection3d(coll)

    # Ground: disc, green, river, roads, rails - drawn before the buildings
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    ax.computed_zorder = False
    coll.set_zorder(10)
    th = np.linspace(0, 2 * np.pi, 180)
    ax.add_collection3d(Poly3DCollection([list(zip(r * np.cos(th), r * np.sin(th), np.zeros_like(th)))],
                                         facecolors="#1c1c1a", edgecolors="none", zorder=1))
    gd = scene["ground"]

    def rings_of(gdf):
        out = []
        for geom in gdf.geometry.simplify(1.0):
            for pp in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
                c = np.asarray(pp.exterior.coords) - (cx, cy)
                out.append(list(zip(c[:, 0], c[:, 1], np.zeros(len(c)))))
        return out

    def lines_of(gdf):
        out = []
        for geom in gdf.geometry.simplify(1.0):
            for ln in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
                c = np.asarray(ln.coords) - (cx, cy)
                out.append(list(zip(c[:, 0], c[:, 1], np.zeros(len(c)))))
        return out

    if "green" in gd:
        ax.add_collection3d(Poly3DCollection(rings_of(gd["green"]), facecolors="#1f2a1e",
                                             edgecolors="none", zorder=2))
    if "water" in gd:
        ax.add_collection3d(Line3DCollection(lines_of(gd["water"]), colors="#1b2633",
                                             linewidths=RIVER_W / (2 * r) * 20 * 72 / 1.6, zorder=3,
                                             capstyle="round"))
    if "roads" in gd:
        for w, sub in gd["roads"].groupby("width"):
            ax.add_collection3d(Line3DCollection(lines_of(sub), colors="#2e2e2b",
                                                 linewidths=w * 0.5, zorder=4))
    if "rail" in gd:
        ax.add_collection3d(Line3DCollection(lines_of(gd["rail"]), colors="#3a3935",
                                             linewidths=0.6, zorder=4))
    zmax = 240
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, zmax)
    ax.set_box_aspect([1, 1, zmax / (2 * r)], zoom=1.38)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # Landmarks: the tallest named buildings
    pinfo = scene["pinfo"]
    tall = pinfo[pinfo["name"].notna()].sort_values("height", ascending=False).head(5)
    for p, row in tall.iterrows():
        g = b3d[b3d["parent"] == p].geometry.unary_union.centroid
        ax.plot([g.x - cx], [g.y - cy], [row["height"] + 2], "o", ms=2.5, color="#e8e6df", zorder=11)
        ax.text(g.x - cx, g.y - cy, row["height"] + 10, row["name"], color="#e8e6df", fontsize=9.5,
                ha="center", va="bottom", zorder=11)

    fig.text(0.02, 0.965, "Central Warsaw, floor by floor", color="white", fontsize=22, fontweight="bold",
             va="top")
    fig.text(0.02, 0.925, "Each coloured slab is one storey, coloured by the median price/m² of flats sold on it "
             "— every sale re-priced to 2025 PLN.\nDark massing: no recorded flat sales on that storey.",
             color="#c3c2b7", fontsize=12.5, va="top")
    cax = fig.add_axes([0.022, 0.83, 0.24, 0.012])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=style.PRICE_CMAP, norm=norm), cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors="#c3c2b7", labelsize=10, length=0)
    cb.ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: style.fmt_int(v)))
    cb.set_label("PLN/m², 2025 prices", color="#c3c2b7", fontsize=11)
    fig.text(0.02, 0.02, style.SOURCE + " Heights: OSM building / building:part tags.", color="#6f6e69",
             fontsize=9)
    fig.savefig(path, dpi=170, facecolor=GROUND)
    plt.close(fig)
    print(f"  {path}")


# ---------------------------------------------------------------------------
# Interactive web scene
# ---------------------------------------------------------------------------

def export_web(scene, path):
    b3d, seg, pinfo = scene["b3d"], scene["seg"], scene["pinfo"]
    norm = price_norm(seg)
    g = b3d.to_crs(4326).geometry.simplify(0.000004)
    lon0, lat0 = [round(v, 5) for v in g.unary_union.centroid.coords[0]]

    def enc(coords):
        a = np.asarray(coords)[:, :2]
        out = np.round((a - (lon0, lat0)) * 1e6).astype(int)
        return out.ravel().tolist()

    fps = []
    for geom in g:
        parts = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        fps.append([[enc(pp.exterior.coords)] + [enc(r.coords) for r in pp.interiors] for pp in parts])

    ground = {"green": [], "roads": [], "rail": [], "water": []}
    gd = scene["ground"]
    if "green" in gd:
        for geom in gd["green"].to_crs(4326).geometry.simplify(0.00001):
            for pp in (geom.geoms if geom.geom_type == "MultiPolygon" else [geom]):
                ground["green"].append([enc(pp.exterior.coords)])
    for key in ("roads", "rail", "water"):
        if key not in gd:
            continue
        sub = gd[key].to_crs(4326)
        widths = sub["width"] if key == "roads" else pd.Series(1.5 if key == "rail" else RIVER_W, index=sub.index)
        for geom, w in zip(sub.geometry.simplify(0.000008), widths):
            for ln in (geom.geoms if geom.geom_type == "MultiLineString" else [geom]):
                ground[key].append([float(w), enc(ln.coords)])

    pids = {p: i for i, p in enumerate(pinfo.index)}
    blds = []
    for p, r in pinfo.iterrows():
        label = next((v for v in (r["addr"], r["name"]) if isinstance(v, str) and v), "")
        blds.append([label, int(r["storeys"]),
                     None if pd.isna(r["value_2025"]) else int(round(r["value_2025"], -1)),
                     int(r["n_sales"])])
    s = seg
    data = {
        "origin": [lon0, lat0],
        "norm": [norm.vmin, norm.vmax],
        "lut": [mcolors.to_hex(style.PRICE_CMAP(x)) for x in np.linspace(0, 1, 64)],
        "fp": fps,
        "ground": ground,
        "bld": blds,
        "seg": {
            "fp": s["fp"].tolist(), "b": s["parent"].map(pids).tolist(),
            "z0": s["z0"].round(1).tolist(), "z1": s["z1"].round(1).tolist(),
            "f": s["floor"].astype(int).tolist(),
            "pa": s["p_adj"].fillna(0).round(-1).astype(int).tolist(),
            "pn": s["p_nom"].fillna(0).round(-1).astype(int).tolist(),
            "n": s["n"].astype(int).tolist(), "y0": s["y0"].astype(int).tolist(),
            "y1": s["y1"].astype(int).tolist(),
        },
    }
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "warsaw_3d_template.html"),
               encoding="utf-8").read()
    html = tpl.replace("/*__SCENE__*/", payload.replace("</", "<\\/"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  {path}  ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    style.apply()
    scene = build_scene(radius=900)
    render_static(scene, os.path.join("output", "analysis", "map_07_3d_center.png"))
    web = build_scene(radius=3000)
    export_web(web, os.path.join("output", "web", "warsaw_3d.html"))


if __name__ == "__main__":
    main()
