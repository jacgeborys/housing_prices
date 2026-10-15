"""
Export a Warsaw crop for Blender: building footprints with heights, coloured by a per-building
value from the housing-price model, plus dark ground layers (water, green, roads, railways).

    python 3d/export_scene.py --name prices_wide --set price_m2 \
        --bounds=-7000,-15000,15000,9000 --detail-bounds=-7000,-6500,7000,9000

Writes 3d/scenes/<name>/scene.json for 3d/build_blend.py (coordinates in metres, local to the
centre, EPSG:2180, z up). palette[0] = no value (dark grey), palette[k] = value class k.

Adapted from D:/QGIS/TransitFrequency/3d/export_scene.py (read only, not imported): same heights
(OSM height, else levels x 3.2 m, else 9 m; building:part drawn instead of a parent they cover
>= 85 %; parts take the parent's value), same tiled ground and the transit poster's dark colours,
so renders match the transit scenes. The scene origin is PKiN, so the saved views in 3d/views fit.

Values: buildings with fewer than MIN_SALES recorded sales get none (their hedonic value is mostly
the district mean through empirical-Bayes shrinkage).
"""
import argparse
import json
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from matplotlib import colormaps
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Polygon, box

warnings.filterwarnings('ignore')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CRS = 'EPSG:2180'
CENTER = (52.2318, 21.0060)                               # PKiN, origin of the saved views
OSM_DIR = Path('D:/QGIS/osm_basemap/warsaw')              # buildings.gpkg (+ parts), underground_ids.txt
GROUND_DIR = Path('D:/QGIS/mapy_warszawy_misc/data/osm')  # the transit poster's ground layers
GROUND = {  # stem: (file, layer, pandas query)
    'water': ('water.gpkg', 'water', None),
    'parks': ('parks.gpkg', 'parks', None), 'meadow': ('meadow.gpkg', 'meadow', None),
    'grass': ('grass.gpkg', 'grass', None), 'cemeteries': ('cemeteries.gpkg', 'cemeteries', None),
    'allotments': ('allotments.gpkg', 'allotments', None),
    'leisure_relations': ('leisure_relations.gpkg', 'leisure_relations', None),
    'forests': ('forests.gpkg', 'forests', None),
    'roads': ('roads.shp', None, None),
    'railways': ('railways.gpkg', 'railways', "railway == 'rail'"),
}
GREEN = ['parks', 'meadow', 'grass', 'cemeteries', 'allotments', 'leisure_relations']
# Transit poster colours (render_poster.poster_colours('warsaw'), 2026-10-15), 0-255 RGB
COLOURS = {'uncovered': (34, 34, 38), 'background': (12, 12, 14), 'green': (12, 12, 14),
           'forest': (12, 12, 14), 'water': (14, 24, 40), 'roads': (0, 0, 0),
           'buildings': (34, 34, 38), 'railways': (12, 12, 14)}
CONTEXT_RGB = (36, 37, 42)

LEVEL_M, DEFAULT_M = 3.2, 9.0
PARTS_COVER = 0.85
FAR_SIMPLIFY_M = 1.0
MIN_SALES = 3
ROAD_WIDTH = {
    'motorway': 16, 'trunk': 16, 'primary': 13, 'secondary': 11, 'tertiary': 9,
    'motorway_link': 7, 'trunk_link': 7, 'primary_link': 7, 'secondary_link': 7, 'tertiary_link': 7,
    'residential': 7, 'unclassified': 7, 'living_street': 6, 'service': 4, 'pedestrian': 5,
    'footway': 2.5, 'path': 2, 'cycleway': 2.5, 'steps': 2.5, 'track': 3,
}

# value sets: (layer, column, class breaks, legend title)
SETS = {
    'price_m2': ('building_value', 'value_2025',
                 [12000, 13000, 14000, 15000, 16000, 17500, 19000, 21000, 24000, 28000],
                 'Typical flat, PLN/m² (2025)'),
}


def ramp(n, lo=0.12, hi=1.0):
    """Sequential ramp for the dark scene: viridis, dark blue (low) -> yellow (high).
    Deliberately not the transit posters' blue-magenta-yellow, so the two never read alike."""
    cm = colormaps['viridis']
    return [[int(round(c * 255)) for c in cm(x)[:3]] for x in np.linspace(lo, hi, n)]


def read(path, bbox_m, layer=None, query=None):
    import fiona
    with fiona.open(path, layer=layer) as src:
        src_crs = src.crs_wkt
    bb = Transformer.from_crs(CRS, src_crs, always_xy=True).transform_bounds(*bbox_m)
    gdf = gpd.read_file(path, layer=layer, bbox=bb)
    if query and len(gdf):
        gdf = gdf.query(query, engine='python')
    return gdf.to_crs(CRS)


MAX_HOLES = 3  # Blender's polygon fill (tessellate_polygon) leaves polygons with many holes empty


def split_holes(p, max_holes=MAX_HOLES):
    """Halve a polygon until every piece has <= max_holes holes."""
    if len(p.interiors) <= max_holes:
        return [p]
    x0, y0, x1, y1 = p.bounds
    halves = (box(x0, y0, (x0 + x1) / 2, y1), box((x0 + x1) / 2, y0, x1, y1)) if x1 - x0 >= y1 - y0 \
        else (box(x0, y0, x1, (y0 + y1) / 2), box(x0, (y0 + y1) / 2, x1, y1))
    out = []
    for h in halves:
        part = p.intersection(h)
        for q in (part.geoms if hasattr(part, 'geoms') else [part]):
            if isinstance(q, Polygon) and not q.is_empty:
                out += split_holes(q, max_holes)
    return out


def rings(geom, cx, cy):
    """Polygon/MultiPolygon -> [[exterior, *holes], ...] as [[x, y], ...] local to the centre."""
    polys = geom.geoms if isinstance(geom, MultiPolygon) else [geom] if isinstance(geom, Polygon) else []
    polys = [q for p in polys if not p.is_empty for q in split_holes(p)]
    out = []
    for p in polys:
        if p.is_empty or p.area < 0.05:
            continue
        rs = [p.exterior] + list(p.interiors)
        out.append([[[round(x - cx, 1), round(y - cy, 1)] for x, y in r.coords[:-1]] for r in rs])
    return out


def ground_polys(gdf, crop, cx, cy, tile=250.0, simplify=0.3):
    """Dissolved layer cut into tile x tile m pieces (one huge dissolved polygon with thousands of
    holes is filled wrongly by Blender)."""
    import shapely
    if gdf is None or gdf.empty:
        return []
    g = gdf.geometry
    g = g[g.geom_type.isin(['Polygon', 'MultiPolygon'])].buffer(0)
    g = g[~g.is_empty]
    if not len(g):
        return []
    geoms, sidx = np.asarray(g.values), g.sindex
    x0, y0, x1, y1 = crop.bounds
    out = []
    for x in np.arange(x0, x1, tile):
        for y in np.arange(y0, y1, tile):
            cell = box(x, y, min(x + tile, x1), min(y + tile, y1)).intersection(crop)
            idx = sidx.query(cell, predicate='intersects')
            if not len(idx):
                continue
            u = shapely.union_all(shapely.intersection(geoms[idx], cell))
            if simplify:
                u = shapely.make_valid(u.simplify(simplify))
            parts = u.geoms if hasattr(u, 'geoms') else [u]
            out += [r for p in parts if p.geom_type in ('Polygon', 'MultiPolygon') for r in rings(p, cx, cy)]
    return out


def building_values(set_name):
    layer, col, breaks, title = SETS[set_name]
    v = gpd.read_file(ROOT / 'output' / 'price_model.gpkg', layer=layer, ignore_geometry=True)
    v = v[v['n'] >= MIN_SALES]
    return v.drop_duplicates('osm_id').set_index('osm_id')[col], breaks, title


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--name', required=True)
    ap.add_argument('--set', default='price_m2', choices=SETS)
    ap.add_argument('--bounds', default='-1500,-1500,1500,1500',
                    help='xmin,ymin,xmax,ymax in m relative to PKiN (the scene origin)')
    ap.add_argument('--detail-bounds', help='full detail inside; outlines simplified outside')
    args = ap.parse_args()

    lat, lon = CENTER
    cx, cy = Transformer.from_crs('EPSG:4326', CRS, always_xy=True).transform(lon, lat)
    x0, y0, x1, y1 = (float(v) for v in args.bounds.split(','))
    bbox_m = (cx + x0, cy + y0, cx + x1, cy + y1)
    crop = box(*bbox_m)
    print(f"Warsaw {(x1 - x0) / 1000:g} x {(y1 - y0) / 1000:g} km around PKiN, values: {args.set}")

    # Geometry + heights (OSM basemap, buildings and building:part)
    ug = OSM_DIR / 'underground_ids.txt'
    underground = {int(v) for v in ug.read_text().split()} if ug.exists() else set()
    b = read(OSM_DIR / 'buildings.gpkg', bbox_m)
    b['osm_id'] = b['osm_id'].astype('int64')
    b = b[~b['osm_id'].isin(underground) & b.geometry.notna()]
    b = b[b.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
    num = lambda c: pd.to_numeric(b[c], errors='coerce') if c in b else pd.Series(np.nan, b.index)  # noqa: E731
    h, lv = num('height'), num('building_levels')
    b['z1'] = h.fillna(lv * LEVEL_M).fillna(DEFAULT_M)
    b['z0'] = num('min_height').fillna(num('building_min_level') * LEVEL_M).fillna(0.0).clip(lower=0)
    roof = b['building'].isin(['roof', 'carport']) & h.isna()
    b.loc[roof, 'z1'] = 4.0
    b.loc[roof & (b['z0'] == 0), 'z0'] = 3.5
    b = b[b['z1'] > b['z0'] + 0.1]
    b = b[b.geometry.representative_point().within(crop)]
    is_part = b['building_part'].notna()
    parents, parts = b[~is_part], b[is_part]

    # Parent of each part (its inner point in the parent's footprint; largest parent wins)
    pp = gpd.sjoin(gpd.GeoDataFrame({'pid': parts.index.values},
                                    geometry=parts.geometry.representative_point().values, crs=CRS),
                   gpd.GeoDataFrame({'parent': parents.index.values, 'parea': parents.area.values},
                                    geometry=parents.geometry.values, crs=CRS),
                   predicate='within', how='inner').sort_values('parea').drop_duplicates('pid', keep='last')
    part_parent = pp.set_index('pid')['parent']
    drop = []
    for parent, pids in pp.groupby('parent')['pid']:
        g = parents.geometry.loc[parent]
        if parts.geometry.loc[pids.values].unary_union.intersection(g).area >= PARTS_COVER * g.area:
            drop.append(parent)
    # a part is looked up under its parent's osm_id (one colour per building)
    p_key = part_parent.map(parents['osm_id']).reindex(parts.index).fillna(parts['osm_id']).astype('int64')
    b = pd.concat([parents.assign(key=parents['osm_id']).drop(index=drop), parts.assign(key=p_key)])
    print(f"  {len(parents) - len(drop):,} buildings + {len(parts):,} parts ({len(drop):,} drawn by their parts)")

    if args.detail_bounds:
        d0, d1, d2, d3 = (float(v) for v in args.detail_bounds.split(','))
        far = ~b.geometry.representative_point().within(box(cx + d0, cy + d1, cx + d2, cy + d3))
        b.loc[far, 'geometry'] = b.loc[far].geometry.simplify(FAR_SIMPLIFY_M)
        print(f"  far background: {far.sum():,} outlines simplified by {FAR_SIMPLIFY_M:g} m")

    # Value classes
    vals, breaks, title = building_values(args.set)
    v = b['key'].map(vals).to_numpy(dtype=float)
    cls = np.where(np.isnan(v), 0, np.digitize(np.nan_to_num(v), breaks) + 1)
    palette = [list(COLOURS['uncovered'])] + ramp(len(breaks) + 1)
    context_class = len(palette)
    palette.append(list(CONTEXT_RGB))
    labels = ['no data', f'< {breaks[0]:g}'] + [f'{a:g}-{c:g}' for a, c in zip(breaks[:-1], breaks[1:])] \
        + [f'>= {breaks[-1]:g}', 'context']
    print(f"  coloured: {np.isfinite(v).sum():,} of {len(v):,} polygons ({len(vals):,} buildings "
          f"with >= {MIN_SALES} sales in the model)")

    buildings = []
    for geom, z0, z1, c in zip(b.geometry, b['z0'], b['z1'], cls):
        for poly in rings(geom, cx, cy):
            buildings.append({'rings': poly, 'z0': round(float(z0), 2), 'z1': round(float(z1), 2), 'c': int(c)})

    # Ground
    def load(stem):
        f, layer, q = GROUND[stem]
        p = GROUND_DIR / f
        return read(p, bbox_m, layer=layer, query=q) if p.exists() else None
    ground = {'water': ground_polys(load('water'), crop, cx, cy)}
    greens = [g for g in (load(s) for s in GREEN) if g is not None and len(g)]
    ground['green'] = ground_polys(pd.concat(greens) if greens else None, crop, cx, cy)
    ground['forest'] = ground_polys(load('forests'), crop, cx, cy)
    roads = load('roads')
    if roads is not None and len(roads):
        tun = roads['tunnel'].fillna('no').isin(['no', '', 'F']) if 'tunnel' in roads else True
        roads = roads[tun & roads['highway'].isin(ROAD_WIDTH.keys())]
        roads = roads.set_geometry(roads.geometry.buffer(roads['highway'].map(ROAD_WIDTH) / 2, cap_style=2))
        ground['roads'] = ground_polys(roads, crop, cx, cy)
    rail = load('railways')
    if rail is not None and len(rail):
        if 'tunnel' in rail:
            rail = rail[rail['tunnel'].fillna('no').isin(['no', '', 'F'])]
        ground['railways'] = ground_polys(rail.set_geometry(rail.geometry.buffer(1.5, cap_style=2)), crop, cx, cy)
    ground['base'] = rings(crop, cx, cy)

    out_dir = HERE / 'scenes' / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = {'city': 'warsaw', 'crs': CRS, 'center_lonlat': [lon, lat], 'center_xy': [cx, cy],
             'bbox_local': [x0, y0, x1, y1], 'set': args.set, 'title': title, 'breaks': breaks,
             'palette': palette, 'context_class': context_class, 'class_labels': labels,
             'colours': COLOURS, 'buildings': buildings, 'ground': ground}
    out = out_dir / 'scene.json'
    out.write_text(json.dumps(scene, separators=(',', ':')))
    print(f"  wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(buildings):,} building polygons, "
          + ', '.join(f'{k} {len(v)}' for k, v in ground.items()))
    print('  class counts: ' + ' '.join(f'{k}:{n}' for k, n in zip(*np.unique(cls, return_counts=True))))


if __name__ == '__main__':
    main()
