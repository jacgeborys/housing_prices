"""
Per-building values for the Blender scenes of D:/QGIS/TransitFrequency/3d (export_scene.py --values).

Writes output/3d/<name>.csv (osm_id,value) plus the matching --breaks / --colors arguments, and
prints the export command. Buildings with fewer than MIN_SALES recorded sales get no value: their
hedonic value is mostly the district mean (empirical-Bayes shrinkage), so they stay dark ("no data").

    python export_3d_values.py                 # price_m2: hedonic value of a typical flat, 2025 PLN/m²
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
from matplotlib import colormaps

OUT_DIR = Path("output/3d")
MIN_SALES = 3

# value column, class breaks, legend title
SETS = {
    "price_m2": ("value_2025",
                 [12000, 13000, 14000, 15000, 16000, 17500, 19000, 21000, 24000, 28000],
                 "Typical flat, PLN/m² (2025)"),
}


def ramp(n, lo=0.12, hi=1.0):
    """Sequential ramp for a dark scene: viridis, dark blue (low) -> yellow (high)."""
    cm = colormaps["viridis"]
    return ["#%02x%02x%02x" % tuple(int(round(c * 255)) for c in cm(x)[:3])
            for x in np.linspace(lo, hi, n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="price_m2", choices=SETS)
    args = ap.parse_args()
    col, breaks, title = SETS[args.set]

    v = gpd.read_file("output/price_model.gpkg", layer="building_value", ignore_geometry=True)
    v = v[v["n"] >= MIN_SALES]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{args.set}.csv"
    v[["osm_id", col]].rename(columns={col: "value"}).to_csv(out, index=False)

    colors = ramp(len(breaks) + 1)
    counts = np.bincount(np.digitize(v[col], breaks), minlength=len(breaks) + 1)
    print(f"{out}: {len(v):,} buildings with >= {MIN_SALES} sales ({title})")
    print("class sizes: " + " ".join(str(c) for c in counts))
    print("\nexport (run in D:/QGIS/TransitFrequency):")
    print(f"python 3d/export_scene.py --city warsaw --center 52.2318,21.0060 "
          f"--bounds=-7000,-15000,15000,9000 --detail-bounds=-7000,-6500,7000,9000 "
          f"--name prices_wide --no-coverage --values {out.resolve().as_posix()} "
          f"--breaks {','.join(str(b) for b in breaks)} --colors {','.join(colors)}")


if __name__ == "__main__":
    main()
