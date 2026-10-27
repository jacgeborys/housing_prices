"""
The report: one deck in story order, built from the figure scripts of this project.

    python report/build_report.py             assemble output/report/ from the figures already built
    python report/build_report.py --rebuild   rerun every figure script first (pipeline order, ~1 h)

Output: output/report/NN_<slide>.png (numbered copies) and output/report/warsaw_flats.pdf (one 8 x 10 in page per
slide, images embedded losslessly). Slides come from output/summary/ (figure scripts in the project root and
access/) and output/report/src/ (report/descriptive.py: plain descriptive slides and the methods page).
"""
import argparse
import os
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "output", "report")
PDF = os.path.join(OUT, "warsaw_flats.pdf")

SLIDES = [   # (source folder, file stem); sections as in the README
    # opening
    ("summary", "31_price_ridgelines"), ("summary", "32_price_points"),
    # 1 what's in the data
    ("src", "r01_data"), ("summary", "23_blind_spot"),
    # 2 sizes
    ("src", "r02_sizes"), ("src", "r03_new_resale"), ("summary", "29_flat_sizes"),
    # 3 prices
    ("summary", "01_prices_over_time"), ("summary", "33_what_1m_buys"), ("summary", "34_family_flats"),
    ("summary", "38_house_map"), ("summary", "35_house_or_flat"),
    # 4 what goes with price
    ("summary", "03_what_drives_price"), ("summary", "18_building_age"), ("summary", "21_infill"),
    ("summary", "08_cafes"), ("summary", "09_noise"), ("summary", "20_trees"), ("summary", "11_transit_x_price"),
    ("summary", "12_price_compass"), ("summary", "24_transit_access"),
    # 5 reputation
    ("summary", "10_district_anatomy"), ("summary", "06_cheaper_than_expected"), ("summary", "15_estates"),
    ("summary", "14_reputation_over_time"),
    # 6 market behaviour
    ("summary", "16_flippers"), ("summary", "26_new_flats_on_offer"),
    # end
    ("src", "r99_methods"),
]

PIPELINE = [   # figure scripts in dependency order (data fetches and price_index.py are not rerun here)
    "age_check.py", "price_drivers.py", "summary_charts.py", "summary_maps.py", "summary_advanced.py",
    "reputation_time.py", "estates.py", "infill.py", "flips.py", "building_age_story.py", "flat_sizes.py",
    "offers/offers_story.py", "access/access_preview.py", "houses.py", "house_map.py", "family_flats.py", "price_ridgelines.py",
    "price_points.py", "report/descriptive.py",
]


def rebuild():
    env = dict(os.environ, LOKY_MAX_CPU_COUNT="8")
    for script in PIPELINE:
        print(f"=== {script}", flush=True)
        r = subprocess.run([sys.executable, "-W", "ignore", script], cwd=ROOT, env=env)
        if r.returncode:
            sys.exit(f"{script} failed")


def assemble():
    os.makedirs(OUT, exist_ok=True)
    for f in os.listdir(OUT):                     # old numbered copies only; src/ and other files stay
        if f[:2].isdigit() and f[2] == "_" and f.endswith(".png"):
            os.remove(os.path.join(OUT, f))
    folders = {"summary": os.path.join(ROOT, "output", "summary"), "src": os.path.join(OUT, "src")}
    missing = [f"{d}/{s}.png" for d, s in SLIDES if not os.path.exists(os.path.join(folders[d], f"{s}.png"))]
    if missing:
        sys.exit("missing slides (run with --rebuild): " + ", ".join(missing))
    with PdfPages(PDF) as pdf:
        for i, (d, stem) in enumerate(SLIDES, 1):
            src = os.path.join(folders[d], f"{stem}.png")
            shutil.copyfile(src, os.path.join(OUT, f"{i:02d}_{stem}.png"))
            fig = plt.figure(figsize=(8, 10), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.imshow(Image.open(src).convert("RGB"), interpolation="antialiased")
            ax.axis("off")
            pdf.savefig(fig, dpi=300 if stem == "32_price_points" else 150)
            plt.close(fig)
    print(f"{len(SLIDES)} slides -> {OUT}, {PDF} ({os.path.getsize(PDF) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rerun the figure scripts first")
    if ap.parse_args().rebuild:
        rebuild()
    assemble()
