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
    ("src", "r01_data"),
    # 2 sizes
    ("src", "r02_sizes"), ("src", "r03_new_resale"), ("summary", "29_flat_sizes"),
    # 3 prices
    ("summary", "01_prices_over_time"), ("summary", "40_where_flats_are_big"),
    ("summary", "38_house_map"), ("summary", "39_house_cells"), ("summary", "35_house_or_flat"),
    # 4 what goes with price
    ("summary", "03_what_drives_price"), ("summary", "18_building_age"), ("summary", "21_infill"),
    ("summary", "08_cafes"), ("summary", "09_noise"), ("summary", "20_trees"), ("summary", "11_transit_x_price"),
    ("summary", "12_price_compass"), ("summary", "24_transit_access"),
    # 5 reputation
    ("summary", "44_reputation_glow"), ("summary", "41_pairs"), ("summary", "10_district_anatomy"), ("summary", "06_cheaper_than_expected"), ("summary", "15_estates"),
    ("summary", "14_reputation_over_time"), ("summary", "30_forward_test"),
    # 6 market behaviour
    ("summary", "16_flippers"), ("summary", "26_new_flats_on_offer"),
    # end
    ("src", "r99_methods"),
]

SHORT = ["31_price_ridgelines", "01_prices_over_time", "18_building_age", "20_trees", "26_new_flats_on_offer",
         "41_pairs", "06_cheaper_than_expected", "30_forward_test", "16_flippers",
         "r99_methods"]   # the LinkedIn cut (2026-11-03): question + answer titles; old cuts in output/report/archive_*/

SELECTED = [   # the posting set, two batches (2026-11-04) -> output/report/selected/ (numbered, no PDF)
    # batch 1, "Warsaw, in price": the city, how dear, what you get, who you buy from
    "31_price_ridgelines", "32_price_points", "r01_data", "23_blind_spot", "01_prices_over_time",
    "40_where_flats_are_big", "r02_sizes", "38_house_map", "39_house_cells", "35_house_or_flat",
    "26_new_flats_on_offer", "16_flippers", "12_price_compass",
    "flat_aleja_solidarnosci_129_131_0", "r98_other_cities",   # one flat, the zoom-in closer (Muranów edge: 1950s blocks + new builds):
                                   #   python flat_slide.py --street "Aleja „Solidarności”" --no 129/131 --flat 0 --m2 33
                                   #   --rooms 2 --floor 3 --public --label "A 33 m² flat on al. Solidarności"
    # batch 2, "Warsaw, beyond price": what goes with price, and what's left (the address)
    "44_reputation_glow",          # cover: night lights (2026-11-04; alternatives 42 ridgelines, 43 spikes)
    "03_what_drives_price", "18_building_age", "21_infill", "11_transit_x_price", "24_transit_access",
    "08_cafes", "09_noise", "20_trees",
    "10_district_anatomy", "06_cheaper_than_expected", "15_estates", "14_reputation_over_time",
]   # out: 41_pairs (user, 2026-11-04), 30_forward_test (repeats 14; its Stary Żoliborz +31 % clashes with 15's +23 %)

PIPELINE = [   # figure scripts in dependency order (data fetches and price_index.py are not rerun here)
    "age_check.py", "price_drivers.py", "summary_charts.py", "summary_maps.py", "summary_advanced.py",
    "reputation_time.py", "estates.py", "infill.py", "flips.py", "building_age_story.py", "flat_sizes.py",
    "offers/offers_story.py", "access/access_preview.py", "houses.py", "house_map.py", "house_cells.py", "family_flats.py", "price_ridgelines.py",
    "price_points.py", "reputation_ridgelines.py", "reputation_covers.py", "pairs.py", "forward_test.py", "report/descriptive.py", "report/other_cities.py",
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
    folders = {"summary": os.path.join(ROOT, "output", "summary"), "src": os.path.join(OUT, "src"),
               "flats": os.path.join(OUT, "flats")}
    missing = [f"{d}/{s}.png" for d, s in SLIDES if not os.path.exists(os.path.join(folders[d], f"{s}.png"))]
    if missing:
        sys.exit("missing slides (run with --rebuild): " + ", ".join(missing))
    def write_pdf(path, items):
        with PdfPages(path) as pdf:
            for d, stem in items:
                src = os.path.join(folders[d], f"{stem}.png")
                fig = plt.figure(figsize=(8, 10), dpi=150)
                ax = fig.add_axes([0, 0, 1, 1])
                ax.imshow(Image.open(src).convert("RGB"), interpolation="antialiased")
                ax.axis("off")
                pdf.savefig(fig, dpi=300 if stem == "32_price_points" else 150)
                plt.close(fig)

    for i, (d, stem) in enumerate(SLIDES, 1):
        shutil.copyfile(os.path.join(folders[d], f"{stem}.png"), os.path.join(OUT, f"{i:02d}_{stem}.png"))
    write_pdf(PDF, SLIDES)
    by_stem = {stem: (d, stem) for d, stem in SLIDES}
    by_stem["r98_other_cities"] = ("src", "r98_other_cities")
    by_stem["23_blind_spot"] = ("summary", "23_blind_spot")
    by_stem["flat_aleja_solidarnosci_129_131_0"] = ("flats", "flat_aleja_solidarnosci_129_131_0")   # sample flat, posting set only
    write_pdf(PDF.replace(".pdf", "_short.pdf"), [by_stem[x] for x in SHORT])
    sel = os.path.join(OUT, "selected")   # the set to post; numbered in order, not merged (breaks between posts by hand)
    os.makedirs(sel, exist_ok=True)
    for f in os.listdir(sel):
        if f.endswith(".png"):
            os.remove(os.path.join(sel, f))
    for i, stem in enumerate(SELECTED, 1):
        d, _ = by_stem[stem]
        shutil.copyfile(os.path.join(folders[d], f"{stem}.png"), os.path.join(sel, f"{i:02d}_{stem}.png"))
    b1 = SELECTED[:SELECTED.index("44_reputation_glow")]          # batch 1 = everything before the batch-2 cover
    write_pdf(os.path.join(OUT, "warsaw_in_price_batch1.pdf"), [by_stem[x] for x in b1])
    b2 = SELECTED[len(b1):]                                        # batch 2 = the cover and everything after it
    write_pdf(os.path.join(OUT, "warsaw_beyond_price_batch2.pdf"), [by_stem[x] for x in b2])
    print(f"batch 2 ({len(b2)} slides) -> warsaw_beyond_price_batch2.pdf")
    print(f"selected deck: {len(SELECTED)} slides -> {sel}; batch 1 ({len(b1)} slides) -> warsaw_in_price_batch1.pdf")
    print(f"short deck: {len(SHORT)} slides -> {PDF.replace('.pdf', '_short.pdf')}")
    print(f"{len(SLIDES)} slides -> {OUT}, {PDF} ({os.path.getsize(PDF) / 1e6:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="rerun the figure scripts first")
    if ap.parse_args().rebuild:
        rebuild()
    assemble()
