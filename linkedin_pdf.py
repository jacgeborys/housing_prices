"""
LinkedIn document post: the selected summary figures, in story order, copied to output/linkedin/ (numbered) and
merged into one PDF (lossless images, one 8 x 10 in page per figure).

    python linkedin_pdf.py   -> output/linkedin/NN_<figure>.png, output/linkedin/warsaw_flat_prices.pdf
"""
import os
import shutil

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

SRC, OUT = "output/summary", "output/linkedin"
PDF = f"{OUT}/warsaw_flat_prices.pdf"
ORDER = [   # 1 what a flat is worth, 2 what you pay for around it, 3 reputation
    "31_price_ridgelines", "23_blind_spot", "01_prices_over_time", "33_what_1m_buys", "34_family_flats", "35_house_or_flat",
    "18_building_age", "21_infill", "26_new_flats_on_offer", "16_flippers",
    "03_what_drives_price", "08_cafes", "09_noise", "20_trees", "24_transit_access", "25_what_reach_pays",
    "11_transit_x_price", "12_price_compass", "13_shape_of_value",
    "10_district_anatomy", "06_cheaper_than_expected", "15_estates", "14_reputation_over_time", "30_forward_test",
]


def main():
    os.makedirs(OUT, exist_ok=True)
    with PdfPages(PDF) as pdf:
        for i, name in enumerate(ORDER, 1):
            src = f"{SRC}/{name}.png"
            shutil.copyfile(src, f"{OUT}/{i:02d}_{name}.png")
            im = Image.open(src).convert("RGB")
            w, h = im.size
            fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
            fig.add_axes([0, 0, 1, 1]).imshow(im, interpolation="none")
            fig.axes[0].axis("off")
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
    print(f"{len(ORDER)} figures -> {OUT}/, {PDF} ({os.path.getsize(PDF) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
