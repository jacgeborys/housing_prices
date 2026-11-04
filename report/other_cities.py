"""
Closing slide of the first post: the Gdańsk and Lublin ridgeline maps (gdansk/maps.py, lublin/maps.py, rebuilt in the
deck's font) cropped to the map and stacked on the dark cover background, with a question for the comments.

Output: output/report/src/r98_other_cities.png (2400 x 3000, dark)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import style  # noqa: E402
from price_ridgelines import BG, INK, MUTED  # noqa: E402
from summary_charts import DPI, FIG  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CROP = (0.04, 0.17, 0.96, 0.885)        # map area of a city ridgeline slide: below the title, above the color bar
CITIES = [("Gdańsk", "gdansk", "10–19K+ PLN/m²", [-0.04, 0.44, 0.80, 0.45], (0.70, 0.74)),
          ("Lublin", "lublin", "7–13K+ PLN/m²", [0.24, 0.045, 0.80, 0.45], (0.06, 0.395))]


def main():
    style.apply()
    fig = plt.figure(figsize=FIG, facecolor=BG)
    for city, folder, scale, box, (lx, ly) in CITIES:
        im = Image.open(os.path.join(ROOT, folder, "output", "summary", "01_price_ridgelines.png")).convert("RGB")
        w, h = im.size
        im = im.crop((int(CROP[0] * w), int(CROP[1] * h), int(CROP[2] * w), int(CROP[3] * h)))
        ax = fig.add_axes(box)
        ax.imshow(np.asarray(im), interpolation="antialiased")
        ax.axis("off")
        fig.text(lx, ly, city, color=INK, fontsize=24, fontweight="bold", va="bottom")
        fig.text(lx, ly - 0.006, f"colors: {scale}", color=MUTED, fontsize=10.5, va="top")
    style.byline(fig, y=0.975, color=MUTED, fontsize=10, x=0.95)
    fig.text(0.05, 0.955, "Some other cities", color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, "Same register, same method, same waves: Gdańsk and Lublin.", color=MUTED, fontsize=12.5,
             va="top")
    fig.text(0.05, 0.022, "Typical 2-room resale flat, 2025, per building; each map on its own city's price scale "
                          "(Warsaw: 12–24K+). Source: GUGiK RCN, BDOT10k.", color=MUTED, fontsize=7.5, va="bottom")
    out = os.path.join(ROOT, "output", "report", "src", "r98_other_cities.png")
    mpl.rcParams["savefig.facecolor"] = BG
    fig.savefig(out, dpi=DPI, facecolor=BG)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
