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
REGION = (0.14, 0.865)         # band of a city ridgeline slide that holds the map (below the title, above the color bar)
CITIES = [   # name, folder, color scale, axes box, label position
    ("Kraków", "krakow", "10–20K+ PLN/m²", [0.14, 0.535, 0.86, 0.33], (0.05, 0.835)),
    ("Gdańsk", "gdansk", "10–19K+ PLN/m²", [0.00, 0.06, 0.52, 0.44], (0.05, 0.50)),
    ("Lublin", "lublin", "7–13K+ PLN/m²", [0.49, 0.06, 0.51, 0.44], (0.56, 0.50)),
]


def map_crop(path, pad=20):
    """The map itself: the bounding box of everything that is not background inside REGION."""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im).astype(int)
    h = a.shape[0]
    y0, y1 = int(REGION[0] * h), int(REGION[1] * h)
    bg = a[5, 5]
    ink = (np.abs(a[y0:y1] - bg).sum(axis=2) > 40)
    rows, cols = np.flatnonzero(ink.any(axis=1)), np.flatnonzero(ink.any(axis=0))
    return im.crop((max(cols[0] - pad, 0), y0 + max(rows[0] - pad, 0), cols[-1] + pad, y0 + rows[-1] + pad))


def main():
    style.apply()
    fig = plt.figure(figsize=FIG, facecolor=BG)
    for city, folder, scale, box, (lx, ly) in CITIES:
        im = map_crop(os.path.join(ROOT, folder, "output", "summary", "01_price_ridgelines.png"))
        ax = fig.add_axes(box)
        ax.imshow(np.asarray(im), interpolation="antialiased")
        ax.axis("off")
        fig.text(lx, ly, city, color=INK, fontsize=22, fontweight="bold", va="bottom")
        fig.text(lx, ly - 0.006, f"colors: {scale}", color=MUTED, fontsize=10, va="top")
    style.byline(fig, y=0.975, color=MUTED, fontsize=10, x=0.95)
    fig.text(0.05, 0.955, "Some other cities", color=INK, fontsize=30, fontweight="bold", va="top")
    fig.text(0.05, 0.905, "Same register, same method, same waves: Kraków, Gdańsk and Lublin.", color=MUTED, fontsize=12.5,
             va="top")
    fig.text(0.05, 0.022, "Typical 2-room resale flat, 2025, per building; each map on its own city's price scale (Warsaw: 12–24K+).\n"
                          "Kraków's register is thin before 2021. Source: GUGiK RCN, BDOT10k.", color=MUTED, fontsize=7.5, va="bottom")
    out = os.path.join(ROOT, "output", "report", "src", "r98_other_cities.png")
    mpl.rcParams["savefig.facecolor"] = BG
    fig.savefig(out, dpi=DPI, facecolor=BG)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
