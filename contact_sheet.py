"""
Contact sheet of every summary figure: output/summary/*.png as numbered thumbnails in one image
-> output/summary/00_contact_sheet.png
"""
import glob
import os
import re

from PIL import Image, ImageDraw, ImageFont

SRC = "output/summary"
OUT = os.path.join(SRC, "00_contact_sheet.png")
COLS = 6
THUMB_W = 400
PAD = 24
LABEL_H = 34
BG = (249, 249, 247)


def font(size):
    for f in ("segoeuib.ttf", "segoeui.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    files = sorted(f for f in glob.glob(os.path.join(SRC, "*.png")) if not os.path.basename(f).startswith("00_"))
    thumbs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        h = int(im.height * THUMB_W / im.width)
        thumbs.append((os.path.basename(f), im.resize((THUMB_W, h), Image.LANCZOS)))
    th = max(t.height for _, t in thumbs)
    rows = (len(thumbs) + COLS - 1) // COLS
    W = COLS * THUMB_W + (COLS + 1) * PAD
    H = 110 + rows * (th + LABEL_H + PAD) + PAD
    sheet = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(sheet)
    d.text((PAD, 30), f"Warsaw flat prices: all {len(thumbs)} summary figures", fill=(11, 11, 11), font=font(40))
    f_lab = font(19)
    for k, (name, t) in enumerate(thumbs):
        r, c = divmod(k, COLS)
        x = PAD + c * (THUMB_W + PAD)
        y = 110 + r * (th + LABEL_H + PAD)
        label = re.sub(r"\.png$", "", name).replace("_", " ")
        d.text((x, y), label, fill=(82, 81, 78), font=f_lab)
        sheet.paste(t, (x, y + LABEL_H))
        d.rectangle([x - 1, y + LABEL_H - 1, x + THUMB_W, y + LABEL_H + t.height], outline=(225, 224, 217))
    sheet.save(OUT, optimize=True)
    print(f"Saved {OUT}: {len(thumbs)} figures, {W} x {H} px")


if __name__ == "__main__":
    main()
