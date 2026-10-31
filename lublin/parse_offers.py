"""
Developer price lists (dane.gov.pl, 2025 law; files already downloaded by offers/fetch_price_lists.py into
offers/data/raw/) -> flats on offer in Lublin. Warsaw's parser (offers/parse_price_lists.py) is imported, not edited;
only the location filter differs: the investment's town or gmina must be the word "Lublin" (not Lubliniec).
Run from the lublin/ folder:  python parse_offers.py   -> data/offers_lublin.csv
"""
import glob
import os
import re
import sys

sys.dont_write_bytecode = True
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "offers"))
import pandas as pd  # noqa: E402

import parse_price_lists as ppl  # noqa: E402

OUT = "data/offers_lublin.csv"
TOWN = re.compile(r"\blublin\b")


def main():
    idx = pd.read_csv(os.path.join(ROOT, "offers", "data", "index.csv"))
    title = dict(zip(idx["id"].astype(str), idx["title"]))
    rows = []
    for f in sorted(glob.glob(os.path.join(ppl.RAW, "*"))):
        did = os.path.basename(f).split(".")[0]
        try:
            raw = ppl.read_raw(f)
            d = None if raw is None else ppl.with_header(raw)
            p = None if d is None else ppl.pick(d)
            if p is None or ("inv_town" not in p and "inv_gmina" not in p):
                continue
            loc = (p.get("inv_town", pd.Series("", index=p.index)).astype(str) + " " +
                   p.get("inv_gmina", pd.Series("", index=p.index)).astype(str)).str.lower()
            p = p[loc.map(lambda s: bool(TOWN.search(s)))].copy()
            if len(p):
                p["dataset_id"], p["dataset_title"] = did, title.get(did, "")
                rows.append(p)
        except Exception:
            continue
    flats = pd.concat(rows, ignore_index=True)
    for c in ("price_m2", "price", "price_total"):
        if c in flats:
            flats[c] = ppl.to_num(flats[c])
    flats["area"] = (flats["price"] / flats["price_m2"]).round(2)
    if "kind" in flats:
        flats = flats[~flats["kind"].astype(str).str.contains("dom", case=False, na=False)]
    flats = flats[flats["price_m2"].between(4000, 60000)]
    flats.to_csv(OUT, index=False, encoding="utf-8")
    print(f"{flats['dataset_id'].nunique():,} datasets, {len(flats):,} flats on offer in Lublin "
          f"(median {flats['price_m2'].median():,.0f} PLN/m², {flats['area'].median():.1f} m²) -> {OUT}")


if __name__ == "__main__":
    main()
