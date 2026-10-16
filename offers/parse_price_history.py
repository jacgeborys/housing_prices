"""
Parse the monthly snapshots (fetch_price_history.py) into one long table: flat x month -> asking price.

Same template logic as parse_price_lists.py (columns by name, Warsaw by the investment's location). A flat is
developer + investment street + number + unit; its asking price per m² is followed month by month; a flat that
disappears from a list after appearing is taken as sold or withdrawn ("off the list"). Several rows for one flat
in a snapshot: the latest-dated price wins; undecidable ones (> 5 % apart) are dropped.

    python offers/parse_price_history.py   -> offers/data/history_flats.csv
"""
import glob
import os

import pandas as pd

from parse_price_lists import pick, read_raw, to_num, with_header

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "history_flats.csv")


def main():
    rows, bad = [], 0
    files = sorted(glob.glob(os.path.join(HERE, "data", "history", "*", "*")))
    for k, f in enumerate(files):
        did = os.path.basename(os.path.dirname(f))
        month = os.path.basename(f).split(".")[0]
        try:
            raw = read_raw(f)
            d = with_header(raw) if raw is not None else None
            p = pick(d) if d is not None else None
        except Exception:
            p = None
        if p is None or ("inv_town" not in p and "inv_gmina" not in p):
            bad += 1
            continue
        loc = (p.get("inv_town", pd.Series("", index=p.index)).astype(str) + " " +
               p.get("inv_gmina", pd.Series("", index=p.index)).astype(str)).str.lower()
        p = p[loc.str.contains("warszaw", na=False)].copy()
        if len(p):
            p["dataset_id"], p["month"] = did, month
            rows.append(p[[c for c in ("developer", "inv_street", "inv_no", "unit", "price_m2", "price_m2_from", "price",
                                       "dataset_id", "month") if c in p]])
        if k % 500 == 0:
            print(f"  {k:,}/{len(files):,}", flush=True)
    h = pd.concat(rows, ignore_index=True)
    for c in ("price_m2", "price"):
        h[c] = to_num(h[c])
    h = h[h["price_m2"].between(5000, 150000)]
    h["flat"] = h[["developer", "inv_street", "inv_no", "unit"]].astype(str).agg(" | ".join, axis=1)
    # some files list one flat number several times (price-history rows, or two buildings sharing numbering):
    # keep the row with the latest price date; if the date can't decide and the prices differ by > 5 %, the flat
    # is ambiguous and leaves the history altogether (a guess would create fake price changes)
    h["pdate"] = pd.to_datetime(h["price_m2_from"], errors="coerce", dayfirst=True)
    h = h.sort_values(["flat", "month", "pdate"])
    grp = h.groupby(["flat", "month"])
    spread = grp["price_m2"].transform("max") / grp["price_m2"].transform("min") - 1
    top = grp["pdate"].transform("max")
    n_top = h.assign(is_top=h["pdate"].eq(top)).groupby(["flat", "month"])["is_top"].transform("sum")
    ambiguous = (spread > 0.05) & ((n_top != 1) | top.isna())
    bad_flats = set(h.loc[ambiguous, "flat"])
    h = h[~h["flat"].isin(bad_flats)].drop_duplicates(["flat", "month"], keep="last").drop(columns=["pdate"])
    print(f"  {len(bad_flats):,} flats dropped as ambiguous (several rows per snapshot, prices > 5 % apart)")
    h.to_csv(OUT, index=False, encoding="utf-8")
    per = h.groupby("flat")["month"].nunique()
    print(f"{len(files) - bad:,} of {len(files):,} snapshots parsed; {len(h):,} flat-months, {h['flat'].nunique():,} flats, "
          f"{(per >= 3).sum():,} followed for >= 3 months")
    print(h.groupby("month")["flat"].nunique().to_string())


if __name__ == "__main__":
    main()
