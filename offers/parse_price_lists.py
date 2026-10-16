"""
Parse the developer price lists downloaded by fetch_price_lists.py into one table of flats on offer in Warsaw.

The official template (dane.gov.pl, 2025 law) is one wide table: developer details, the sales office address, the
INVESTMENT location (województwo, powiat, gmina, miejscowość, ulica, nr nieruchomości, kod pocztowy), then one row
per unit with its type, number, price per m², total price and the dates the prices apply from. Developers' files
differ in header row, column order, spelling and format, so columns are matched by name patterns, and only the
investment-location block decides whether a row is in Warsaw (many developers are registered in Warsaw but build
elsewhere).

    python offers/parse_price_lists.py   -> offers/data/flats_warszawa.csv, offers/data/parse_report.csv
"""
import glob
import io
import json
import os
import re

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
OUT = os.path.join(HERE, "data", "flats_warszawa.csv")
REPORT = os.path.join(HERE, "data", "parse_report.csv")

LOC = r"(lokalizacji przedsi|lokalizacja przedsi|przedsi\w* deweloper)"
COLS = {   # output column: regex on the normalised header
    "developer": r"^nazwa dewelopera",
    "inv_voiv": r"wojew\w*" + r".*" + LOC,
    "inv_county": r"powiat" + r".*" + LOC,
    "inv_gmina": r"gmina" + r".*" + LOC,
    "inv_town": r"miejscowo\w*" + r".*" + LOC,
    "inv_street": r"ulica" + r".*" + LOC,
    "inv_no": r"nr nieruchomo\w*" + r".*" + LOC,
    "inv_zip": r"kod pocztowy" + r".*" + LOC,
    "kind": r"^rodzaj nieruchomo",
    "unit": r"^nr lokalu lub domu|^numer lokalu|^nr lokalu$",
    "price_m2": r"^cena (za )?m ?2|^cena za metr|^cena 1 ?m",
    "price_m2_from": r"^data od kt\w* (obowi\w* )?cena (obowi\w* )?cena m ?2|^data od kt\w* obowi\w* cena m ?2",
    "price": r"^cena lokalu mieszkalnego lub domu jednorodzinnego b\w* przedmiotem umowy stanowi\w* iloczyn",
    "price_total": r"^cena lokalu mieszkalnego lub domu jednorodzinnego uwzgl",
}
TEMPLATE_HINT = re.compile(r"nazwa dewelopera|lokalizacji przedsi|cena m ?2", re.I)


def norm(s):
    s = str(s).lower().replace("\n", " ").replace("²", "2").replace("  ", " ")
    return re.sub(r"\s+", " ", s).strip()


def read_raw(path):
    ext = path.rsplit(".", 1)[-1].lower()
    data = open(path, "rb").read()
    if not data:
        return None
    if ext in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(data), header=None, dtype=str)
    if ext == "json":
        j = json.loads(data.decode("utf-8", "ignore"))
        if isinstance(j, dict):
            j = next((v for v in j.values() if isinstance(v, list)), [j])
        d = pd.json_normalize(j)
        return pd.concat([pd.DataFrame([d.columns], columns=d.columns), d.astype(str)], ignore_index=True)
    for enc in ("utf-8-sig", "cp1250"):
        for sep in (";", ",", "\t"):
            try:
                d = pd.read_csv(io.BytesIO(data), header=None, sep=sep, dtype=str, encoding=enc,
                                on_bad_lines="skip", engine="python")
                if d.shape[1] > 5:
                    return d
            except Exception:
                continue
    return None


def with_header(raw):
    """First row that looks like the template header becomes the header."""
    for i in range(min(15, len(raw))):
        row = " | ".join(norm(v) for v in raw.iloc[i].tolist())
        if len(TEMPLATE_HINT.findall(row)) >= 2:
            d = raw.iloc[i + 1:].copy()
            d.columns = [norm(c) for c in raw.iloc[i].tolist()]
            return d
    return None


def pick(d):
    out = {}
    for name, pat in COLS.items():
        hit = [c for c in d.columns if re.search(pat, c)]
        if hit:
            out[name] = d[hit[0]].to_numpy()
    return pd.DataFrame(out) if out else None


def to_num(s):
    s = pd.Series(s, dtype=str).str.replace(r"[^\d,.\-]", "", regex=True)
    s = s.str.replace(",", ".", regex=False).str.replace(r"\.(?=.*\.)", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def main():
    files = sorted(glob.glob(os.path.join(RAW, "*")))
    idx = pd.read_csv(os.path.join(HERE, "data", "index.csv"))
    title = dict(zip(idx["id"].astype(str), idx["title"]))
    rows, rep = [], []
    for f in files:
        did = os.path.basename(f).split(".")[0]
        status, n, nw = "", 0, 0
        try:
            raw = read_raw(f)
            if raw is None:
                status = "unreadable"
            else:
                d = with_header(raw)
                if d is None:
                    status = "no template header"
                else:
                    p = pick(d)
                    if p is None or "inv_town" not in p and "inv_gmina" not in p:
                        status = "no investment location"
                    else:
                        loc = (p.get("inv_town", pd.Series("", index=p.index)).astype(str) + " " +
                               p.get("inv_gmina", pd.Series("", index=p.index)).astype(str)).str.lower()
                        p = p[loc.str.contains("warszaw", na=False)].copy()
                        n, nw = len(d), len(p)
                        status = "ok"
                        if nw:
                            p["dataset_id"] = did
                            p["dataset_title"] = title.get(did, "")
                            rows.append(p)
        except Exception as e:
            status = f"error {type(e).__name__}"
        rep.append({"id": did, "status": status, "rows": n, "warsaw_rows": nw})
    rep = pd.DataFrame(rep)
    rep.to_csv(REPORT, index=False)
    flats = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    for c in ("price_m2", "price", "price_total"):
        if c in flats:
            flats[c] = to_num(flats[c])
    if {"price", "price_m2"} <= set(flats.columns):
        flats["area"] = (flats["price"] / flats["price_m2"]).round(2)
    if "kind" in flats:
        flats = flats[~flats["kind"].astype(str).str.contains("dom", case=False, na=False)]
    flats = flats[flats["price_m2"].between(5000, 150000)] if "price_m2" in flats else flats
    flats.to_csv(OUT, index=False, encoding="utf-8")
    print(rep["status"].value_counts().to_string())
    print(f"{rep['warsaw_rows'].gt(0).sum():,} datasets with Warsaw flats; {len(flats):,} flats on offer kept "
          f"(median {flats['price_m2'].median():,.0f} PLN/m², median area {flats.get('area', pd.Series()).median():.1f} m²)")
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
