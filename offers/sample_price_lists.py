"""
How usable are the developers' price lists on dane.gov.pl? (2025 price-transparency law: every developer
publishes offer prices and their history; one dataset per developer and year, "Ceny ofertowe mieszkań
dewelopera ..."). Formats vary, so before downloading everything this samples N_SAMPLE datasets and reports
which files carry a location (columns or values naming the municipality / street) and how many are in Warsaw.

  python offers/sample_price_lists.py      -> offers/data/sample/*.xlsx|csv, offers/data/sample_report.csv
"""
import io
import os
import random
import re
import time

import pandas as pd
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data", "sample")
API = "https://api.dane.gov.pl/1.4"
N_SAMPLE = 300
PAUSE = 0.5
LOC_COL = re.compile(r"miejscowo|gmin|powiat|wojew|ulic|adres|lokalizac|miasto|kod poczt", re.I)
HDRS = {"User-Agent": "ceny-mieszkan research (non-commercial)"}


def list_datasets():
    ids, page = [], 1
    while True:
        r = requests.get(f"{API}/datasets", params={"title[phrase]": "Ceny ofertowe mieszkań dewelopera",
                                                     "per_page": 100, "page": page}, headers=HDRS, timeout=120)
        d = r.json().get("data", [])
        if not d:
            break
        ids += [(x["id"], x["attributes"]["title"]) for x in d]
        page += 1
        time.sleep(PAUSE)
        if page % 20 == 0:
            print(f"  listed {len(ids):,}", flush=True)
    return ids


def read_any(content, fmt):
    if fmt in ("xlsx", "xls"):
        return pd.read_excel(io.BytesIO(content), header=None)
    for enc in ("utf-8", "cp1250"):
        try:
            return pd.read_csv(io.BytesIO(content), header=None, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return None


def main():
    os.makedirs(OUT, exist_ok=True)
    ids = list_datasets()
    print(f"{len(ids):,} developer datasets", flush=True)
    random.seed(0)
    rows = []
    for did, title in random.sample(ids, min(N_SAMPLE, len(ids))):
        r = requests.get(f"{API}/datasets/{did}/resources", headers=HDRS, timeout=120).json()
        time.sleep(PAUSE)
        res = sorted(r.get("data", []), key=lambda x: x["attributes"].get("modified", ""), reverse=True)
        if not res:
            rows.append({"id": did, "title": title, "status": "no resource"})
            continue
        a = res[0]["attributes"]
        url, fmt = a.get("file_url") or a.get("link"), (a.get("format") or "").lower()
        try:
            c = requests.get(url, headers=HDRS, timeout=120).content
            time.sleep(PAUSE)
            df = read_any(c, fmt)
        except Exception as e:
            rows.append({"id": did, "title": title, "status": f"error {type(e).__name__}", "url": url})
            continue
        if df is None:
            rows.append({"id": did, "title": title, "status": "unreadable", "url": url, "format": fmt})
            continue
        head = " | ".join(str(v) for v in df.head(8).astype(str).to_numpy().ravel())
        text = " ".join(df.astype(str).to_numpy().ravel()[:5000])
        open(os.path.join(OUT, f"{did}.{fmt or 'bin'}"), "wb").write(c)
        rows.append({"id": did, "title": title, "status": "ok", "format": fmt, "rows": len(df), "cols": df.shape[1],
                     "has_loc_col": bool(LOC_COL.search(head)), "mentions_warszawa": "warszaw" in text.lower(),
                     "header": head[:400], "url": url})
    rep = pd.DataFrame(rows)
    rep.to_csv(os.path.join(HERE, "data", "sample_report.csv"), index=False, encoding="utf-8")
    ok = rep[rep["status"] == "ok"]
    print(rep["status"].value_counts().to_string())
    print(f"readable {len(ok)}: with location columns {ok['has_loc_col'].mean():.0%}, "
          f"mention Warszawa {ok['mentions_warszawa'].mean():.0%}, median {ok['rows'].median():.0f} rows, "
          f"formats {ok['format'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
