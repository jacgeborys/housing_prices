"""
Download the newest file of every developer price-list dataset on dane.gov.pl ("Ceny ofertowe mieszkań
dewelopera ...", 2025 price-transparency law) -> offers/data/raw/<dataset id>.<format>, plus
offers/data/index.csv (dataset id, title, resource id, modified, format, url, status).

Resumable: datasets already downloaded are skipped. Polite: one request at a time, PAUSE seconds apart.
Older resources (the daily price history) are not fetched here.

    python offers/fetch_price_lists.py
"""
import os
import time

import pandas as pd
import requests

from sample_price_lists import API, HDRS, list_datasets

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
INDEX = os.path.join(HERE, "data", "index.csv")
PAUSE = 0.5


def main():
    os.makedirs(RAW, exist_ok=True)
    ids = list_datasets()
    done = set(pd.read_csv(INDEX)["id"]) if os.path.exists(INDEX) else set()
    print(f"{len(ids):,} datasets, {len(done):,} already indexed", flush=True)
    out = open(INDEX, "a", encoding="utf-8")
    if not done:
        out.write("id,resource_id,modified,format,status,url,title\n")
    n = 0
    for did, title in ids:
        if did in done:
            continue
        status, rid, mod, fmt, url = "no resource", "", "", "", ""
        try:
            r = requests.get(f"{API}/datasets/{did}/resources", headers=HDRS, timeout=120).json()
            time.sleep(PAUSE)
            res = sorted(r.get("data", []), key=lambda x: x["attributes"].get("modified", ""), reverse=True)
            if res:
                a = res[0]["attributes"]
                rid, mod = res[0]["id"], a.get("modified", "")[:19]
                fmt, url = (a.get("format") or "bin").lower(), a.get("file_url") or a.get("link") or ""
                c = requests.get(url, headers=HDRS, timeout=180).content
                time.sleep(PAUSE)
                open(os.path.join(RAW, f"{did}.{fmt}"), "wb").write(c)
                status = "ok"
        except Exception as e:
            status = f"error {type(e).__name__}"
        safe_title = '"' + title.replace('"', "'") + '"'
        out.write(f"{did},{rid},{mod},{fmt},{status},{url},{safe_title}\n")
        out.flush()
        n += 1
        if n % 500 == 0:
            print(f"  {n:,} new datasets fetched", flush=True)
    out.close()
    idx = pd.read_csv(INDEX)
    print(idx["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
