"""
Download the newest file of every developer price-list dataset on dane.gov.pl ("Ceny ofertowe mieszkań
dewelopera ...", 2025 price-transparency law) -> offers/data/raw/<dataset id>.<format>, plus
offers/data/index.csv (dataset id, resource id, modified, format, status, url, title).

Every dataset is tagged region "Polska", so the investment's location is only inside the file: all ~13k are
fetched. WORKERS parallel connections (each dataset = 2 requests: resource list, then the file); the first
version did one request at a time with a 0.5 s pause, ~9 h in total, almost all of it waiting.
Resumable: datasets already in index.csv are skipped; errors are retried with --retry-errors.

    python offers/fetch_price_lists.py [--retry-errors]
"""
import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from sample_price_lists import API, HDRS, list_datasets

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
INDEX = os.path.join(HERE, "data", "index.csv")
WORKERS = 8
_lock = threading.Lock()
_local = threading.local()


def session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HDRS)
    return _local.s


def fetch(did, title):
    status, rid, mod, fmt, url = "no resource", "", "", "", ""
    try:
        r = session().get(f"{API}/datasets/{did}/resources", timeout=120).json()
        res = sorted(r.get("data", []), key=lambda x: x["attributes"].get("modified", ""), reverse=True)
        if res:
            a = res[0]["attributes"]
            rid, mod = res[0]["id"], a.get("modified", "")[:19]
            fmt, url = (a.get("format") or "bin").lower(), a.get("file_url") or a.get("link") or ""
            c = session().get(url, timeout=180).content
            open(os.path.join(RAW, f"{did}.{fmt}"), "wb").write(c)
            status = "ok"
    except Exception as e:
        status = f"error {type(e).__name__}"
    return did, rid, mod, fmt, status, url, title


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retry-errors", action="store_true")
    args = ap.parse_args()
    os.makedirs(RAW, exist_ok=True)
    ids = list_datasets()
    idx = pd.read_csv(INDEX) if os.path.exists(INDEX) else pd.DataFrame(columns=["id", "status"])
    if args.retry_errors and len(idx):
        idx = idx[~idx["status"].astype(str).str.startswith("error")]
        idx.to_csv(INDEX, index=False)
    done = set(idx["id"].astype(str))
    todo = [(d, t) for d, t in ids if str(d) not in done]
    print(f"{len(ids):,} datasets, {len(done):,} done, {len(todo):,} to fetch with {WORKERS} workers", flush=True)
    new = not os.path.exists(INDEX) or not len(idx)
    with open(INDEX, "a" if not new else "w", encoding="utf-8") as out:
        if new:
            out.write("id,resource_id,modified,format,status,url,title\n")
        with ThreadPoolExecutor(WORKERS) as ex:
            futs = [ex.submit(fetch, d, t) for d, t in todo]
            for n, f in enumerate(as_completed(futs), 1):
                did, rid, mod, fmt, status, url, title = f.result()
                with _lock:
                    out.write(f'{did},{rid},{mod},{fmt},{status},{url},"{title.replace(chr(34), chr(39))}"\n')
                    out.flush()
                if n % 500 == 0:
                    print(f"  {n:,}/{len(todo):,}", flush=True)
    print(pd.read_csv(INDEX)["status"].value_counts().to_string())


if __name__ == "__main__":
    main()
