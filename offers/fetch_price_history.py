"""
Monthly price history of the Warsaw developer price lists: for every dataset with Warsaw flats
(flats_warszawa.csv), list all its resources (daily snapshots since the 2025 law, ~1/3 of datasets keep them) and
download the last snapshot of each month (by data_date) -> offers/data/history/<dataset>/<YYYY-MM>.<fmt>, plus
offers/data/history_index.csv. Parallel, resumable (existing files are skipped).

    python offers/fetch_price_history.py
"""
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from fetch_price_lists import WORKERS, session
from sample_price_lists import API

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "data", "history")
INDEX = os.path.join(HERE, "data", "history_index.csv")


def monthly(did):
    """Last snapshot of each month for one dataset."""
    rows, page = [], 1
    while True:
        r = {}
        for attempt in range(4):                          # the API sometimes stalls on long resource lists
            try:
                r = session().get(f"{API}/datasets/{did}/resources", params={"per_page": 100, "page": page},
                                  timeout=60).json()
                break
            except Exception:
                time.sleep(5 * (attempt + 1))
        d = r.get("data", [])
        rows += [(x["id"], x["attributes"].get("data_date") or x["attributes"].get("created", "")[:10],
                  (x["attributes"].get("format") or "bin").lower(), x["attributes"].get("file_url") or "")
                 for x in d]
        if len(d) < 100:
            break
        page += 1
    df = pd.DataFrame(rows, columns=["rid", "date", "fmt", "url"])
    df = df[df["date"].astype(str).str.match(r"\d{4}-\d{2}-\d{2}")]
    df["month"] = df["date"].str[:7]
    return df.sort_values("date").groupby("month").tail(1).assign(did=did)


def download(row):
    out = os.path.join(HIST, str(row.did), f"{row.month}.{row.fmt}")
    if os.path.exists(out):
        return out, "cached"
    try:
        c = session().get(row.url, timeout=180).content
        os.makedirs(os.path.dirname(out), exist_ok=True)
        open(out, "wb").write(c)
        return out, "ok"
    except Exception as e:
        return out, f"error {type(e).__name__}"


def main():
    ids = pd.read_csv(os.path.join(HERE, "data", "flats_warszawa.csv"), low_memory=False,
                      usecols=["dataset_id"])["dataset_id"].astype(str).unique()
    print(f"{len(ids)} Warsaw datasets: listing snapshots ...", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        lists = [f.result() for f in as_completed([ex.submit(monthly, d) for d in ids])]
    snaps = pd.concat(lists, ignore_index=True)
    snaps.to_csv(INDEX, index=False)
    per = snaps.groupby("did").size()
    print(f"{len(snaps):,} monthly snapshots; datasets with >= 3 months: {(per >= 3).sum()} of {len(per)}; "
          f"months {snaps['month'].min()} .. {snaps['month'].max()}", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        res = [f.result() for f in as_completed([ex.submit(download, r) for r in snaps.itertuples()])]
    print(pd.Series([s for _, s in res]).value_counts().to_string())


if __name__ == "__main__":
    main()
