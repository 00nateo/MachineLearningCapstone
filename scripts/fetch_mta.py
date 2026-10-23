#!/usr/bin/env python
"""MTA subway hourly ridership by station complex, and MTA service alerts (NY Open Data / Socrata).

Sources: wujg-7c2s  MTA Subway Hourly Ridership: 2020-2024 (starts 2020-07)
         5wq4-mkjj  MTA Subway Hourly Ridership: Beginning 2025
         3h5b-5ktz  MTA Service Alerts: 2012 - 2020 (ends 2020-03-31)
         7kct-peq7  MTA Service Alerts: Beginning April 2020 (starts 2020-04-28)
Output:  data/external/mta/subway_hourly_YYYY-MM.parquet
         transit_timestamp (local hour), transit_mode, station_complex_id, station_complex,
         borough, latitude, longitude, ridership, transfers (summed over payment/fare class)
         data/external/mta/alerts_2012_2020.parquet, alerts_2020_present.parquet
         one row per alert update (post time only, no end time), local time, from 2018
No hourly ridership exists before 2020-07. Uses SOCRATA_APP_TOKEN if set (run_pipeline.sh exports it).
Ridership months already downloaded are skipped unless they are within the last ~2 months.
The alerts archive is fetched once; current alerts are re-fetched on every run.
"""
import json
import os
import sys
import tempfile
import time
from datetime import date, timedelta
from pathlib import Path
from http.client import HTTPException
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "mta"
DATASETS = [  # (id, first month, first month NOT covered)
    ("wujg-7c2s", date(2020, 7, 1), date(2025, 1, 1)),
    ("5wq4-mkjj", date(2025, 1, 1), None),
]
GROUP = "transit_timestamp, transit_mode, station_complex_id, station_complex, borough, latitude, longitude"
PAGE = 100_000
ALERTS = [  # (dataset, output, columns, refresh every run)
    ("3h5b-5ktz", "alerts_2012_2020.parquet", "status_id, date, agency, status_label, affected, header", False),
    ("7kct-peq7", "alerts_2020_present.parquet",
     "alert_id, event_id, update_number, date, agency, status_label, affected, header", True),
]
ALERTS_START = "2018-01-01"


def get(url, params, tries=5, timeout=600):
    url = f"{url}?{urlencode(params, doseq=True)}"
    headers = {"User-Agent": "rideshare-capstone"}
    if os.environ.get("SOCRATA_APP_TOKEN"):
        headers["X-App-Token"] = os.environ["SOCRATA_APP_TOKEN"]
    for attempt in range(1, tries + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as r:
                return r.read()
        except (OSError, HTTPException) as e:  # incl. timeouts, resets, IncompleteRead
            print(f"  retry {attempt}/{tries}: {e}", file=sys.stderr, flush=True)
            time.sleep(15 * attempt)
    raise RuntimeError(f"giving up on {url}")


def next_month(d):
    return (d.replace(day=28) + timedelta(days=4)).replace(day=1)


def fetch_month(dataset, start, dest):
    header, rows, offset = None, [], 0
    while True:
        lines = get(f"https://data.ny.gov/resource/{dataset}.csv", {
            "$select": f"{GROUP}, sum(ridership) AS ridership, sum(transfers) AS transfers",
            "$where": f"transit_timestamp >= '{start}T00:00:00' "
                      f"AND transit_timestamp < '{next_month(start)}T00:00:00'",
            "$group": GROUP, "$order": GROUP, "$limit": PAGE, "$offset": offset,
        }).decode().splitlines()
        header, page = lines[0], lines[1:]
        rows += page
        if len(page) < PAGE:
            break
        offset += PAGE
    if not rows:
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        csv = Path(tmp) / "month.csv"
        csv.write_text("\n".join([header, *rows]) + "\n")
        part = dest.with_name(dest.name + ".part")
        duckdb.execute(f"""
            COPY (
                SELECT CAST(transit_timestamp AS TIMESTAMP) AS transit_timestamp,
                       transit_mode, station_complex_id, station_complex, borough,
                       TRY_CAST(latitude AS DOUBLE) AS latitude,
                       TRY_CAST(longitude AS DOUBLE) AS longitude,
                       TRY_CAST(ridership AS DOUBLE) AS ridership,
                       TRY_CAST(transfers AS DOUBLE) AS transfers
                FROM read_csv('{csv}', header = true, all_varchar = true)
            ) TO '{part}' (FORMAT parquet, COMPRESSION zstd)""")
        part.replace(dest)
    return len(rows)


def fetch_json(dataset, params):
    rows, offset = [], 0
    while True:
        page = json.loads(get(f"https://data.ny.gov/resource/{dataset}.json",
                              {**params, "$limit": PAGE, "$offset": offset}))
        rows += page
        if len(page) < PAGE:
            return pd.DataFrame(rows)
        offset += PAGE


def fetch_alerts():
    for dataset, name, select, refresh in ALERTS:
        dest = OUT / name
        if dest.exists() and not refresh:
            continue
        cols = [c.strip() for c in select.split(",")]
        df = fetch_json(dataset, {"$select": select, "$where": f"date >= '{ALERTS_START}T00:00:00'",
                                  "$order": f"date, {cols[0]}"}).reindex(columns=cols)
        df["date"] = pd.to_datetime(df["date"])
        part = dest.with_name(dest.name + ".part")
        df.to_parquet(part, compression="zstd", index=False)
        part.replace(dest)
        print(f"alerts {dataset}: {len(df):,} rows", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today()
    refresh_after = today.replace(day=1) - timedelta(days=62)
    for dataset, first, stop in DATASETS:
        month = first
        while month <= today and (stop is None or month < stop):
            dest = OUT / f"subway_hourly_{month:%Y-%m}.parquet"
            if not (dest.exists() and month < refresh_after):
                n = fetch_month(dataset, month, dest)
                print(f"{month:%Y-%m} ({dataset}): {n:,} rows", flush=True)
            month = next_month(month)
    fetch_alerts()


if __name__ == "__main__":
    main()
