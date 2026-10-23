#!/usr/bin/env python
"""CTA 'L' ridership and station locations (Chicago Data Portal / Socrata).

Sources: 5neh-572f  CTA - Ridership - 'L' Station Entries - Daily Totals (~2 month lag)
         8pix-ypme  CTA - System Information - List of 'L' Stops
Output:  data/external/cta/l_station_day.parquet   station_id, stationname, date, daytype, rides
         data/external/cta/l_stations.parquet      station_id, station_name, latitude, longitude
CTA publishes no hourly ridership: features built from this must use the previous day.
Both tables are small and re-fetched in full on every run. Uses SOCRATA_APP_TOKEN if set.
"""
import json
import os
import sys
import time
from pathlib import Path
from http.client import HTTPException
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "cta"
BASE = "https://data.cityofchicago.org/resource"
START = "2018-01-01"
PAGE = 50_000


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


def fetch_all(dataset, params):
    rows, offset = [], 0
    while True:
        page = json.loads(get(f"{BASE}/{dataset}.json", {**params, "$limit": PAGE, "$offset": offset}))
        rows += page
        if len(page) < PAGE:
            return pd.DataFrame(rows)
        offset += PAGE


def write(df, name):
    out = OUT / name
    part = out.with_name(out.name + ".part")
    df.to_parquet(part, compression="zstd", index=False)
    part.replace(out)
    print(f"wrote {out} ({len(df):,} rows)", flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    raw = fetch_all("5neh-572f", {"$where": f"date >= '{START}T00:00:00'", "$order": "date, station_id"})
    days = pd.DataFrame({
        "station_id": raw["station_id"].astype(int),
        "stationname": raw["stationname"],
        "date": pd.to_datetime(raw["date"]).dt.date,
        "daytype": raw["daytype"],
        "rides": pd.to_numeric(raw["rides"]).astype(int),
    }).drop_duplicates()
    dups = days.duplicated(["station_id", "date"]).sum()
    if dups:
        print(f"WARNING: {dups} conflicting station-day rows, keeping the last", flush=True)
        days = days.drop_duplicates(["station_id", "date"], keep="last")
    write(days, "l_station_day.parquet")

    stops = fetch_all("8pix-ypme", {"$order": "stop_id"})
    stations = pd.DataFrame({
        "station_id": stops["map_id"].astype(int),
        "station_name": stops["station_name"],
        "latitude": stops["location"].map(lambda l: float(l["latitude"])),
        "longitude": stops["location"].map(lambda l: float(l["longitude"])),
    }).groupby("station_id", as_index=False).agg(
        station_name=("station_name", "first"), latitude=("latitude", "mean"), longitude=("longitude", "mean"))
    write(stations, "l_stations.parquet")


if __name__ == "__main__":
    main()
