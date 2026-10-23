#!/usr/bin/env python
"""BTS Reporting Carrier On-Time Performance, filtered to flights touching Chicago/NYC airports.

Source: https://www.transtats.bts.gov/  (monthly PREZIP archives; ~2-3 month publication lag)
Output: data/external/bts/flights_YYYY-MM.parquet
Scheduled/actual times are hhmm in local time at the respective airport (2400 = midnight).
Existing months are skipped (BTS publishes whole months); unpublished months are skipped.
"""
import sys
import tempfile
import time
import zipfile
from datetime import date
from pathlib import Path
from http.client import HTTPException
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "bts"
URL = "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{y}_{m}.zip"
START_YEAR = 2018
AIRPORTS = ("ORD", "MDW", "JFK", "LGA", "EWR")

COLS = {
    "FlightDate": "DATE", "Reporting_Airline": "VARCHAR",
    "Flight_Number_Reporting_Airline": "VARCHAR", "Tail_Number": "VARCHAR",
    "Origin": "VARCHAR", "Dest": "VARCHAR",
    "CRSDepTime": "SMALLINT", "DepTime": "SMALLINT", "DepDelay": "DOUBLE",
    "CRSArrTime": "SMALLINT", "ArrTime": "SMALLINT", "ArrDelay": "DOUBLE",
    "Cancelled": "FLAG", "CancellationCode": "VARCHAR", "Diverted": "FLAG",
    "CarrierDelay": "DOUBLE", "WeatherDelay": "DOUBLE", "NASDelay": "DOUBLE",
    "SecurityDelay": "DOUBLE", "LateAircraftDelay": "DOUBLE",
}


def get(url, tries=5, timeout=600):
    """Return the body, or None if the file isn't published (404)."""
    for attempt in range(1, tries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": "rideshare-capstone"}), timeout=timeout) as r:
                return r.read()
        except HTTPError as e:
            if e.code == 404:
                return None
            err = e
        except (OSError, HTTPException) as e:  # incl. timeouts, resets, IncompleteRead
            err = e
        print(f"  retry {attempt}/{tries}: {err}", file=sys.stderr, flush=True)
        time.sleep(15 * attempt)
    raise RuntimeError(f"giving up on {url}")


def cast(col, typ):
    if typ == "FLAG":  # served as "0.00" / "1.00"
        return f"TRY_CAST(\"{col}\" AS DOUBLE) = 1 AS {col}"
    return f"TRY_CAST(NULLIF(TRIM(\"{col}\"), '') AS {typ}) AS {col}"


def fetch_month(year, month, dest):
    body = get(URL.format(y=year, m=month))
    with tempfile.TemporaryDirectory() as tmp:
        zpath = Path(tmp) / "month.zip"
        zpath.write_bytes(body or b"")
        if not zipfile.is_zipfile(zpath):
            return False
        with zipfile.ZipFile(zpath) as z:
            name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
            csv = z.extract(name, tmp)
        airports = ", ".join(f"'{a}'" for a in AIRPORTS)
        part = dest.with_name(dest.name + ".part")
        duckdb.execute(f"""
            COPY (
                SELECT {", ".join(cast(c, t) for c, t in COLS.items())}
                FROM read_csv('{csv}', header = true, all_varchar = true)
                WHERE Origin IN ({airports}) OR Dest IN ({airports})
            ) TO '{part}' (FORMAT parquet, COMPRESSION zstd)""")
        part.replace(dest)
    return True


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    today = date.today()
    for year in range(START_YEAR, today.year + 1):
        for month in range(1, 13):
            if date(year, month, 1) > today:
                break
            dest = OUT / f"flights_{year}-{month:02d}.parquet"
            if dest.exists():
                continue
            if fetch_month(year, month, dest):
                n, = duckdb.execute(f"SELECT count(*) FROM '{dest}'").fetchone()
                print(f"{year}-{month:02d}: {n:,} flights", flush=True)
            else:
                print(f"{year}-{month:02d}: not published", flush=True)


if __name__ == "__main__":
    main()
