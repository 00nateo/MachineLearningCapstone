#!/usr/bin/env python
"""ASOS surface observations (Iowa Environmental Mesonet) for Chicago and NYC stations.

Source: https://mesonet.agron.iastate.edu/request/download.phtml
Raw:    data/external/weather/raw/<station>_<year>.csv   (UTC timestamps)
Output: data/external/weather/asos.parquet
        city, station, valid_utc, valid_local, <VARS>
Fetched in UTC and converted, so fall-back hour observations stay distinct via valid_utc.
Includes routine hourly and special METARs: specials catch sudden changes (storm onset),
but their p01i is accumulated since the last routine report. Trace precip = 0.0001.
Past years are skipped if already downloaded; the current year is always refreshed.
"""
import sys
import time
from datetime import date
from pathlib import Path
from http.client import HTTPException
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "weather"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
START_YEAR = 2018

STATIONS = {  # station -> (city, tz)
    "ORD": ("chicago", "America/Chicago"),
    "MDW": ("chicago", "America/Chicago"),
    "LGA": ("nyc", "America/New_York"),
    "JFK": ("nyc", "America/New_York"),
    "NYC": ("nyc", "America/New_York"),  # Central Park
}
VARS = {
    "tmpf": "DOUBLE", "dwpf": "DOUBLE", "relh": "DOUBLE", "feel": "DOUBLE",
    "sknt": "DOUBLE", "gust": "DOUBLE", "vsby": "DOUBLE", "p01i": "DOUBLE",
    "ice_accretion_1hr": "DOUBLE", "snowdepth": "DOUBLE",
    "skyc1": "VARCHAR", "skyl1": "DOUBLE", "wxcodes": "VARCHAR",
}


def get(url, params, tries=5, timeout=600):
    url = f"{url}?{urlencode(params, doseq=True)}"
    for attempt in range(1, tries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": "rideshare-capstone"}), timeout=timeout) as r:
                return r.read()
        except (OSError, HTTPException) as e:  # incl. timeouts, resets, IncompleteRead
            print(f"  retry {attempt}/{tries}: {e}", file=sys.stderr, flush=True)
            time.sleep(15 * attempt)
    raise RuntimeError(f"giving up on {url}")


def fetch(station, year, dest):
    body = get(URL, {
        "station": station, "data": list(VARS),
        "year1": year, "month1": 1, "day1": 1, "year2": year + 1, "month2": 1, "day2": 1,
        "tz": "Etc/UTC", "format": "onlycomma", "latlon": "no", "elev": "no",
        "missing": "empty", "trace": "0.0001", "direct": "no", "report_type": [3, 4],
    })
    if not body.startswith(b"station,"):
        raise RuntimeError(f"unexpected response for {station} {year}: {body[:200]!r}")
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.replace(dest)


def main():
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    this_year = date.today().year
    for station in STATIONS:
        for year in range(START_YEAR, this_year + 1):
            dest = raw / f"{station}_{year}.csv"
            if dest.exists() and year < this_year:
                continue
            fetch(station, year, dest)
            print(f"{station} {year}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)
            time.sleep(2)  # be polite to IEM

    stations = ", ".join(f"('{s}', '{c}', '{tz}')" for s, (c, tz) in STATIONS.items())
    cols = ", ".join(f"TRY_CAST(NULLIF(\"{v}\", '') AS {t}) AS {v}" for v, t in VARS.items())
    out = OUT / "asos.parquet"
    part = out.with_name(out.name + ".part")
    duckdb.execute(f"""
        COPY (
            SELECT DISTINCT m.city, r.station,
                   CAST(r.valid AS TIMESTAMP) AS valid_utc,
                   timezone(m.tz, timezone('UTC', CAST(r.valid AS TIMESTAMP))) AS valid_local,
                   {cols}
            FROM read_csv('{raw}/*.csv', header = true, all_varchar = true, union_by_name = true) r
            JOIN (VALUES {stations}) m(station, city, tz) USING (station)
            ORDER BY r.station, valid_utc
        ) TO '{part}' (FORMAT parquet, COMPRESSION zstd)""")
    part.replace(out)
    n, = duckdb.execute(f"SELECT count(*) FROM '{out}'").fetchone()
    print(f"wrote {out} ({n:,} observations)")


if __name__ == "__main__":
    main()
