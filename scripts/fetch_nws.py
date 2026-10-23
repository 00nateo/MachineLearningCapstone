#!/usr/bin/env python
"""NWS watches/warnings/advisories (VTEC) issued by the Chicago (LOT) and New York (OKX) offices.

Source: IEM archive, https://mesonet.agron.iastate.edu/request/gis/watchwarn.phtml
Raw:    data/external/nws/raw/warnings_<year>.csv
Output: data/external/nws/warnings.parquet — all rows from both offices plus a city column,
        with the utc_* columns parsed to TIMESTAMP (all UTC; convert to local before joining
        to trip hours).
Rows with gtype = 'C' are per county/zone and carry a UGC code (e.g. NYZ074, ILZ014);
gtype = 'P' rows are storm-based polygons with an empty ugc. Filtering to the UGCs that
cover each city happens at feature-building time.
Past years are skipped if already downloaded; the current year is always refreshed.
"""
import sys
import time
from datetime import date
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import duckdb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external" / "nws"
URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/watchwarn.py"
START_YEAR = 2018
WFOS = {"LOT": "chicago", "OKX": "nyc"}
TIMES = {  # column -> format as served
    "utc_issue": "%Y-%m-%d %H:%M",
    "utc_expire": "%Y-%m-%d %H:%M",
    "utc_prodissue": "%Y-%m-%d %H:%M",
    "utc_init_expire": "%Y%m%d%H%M",
    "utc_updated": "%Y-%m-%d %H:%M",
    "utc_polygon_begin": "%Y-%m-%d %H:%M",
    "utc_polygon_end": "%Y-%m-%d %H:%M",
}


def get(url, params, tries=5, timeout=600):
    url = f"{url}?{urlencode(params, doseq=True)}"
    for attempt in range(1, tries + 1):
        try:
            with urlopen(Request(url, headers={"User-Agent": "rideshare-capstone"}), timeout=timeout) as r:
                return r.read()
        except (HTTPError, URLError, TimeoutError) as e:
            print(f"  retry {attempt}/{tries}: {e}", file=sys.stderr, flush=True)
            time.sleep(15 * attempt)
    raise RuntimeError(f"giving up on {url}")


def fetch(year, dest):
    body = get(URL, {
        "accept": "csv", "timeopt": 1, "wfo[]": list(WFOS),
        "year1": year, "month1": 1, "day1": 1, "hour1": 0, "minute1": 0,
        "year2": year + 1, "month2": 1, "day2": 1, "hour2": 0, "minute2": 0,
    })
    if not body.startswith(b"wfo,"):
        raise RuntimeError(f"unexpected response for {year}: {body[:200]!r}")
    part = dest.with_name(dest.name + ".part")
    part.write_bytes(body)
    part.replace(dest)


def main():
    raw = OUT / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    this_year = date.today().year
    for year in range(START_YEAR, this_year + 1):
        dest = raw / f"warnings_{year}.csv"
        if dest.exists() and year < this_year:
            continue
        fetch(year, dest)
        print(f"{year}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)
        time.sleep(2)

    city = " ".join(f"WHEN '{w}' THEN '{c}'" for w, c in WFOS.items())
    parsed = ", ".join(f"try_strptime(NULLIF({col}, ''), '{fmt}') AS {col}" for col, fmt in TIMES.items())
    out = OUT / "warnings.parquet"
    part = out.with_name(out.name + ".part")
    duckdb.execute(f"""
        COPY (
            SELECT DISTINCT CASE wfo {city} END AS city, * REPLACE ({parsed})
            FROM read_csv('{raw}/*.csv', header = true, all_varchar = true, union_by_name = true)
            ORDER BY utc_issue
        ) TO '{part}' (FORMAT parquet, COMPRESSION zstd)""")
    part.replace(out)
    n, = duckdb.execute(f"SELECT count(*) FROM '{out}'").fetchone()
    print(f"wrote {out} ({n:,} rows)")


if __name__ == "__main__":
    main()
