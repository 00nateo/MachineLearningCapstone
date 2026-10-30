#!/usr/bin/env python
"""Convert raw trip downloads into typed, zstd-compressed monthly parquet.

  Chicago  <RAW_DIR>/Chicago-Ride-Data/<year>/tnp_trips_YYYY-MM-DD.csv   (one CSV per day)
        -> data/processed/chicago_trips/<year>/tnp_trips_YYYY-MM.parquet
  NYC      <RAW_DIR>/NYC-Ride-Data/fhvhv/fhvhv_tripdata_YYYY-MM.parquet
        -> data/processed/nyc_fhvhv/fhvhv_YYYY-MM.parquet

Columns are cast to one fixed schema per city; columns absent from a month's source files
become typed NULLs and are listed in that month's log line. Timestamps stay naive local
wall-clock time, as published. Raw files are only read.
A month is rebuilt only if its output is missing or older than one of its inputs, so the
partial current month is picked up again as new days arrive.

Usage: python scripts/convert.py [chicago|nyc ...]     (default: both)
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("RAW_DIR", f"/scratch/{os.environ['USER']}/raw"))
OUT = ROOT / "data" / "processed"

CHICAGO_COLS = {
    "trip_id": "VARCHAR",
    "trip_start_timestamp": "TIMESTAMP",  # rounded to 15 min by the city
    "trip_end_timestamp": "TIMESTAMP",
    "trip_seconds": "INTEGER",
    "trip_miles": "DOUBLE",
    "percent_time_chicago": "DOUBLE",  # 2024+
    "percent_distance_chicago": "DOUBLE",  # 2024+
    "pickup_census_tract": "VARCHAR",  # 11-digit GEOID, often privacy-masked
    "dropoff_census_tract": "VARCHAR",
    "pickup_community_area": "SMALLINT",  # NULL = outside city or masked
    "dropoff_community_area": "SMALLINT",
    "fare": "DOUBLE",
    "tip": "DOUBLE",
    "additional_charges": "DOUBLE",
    "trip_total": "DOUBLE",
    "shared_trip_authorized": "BOOLEAN",
    "shared_trip_match": "BOOLEAN",  # 2024+
    "trips_pooled": "INTEGER",
    "pickup_centroid_latitude": "DOUBLE",
    "pickup_centroid_longitude": "DOUBLE",
    "dropoff_centroid_latitude": "DOUBLE",
    "dropoff_centroid_longitude": "DOUBLE",
}

NYC_COLS = {
    "hvfhs_license_num": "VARCHAR",
    "dispatching_base_num": "VARCHAR",
    "originating_base_num": "VARCHAR",
    "request_datetime": "TIMESTAMP",
    "on_scene_datetime": "TIMESTAMP",
    "pickup_datetime": "TIMESTAMP",
    "dropoff_datetime": "TIMESTAMP",
    "PULocationID": "SMALLINT",
    "DOLocationID": "SMALLINT",
    "trip_miles": "DOUBLE",
    "trip_time": "INTEGER",
    "base_passenger_fare": "DOUBLE",
    "tolls": "DOUBLE",
    "bcf": "DOUBLE",
    "sales_tax": "DOUBLE",
    "congestion_surcharge": "DOUBLE",
    "airport_fee": "DOUBLE",
    "cbd_congestion_fee": "DOUBLE",  # 2025+
    "tips": "DOUBLE",
    "driver_pay": "DOUBLE",
    "shared_request_flag": "VARCHAR",
    "shared_match_flag": "VARCHAR",
    "access_a_ride_flag": "VARCHAR",
    "wav_request_flag": "VARCHAR",
    "wav_match_flag": "VARCHAR",
}

CSV_CAST = "TRY_CAST(NULLIF(TRIM(\"{c}\"), '') AS {t})"
PARQUET_CAST = "CAST(\"{c}\" AS {t})"


def connect():
    con = duckdb.connect()
    con.execute(f"SET threads = {int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))}")
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 16 * 1024))
    con.execute(f"SET memory_limit = '{int(mem_mb * 0.75)}MB'")
    con.execute("SET preserve_insertion_order = false")
    if os.environ.get("TMPDIR"):
        con.execute(f"SET temp_directory = '{os.environ['TMPDIR']}/duckdb'")
    return con


def is_stale(out, inputs):
    return not out.exists() or out.stat().st_mtime < max(p.stat().st_mtime for p in inputs)


def select_list(con, src, cols, cast):
    """Cast each wanted column; columns absent from the source become typed NULLs.

    Returns (select list SQL, names of the NULL-filled columns).
    """
    present = {r[0].lower(): r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()}
    exprs, missing = [], []
    for name, typ in cols.items():
        src_name = present.get(name.lower())
        if src_name:
            exprs.append(f'{cast.format(c=src_name, t=typ)} AS "{name}"')
        else:
            exprs.append(f'NULL::{typ} AS "{name}"')
            missing.append(name)
    return ", ".join(exprs), missing


def write_parquet(con, query, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_name(out.name + ".part")
    con.execute(f"COPY ({query}) TO '{part}' (FORMAT parquet, COMPRESSION zstd)")
    part.replace(out)


def chicago_jobs():
    days = defaultdict(list)
    for f in sorted((RAW / "Chicago-Ride-Data").glob("*/tnp_trips_*.csv")):
        if m := re.fullmatch(r"tnp_trips_(\d{4}-\d{2})-\d{2}\.csv", f.name):
            days[m.group(1)].append(f)
    for month, files in sorted(days.items()):
        out = OUT / "chicago_trips" / month[:4] / f"tnp_trips_{month}.parquet"
        if is_stale(out, files):
            file_list = ", ".join(f"'{f}'" for f in files)
            src = f"read_csv([{file_list}], header = true, all_varchar = true, union_by_name = true)"
            yield month, out, src, f" from {len(files)} day files"


def nyc_jobs():
    for f in sorted((RAW / "NYC-Ride-Data" / "fhvhv").glob("fhvhv_tripdata_*.parquet")):
        month = f.stem.removeprefix("fhvhv_tripdata_")
        out = OUT / "nyc_fhvhv" / f"fhvhv_{month}.parquet"
        if is_stale(out, [f]):
            yield month, out, f"read_parquet('{f}')", ""


CITIES = {
    "chicago": dict(jobs=chicago_jobs, cols=CHICAGO_COLS, cast=CSV_CAST, ts="trip_start_timestamp"),
    "nyc": dict(jobs=nyc_jobs, cols=NYC_COLS, cast=PARQUET_CAST, ts="pickup_datetime"),
}


def main():
    con = connect()
    failed = []
    for city in sys.argv[1:] or list(CITIES):
        c = CITIES[city]
        for month, out, src, note in c["jobs"]():
            try:
                select, missing = select_list(con, src, c["cols"], c["cast"])
                write_parquet(con, f"SELECT {select} FROM {src}", out)
                n, bad = con.execute(
                    f"SELECT count(*), count(*) FILTER ({c['ts']} IS NULL) FROM '{out}'").fetchone()
                nulls = f"  NULL-filled: {', '.join(missing)}" if missing else ""
                warn = f"  WARNING: {bad:,} unparseable {c['ts']}" if bad else ""
                print(f"{city} {month}: {n:,} trips{note}{nulls}{warn}", flush=True)
            except duckdb.Error as e:
                print(f"{city} {month}: FAILED: {e}", file=sys.stderr, flush=True)
                failed.append(f"{city} {month}")
    if failed:
        sys.exit(f"{len(failed)} month(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
