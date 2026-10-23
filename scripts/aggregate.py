#!/usr/bin/env python
"""Aggregate processed trips into zone-hour pickup panels and OD drive-time tables.

Outputs in data/processed/:
  {city}_zone_hour.parquet    zone, hour, pickups — zero-filled over every zone x hour
  {city}_city_hour.parquet    hour, dst_merged_hour, pickups_total, pickups_unzoned
  {city}_drive_times.parquet  year, pu_zone, do_zone, hour_of_day, is_weekend,
                              n_trips, median_trip_seconds

Zones: Chicago community areas 1-77, NYC taxi zones 1-263. Pickups with no valid zone
(Chicago NULL = outside city / privacy-masked; NYC 264/265 = unknown) are kept out of the
zone panel and counted in pickups_unzoned instead.

`hour` is the naive local wall-clock hour the trip started in. The hour grid is built from
real UTC hours converted to local time, so the nonexistent spring-forward hour is absent and
the fall-back 01:00 hour holds two real hours of trips (trip data cannot tell them apart);
those hours have dst_merged_hour = true.

Trips whose start month differs from their file's month (stray TLC records) are dropped.
drive_times are per calendar year: a feature for an hour in year y must use year y-1.
Hotspot labels are not computed here.

Usage: python scripts/aggregate.py [chicago|nyc ...]     (default: both)
"""
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

CITIES = {
    "chicago": dict(
        files="chicago_trips/*/tnp_trips_*.parquet", month_re=r"tnp_trips_(\d{4}-\d{2})\.parquet",
        tz="America/Chicago", n_zones=77, ts="trip_start_timestamp",
        pu="pickup_community_area", do="dropoff_community_area", secs="trip_seconds"),
    "nyc": dict(
        files="nyc_fhvhv/fhvhv_*.parquet", month_re=r"fhvhv_(\d{4}-\d{2})\.parquet",
        tz="America/New_York", n_zones=263, ts="pickup_datetime",
        pu="PULocationID", do="DOLocationID", secs="trip_time"),
}
MIN_TRIP_SECONDS, MAX_TRIP_SECONDS = 60, 3 * 3600  # drive-time sanity bounds


def connect():
    con = duckdb.connect()
    con.execute(f"SET threads = {int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))}")
    mem_mb = int(os.environ.get("SLURM_MEM_PER_NODE", 16 * 1024))
    con.execute(f"SET memory_limit = '{int(mem_mb * 0.75)}MB'")
    con.execute("SET preserve_insertion_order = false")
    if os.environ.get("TMPDIR"):
        con.execute(f"SET temp_directory = '{os.environ['TMPDIR']}/duckdb'")
    return con


def write_parquet(con, query, out):
    part = out.with_name(out.name + ".part")
    con.execute(f"COPY ({query}) TO '{part}' (FORMAT parquet, COMPRESSION zstd)")
    part.replace(out)


def trips_sql(c, files):
    """One row per trip: start hour, pickup/dropoff zone (NULL if invalid), duration."""
    file_list = ", ".join(f"'{f}'" for f in files)
    z = c["n_zones"]
    return f"""
        SELECT date_trunc('hour', {c['ts']}) AS hour,
               CASE WHEN {c['pu']} BETWEEN 1 AND {z} THEN {c['pu']} END AS pu_zone,
               CASE WHEN {c['do']} BETWEEN 1 AND {z} THEN {c['do']} END AS do_zone,
               {c['secs']} AS trip_seconds
        FROM read_parquet([{file_list}], filename = true)
        WHERE strftime({c['ts']}, '%Y-%m') = regexp_extract(filename, '{c['month_re']}', 1)"""


def build_panels(con, city, c, files):
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE counts AS
        SELECT hour, pu_zone AS zone, count(*)::INTEGER AS pickups
        FROM ({trips_sql(c, files)}) GROUP BY ALL""")
    lo, hi = con.execute("SELECT min(hour), max(hour) FROM counts").fetchone()
    tz = c["tz"]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE grid AS
        SELECT timezone('{tz}', t) AS hour, count(*) > 1 AS dst_merged_hour
        FROM generate_series(timezone('{tz}', TIMESTAMP '{lo}'),
                             timezone('{tz}', TIMESTAMP '{hi}'), INTERVAL 1 HOUR) g(t)
        GROUP BY ALL""")

    write_parquet(con, f"""
        SELECT z.zone::SMALLINT AS zone, g.hour, coalesce(c.pickups, 0) AS pickups
        FROM grid g CROSS JOIN range(1, {c['n_zones'] + 1}) z(zone)
        LEFT JOIN counts c ON c.hour = g.hour AND c.zone = z.zone
        ORDER BY g.hour, zone""", PROC / f"{city}_zone_hour.parquet")
    write_parquet(con, """
        SELECT hour, dst_merged_hour,
               coalesce(sum(pickups), 0)::INTEGER AS pickups_total,
               coalesce(sum(pickups) FILTER (zone IS NULL), 0)::INTEGER AS pickups_unzoned
        FROM grid LEFT JOIN counts USING (hour)
        GROUP BY hour, dst_merged_hour ORDER BY hour""", PROC / f"{city}_city_hour.parquet")

    total, unzoned = con.execute(
        "SELECT sum(pickups), sum(pickups) FILTER (zone IS NULL) FROM counts").fetchone()
    off_grid = con.execute(
        "SELECT coalesce(sum(pickups), 0) FROM counts ANTI JOIN grid USING (hour)").fetchone()[0]
    merged = con.execute("SELECT count(*) FROM grid WHERE dst_merged_hour").fetchone()[0]
    empty_days = [d for (d,) in con.execute("""
        SELECT hour::DATE AS d FROM grid LEFT JOIN counts USING (hour)
        GROUP BY d HAVING coalesce(sum(pickups), 0) = 0 ORDER BY d""").fetchall()]
    print(f"{city}: {lo} -> {hi}, {total:,} pickups, {(unzoned or 0) / total:.2%} unzoned, "
          f"{merged} DST-merged hours, {off_grid:,} pickups in nonexistent DST hours", flush=True)
    if empty_days:
        print(f"{city}: WARNING {len(empty_days)} days with zero trips (missing downloads?): "
              f"{', '.join(map(str, empty_days[:20]))}{' ...' if len(empty_days) > 20 else ''}",
              flush=True)


def build_drive_times(con, city, c, files):
    by_year = defaultdict(list)
    for f in files:
        by_year[re.search(c["month_re"], f.name).group(1)[:4]].append(f)
    con.execute("""
        CREATE OR REPLACE TEMP TABLE drive_times (
            year SMALLINT, pu_zone SMALLINT, do_zone SMALLINT, hour_of_day TINYINT,
            is_weekend BOOLEAN, n_trips INTEGER, median_trip_seconds DOUBLE)""")
    for year, year_files in sorted(by_year.items()):
        con.execute(f"""
            INSERT INTO drive_times
            SELECT {year}, pu_zone, do_zone, hour(hour), dayofweek(hour) IN (0, 6),
                   count(*), median(trip_seconds)
            FROM ({trips_sql(c, year_files)})
            WHERE pu_zone IS NOT NULL AND do_zone IS NOT NULL
              AND trip_seconds BETWEEN {MIN_TRIP_SECONDS} AND {MAX_TRIP_SECONDS}
            GROUP BY ALL""")
        print(f"{city}: drive times {year} from {len(year_files)} months", flush=True)
    write_parquet(con, "SELECT * FROM drive_times ORDER BY ALL", PROC / f"{city}_drive_times.parquet")


def main():
    con = connect()
    for city in sys.argv[1:] or list(CITIES):
        c = CITIES[city]
        files = sorted(PROC.glob(c["files"]))
        if not files:
            print(f"{city}: no processed trip files, skipping", flush=True)
            continue
        build_panels(con, city, c, files)
        build_drive_times(con, city, c, files)


if __name__ == "__main__":
    main()
