#!/usr/bin/env python
"""Aggregate processed trips into zone-hour pickup panels and OD drive-time tables.

Outputs in data/processed/:
  {city}_zone_hour.parquet    zone, hour, pickups, dropoffs (zero-filled over every zone x hour),
                              median_wait_s, median_fare, median_driver_pay, median_trip_miles
                              (medians over that zone-hour's pickups; NULL when it has none)
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
Dropoffs are counted at the dropoff zone in the hour the trip ended.
NYC median_wait_s is request -> pickup; Chicago publishes no request time or driver pay, so
median_wait_s and median_driver_pay are NULL there. median_fare is rider fare + tip.

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
        pu="pickup_community_area", do="dropoff_community_area", secs="trip_seconds",
        end="trip_end_timestamp", wait="NULL::DOUBLE", fare="fare + coalesce(tip, 0)",
        pay="NULL::DOUBLE", miles="trip_miles"),
    "nyc": dict(
        files="nyc_fhvhv/fhvhv_*.parquet", month_re=r"fhvhv_(\d{4}-\d{2})\.parquet",
        tz="America/New_York", n_zones=263, ts="pickup_datetime",
        pu="PULocationID", do="DOLocationID", secs="trip_time",
        end="dropoff_datetime", wait="date_diff('second', request_datetime, pickup_datetime)",
        fare="base_passenger_fare + coalesce(tips, 0)", pay="driver_pay", miles="trip_miles"),
}
MIN_TRIP_SECONDS, MAX_TRIP_SECONDS = 60, 3 * 3600  # drive-time sanity bounds
MAX_WAIT_SECONDS = 3600  # request -> pickup sanity bound


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
    """One row per trip: start/end hour, pickup/dropoff zone (NULL if invalid), duration,
    wait, fare, driver pay, miles."""
    file_list = ", ".join(f"'{f}'" for f in files)
    z = c["n_zones"]
    return f"""
        SELECT date_trunc('hour', {c['ts']}) AS hour,
               date_trunc('hour', {c['end']}) AS dropoff_hour,
               CASE WHEN {c['pu']} BETWEEN 1 AND {z} THEN {c['pu']} END AS pu_zone,
               CASE WHEN {c['do']} BETWEEN 1 AND {z} THEN {c['do']} END AS do_zone,
               {c['secs']} AS trip_seconds, {c['wait']} AS wait_seconds,
               {c['fare']} AS fare, {c['pay']} AS driver_pay, {c['miles']} AS trip_miles
        FROM read_parquet([{file_list}], filename = true)
        WHERE strftime({c['ts']}, '%Y-%m') = regexp_extract(filename, '{c['month_re']}', 1)"""


def build_panels(con, city, c, files):
    con.execute("""
        CREATE OR REPLACE TEMP TABLE counts (
            hour TIMESTAMP, zone SMALLINT, pickups INTEGER, median_wait_s DOUBLE,
            median_fare DOUBLE, median_driver_pay DOUBLE, median_trip_miles DOUBLE)""")
    con.execute("CREATE OR REPLACE TEMP TABLE drops (hour TIMESTAMP, zone SMALLINT, dropoffs INTEGER)")
    # One month file at a time keeps the medians' memory bounded. A pickup hour never spans
    # files, so the medians are exact; dropoff hours can (trips ending after midnight on the
    # 1st), so dropoffs are summed again when the panel is written.
    for f in files:
        trips = trips_sql(c, [f])
        con.execute(f"""
            INSERT INTO counts
            SELECT hour, pu_zone, count(*),
                   median(wait_seconds) FILTER (wait_seconds BETWEEN 0 AND {MAX_WAIT_SECONDS}),
                   median(fare), median(driver_pay), median(trip_miles)
            FROM ({trips}) GROUP BY hour, pu_zone""")
        con.execute(f"""
            INSERT INTO drops
            SELECT dropoff_hour, do_zone, count(*) FROM ({trips})
            WHERE do_zone IS NOT NULL AND dropoff_hour IS NOT NULL
            GROUP BY dropoff_hour, do_zone""")
    lo, hi = con.execute("SELECT min(hour), max(hour) FROM counts").fetchone()
    tz = c["tz"]
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE grid AS
        SELECT timezone('{tz}', t) AS hour, count(*) > 1 AS dst_merged_hour
        FROM generate_series(timezone('{tz}', TIMESTAMP '{lo}'),
                             timezone('{tz}', TIMESTAMP '{hi}'), INTERVAL 1 HOUR) g(t)
        GROUP BY ALL""")

    write_parquet(con, f"""
        SELECT z.zone::SMALLINT AS zone, g.hour,
               coalesce(c.pickups, 0) AS pickups, coalesce(d.dropoffs, 0) AS dropoffs,
               c.median_wait_s, c.median_fare, c.median_driver_pay, c.median_trip_miles
        FROM grid g CROSS JOIN range(1, {c['n_zones'] + 1}) z(zone)
        LEFT JOIN counts c ON c.hour = g.hour AND c.zone = z.zone
        LEFT JOIN (SELECT hour, zone, sum(dropoffs)::INTEGER AS dropoffs FROM drops GROUP BY hour, zone) d
               ON d.hour = g.hour AND d.zone = z.zone
        ORDER BY g.hour, zone""", PROC / f"{city}_zone_hour.parquet")
    write_parquet(con, """
        SELECT hour, dst_merged_hour,
               coalesce(sum(pickups), 0)::INTEGER AS pickups_total,
               coalesce(sum(pickups) FILTER (zone IS NULL), 0)::INTEGER AS pickups_unzoned
        FROM grid LEFT JOIN counts USING (hour)
        GROUP BY hour, dst_merged_hour ORDER BY hour""", PROC / f"{city}_city_hour.parquet")

    total, unzoned = con.execute(
        "SELECT sum(pickups), sum(pickups) FILTER (zone IS NULL) FROM counts").fetchone()
    dropped_off = con.execute("SELECT sum(dropoffs) FROM drops").fetchone()[0]
    off_grid = con.execute(
        "SELECT coalesce(sum(pickups), 0) FROM counts ANTI JOIN grid USING (hour)").fetchone()[0]
    merged = con.execute("SELECT count(*) FROM grid WHERE dst_merged_hour").fetchone()[0]
    empty_days = [d for (d,) in con.execute("""
        SELECT hour::DATE AS d FROM grid LEFT JOIN counts USING (hour)
        GROUP BY d HAVING coalesce(sum(pickups), 0) = 0 ORDER BY d""").fetchall()]
    print(f"{city}: {lo} -> {hi}, {total:,} pickups, {(unzoned or 0) / total:.2%} unzoned, "
          f"{dropped_off:,} zoned dropoffs, "
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
