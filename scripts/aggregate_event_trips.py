#!/usr/bin/env python
"""Stream monthly trip Parquet on the cluster into small event-associated OD counts.

NYC: converted nyc_fhvhv files OR original TLC fhvhv monthly Parquet.
Chicago: converted chicago_trips monthly Parquet (run convert.py on raw CSV first).
Arrival remote zone = pickup zone of a trip ending at the venue before/during.
Departure remote zone = dropoff zone of a trip starting at the venue during/after.
Only remote zones are output. Active venue zones are excluded there. This measures
event-associated traffic, not proven attendee movement or causal event effects.

Prior-week controls use the SAME local-time windows and candidate-zone exclusions.
Output shards are additive across input months (including midnight dropoffs).
Absent months are missing coverage, never zero demand. Inspect exposures.parquet
and manifest.json before forming training labels. No raw trips leave the cluster.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import re

import duckdb
import pandas as pd

from event_context import ROOT, categorize, make_windows


def merged_intervals(events, before_hours, after_hours):
    """Union overlapping intervals per venue/category/phase/direction before joining trips."""
    groups = {}
    for e in events.itertuples(index=False):
        if pd.isna(e.zone) or e.category not in ("sports", "concerts"):
            continue
        intervals = [("before", "arrival", e.start_local - pd.Timedelta(hours=before_hours), e.start_local),
                     ("during", "arrival", e.start_local, e.end_local_est),
                     ("during", "departure", e.start_local, e.end_local_est),
                     ("after", "departure", e.end_local_est, e.end_local_est + pd.Timedelta(hours=after_hours))]
        for phase, direction, start, end in intervals:
            groups.setdefault((int(e.zone), e.category, phase, direction), []).append((start, end))
    rows = []
    for key, intervals in groups.items():
        merged = []
        for start, end in sorted(intervals):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        for start, end in merged:
            rows.append(dict(venue_zone=key[0], category=key[1], phase=key[2], direction=key[3],
                             start=start, end=end))
    return pd.DataFrame(rows, columns=["venue_zone", "category", "phase", "direction", "start", "end"])


def control_intervals(windows, weeks):
    frames = []
    for week in [0, *weeks]:
        d = windows.copy()
        d["shift_days"] = week * 7
        d["period"] = "event" if week == 0 else f"control_week_{week}"
        d["start"] -= pd.Timedelta(days=week * 7)
        d["end"] -= pd.Timedelta(days=week * 7)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def aggregate_file(con, path, city, windows, exclusions, month):
    """Scan only needed columns; distinct merged windows prevent within-group duplication."""
    cols = (dict(start="pickup_datetime", end="dropoff_datetime", pu="PULocationID", do="DOLocationID", secs="trip_time")
            if city == "nyc" else dict(start="trip_start_timestamp", end="trip_end_timestamp",
                                     pu="pickup_community_area", do="dropoff_community_area", secs="trip_seconds"))
    n = 263 if city == "nyc" else 77
    lo, hi = windows.start.min(), windows.end.max()
    con.register("event_windows", windows)
    con.register("excluded_zones", exclusions)
    # A materialized projection avoids scanning wide Parquet twice for arrivals/departures.
    con.execute(f"""CREATE OR REPLACE TEMP TABLE event_trips AS
        SELECT {cols['start']}::TIMESTAMP AS start_time, {cols['end']}::TIMESTAMP AS end_time,
               {cols['pu']}::SMALLINT AS pu_zone, {cols['do']}::SMALLINT AS do_zone
        FROM read_parquet(?) WHERE strftime({cols['start']}, '%Y-%m') = ?
          AND {cols['start']} < ? AND {cols['end']} >= ?
          AND {cols['end']} >= {cols['start']} AND {cols['secs']} BETWEEN 60 AND 10800
          AND {cols['pu']} BETWEEN 1 AND {n} AND {cols['do']} BETWEEN 1 AND {n}
    """, [str(path), month, hi.to_pydatetime(), lo.to_pydatetime()])
    return con.execute("""WITH linked AS (
        SELECT w.category, w.phase, w.direction, w.period,
               date_trunc('hour', t.end_time) + w.shift_days * INTERVAL 1 DAY AS reference_hour,
               t.pu_zone AS remote_zone
        FROM event_trips t JOIN event_windows w ON w.direction = 'arrival'
             AND t.do_zone = w.venue_zone AND t.end_time >= w.start AND t.end_time < w.end
        UNION ALL
        SELECT w.category, w.phase, w.direction, w.period,
               date_trunc('hour', t.start_time) + w.shift_days * INTERVAL 1 DAY AS reference_hour,
               t.do_zone AS remote_zone
        FROM event_trips t JOIN event_windows w ON w.direction = 'departure'
             AND t.pu_zone = w.venue_zone AND t.start_time >= w.start AND t.start_time < w.end)
        SELECT reference_hour, category, phase, direction, period, remote_zone, count(*)::BIGINT AS trips
        FROM linked l ANTI JOIN excluded_zones e ON l.reference_hour = e.hour AND l.remote_zone = e.zone
        GROUP BY ALL ORDER BY reference_hour, category, phase, direction, period, remote_zone
    """).df()


def exposures(windows, available_months):
    """Explicit observed windows distinguish a zero count from absent trip files.

    Arrival coverage also requires the prior month on the first of each month,
    because a valid trip may start before midnight (max duration three hours).
    """
    rows = []
    for w in windows.itertuples(index=False):
        for hour in pd.date_range(w.start.floor("h"), w.end.ceil("h"), inclusive="left", freq="h"):
            required = {hour.strftime("%Y-%m")}
            if w.direction == "arrival":
                required.add((hour - pd.Timedelta(hours=3)).strftime("%Y-%m"))
            rows.append(dict(reference_hour=hour + pd.Timedelta(days=w.shift_days), observed_hour=hour,
                             venue_zone=w.venue_zone, category=w.category, phase=w.phase, direction=w.direction,
                             period=w.period, minutes=(min(hour + pd.Timedelta(hours=1), w.end) - max(hour, w.start)).total_seconds()/60,
                             required_month_files_present=required.issubset(available_months)))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", required=True, choices=["nyc", "chicago"])
    ap.add_argument("--trip-glob", required=True, help="Quote the glob; one canonical monthly file per month")
    ap.add_argument("--events", type=Path, default=ROOT / "data" / "export" / "events.parquet")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--before-hours", type=int, default=3)
    ap.add_argument("--after-hours", type=int, default=3)
    ap.add_argument("--control-weeks", type=int, nargs="*", default=[1, 2, 3, 4])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--memory-limit", default="8GB", help="DuckDB memory budget, independent of total dataset size")
    ap.add_argument("--temp-dir", type=Path, help="Cluster scratch directory for DuckDB spill")
    ap.add_argument("--start-date", help="First event date, e.g. 2023-01-01")
    ap.add_argument("--end-date", help="Inclusive last event date, e.g. 2024-12-31")
    args = ap.parse_args()
    if min(args.before_hours, args.after_hours, args.threads) < 1 or any(w < 1 for w in args.control_weeks):
        ap.error("Window lengths, threads, and control weeks must be positive")
    if len(set(args.control_weeks)) != len(args.control_weeks):
        ap.error("Control weeks must be unique")
    files = sorted(Path(p) for p in glob.glob(args.trip_glob, recursive=True))
    if not files:
        ap.error("No trip files matched")
    months = {}
    for path in files:
        match = re.search(r"(20\d{2}-\d{2})(?:\.parquet)$", path.name)
        if not match:
            ap.error(f"Expected a monthly YYYY-MM.parquet filename: {path}")
        month = match.group(1)
        if month in months:
            ap.error(f"Multiple files for {month}; select one canonical monthly dataset")
        months[month] = path
    events = categorize(pd.read_parquet(args.events), args.city)
    if args.start_date:
        events = events[events.start_local >= pd.Timestamp(args.start_date)]
    if args.end_date:
        events = events[events.start_local < pd.Timestamp(args.end_date) + pd.Timedelta(days=1)]
    if events.empty:
        ap.error("No events in selected dates")
    windows = control_intervals(merged_intervals(events, args.before_hours, args.after_hours), args.control_weeks)
    if windows.empty:
        ap.error("No mapped venue events; cannot infer venue-associated OD")
    # Mask all contemporaneous events, including ones outside the selected event-date range.
    all_events = categorize(pd.read_parquet(args.events), args.city)
    _, exclusions = make_windows(all_events, args.before_hours, args.after_hours)
    args.out.mkdir(parents=True, exist_ok=True)
    # Different reruns need separate directories, preventing accidental mixing of shards.
    if list(args.out.glob("part-*.parquet")):
        ap.error("Output directory already contains shards; use a new --out directory")
    manifest = dict(city=args.city, before_hours=args.before_hours, after_hours=args.after_hours,
                    control_weeks=args.control_weeks, event_catalog_sha256=hashlib.sha256(args.events.read_bytes()).hexdigest(),
                    mapped_events=int(events.zone.notna().sum()), unmapped_or_outside_events=int(events.zone.isna().sum()),
                    trip_filter="both zones valid, duration 60..10800 seconds, start within file month",
                    interpretation="event-associated remote endpoints; not verified attendance or causal effects",
                    calendar="venue-anchored sports/concerts only; citywide holiday/election flags remain in event_context",
                    available_months=sorted(months), status="in_progress", processed_files=[])
    exposures(windows, set(months)).to_parquet(args.out / "exposures.parquet", index=False)
    with duckdb.connect() as con:
        con.execute(f"SET threads = {args.threads}")
        con.execute("SET memory_limit = ?", [args.memory_limit])
        if args.temp_dir:
            args.temp_dir.mkdir(parents=True, exist_ok=True)
            con.execute("SET temp_directory = ?", [str(args.temp_dir)])
        for month, path in sorted(months.items()):
            start = pd.Timestamp(month + "-01")
            end = start + pd.offsets.MonthBegin(1) + pd.Timedelta(hours=3)
            subset = windows[(windows.end > start) & (windows.start < end)]
            if subset.empty:
                continue
            result = aggregate_file(con, path, args.city, subset, exclusions, month)
            result.to_parquet(args.out / f"part-{month}.parquet", compression="zstd", index=False)
            manifest["processed_files"].append(dict(month=month, path=str(path), rows=len(result),
                                                     size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns))
            print(f"{month}: {len(result):,} aggregate rows", flush=True)
            (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest["status"] = "complete"
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"-> {args.out} (sum part-*.parquet by keys when combining months; inspect exposure coverage)")


if __name__ == "__main__":
    main()
