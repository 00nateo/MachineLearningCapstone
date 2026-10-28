#!/usr/bin/env python
"""Surge ("unexpected hotspot") labels: zone-hours much busier than that zone usually is.

Design: SURGE_LABEL_DESIGN.md. Builds data/processed_ml/{city}_surge.parquet from the feature
table (features.py output) without touching the existing top-10% label.

Baseline (expected pickups under normal conditions), per zone and hour:
  median pickups at the same hour on the same weekday over the previous BASELINE_WEEKS weeks,
  ignoring hours flagged extreme (any component) and public holidays, so past storms, events
  and holidays don't define "normal". Past data only. NULL if fewer than MIN_CLEAN_WEEKS clean
  weeks are available. It deliberately does NOT condition on weather / events / outages: those
  are the anomalies to detect.
Labels:
  surge_125 / surge_150 / surge_200 / surge_300: baseline >= VOLUME_FLOOR and
      pickups >= 1.25x / 1.5x / 2x / 3x baseline (ratio labels; the floor stops 1 -> 3 rides
      from counting as +200%).
  surge_z3: count-aware alternative, (pickups - baseline) / sqrt(baseline) >= 3 (a Poisson-style
      z-score: small, noisy zones need a bigger jump), with baseline >= Z_FLOOR.
Labels describe hour h itself (like `hotspot`); never feed baseline-of-h or the labels to a model
for the same hour beyond what is already lagged.

Also writes data/results/{city}_surge_label_summary.csv: per label, positive rate overall, by
fold-3 split, normal vs extreme hours and per condition, by time of day, zone spread, and
overlap with the top-10% hotspot label.

Usage: python scripts/surge_label.py [nyc|chicago ...]     (default: both)
"""
import os
import sys
from pathlib import Path

import duckdb
import holidays
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"
RESULTS = ROOT / "data" / "results"
CITIES = {"nyc": "NY", "chicago": "IL"}
BASELINE_WEEKS = 8
MIN_CLEAN_WEEKS = 4
VOLUME_FLOOR = 10  # expected rides per hour for the ratio labels
Z_FLOOR = 5        # expected rides per hour for the count-aware label
THRESHOLDS = {"surge_125": 1.25, "surge_150": 1.5, "surge_200": 2.0, "surge_300": 3.0}
FLAGS = ["extreme_flag", "severe_weather_flag", "nws_warning_flag", "major_event_flag",
         "transit_disruption_flag", "flight_delay_flag"]
TIME_OF_DAY = """CASE WHEN hour(hour) < 6 THEN '00-05 night' WHEN hour(hour) < 10 THEN '06-09 morning rush'
                      WHEN hour(hour) < 16 THEN '10-15 midday' WHEN hour(hour) < 20 THEN '16-19 evening rush'
                      ELSE '20-23 evening' END"""


def connect():
    con = duckdb.connect()
    con.execute(f"SET threads = {int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))}")
    con.execute("SET preserve_insertion_order = false")
    return con


def build(con, city):
    src = PROC_ML / f"{city}_features.parquet"
    years = con.execute(f"SELECT year(min(hour)), year(max(hour)) FROM read_parquet('{src}')").fetchone()
    hol = holidays.country_holidays("US", subdiv=CITIES[city], years=range(years[0], years[1] + 1))
    con.register("holiday_dates", pd.DataFrame({"d": sorted(hol)}))
    labels = ",\n".join(f"baseline >= {VOLUME_FLOOR} AND pickups >= {t} * baseline AS {name}"
                        for name, t in THRESHOLDS.items())
    out = PROC_ML / f"{city}_surge.parquet"
    con.execute(f"""
        COPY (
            WITH f AS (
                SELECT zone, hour, pickups, hotspot, fold1, fold2, fold3, {', '.join(FLAGS)},
                       hour::DATE IN (SELECT d FROM holiday_dates) AS is_holiday
                FROM read_parquet('{src}')),
            b AS (
                SELECT *,
                       median(CASE WHEN NOT extreme_flag AND NOT is_holiday THEN pickups END) OVER w AS med,
                       count(CASE WHEN NOT extreme_flag AND NOT is_holiday THEN pickups END) OVER w AS clean_weeks
                FROM f
                WINDOW w AS (PARTITION BY zone, dayofweek(hour), hour(hour) ORDER BY hour
                             ROWS BETWEEN {BASELINE_WEEKS} PRECEDING AND 1 PRECEDING)),
            s AS (
                SELECT * EXCLUDE (med, clean_weeks),
                       CASE WHEN clean_weeks >= {MIN_CLEAN_WEEKS} THEN med END AS baseline
                FROM b)
            SELECT *, pickups / nullif(baseline, 0) AS ratio,
                   (pickups - baseline) / sqrt(nullif(baseline, 0)) AS surge_z,
                   {labels},
                   baseline >= {Z_FLOOR} AND (pickups - baseline) / sqrt(baseline) >= 3 AS surge_z3
            FROM s
            ORDER BY hour, zone
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    con.unregister("holiday_dates")
    return out


def summarize(con, city, path):
    names = [*THRESHOLDS, "surge_z3"]
    t = f"read_parquet('{path}')"
    rows = []
    for name in names:
        def rate(where):
            return con.execute(f"SELECT avg(coalesce({name}, false)::INT), count(*) FROM {t} WHERE {where}").fetchone()
        r = {"label": name}
        r["rate_all"], _ = rate("baseline IS NOT NULL")
        for split in ("train", "early_stop", "calibrate", "test"):
            r[f"rate_fold3_{split}"], _ = rate(f"baseline IS NOT NULL AND fold3 = '{split}'")
        r["rate_normal"], _ = rate("baseline IS NOT NULL AND NOT extreme_flag")
        for flag in FLAGS:
            r[f"rate_{flag.removesuffix('_flag')}"], _ = rate(f"baseline IS NOT NULL AND {flag}")
        zone = con.execute(f"""SELECT count(*) FILTER (r > 0), max(r) FROM (
            SELECT zone, avg(coalesce({name}, false)::INT) AS r FROM {t} WHERE baseline IS NOT NULL GROUP BY zone)""").fetchone()
        r["zones_with_any"], r["max_zone_rate"] = zone
        r["share_also_top10_hotspot"] = con.execute(
            f"SELECT avg(hotspot::INT) FROM {t} WHERE {name}").fetchone()[0]
        for tod, v in con.execute(f"""SELECT {TIME_OF_DAY} AS tod, avg(coalesce({name}, false)::INT)
                                       FROM {t} WHERE baseline IS NOT NULL GROUP BY 1 ORDER BY 1""").fetchall():
            r[f"rate_{tod.split()[0]}"] = v
        rows.append(r)
    res = pd.DataFrame(rows)
    base_cov = con.execute(f"SELECT avg((baseline IS NOT NULL)::INT), avg((baseline >= {VOLUME_FLOOR})::INT) FROM {t}").fetchone()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{city}_surge_label_summary.csv"
    res.to_csv(out, index=False)
    show = ["label", "rate_all", "rate_normal", "rate_extreme", "rate_major_event", "rate_severe_weather",
            "rate_nws_warning", "rate_transit_disruption", "zones_with_any", "max_zone_rate", "share_also_top10_hotspot"]
    print(f"{city}: baseline available for {base_cov[0]:.1%} of zone-hours, >= {VOLUME_FLOOR}/h for {base_cov[1]:.1%}\n"
          + res[show].round(4).to_string(index=False)
          + "\nby time of day:\n" + res[["label"] + [c for c in res.columns if c[5:7].isdigit()]].round(4).to_string(index=False)
          + "\nby fold-3 split:\n" + res[["label"] + [c for c in res.columns if c.startswith("rate_fold3")]].round(4).to_string(index=False)
          + f"\n-> {path}\n-> {out}", flush=True)


def main():
    con = connect()
    for city in sys.argv[1:] or list(CITIES):
        summarize(con, city, build(con, city))


if __name__ == "__main__":
    main()
