#!/usr/bin/env python
"""Build data/processed_ml/{city}_labeled.parquet: hotspot label + rolling-origin fold tags.

Hotspot label: pickups > the 90th percentile of pickups across zones in that same hour
(quantile threshold, not a rank/ntile, so nights where many zones tie at 0 pickups don't
force a handful of zero-pickup zones to be labelled hotspots).

Fold tags: three rolling-origin folds, each a column (fold1, fold2, fold3) whose value is
one of 'train', 'early_stop', 'calibrate', 'test', or NULL (hour not used in that fold --
either the 2020-03..2021-06 COVID gap, or outside that fold's defined window). Fold 3 is the
headline fold and matches the dataset card's documented split exactly. See MODELING_PLAN.md.

Usage: python scripts/label.py [nyc|chicago ...]     (default: both)
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / "data" / "export"
OUT = ROOT / "data" / "processed_ml"

CITIES = ("nyc", "chicago")

COVID_GAP = ("2020-03-01", "2021-07-01")  # [start, end) -- excludes Mar 2020 .. Jun 2021

# (train_start, train_end, early_stop_start, early_stop_end, calibrate_end, test_end) --
# all [start, end) on `hour`. test_end = None means open-ended (through whatever data exists).
# fold1's early_stop_start is NOT train_end: the COVID gap sits between them, and that gap
# must land in neither bucket (handled by the ELSE NULL below, not by early_stop's own bounds).
FOLDS = {
    "fold1": ("2019-01-01", "2020-03-01", "2021-07-01", "2022-01-01", "2023-01-01", "2024-01-01"),
    "fold2": ("2019-01-01", "2022-01-01", "2022-01-01", "2023-01-01", "2024-01-01", "2025-01-01"),
    "fold3": ("2019-01-01", "2023-01-01", "2023-01-01", "2024-01-01", "2025-01-01", None),
}


def connect():
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    return con


def fold_case(name, bounds):
    train_start, train_end, es_start, es_end, cal_end, test_end = bounds
    not_covid = f"NOT (hour >= TIMESTAMP '{COVID_GAP[0]}' AND hour < TIMESTAMP '{COVID_GAP[1]}')"
    test_hi = f"AND hour < TIMESTAMP '{test_end}'" if test_end else ""
    return f"""
        CASE
            WHEN hour >= TIMESTAMP '{train_start}' AND hour < TIMESTAMP '{train_end}'
                 AND {not_covid} THEN 'train'
            WHEN hour >= TIMESTAMP '{es_start}' AND hour < TIMESTAMP '{es_end}'
                THEN 'early_stop'
            WHEN hour >= TIMESTAMP '{es_end}' AND hour < TIMESTAMP '{cal_end}'
                THEN 'calibrate'
            WHEN hour >= TIMESTAMP '{cal_end}' {test_hi}
                THEN 'test'
            ELSE NULL
        END AS {name}"""


def label_city(con, city):
    src = EXPORT / f"{city}_zone_hour.parquet"
    if not src.exists():
        print(f"{city}: {src} not found, skipping", flush=True)
        return
    fold_cols = ",\n".join(fold_case(name, bounds) for name, bounds in FOLDS.items())
    out = OUT / f"{city}_labeled.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT zone, hour, pickups, dropoffs,
                   median_wait_s, median_fare, median_driver_pay, median_trip_miles,
                   (pickups > quantile_cont(pickups, 0.9) OVER (PARTITION BY hour)) AS hotspot,
                   {fold_cols}
            FROM read_parquet('{src}')
            ORDER BY hour, zone
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    print(f"{city}: wrote {out}", flush=True)


def main():
    con = connect()
    for city in sys.argv[1:] or list(CITIES):
        label_city(con, city)


if __name__ == "__main__":
    main()
