#!/usr/bin/env python
"""Stratified day sampling of training rows: data/processed_ml/{city}_{fold}_train.parquet.

Lags were computed on the full continuous series in features.py; only the choice of training
rows is subsampled here (MODELING_PLAN.md). Per city and fold, from the fold's train rows:
  - keep DAYS_PER_MONTH random calendar days per month;
  - always keep a zone's rows on a day that zone had any extreme hour. This is zone-day, not
    city-day: major events and flight delays fire in some zone almost every day, so a city-day
    rule would keep nearly every day and sample nothing;
  - weight = EXTREME_WEIGHT on rows with extreme_flag, 1 otherwise.
Early-stop, calibrate and test rows are never sampled. The draw is deterministic (days ordered
by md5(date || SEED)), so reruns pick the same days.

Usage: python scripts/sample.py [--city nyc chicago] [--fold fold3 ...]
"""
import argparse
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"
DAYS_PER_MONTH = 20
EXTREME_WEIGHT = 3.0
SEED = "rideshare-capstone"


def sample(con, city, fold):
    src = PROC_ML / f"{city}_features.parquet"
    out = PROC_ML / f"{city}_{fold}_train.parquet"
    con.execute(f"""
        COPY (
            WITH t AS (SELECT * FROM read_parquet('{src}') WHERE {fold} = 'train'),
            sampled_days AS (
                SELECT d FROM (
                    SELECT d, row_number() OVER (PARTITION BY date_trunc('month', d)
                                                 ORDER BY md5(d::VARCHAR || '{SEED}')) AS r
                    FROM (SELECT DISTINCT hour::DATE AS d FROM t))
                WHERE r <= {DAYS_PER_MONTH}),
            extreme_zone_days AS (SELECT DISTINCT zone, hour::DATE AS d FROM t WHERE extreme_flag)
            SELECT t.*, CASE WHEN t.extreme_flag THEN {EXTREME_WEIGHT} ELSE 1.0 END AS weight
            FROM t
            LEFT JOIN sampled_days s ON s.d = t.hour::DATE
            LEFT JOIN extreme_zone_days x ON x.zone = t.zone AND x.d = t.hour::DATE
            WHERE s.d IS NOT NULL OR x.d IS NOT NULL
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    total = con.execute(f"SELECT count(*) FROM read_parquet('{src}') WHERE {fold} = 'train'").fetchone()[0]
    kept, extreme, days = con.execute(f"""
        SELECT count(*), avg(extreme_flag::INT), count(DISTINCT hour::DATE) FROM read_parquet('{out}')""").fetchone()
    print(f"{city} {fold}: kept {kept:,} of {total:,} train rows ({kept / total:.0%}) on {days} days, "
          f"{extreme:.1%} extreme -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["nyc", "chicago"])
    ap.add_argument("--fold", nargs="+", default=["fold3"])
    a = ap.parse_args()
    con = duckdb.connect()
    for city in a.city:
        for fold in a.fold:
            sample(con, city, fold)


if __name__ == "__main__":
    main()
