#!/usr/bin/env python
"""Add autoregressive lag/rolling features to data/processed_ml/{city}_labeled.parquet.

Week-1 scope: own-zone pickup history only (lag 1h/24h/168h, trailing 24h/168h mean+std).
External joins (weather, events, transit, flights) and cross-zone neighbour features land
in week 2 -- see MODELING_PLAN.md.

All lag/rolling values are computed over the FULL continuous per-zone hour series before any
fold or day-sampling filter is applied, so a kept training row's lag always reflects real
history rather than a hole left by subsampling. Rolling windows exclude the current hour
(1 PRECEDING, not CURRENT ROW) so hour h's own pickups never leak into its own features.

Usage: python scripts/features.py [nyc|chicago ...]     (default: both)
"""
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"

CITIES = ("nyc", "chicago")


def connect():
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    return con


def features_city(con, city):
    src = PROC_ML / f"{city}_labeled.parquet"
    if not src.exists():
        print(f"{city}: {src} not found, run label.py first", flush=True)
        return
    out = PROC_ML / f"{city}_features.parquet"
    con.execute(f"""
        COPY (
            SELECT *,
                   lag(pickups, 1) OVER w AS pickup_lag1,
                   lag(pickups, 24) OVER w AS pickup_lag24,
                   lag(pickups, 168) OVER w AS pickup_lag168,
                   avg(pickups) OVER w24 AS pickup_roll24_mean,
                   stddev_samp(pickups) OVER w24 AS pickup_roll24_std,
                   avg(pickups) OVER w168 AS pickup_roll168_mean,
                   stddev_samp(pickups) OVER w168 AS pickup_roll168_std
            FROM read_parquet('{src}')
            WINDOW
                w AS (PARTITION BY zone ORDER BY hour),
                w24 AS (PARTITION BY zone ORDER BY hour ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING),
                w168 AS (PARTITION BY zone ORDER BY hour ROWS BETWEEN 168 PRECEDING AND 1 PRECEDING)
            ORDER BY hour, zone
        ) TO '{out}' (FORMAT parquet, COMPRESSION zstd)""")
    print(f"{city}: wrote {out}", flush=True)


def main():
    con = connect()
    for city in sys.argv[1:] or list(CITIES):
        features_city(con, city)


if __name__ == "__main__":
    main()
