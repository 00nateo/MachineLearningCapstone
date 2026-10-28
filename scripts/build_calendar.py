#!/usr/bin/env python
"""Flag recurring holidays/events across the full local-date demand panels.

Writes {city}_calendar_day.parquet and sourced {city}_calendar_events.csv under
data/processed_ml/. By default, adds calendar_* columns to existing local feature
tables atomically. Model allowlists, labels, samples, and cached HF exports stay
as they were. --no-annotate produces sidecars only.

Usage: python scripts/build_calendar.py --city nyc chicago
"""
import argparse
from datetime import datetime, timezone
import json

import holidays

from calendar_flags import PROC_ML, SOURCES, build_for_city


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", nargs="+", choices=["nyc", "chicago"], default=["nyc", "chicago"])
    ap.add_argument("--no-annotate", action="store_true")
    args = ap.parse_args()
    for city in args.city:
        days, events = build_for_city(city, annotate=not args.no_annotate)
        print(f"{city}: {len(days):,} calendar days, {len(events):,} event records; "
              f"{days.calendar_any_special_day.sum()} special/context days", flush=True)
    metadata = dict(generated_utc=datetime.now(timezone.utc).isoformat(), sources=SOURCES,
                    holidays_version=holidays.__version__, audited_years=[2018, 2026],
                    election_scope="NYC annual November general; Chicago even-year general plus 2019/2023 municipal rounds",
                    excluded_elections="primaries, special elections, early voting",
                    canceled_marathons=[2020], flags_are_model_inputs=False)
    (PROC_ML / "calendar_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
