#!/usr/bin/env python
"""Build overlapping broad categories, citywide event-window inputs and venue masks."""
import argparse

from event_context import build


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", nargs="+", choices=["nyc", "chicago"], default=["nyc", "chicago"])
    ap.add_argument("--before-hours", type=int, default=3)
    ap.add_argument("--after-hours", type=int, default=3)
    args = ap.parse_args()
    if min(args.before_hours, args.after_hours) < 1:
        ap.error("Windows must be >= 1 hour")
    for city in args.city:
        context, excluded, coverage = build(city, before_hours=args.before_hours, after_hours=args.after_hours)
        print(f"{city}: {len(context):,} hours; {len(excluded):,} excluded venue zone-hours\n"
              + coverage.groupby("category")[["events", "mapped_venue_events"]].sum().to_string(), flush=True)


if __name__ == "__main__":
    main()
