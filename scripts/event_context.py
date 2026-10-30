"""Citywide event categories/windows and off-venue target masks.

These are overlapping contexts, not claims that a particular trip attended an
event. Venue event times come from the cached schedule; ends are estimates and
concert starts are assumed by the source pipeline. Calendar events are date-only.
"""
from pathlib import Path
import json

import duckdb
import numpy as np
import pandas as pd

from calendar_flags import FLAG_COLUMNS, PROC_ML, ROOT, build_for_city

CATEGORIES = ("sports", "concerts", "federal_holidays", "elections", "cultural", "road_races")
PHASES = ("before", "during", "after")
TIMED_FEATURES = [f"context_{category}_{phase}" for category in ("sports", "concerts") for phase in PHASES]
DAY_FLAGS = {"federal_holidays": "calendar_federal_holiday", "elections": "calendar_any_election",
             "cultural": "calendar_halloween", "road_races": "calendar_nyc_marathon"}
CONTEXT_FEATURES = TIMED_FEATURES + FLAG_COLUMNS


def categorize(events, city):
    """Keep unknown event sources visible rather than guessing their categories."""
    df = events[events.metro == city].drop_duplicates().copy()
    df["category"] = np.select([df.league.eq("concert"), df.source.eq("espn")],
                                ["concerts", "sports"], default="unclassified")
    df["start_local"] = pd.to_datetime(df.start_local)
    df["end_local_est"] = pd.to_datetime(df.end_local_est)
    if df[["source", "league", "event_id"]].duplicated().any():
        raise ValueError("Duplicate source/league/event_id in metro schedule; resolve before counting events")
    if df.start_local.isna().any() or (df.end_local_est <= df.start_local).any() or df.end_local_est.isna().any():
        raise ValueError("Every event needs a valid start and estimated end")
    return df


def make_windows(events, before_hours=3, after_hours=3):
    """Intervals are half-open; each hourly bin counts any interval overlap.

    Boundary hours can overlap phases (e.g. a 19:30 start). Never assign invented
    exact hourly start/end times. Exclusion spans before through after, including
    midnight spillover. Unknown/outside-city venue zones cannot be excluded.
    """
    rows, excluded = [], []
    for e in events.itertuples(index=False):
        if e.category not in ("sports", "concerts"):
            continue
        intervals = (("before", e.start_local - pd.Timedelta(hours=before_hours), e.start_local),
                     ("during", e.start_local, e.end_local_est),
                     ("after", e.end_local_est, e.end_local_est + pd.Timedelta(hours=after_hours)))
        for phase, start, end in intervals:
            for hour in pd.date_range(start.floor("h"), end.ceil("h"), inclusive="left", freq="h"):
                rows.append(dict(hour=hour, category=e.category, phase=phase,
                                 source=e.source, event_id=e.event_id, zone=e.zone))
                if pd.notna(e.zone):
                    excluded.append((hour, int(e.zone)))
    windows = pd.DataFrame(rows, columns=["hour", "category", "phase", "source", "event_id", "zone"])
    exclusions = pd.DataFrame(excluded, columns=["hour", "zone"]).drop_duplicates()
    return windows, exclusions


def build(city, proc_ml=PROC_ML, export=None, before_hours=3, after_hours=3):
    proc_ml = Path(proc_ml)
    export = Path(export) if export is not None else ROOT / "data" / "export"
    days, _ = build_for_city(city, annotate=False, proc_ml=proc_ml, export=export)
    events = categorize(pd.read_parquet(export / "events.parquet"), city)
    windows, exclusions = make_windows(events, before_hours, after_hours)
    with duckdb.connect() as con:
        hours = con.execute("SELECT DISTINCT hour FROM read_parquet(?) ORDER BY hour",
                            [str(export / f"{city}_zone_hour.parquet")]).df()
    if windows.empty:
        context = hours.copy()
    else:
        counts = windows.groupby(["hour", "category", "phase"]).size().unstack(["category", "phase"], fill_value=0)
        counts.columns = [f"context_{c}_{p}" for c, p in counts.columns]
        context = hours.merge(counts, on="hour", how="left", validate="one_to_one")
    for c in TIMED_FEATURES:
        context[c] = context[c].fillna(0).astype("int32") if c in context else 0
    context["date"] = context.hour.dt.date
    context = context.merge(days, on="date", how="left", validate="many_to_one").drop(columns="date")
    if context.calendar_any_special_day.isna().any():
        raise ValueError("Calendar did not cover every demand hour")
    exclusions = exclusions.merge(hours, on="hour", how="inner", validate="many_to_one")
    context.to_parquet(proc_ml / f"{city}_event_context.parquet", index=False)
    exclusions.to_parquet(proc_ml / f"{city}_event_exclusions.parquet", index=False)
    events.to_parquet(proc_ml / f"{city}_event_catalog.parquet", index=False)
    coverage = events.groupby(["category", "source", events.start_local.dt.year.rename("year")]).agg(
        events=("event_id", "size"), mapped_venue_events=("zone", "count"),
        venues=("venue_name", "nunique")).reset_index()
    coverage.to_csv(proc_ml / f"{city}_event_coverage.csv", index=False)
    metadata = dict(categories=list(CATEGORIES), before_hours=before_hours, after_hours=after_hours,
                    venue_exclusion="union of mapped venue zones for ALL sports/concert windows, including spillover",
                    unmapped_venues="outside-city and unmapped venues cannot be masked; inspect coverage",
                    spatial_scope="calendar days are citywide; road races have no route mask yet",
                    event_times="estimated ends; concert start 19:30/end 23:00 assumed upstream",
                    timing="half-open intervals, hourly overlap; boundary hours can have multiple phases",
                    knowledge="retrospective schedule snapshot; not a historical publication-time archive",
                    predictors=CONTEXT_FEATURES)
    (proc_ml / f"{city}_event_context.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return context, exclusions, coverage


def attach(df, city, data_dir=PROC_ML, off_venue=False):
    """Add context to training/inference rows; never add target-hour demand."""
    data_dir = Path(data_dir)
    context = pd.read_parquet(data_dir / f"{city}_event_context.parquet", columns=["hour", *CONTEXT_FEATURES])
    # Supports frames that already carry calendar annotations.
    df = df.drop(columns=[c for c in CONTEXT_FEATURES if c in df])
    joined = df.merge(context, on="hour", how="left", validate="many_to_one", sort=False)
    if joined[CONTEXT_FEATURES].isna().any().any():
        raise ValueError("Event context must cover every input hour; rebuild build_event_context.py")
    exclusions = pd.read_parquet(data_dir / f"{city}_event_exclusions.parquet")
    excluded = pd.MultiIndex.from_frame(joined[["hour", "zone"]]).isin(pd.MultiIndex.from_frame(exclusions[["hour", "zone"]]))
    joined["venue_excluded"] = excluded
    return joined.loc[~excluded].reset_index(drop=True) if off_venue else joined


def context_masks(df):
    masks = {"overall": np.ones(len(df), bool)}
    any_context = np.zeros(len(df), bool)
    for category in ("sports", "concerts"):
        group = np.zeros(len(df), bool)
        for phase in PHASES:
            mask = df[f"context_{category}_{phase}"].to_numpy() > 0
            masks[f"{category}/{phase}"] = mask
            group |= mask
        masks[category] = group
        any_context |= group
    for category, flag in DAY_FLAGS.items():
        mask = df[flag].to_numpy(bool)
        masks[category] = mask
        any_context |= mask
    masks["ordinary"] = ~any_context & ~df.calendar_any_special_day.to_numpy(bool)
    return masks
