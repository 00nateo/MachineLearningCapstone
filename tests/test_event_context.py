"""Calendar accuracy, off-venue masks, and trip aggregation semantics."""
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aggregate_event_trips import aggregate_file, control_intervals, exposures, merged_intervals
from calendar_flags import FLAG_COLUMNS, annotate_features, build_calendar
from event_context import CONTEXT_FEATURES, TIMED_FEATURES, attach, categorize, make_windows
from evaluate_event_context import metrics


def event(start="2024-01-02 19:30", end="2024-01-02 21:30", zone=1, event_id="one", league="nba", source="espn"):
    return dict(metro="nyc", source=source, league=league, event_id=event_id, zone=zone,
                venue_name="Test Arena", start_local=pd.Timestamp(start), end_local_est=pd.Timestamp(end))


class CalendarTests(unittest.TestCase):
    def test_dates_observance_and_cancellations(self):
        d, events = build_calendar("nyc", "2019-01-01", "2026-12-31")
        d = d.set_index(pd.to_datetime(d.date))
        self.assertTrue(d.loc["2024-11-28", "calendar_thanksgiving"])
        self.assertTrue(d.loc["2021-12-31", "calendar_new_years_day"])
        self.assertTrue(d.loc["2021-12-31", "calendar_federal_holiday_observed"])
        self.assertFalse(d.loc["2021-12-31", "calendar_federal_holiday_actual"])
        self.assertTrue(d.loc["2021-06-18", "calendar_juneteenth"])
        self.assertFalse(d.loc["2020-06-19", "calendar_juneteenth"])
        self.assertTrue(d.loc["2024-11-03", "calendar_nyc_marathon"])
        self.assertFalse(d.loc["2020-11-01", "calendar_nyc_marathon"])
        self.assertTrue((events.status == "canceled").any())
        self.assertTrue(d.loc["2024-11-03", "calendar_dst_fall"])
        self.assertTrue(d.loc["2022-11-08", "calendar_general_election"])
        self.assertFalse(d.loc["2022-11-01", "calendar_general_election"])
        self.assertTrue(d.loc["2023-11-07", "calendar_local_election"])
        chi, _ = build_calendar("chicago", "2023-01-01", "2023-12-31")
        chi = chi.set_index(pd.to_datetime(chi.date))
        self.assertTrue(chi.loc["2023-04-04", "calendar_local_election"])
        self.assertFalse(chi.loc["2023-11-07", "calendar_general_election"])
        self.assertFalse(chi.calendar_nyc_marathon.any())

    def test_annotation_preserves_rows_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.parquet"
            original = pd.DataFrame(dict(zone=[1, 2, 1], hour=pd.to_datetime(["2024-10-31", "2024-10-31", "2024-11-01"]), y=[1, 0, 0]))
            original.to_parquet(path, index=False)
            days, _ = build_calendar("nyc", "2024-10-31", "2024-11-01")
            annotate_features(path, days)
            first = pd.read_parquet(path)
            pd.testing.assert_frame_equal(first[original.columns], original)
            annotate_features(path, days)
            pd.testing.assert_frame_equal(pd.read_parquet(path), first)


class ContextTests(unittest.TestCase):
    def test_half_open_hour_overlap_midnight_and_duplicates(self):
        raw = pd.DataFrame([event(start="2024-01-02 23:30", end="2024-01-03 01:00")] * 2)
        events = categorize(raw, "nyc")
        self.assertEqual(len(events), 1)
        windows, excluded = make_windows(events, 3, 3)
        after = windows[windows.phase == "after"]
        self.assertEqual(set(after.hour), set(pd.date_range("2024-01-03 01:00", periods=3, freq="h")))
        self.assertIn(pd.Timestamp("2024-01-03 03:00"), set(excluded.hour))
        self.assertNotIn(pd.Timestamp("2024-01-03 04:00"), set(excluded.hour))
        # A 23:30 start contributes to both before and during in the 23h bin.
        self.assertEqual(set(windows[windows.hour == pd.Timestamp("2024-01-02 23:00")].phase), {"before", "during"})

    def test_attachment_masks_target_zone_only_and_preserves_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            hours = pd.date_range("2024-10-31", periods=2, freq="h")
            context = pd.DataFrame(dict(hour=hours))
            for c in CONTEXT_FEATURES:
                context[c] = 0
            context.to_parquet(root / "nyc_event_context.parquet", index=False)
            pd.DataFrame(dict(hour=[hours[0]], zone=[1])).to_parquet(root / "nyc_event_exclusions.parquet", index=False)
            data = pd.DataFrame(dict(hour=[hours[1], hours[0], hours[0]], zone=[1, 2, 1], y=[0, 1, 1]))
            out = attach(data, "nyc", root, off_venue=True)
            self.assertEqual(list(out.zone), [1, 2])
            self.assertEqual(list(out.y), [0, 1])
            self.assertFalse(out.venue_excluded.any())

    def test_frozen_metrics_keep_negatives_and_never_score_test(self):
        d = pd.DataFrame(dict(zone=[1, 2, 3, 4], hour=pd.to_datetime(["2024-01-01"]*4),
                              split=["calibrate"]*3+["test"], y=[1, 0, 1, 1], score=[.9, .8, .9, .9],
                              call=[1, 1, 1, 1], venue_excluded=[False, False, True, False]))
        for c in CONTEXT_FEATURES:
            d[c] = 0
        d["context_sports_before"] = 1
        result = metrics(d, "toy")
        row = result[(result.scope == "off_venue") & (result.group == "sports")].iloc[0]
        self.assertEqual(row.n, 2)
        self.assertEqual(row.fp, 1)
        self.assertEqual(row.precision, .5)
        self.assertEqual(set(result.split), {"calibrate"})


class TripAggregationTests(unittest.TestCase):
    def test_cli_writes_small_shards_and_completed_manifest(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame([event()]).to_parquet(root / "events.parquet", index=False)
            pd.DataFrame(dict(pickup_datetime=pd.to_datetime(["2024-01-02 18:00"]),
                              dropoff_datetime=pd.to_datetime(["2024-01-02 18:30"]),
                              PULocationID=[2], DOLocationID=[1], trip_time=[1800])).to_parquet(root / "fhvhv_2024-01.parquet")
            script = Path(__file__).resolve().parents[1] / "scripts" / "aggregate_event_trips.py"
            result = subprocess.run([sys.executable, str(script), "--city", "nyc", "--trip-glob", str(root / "fhvhv_*.parquet"),
                "--events", str(root / "events.parquet"), "--out", str(root / "out"), "--threads", "1", "--memory-limit", "128MB"],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((root / "out" / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(len(manifest["processed_files"]), 1)
            counts = pd.read_parquet(root / "out" / "part-2024-01.parquet")
            self.assertEqual(counts.trips.sum(), 1)
            self.assertNotIn("PULocationID", counts)

    def test_overlaps_controls_remote_direction_and_exclusions(self):
        # Two overlapping games at the same venue must not duplicate a trip.
        events = categorize(pd.DataFrame([event(), event(event_id="two")]), "nyc")
        windows = control_intervals(merged_intervals(events, 3, 3), [1])
        self.assertEqual(len(windows), 8)
        _, excluded = make_windows(events, 3, 3)
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con:
            path = Path(tmp) / "fhvhv_2024-01.parquet"
            trips = pd.DataFrame(dict(
                pickup_datetime=pd.to_datetime(["2024-01-02 18:00", "2024-01-02 22:00", "2024-01-02 18:00", "2024-01-02 18:00"]),
                dropoff_datetime=pd.to_datetime(["2024-01-02 18:30", "2024-01-02 22:30", "2024-01-02 18:30", "2024-01-02 18:30"]),
                PULocationID=[2, 1, 1, 264], DOLocationID=[1, 3, 1, 1], trip_time=[1800]*4))
            trips.to_parquet(path, index=False)
            result = aggregate_file(con, path, "nyc", windows, excluded, "2024-01")
        self.assertEqual(result.trips.sum(), 2)
        arrival = result[result.direction == "arrival"].iloc[0]
        departure = result[result.direction == "departure"].iloc[0]
        self.assertEqual(arrival.remote_zone, 2)
        self.assertEqual(arrival.phase, "before")
        self.assertEqual(departure.remote_zone, 3)
        self.assertEqual(departure.phase, "after")
        # Dec control windows are marked missing, rather than turned into zeros.
        support = exposures(windows, {"2024-01"})
        self.assertFalse(support[support.period == "control_week_1"].required_month_files_present.any())
        self.assertTrue(support[support.period == "event"].required_month_files_present.all())

    def test_cross_month_arrival_and_shifted_reference_hour(self):
        events = categorize(pd.DataFrame([event(start="2024-02-08 01:00", end="2024-02-08 03:00")]), "nyc")
        windows = control_intervals(merged_intervals(events, 3, 3), [1])
        excluded = pd.DataFrame(dict(hour=pd.Series(dtype="datetime64[ns]"), zone=pd.Series(dtype="int64")))
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con:
            path = Path(tmp) / "fhvhv_2024-01.parquet"
            pd.DataFrame(dict(pickup_datetime=pd.to_datetime(["2024-01-31 23:45"]),
                              dropoff_datetime=pd.to_datetime(["2024-02-01 00:15"]),
                              PULocationID=[2], DOLocationID=[1], trip_time=[1800])).to_parquet(path)
            result = aggregate_file(con, path, "nyc", windows, excluded, "2024-01")
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].reference_hour, pd.Timestamp("2024-02-08 00:00"))
        self.assertEqual(result.iloc[0].period, "control_week_1")
        support = exposures(windows, {"2024-02"})
        row = support[(support.observed_hour == pd.Timestamp("2024-02-01")) & (support.direction == "arrival")]
        self.assertFalse(row.required_month_files_present.any())


if __name__ == "__main__":
    unittest.main()
