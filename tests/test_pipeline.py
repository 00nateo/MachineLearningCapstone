"""Offline conversion/aggregation integration checks for the pipeline PRs."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import geopandas as gpd
import pandas as pd
from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aggregate
import convert
import export


class PipelineTests(unittest.TestCase):
    def test_adjacency_uses_the_documented_twenty_meter_tolerance(self):
        # Zones 1/2 have a 10 m gap; zones 2/3 have a 30 m gap.
        # Buffering both sides by 20 m incorrectly connects zones 2 and 3.
        zones = gpd.GeoDataFrame({"zone": [1, 2, 3]}, geometry=[
            box(500000, 4500000, 500100, 4500100),
            box(500110, 4500000, 500210, 4500100),
            box(500240, 4500000, 500340, 4500100),
        ], crs=export.UTM["nyc"])
        pairs = export.adjacency(zones, "nyc")
        self.assertEqual(set(zip(pairs.zone, pairs.neighbor)), {(1, 2), (2, 1)})

    def convert_frame(self, con, frame, root, month, city):
        if city == "nyc":
            raw = root / f"source_{month}.parquet"
            frame.to_parquet(raw, index=False)
            source = f"read_parquet('{raw}')"
            out = root / f"fhvhv_{month}.parquet"
        else:
            raw = root / f"source_{month}.csv"
            frame.to_csv(raw, index=False)
            source = f"read_csv('{raw}', header=true, all_varchar=true)"
            out = root / f"tnp_trips_{month}.parquet"
        config = convert.CITIES[city]
        select, missing = convert.select_list(con, source, config["cols"], config["cast"])
        convert.write_parquet(con, f"SELECT {select} FROM {source}", out)
        return out, missing

    def test_nyc_cross_month_dropoffs_medians_and_stray_records(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            january = pd.DataFrame(dict(
                pickup_datetime=pd.to_datetime(["2024-01-31 23:30", "2024-01-31 23:50", "2024-02-01 00:15"]),
                dropoff_datetime=pd.to_datetime(["2024-02-01 00:10", "2024-02-01 00:20", "2024-02-01 00:30"]),
                request_datetime=pd.to_datetime(["2024-01-31 23:27", "2024-01-31 23:43", "2024-02-01 00:10"]),
                PULocationID=[1, 1, 1], DOLocationID=[2, 2, 2], trip_time=[2400, 1800, 900],
                base_passenger_fare=[10., 20., 99.], tips=[2., 4., 99.], driver_pay=[8., 14., 99.], trip_miles=[2., 4., 99.]))
            february = pd.DataFrame(dict(
                pickup_datetime=pd.to_datetime(["2024-02-01 00:05", "2024-02-01 01:10"]),
                dropoff_datetime=pd.to_datetime(["2024-02-01 00:35", "2024-02-01 01:40"]),
                request_datetime=pd.to_datetime(["2024-02-01 00:00", "2024-02-01 01:05"]),
                PULocationID=[1, 1], DOLocationID=[2, 2], trip_time=[1800, 1800],
                base_passenger_fare=[15., 15.], tips=[1., 1.], driver_pay=[10., 10.], trip_miles=[3., 3.]))
            jan, missing = self.convert_frame(con, january, root, "2024-01", "nyc")
            feb, _ = self.convert_frame(con, february, root, "2024-02", "nyc")
            self.assertIn("cbd_congestion_fee", missing)
            self.assertTrue(pd.read_parquet(jan).cbd_congestion_fee.isna().all())
            with patch.object(aggregate, "PROC", root):
                aggregate.build_panels(con, "nyc", aggregate.CITIES["nyc"], [jan, feb])
                aggregate.build_drive_times(con, "nyc", aggregate.CITIES["nyc"], [jan, feb])
            panel = pd.read_parquet(root / "nyc_zone_hour.parquet").set_index(["hour", "zone"])
            self.assertEqual(int(panel.pickups.sum()), 4)  # wrong-file-month row excluded
            self.assertEqual(int(panel.dropoffs.sum()), 4)
            early = panel.loc[(pd.Timestamp("2024-02-01 00:00"), 2)]
            self.assertEqual(early.dropoffs, 3)  # two Jan arrivals plus one Feb arrival
            self.assertEqual(early.pickups, 0)
            medians = panel.loc[(pd.Timestamp("2024-01-31 23:00"), 1)]
            self.assertEqual(medians.median_wait_s, 300.)
            self.assertEqual(medians.median_fare, 18.)
            self.assertEqual(medians.median_driver_pay, 11.)
            self.assertEqual(medians.median_trip_seconds, 2100.)
            self.assertFalse(panel.index.has_duplicates)
            drives = pd.read_parquet(root / "nyc_drive_times.parquet")
            self.assertEqual(int(drives.n_trips.sum()), 4)

    def test_chicago_missing_columns_and_unzoned_pickups(self):
        with tempfile.TemporaryDirectory() as tmp, duckdb.connect() as con, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            frame = pd.DataFrame(dict(trip_start_timestamp=["2024-01-01 00:00:00"] * 2,
                trip_end_timestamp=["2024-01-01 00:30:00"] * 2, trip_seconds=[1800, 1800],
                pickup_community_area=["1", ""], dropoff_community_area=[2, 2],
                fare=[10., 20.], tip=[2., 3.], trip_miles=[3., 4.]))
            source, missing = self.convert_frame(con, frame, root, "2024-01", "chicago")
            self.assertIn("percent_time_chicago", missing)
            with patch.object(aggregate, "PROC", root):
                aggregate.build_panels(con, "chicago", aggregate.CITIES["chicago"], [source])
            panel = pd.read_parquet(root / "chicago_zone_hour.parquet")
            self.assertEqual(len(panel), 77)
            self.assertEqual(int(panel.pickups.sum()), 1)
            self.assertEqual(int(panel.dropoffs.sum()), 2)
            self.assertTrue(panel.median_wait_s.isna().all())
            self.assertTrue(panel.median_driver_pay.isna().all())
            city = pd.read_parquet(root / "chicago_city_hour.parquet").iloc[0]
            self.assertEqual(city.pickups_total, 2)
            self.assertEqual(city.pickups_unzoned, 1)


if __name__ == "__main__":
    unittest.main()
