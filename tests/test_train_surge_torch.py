"""Surge history, onset risk sets, target isolation and CPU/MPS integration."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_surge
import model_data
import surge_data
import train_surge_torch as trainer
from train_torch import predict, restore
from surge_thresholds import apply_policy
import test_train_torch as hotspot_tests


class SurgeTrainingTests(unittest.TestCase):
    def fixture(self, root):
        df = hotspot_tests.TorchTrainingTests().fixture(root)
        y = (df.hour.dt.hour % 4).isin([1, 2]) & (df.zone <= 4)
        df["event_flag"] = y.astype(float)  # toy known-in-advance schedule signal
        surge = df[["zone", "hour"]].copy()
        surge["baseline"] = 20.
        surge["surge_150"] = y.astype("boolean")
        surge["ratio"] = np.where(y, 1.8, 0.8)
        df.to_parquet(root / "nyc_features.parquet", index=False)
        surge.to_parquet(root / "nyc_surge.parquet", index=False)
        return df, surge

    def test_phase_metrics_include_negative_opportunities(self):
        df = pd.DataFrame(dict(split=["calibrate"] * 7, previous_surge=[0, 0, 0, 1, 1, 1, np.nan],
                               y=[1, 0, 0, 1, 0, 1, 1], score=[.9, .8, .1, .9, .8, .1, .9]))
        for flag in model_data.FLAGS:
            df[flag] = False
        rows = evaluate_surge.report(df, "toy", .5)
        rows = rows[rows.condition == "all"].set_index("phase")
        self.assertEqual(rows.loc["onset", "n"], 3)
        self.assertEqual(rows.loc["onset", "fp"], 1)
        self.assertAlmostEqual(rows.loc["onset", "precision"], .5)
        self.assertAlmostEqual(rows.loc["onset", "recall"], 1.)
        self.assertAlmostEqual(rows.loc["continuation", "f1"], .5)
        self.assertEqual(rows.loc["unknown_history", "n"], 1)
        self.assertEqual(rows.drop(index="all").n.sum(), rows.loc["all", "n"])

    def test_history_precedes_eligibility_sampling_and_gap_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features, surge = self.fixture(root)
            t = pd.Timestamp("2019-02-01")
            low = (surge.zone == 1) & (surge.hour == t + pd.Timedelta(hours=2))
            surge.loc[low, "baseline"] = 5.
            surge.loc[low, "surge_150"] = False
            missing_baseline = (surge.zone == 1) & (surge.hour == t)
            surge.loc[missing_baseline, "baseline"] = np.nan
            surge.loc[missing_baseline, "surge_150"] = pd.NA
            # A genuinely absent row creates a two-hour gap for zone 1 at h=5.
            gap = (features.zone == 1) & (features.hour == t + pd.Timedelta(hours=4))
            features = features[~gap]
            surge = surge[~gap]
            features.to_parquet(root / "nyc_features.parquet", index=False)
            surge.to_parquet(root / "nyc_surge.parquet", index=False)
            sample = pd.read_parquet(root / "nyc_fold3_train.parquet")
            sample = sample[~((sample.zone == 2) & (sample.hour == t + pd.Timedelta(hours=1)))]
            sample.to_parquet(root / "nyc_fold3_train.parquet", index=False)
            with duckdb.connect() as con:
                train, _ = surge_data.load(con, "nyc", "fold3", root)
            def prior(zone, hour):
                return train.loc[(train.zone == zone) & (train.hour == t + pd.Timedelta(hours=hour)), "previous_surge"].iloc[0]
            self.assertEqual(prior(1, 3), 0.)  # uses excluded low-volume hour, not h=1
            self.assertEqual(prior(2, 2), 1.)  # uses unsampled hour h=1
            self.assertTrue(pd.isna(prior(1, 1)))  # missing baseline/label
            self.assertTrue(pd.isna(prior(1, 5)))  # genuine clock-hour gap

    def roundtrip(self, device, variant, cached=False, features="basic", threshold_mode="global"):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            self.fixture(root)
            args = trainer.parser().parse_args([
                "--device", device, "--variant", variant, "--epochs", "8", "--patience", "4",
                "--hidden", "16", "8", "--lr", ".02", "--dropout", "0", "--threads", "1",
                "--batch-size", "128", "--eval-batch-size", "64", "--run-name", "test",
                "--data-dir", tmp, "--model-dir", tmp, "--results-dir", tmp,
                "--features", features, "--threshold-mode", threshold_mode,
            ] + (["--cache-device"] if cached else []))
            trainer.train("nyc", "fold3", args)
            tag = surge_data.tag("nyc", "fold3", "torch", variant, "test")
            model, prep, checkpoint = restore(root / f"{tag}.pt")
            self.assertEqual(checkpoint["target"], "y")
            self.assertEqual(checkpoint["label"], "surge_150")
            self.assertTrue(set(surge_data.SURGE_INPUTS).issubset(prep.columns))
            self.assertFalse({"y", "previous_surge", "ratio", "hotspot"}.intersection(prep.columns))
            if variant == "deployable":
                self.assertNotIn("event_attendance", prep.columns)
            pred = pd.read_parquet(root / f"{tag}_pred.parquet")
            self.assertEqual(set(pred.split), {"early_stop", "calibrate", "test"})
            es = pred[pred.split == "early_stop"]
            self.assertEqual(checkpoint["cutoff"], surge_data.best_cutoff(es.y, es.score))
            np.testing.assert_array_equal(pred.call, apply_policy(pred, checkpoint["threshold_policy"])[0])
            with duckdb.connect() as con:
                _, held = surge_data.load(con, "nyc", "fold3", root, enhanced=features == "enhanced")
                old_pred = pred.drop(columns=["previous_hour", "previous_surge"])
                enriched = surge_data.attach_history(con, old_pred, "nyc", root)
            np.testing.assert_allclose(enriched.previous_surge, pred.previous_surge, equal_nan=True)
            np.testing.assert_allclose(predict(model, held, prep, 64, torch.device("cpu")), pred.score, atol=1e-5)
            metrics = pd.read_csv(root / f"{tag}_metrics.csv")
            phases = pd.read_csv(root / f"{tag}_phases.csv")
            self.assertNotIn("test", set(metrics.split) | set(phases.split))
            self.assertTrue({"onset", "continuation"}.issubset(phases.phase))
            persistence_onset = phases[(phases.model == "persistence_label") & (phases.phase == "onset")]
            self.assertTrue((persistence_onset.tp == 0).all())
            self.assertTrue((persistence_onset.fp == 0).all())

    def test_cpu_full_surge_training(self):
        self.roundtrip("cpu", "full")

    def test_cpu_deployable_surge_training(self):
        self.roundtrip("cpu", "deployable")

    def test_cpu_joint_policy_checkpoint_roundtrip(self):
        self.roundtrip("cpu", "deployable", features="enhanced", threshold_mode="joint")

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable")
    def test_mps_surge_training(self):
        self.roundtrip("mps", "full", cached=True, features="enhanced", threshold_mode="auto")

    def test_enhanced_features_are_past_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            features, surge = self.fixture(root)
            t = pd.Timestamp("2019-02-01 06:00")
            with duckdb.connect() as con:
                before, _ = surge_data.load(con, "nyc", "fold3", root, enhanced=True)
            # Change current and future observations/labels; h's new predictors must not change.
            changed = (features.zone == 1) & (features.hour >= t)
            features.loc[changed, "pickups"] = 1e6
            surge.loc[changed, "ratio"] = 100.
            surge.loc[changed, "surge_150"] = True
            features.to_parquet(root / "nyc_features.parquet", index=False)
            surge.to_parquet(root / "nyc_surge.parquet", index=False)
            with duckdb.connect() as con:
                after, _ = surge_data.load(con, "nyc", "fold3", root, enhanced=True)
            a = before.loc[(before.zone == 1) & (before.hour == t), surge_data.ENHANCED_INPUTS]
            b = after.loc[(after.zone == 1) & (after.hour == t), surge_data.ENHANCED_INPUTS]
            np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


if __name__ == "__main__":
    unittest.main()
