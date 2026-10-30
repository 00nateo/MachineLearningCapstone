"""Leakage, artifact round-trip and CPU/MPS training integration checks.

Run: .venv-torch/bin/python -m unittest discover -s tests -v
"""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import duckdb
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import conformal
import evaluate
import model_data
import train_torch as trainer


class TorchTrainingTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.default_rng(123)
        hours = list(pd.date_range("2019-02-01", periods=48, freq="h"))
        hours += list(pd.date_range("2023-01-01", periods=8, freq="h"))
        hours += list(pd.date_range("2024-01-01", periods=8, freq="h"))
        hours += list(pd.date_range("2024-07-01", periods=8, freq="h"))
        hours += list(pd.date_range("2025-01-01", periods=8, freq="h"))
        df = pd.DataFrame([(zone, h) for h in hours for zone in range(1, 17)], columns=["zone", "hour"])
        df["hotspot"] = df["zone"] <= 2
        df["pickups"] = df["hotspot"].astype(int) * 100
        for col in model_data.TABLE_FEATURES + model_data.CITY_FEATURES["nyc"] + model_data.DEPLOYABLE_ONLY:
            df[col] = df["hotspot"].astype(float) * 5 + rng.normal(0, 0.2, len(df))
        df.loc[df.index % 7 == 0, "wx_temp_f_lag1"] = np.nan
        for i, col in enumerate(model_data.FLAGS[1:]):
            df[col] = (df.index + i) % 7 == 0
        df["extreme_flag"] = df[model_data.FLAGS[1:]].any(axis=1)
        df["fold3"] = df["hour"].dt.year.map({2019: "train", 2023: "early_stop", 2024: "calibrate", 2025: "test"})
        df.to_parquet(root / "nyc_features.parquet", index=False)
        train = df[df["fold3"] == "train"].assign(weight=lambda d: np.where(d["extreme_flag"], 3., 1.))
        train.to_parquet(root / "nyc_fold3_train.parquet", index=False)
        return df

    def test_preprocessing_uses_training_statistics_and_excludes_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = self.fixture(Path(tmp))
            train = model_data.add_calendar(df[df["fold3"] == "train"].copy(), "nyc")
            train["wx_vis_mi_lag1"] = np.nan
            prep = trainer.Preprocessor.fit(train, "nyc", "full")
            _, x = prep.transform(train)
            col = prep.columns.index("wx_vis_mi_lag1")
            np.testing.assert_array_equal(x[:, col], 0.)
            np.testing.assert_array_equal(x[:, len(prep.columns) + col], 1.)
            mean = prep.mean.copy()
            future = train.iloc[:2].copy()
            future["pickup_lag1"] = 1e6
            future["wx_vis_mi_lag1"] = 1e6
            future_x = prep.transform(future)[1]
            np.testing.assert_array_equal(future_x[:, col], 0.)
            np.testing.assert_array_equal(future_x[:, len(prep.columns) + col], 1.)
            np.testing.assert_array_equal(prep.mean, mean)
            changed_labels = train.assign(hotspot=~train["hotspot"], pickups=1e9, extreme_flag=False)
            np.testing.assert_array_equal(prep.transform(changed_labels)[1], x)
            self.assertTrue(np.isfinite(x).all())
            for variant in ("full", "deployable"):
                forbidden = {"pickups", "hotspot", "fold3", *model_data.FLAGS}
                self.assertFalse(forbidden.intersection(model_data.inputs("nyc", variant)))

    def test_artifact_names_preserve_existing_lightgbm_paths(self):
        self.assertEqual(model_data.artifact_tag("nyc", "fold3"), "nyc_fold3")
        self.assertEqual(model_data.artifact_tag("nyc", "fold3", variant="deployable"), "nyc_fold3_deployable")
        self.assertEqual(model_data.artifact_tag("nyc", "fold3", "torch", "deployable"), "nyc_fold3_torch_deployable")
        self.assertEqual(len(model_data.inputs("nyc", "full")), 23)
        self.assertEqual(len(model_data.inputs("nyc", "deployable")), 21)

    def training_roundtrip(self, device, variant, cached=False):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            df = self.fixture(root)
            argv = ["--device", device, "--variant", variant, "--epochs", "12", "--patience", "4",
                    "--hidden", "16", "8", "--lr", "0.02", "--dropout", "0", "--threads", "1",
                    "--batch-size", "128", "--eval-batch-size", "64", "--data-dir", str(root),
                    "--model-dir", str(root), "--results-dir", str(root)]
            if cached:
                argv.append("--cache-device")
            args = trainer.parser().parse_args(argv)
            trainer.train("nyc", "fold3", args)
            tag = model_data.artifact_tag("nyc", "fold3", "torch", variant)
            model, prep, checkpoint = trainer.restore(root / f"{tag}.pt")
            self.assertLess(checkpoint["best_logloss"], checkpoint["initial_logloss"])
            saved = pd.read_parquet(root / f"{tag}_pred.parquet")
            self.assertEqual(set(saved["split"]), {"early_stop", "calibrate", "test"})
            self.assertTrue(saved["score"].between(0, 1).all())
            self.assertFalse(saved.duplicated(["zone", "hour"]).any())
            raw = df[df["fold3"] != "train"].sort_values(["hour", "zone"]).reset_index(drop=True)
            raw = model_data.add_calendar(raw, "nyc")
            reloaded = trainer.predict(model, raw, prep, 64, torch.device("cpu"))
            np.testing.assert_allclose(reloaded, saved["score"], atol=1e-5)
            result = pd.read_csv(root / f"{tag}_model.csv")
            self.assertEqual(set(result["split"]), {"early_stop", "calibrate"})
            self.assertEqual(set(result["model"]), {model_data.model_name("torch", variant)})
            if variant == "deployable":
                self.assertNotIn("event_attendance", prep.columns)
                self.assertNotIn("subway_entries_lag1", prep.columns)
                self.assertIn("expected_attendance", prep.columns)
            # Exercise the real downstream loaders, bootstrap comparisons and threshold fitting.
            preds = saved[saved["split"].isin(["early_stop", "calibrate"])]
            pd.concat([preds[["zone", "hour", "split", "score", "call"]].assign(model=name)
                       for name in evaluate.BASELINES]).to_parquet(root / "nyc_fold3_baseline_pred.parquet", index=False)
            with patch.object(evaluate, "PROC_ML", root), patch.object(evaluate, "RESULTS", root):
                evaluate.evaluate("nyc", "fold3", "calibrate", 10, "torch", variant)
            with patch.object(conformal, "PROC_ML", root), patch.object(conformal, "RESULTS", root):
                conformal.run("nyc", "fold3", variant, "global", "2024-07-01", "torch")
            self.assertTrue((root / f"{tag}_eval_ci.csv").exists())
            self.assertTrue((root / f"{tag}_abstention_global.csv").exists())

    def test_cpu_full_end_to_end(self):
        self.training_roundtrip("cpu", "full")

    def test_cpu_deployable_end_to_end(self):
        self.training_roundtrip("cpu", "deployable")

    @unittest.skipUnless(torch.backends.mps.is_available(), "Apple MPS is unavailable")
    def test_mps_cached_training_and_cpu_checkpoint_reload(self):
        self.training_roundtrip("mps", "full", cached=True)


if __name__ == "__main__":
    unittest.main()
