"""Independent brute-force checks of pooled F1 and precision constraints."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from surge_thresholds import apply_policy, global_policy, joint_policy, select_policy, summary


class ThresholdTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(7)
        n = 120
        return pd.DataFrame(dict(y=rng.random(n) < .3, score=rng.integers(0, 10, n) / 10,
                                 previous_surge=np.tile([0., 0., 1., np.nan], n // 4),
                                 split="early_stop", hour=pd.date_range("2023-01-01", periods=n, freq="h")))

    def test_joint_matches_bruteforce_pooled_optimum_with_ties(self):
        df = self.data()
        base = global_policy(df)
        floor = summary(df.y, apply_policy(df, base)[0])["precision"]
        best_f1 = -1
        cuts = [*np.unique(df.score), 1.000001]
        for a in cuts:
            for b in cuts:
                p = dict(base, onset=a, continuation=b)
                m = summary(df.y, apply_policy(df, p)[0])
                if m["predicted_positives"] and m["precision"] >= floor - 1e-12:
                    best_f1 = max(best_f1, m["f1"])
        selected = joint_policy(df, grid_size=201)
        m = summary(df.y, apply_policy(df, selected)[0])
        self.assertAlmostEqual(m["f1"], best_f1)
        self.assertGreaterEqual(m["precision"] + 1e-12, floor)
        self.assertGreaterEqual(m["f1"] + 1e-12, summary(df.y, apply_policy(df, base)[0])["f1"])

    def test_explicit_precision_floor_and_infeasibility(self):
        df = self.data()
        df["score"] = np.where(df.y, .8, .2)
        p = joint_policy(df, min_precision=.9)
        self.assertGreaterEqual(summary(df.y, apply_policy(df, p)[0])["precision"], .9)
        df["score"] = np.where(df.y, .2, .8)
        with self.assertRaises(ValueError):
            global_policy(df, min_precision=.9)

    def test_auto_rejects_a_rule_that_hurts_later_validation_precision(self):
        df = self.data()
        df["score"] = np.where(df.y, .8, .2)
        harmful = dict(mode="joint", global_cutoff=.8, onset=0., continuation=0.,
                       unknown_history=.8, min_precision=0.)
        with patch("surge_thresholds.joint_policy", return_value=harmful):
            p, audit = select_policy(df, mode="auto")
        self.assertEqual(p["mode"], "global")
        self.assertEqual(set(audit.stage), {"fit", "check", "refit"})

    def test_calibrate_and_test_rows_cannot_fit_thresholds(self):
        df = self.data()
        for split in ("calibrate", "test"):
            with self.assertRaises(ValueError):
                select_policy(df.assign(split=split))

    def test_unknown_history_uses_global_fallback(self):
        df = self.data()
        p = dict(global_policy(df), onset=.1, continuation=.9)
        calls, thresholds = apply_policy(df, p)
        unknown = df.previous_surge.isna()
        np.testing.assert_array_equal(thresholds[unknown], p["global_cutoff"])
        np.testing.assert_array_equal(calls[unknown], df.loc[unknown, "score"] >= p["global_cutoff"])


if __name__ == "__main__":
    unittest.main()
