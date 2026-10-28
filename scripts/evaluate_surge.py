#!/usr/bin/env python
"""Evaluate demand-surge onset versus continuation from saved model predictions.

Onset risk set: previous_surge == 0; positives are newly starting surges.
Continuation risk set: previous_surge == 1; negatives are surges that ended.
Both sets contain negatives, so false alarms, precision and F1 are meaningful.
Unknown prior labels or local clock-hour gaps form a separate group. This evaluator
does not fit thresholds: it reuses saved calls from the global or jointly optimized
validation-selected policy. Neither calibrate nor test labels choose thresholds.
Test rows are never scored. This is subgroup analysis of a surge classifier,
not a separately trained onset classifier or a guaranteed early-warning system.

Usage: python scripts/evaluate_surge.py --city nyc --model lightgbm --variant deployable
       python scripts/evaluate_surge.py --city nyc --model torch --run-name improved
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from baseline import PROC_ML, RESULTS
from evaluate import CONDITIONS
from model_data import N_ZONES
from surge_data import attach_history, tag


def phase_masks(df):
    previous = df["previous_surge"]
    return {"all": np.ones(len(df), dtype=bool),
            "onset": previous.eq(0).fillna(False).to_numpy(dtype=bool),
            "continuation": previous.eq(1).fillna(False).to_numpy(dtype=bool),
            "unknown_history": previous.isna().to_numpy()}


def scores(y, score, call):
    y, call = np.asarray(y, dtype=bool), np.asarray(call, dtype=bool)
    tp, fp = int((y & call).sum()), int((~y & call).sum())
    fn, tn = int((y & ~call).sum()), int((~y & ~call).sum())
    return dict(n=len(y), positives=tp + fn, negatives=fp + tn,
                surge_rate=float(y.mean()) if len(y) else np.nan,
                auc_pr=average_precision_score(y, score) if y.any() else np.nan,
                precision=tp / (tp + fp) if tp + fp else np.nan,
                recall=tp / (tp + fn) if tp + fn else np.nan,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan,
                tp=tp, fp=fp, fn=fn, tn=tn,
                false_alarms_per_1000_negative_hours=1000 * fp / (fp + tn) if fp + tn else np.nan)


def report(df, model, cutoff=None):
    """Use saved calls, or one supplied validation-selected cutoff; never tune here."""
    rows = []
    for split in ("early_stop", "calibrate"):
        d = df[df["split"] == split]
        if d.empty:
            continue
        if d["y"].isna().any() or not np.isfinite(d["score"]).all():
            raise ValueError("Evaluation requires known labels and finite scores")
        calls = d["call"].to_numpy(bool) if cutoff is None else d["score"].to_numpy() >= cutoff
        for phase, phase_mask in phase_masks(d).items():
            for condition, flag in {"all": "ALL", **CONDITIONS}.items():
                condition_mask = (np.ones(len(d), bool) if flag == "ALL" else
                                  (~d["extreme_flag"] if flag is None else d[flag]).to_numpy(bool))
                mask = phase_mask & condition_mask
                if not mask.any():
                    continue
                used_cutoff = cutoff
                if used_cutoff is None and "decision_threshold" in d:
                    unique = d.loc[mask, "decision_threshold"].unique()
                    used_cutoff = float(unique[0]) if len(unique) == 1 else None
                rows.append(dict(model=model, split=split, phase=phase, condition=condition,
                                 cutoff=np.nan if used_cutoff is None else used_cutoff,
                                 **scores(d["y"].to_numpy()[mask], d["score"].to_numpy()[mask], calls[mask])))
    return pd.DataFrame(rows)


def saved_report(con, city, fold, model, variant, data_dir=PROC_ML, run_name=""):
    path = Path(data_dir) / f"{tag(city, fold, model, variant, run_name)}_pred.parquet"
    pred = pd.read_parquet(path, filters=[("split", "in", ["early_stop", "calibrate"])])
    pred = attach_history(con, pred, city, data_dir)
    return report(pred, f"{model}_{variant}")


def write_reports(result, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    result[result["phase"] == "all"].to_csv(f"{stem}_metrics.csv", index=False)
    result[result["phase"] != "all"].to_csv(f"{stem}_phases.csv", index=False)
    cal = result[(result["split"] == "calibrate") & (result["condition"] == "all")]
    validation = result[(result["split"] == "early_stop") & (result["condition"] == "all") &
                        (result["phase"] == "all")]
    cols = ["model", "phase", "n", "positives", "auc_pr", "precision", "recall", "f1"]
    print("\nEarly-stop validation (use this split for model selection):\n"
          + validation[["model", "auc_pr", "precision", "recall", "f1"]].round(4).to_string(index=False)
          + "\n\nCalibration split: overall and by prior state (reporting only):\n"
          + cal[cols].round(4).to_string(index=False)
          + f"\n-> {stem}_metrics.csv\n-> {stem}_phases.csv", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", choices=list(N_ZONES), default=["nyc"])
    ap.add_argument("--fold", nargs="+", choices=["fold1", "fold2", "fold3"], default=["fold3"])
    ap.add_argument("--model", choices=["lightgbm", "torch"], default="torch")
    ap.add_argument("--variant", choices=["full", "deployable"], default="full")
    ap.add_argument("--run-name", default="")
    ap.add_argument("--data-dir", type=Path, default=PROC_ML)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    with duckdb.connect() as con:
        for city in args.city:
            for fold in args.fold:
                result = saved_report(con, city, fold, args.model, args.variant, args.data_dir, args.run_name)
                write_reports(result, args.results_dir / f"{tag(city, fold, args.model, args.variant, args.run_name)}_analysis")


if __name__ == "__main__":
    main()
