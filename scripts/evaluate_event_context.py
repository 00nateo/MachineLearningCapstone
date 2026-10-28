#!/usr/bin/env python
"""Compare frozen surge predictions on event categories and named days.

Reuses saved scores/calls, scores early_stop/calibrate only, and includes negative
zone-hours. Reports both all zones and off-venue zones with identical masks for
each model. Groups overlap; dates/hours reveal limited independent event support.
No thresholds are fitted. Dropoff summaries are descriptive, not OD flows.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from calendar_flags import FLAG_COLUMNS
from event_context import PROC_ML, ROOT, attach, context_masks
from evaluate_surge import scores
from surge_data import tag


def metrics(pred, model):
    rows = []
    for split in ("early_stop", "calibrate"):
        d = pred[pred.split == split]
        if d.empty:
            continue
        groups = context_masks(d)
        groups.update({flag.removeprefix("calendar_"): d[flag].to_numpy(bool) for flag in FLAG_COLUMNS})
        y, score, call = d.y.to_numpy(bool), d.score.to_numpy(float), d.call.to_numpy(bool)
        if not np.isfinite(score).all() or d.y.isna().any():
            raise ValueError("Predictions need finite scores and known labels")
        dates = d.hour.dt.normalize().to_numpy()
        hours = d.hour.to_numpy()
        for scope, scope_mask in (("all_scored_zones", np.ones(len(d), bool)),
                                  ("off_venue", ~d.venue_excluded.to_numpy(bool))):
            for group, group_mask in groups.items():
                m = scope_mask & group_mask
                if not m.any():
                    continue
                rows.append(dict(model=model, split=split, scope=scope, group=group,
                                 dates=len(np.unique(dates[m])), hours=len(np.unique(hours[m])),
                                 **scores(y[m], score[m], call[m])))
    result = pd.DataFrame(rows)
    overall = result[result.group == "overall"][["model", "split", "scope", "f1", "precision", "recall", "auc_pr"]]
    result = result.merge(overall.rename(columns={c: f"overall_{c}" for c in ("f1", "precision", "recall", "auc_pr")}),
                          on=["model", "split", "scope"], validate="many_to_one")
    for metric in ("f1", "precision", "recall", "auc_pr"):
        result[f"delta_{metric}_vs_overall"] = result[metric] - result[f"overall_{metric}"]
    return result


def zone_summary(pred, con, city):
    """Marginal demand/probability by zone. Does not infer passenger trip links."""
    con.register("event_predictions", pred[["zone", "hour"]])
    demand = con.execute("""SELECT p.zone, p.hour, z.pickups, z.dropoffs FROM event_predictions p
                             JOIN read_parquet(?) z USING (zone, hour)""",
                         [str(ROOT / "data" / "export" / f"{city}_zone_hour.parquet")]).df()
    d = pred.merge(demand, on=["zone", "hour"], validate="one_to_one")
    rows = []
    for split in ("early_stop", "calibrate"):
        s = d[(d.split == split) & ~d.venue_excluded]
        for group, mask in context_masks(s).items():
            if group in ("ordinary", "overall") or not mask.any():
                continue
            g = s.loc[mask].groupby("zone").agg(hours=("hour", "size"), dates=("hour", lambda h: h.dt.normalize().nunique()),
                mean_pickups=("pickups", "mean"), mean_dropoffs=("dropoffs", "mean"),
                mean_surge_score=("score", "mean"), predicted_surge_rate=("call", "mean"), observed_surge_rate=("y", "mean")).reset_index()
            g["rank_by_mean_surge_score"] = g.mean_surge_score.rank(method="min", ascending=False).astype(int)
            rows.append(g.assign(split=split, group=group))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--city", choices=["nyc", "chicago"], default="nyc")
    ap.add_argument("--fold", default="fold3", choices=["fold1", "fold2", "fold3"])
    ap.add_argument("--pred", nargs="+", type=Path, help="Optional saved *_pred.parquet files; default compares current Torch/CPU references")
    ap.add_argument("--data-dir", type=Path, default=PROC_ML)
    ap.add_argument("--results-dir", type=Path, default=ROOT / "data" / "results")
    ap.add_argument("--zone-summary", action="store_true", help="Write descriptive zone ranks for the FIRST model")
    args = ap.parse_args()
    paths = args.pred or [args.data_dir / f"{tag(args.city, args.fold, m, v, r)}_pred.parquet"
                         for m, v, r in (("torch", "full", "improved"), ("lightgbm", "full", ""), ("lightgbm", "deployable", ""))]
    results = []
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stem = args.results_dir / f"{args.city}_{args.fold}_event_context"
    for i, path in enumerate(paths):
        if not path.exists():
            if args.pred:
                raise FileNotFoundError(path)
            print(f"Skip missing {path}", flush=True)
            continue
        pred = pd.read_parquet(path, filters=[("split", "in", ["early_stop", "calibrate"])])
        if pred[["zone", "hour"]].duplicated().any():
            raise ValueError("Duplicate prediction keys")
        pred = attach(pred, args.city, args.data_dir)
        model = path.stem.removesuffix("_pred")
        results.append(metrics(pred, model))
        print(f"Evaluated {model}", flush=True)
        if args.zone_summary and i == 0:
            with duckdb.connect() as con:
                zone_summary(pred, con, args.city).to_csv(f"{stem}_zones.csv", index=False)
        del pred
    if not results:
        raise FileNotFoundError("No saved predictions found")
    result = pd.concat(results, ignore_index=True)
    result.to_csv(f"{stem}_metrics.csv", index=False)
    show = result[(result.split == "calibrate") & (result.scope == "off_venue") &
                  result.group.isin(["overall", "ordinary", "sports", "concerts", "federal_holidays", "elections", "cultural", "road_races"])]
    print(show[["model", "group", "dates", "n", "positives", "precision", "recall", "f1", "delta_f1_vs_overall"]].round(4).to_string(index=False))
    print(f"-> {stem}_metrics.csv\nTest not scored. Overlapping groups; rare event dates are descriptive evidence only.")


if __name__ == "__main__":
    main()
