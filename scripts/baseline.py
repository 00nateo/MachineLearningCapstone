#!/usr/bin/env python
"""Seasonal-naive and 4-week-average baselines, scored exactly like the model.

Each baseline gives a score per zone-hour, and a hotspot call by the label's own rule: score
above the 90th percentile of scores across the city's zones in that hour (ties at 0 never count).
  seasonal_naive: pickups at h-168 (same hour, same weekday, last week)
  avg_4wk:        mean pickups at h-168, h-336, h-504, h-672
Scored on the fold's early_stop and calibrate splits only; test stays untouched until evaluate.py.
Metrics: average precision (AUC-PR) and top-decile precision / recall / F1, overall and for
normal vs extreme hours. Results: data/results/{city}_{fold}_baselines.csv
train.py imports top_decile_call and report from here, so the model is scored the same way.

Usage: python scripts/baseline.py [--city nyc chicago] [--fold fold3 ...]
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"
RESULTS = ROOT / "data" / "results"
SCORED_SPLITS = ("early_stop", "calibrate")


def top_decile_call(con, df, score="score"):
    """Hotspot call per row of df: score above its hour's 90th percentile (the label's rule)."""
    con.register("scored", df[["hour", score]].assign(_row=np.arange(len(df))))
    calls = con.execute(f"""SELECT {score} > quantile_cont({score}, 0.9) OVER (PARTITION BY hour) AS call
                            FROM scored ORDER BY _row""").df()["call"]
    con.unregister("scored")
    return calls.fillna(False).to_numpy(dtype=bool)


def metrics(y, score, call):
    tp = (call & y).sum()
    precision, recall = tp / max(call.sum(), 1), tp / max(y.sum(), 1)
    return dict(n=len(y), hotspot_rate=y.mean(), auc_pr=average_precision_score(y, score),
                precision=precision, recall=recall,
                f1=2 * precision * recall / max(precision + recall, 1e-12))


def report(df, model, split):
    """Metrics rows for all hours, normal hours and extreme hours."""
    rows = []
    for subset, mask in (("all", slice(None)), ("normal", ~df["extreme_flag"]), ("extreme", df["extreme_flag"])):
        d = df[mask]
        rows.append({"model": model, "split": split, "subset": subset,
                     **metrics(d["hotspot"].to_numpy(dtype=bool), d["score"].to_numpy(),
                               d["call"].to_numpy(dtype=bool))})
    return rows


def baselines(con, city, fold):
    src = PROC_ML / f"{city}_features.parquet"
    df = con.execute(f"""
        WITH s AS (
            SELECT zone, hour, hotspot, extreme_flag, {fold} AS split, pickup_lag168,
                   (pickup_lag168 + lag(pickups, 336) OVER w + lag(pickups, 504) OVER w
                    + lag(pickups, 672) OVER w) / 4.0 AS avg_4wk
            FROM read_parquet('{src}') WINDOW w AS (PARTITION BY zone ORDER BY hour))
        SELECT * FROM s WHERE split IN {SCORED_SPLITS} ORDER BY hour, zone""").df()
    rows = []
    for model, col in (("seasonal_naive", "pickup_lag168"), ("avg_4wk", "avg_4wk")):
        d = df.assign(score=df[col].astype(float).fillna(-1.0))  # no history yet -> lowest score
        d["call"] = top_decile_call(con, d)
        for split in SCORED_SPLITS:
            rows += report(d[d["split"] == split], model, split)
    out = RESULTS / f"{city}_{fold}_baselines.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    print(f"{city} {fold} baselines -> {out}\n"
          + res[res["subset"] == "all"].round(3).to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["nyc", "chicago"])
    ap.add_argument("--fold", nargs="+", default=["fold3"])
    a = ap.parse_args()
    con = duckdb.connect()
    for city in a.city:
        for fold in a.fold:
            baselines(con, city, fold)


if __name__ == "__main__":
    main()
