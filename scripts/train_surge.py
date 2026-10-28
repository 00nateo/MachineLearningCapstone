#!/usr/bin/env python
"""LightGBM on the surge label (surge_150: pickups >= 1.5x the zone's normal-conditions baseline).

Design: SURGE_LABEL_DESIGN.md. Reuses train.py / baseline.py / evaluate.py helpers unchanged.
  - Rows: only zone-hours where a surge is possible (baseline >= the 10 rides/h volume floor);
    below it the label is always "no" and would inflate every score.
  - Training rows: the same sampled, weighted training days as the top-10% model
    (sample.py's {city}_{fold}_train.parquet), with the surge label joined on.
  - Inputs: the top-10% model's whitelist for the chosen variant, plus surge context known at
    forecast time: the zone's baseline for hour h (built from earlier weeks only), last hour's
    ratio to its baseline, and whether the zone surged last hour / at this hour last week.
  - Calls: the label is rare (~3%) and fixed-threshold, so instead of a top-decile call the
    probability cutoff that maximizes F1 on early_stop is applied unchanged to calibrate.
    AUC-PR (cutoff-free) is the headline metric.
  - Baselines: surged last hour; surged at this hour last week; the zone's historical surge rate
    over the fold's training rows (cutoff tuned on early_stop like the model).
Scored on early_stop and calibrate; test is predicted and saved but not scored.

Outputs (tag = {city}_{fold}_surge150[_deployable]): data/models/{tag}.txt,
data/processed_ml/{tag}_pred.parquet, data/results/{tag}_metrics.csv, {tag}_importance.csv.
Usage: python scripts/train_surge.py [--city nyc] [--fold fold3] [--variant full deployable]
"""
import argparse
import time

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from baseline import PROC_ML, RESULTS
from evaluate import CONDITIONS, FLAGS
from train import (CITY_FEATURES, DEPLOYABLE_ONLY, EARLY_STOP_ROUNDS, MAX_ROUNDS, MODELS, N_ZONES, PARAMS,
                   TABLE_FEATURES, add_calendar, inputs)

LABEL = "surge_150"
VOLUME_FLOOR = 10
SURGE_INPUTS = ["surge_baseline", "ratio_lag1", "surge_lag1", "surge_lag168"]


def load(con, city, fold):
    feats = ", ".join(f"f.{c}::DOUBLE AS {c}" for c in TABLE_FEATURES + CITY_FEATURES[city] + DEPLOYABLE_ONLY)
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE surge_rows AS
        WITH j AS (
            SELECT f.zone, f.hour, f.{fold} AS split, {feats}, {', '.join(f'f.{c}' for c in FLAGS)},
                   s.{LABEL} AS y, s.baseline AS surge_baseline,
                   lag(s.ratio, 1) OVER w AS ratio_lag1,
                   lag(s.{LABEL}::DOUBLE, 1) OVER w AS surge_lag1,
                   lag(s.{LABEL}::DOUBLE, 168) OVER w AS surge_lag168,
                   lag(s.ratio, 168) OVER w AS ratio_lag168
            FROM read_parquet('{PROC_ML / f'{city}_features.parquet'}') f
            JOIN read_parquet('{PROC_ML / f'{city}_surge.parquet'}') s USING (zone, hour)
            WINDOW w AS (PARTITION BY f.zone ORDER BY f.hour))
        SELECT * FROM j WHERE surge_baseline >= {VOLUME_FLOOR} AND split IS NOT NULL""")
    train = con.execute(f"""
        SELECT r.*, t.weight FROM surge_rows r
        JOIN read_parquet('{PROC_ML / f'{city}_{fold}_train.parquet'}') t USING (zone, hour)
        WHERE r.split = 'train'""").df()
    held = con.execute("""SELECT * FROM surge_rows WHERE split IN ('early_stop', 'calibrate', 'test')
                          ORDER BY hour, zone""").df()
    for d in (train, held):
        d["y"] = d["y"].astype(bool)
        d[FLAGS] = d[FLAGS].astype(bool)
    return add_calendar(train, city), add_calendar(held, city)


def matrix(df, city, variant):
    X = df[inputs(city, variant) + SURGE_INPUTS].copy()
    X["zone"] = pd.Categorical(X["zone"], categories=range(1, N_ZONES[city] + 1))
    for c in X.columns.drop("zone"):
        X[c] = X[c].astype("float32")
    return X


def best_cutoff(y, score):
    """Score cutoff maximizing F1 (chosen on early_stop, applied unchanged later)."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return t[np.argmax(f1)]


def scores(y, score, call):
    tp = (call & y).sum()
    precision, recall = tp / max(call.sum(), 1), tp / max(y.sum(), 1)
    return dict(n=len(y), surge_rate=y.mean(), auc_pr=average_precision_score(y, score) if y.any() else np.nan,
                precision=precision, recall=recall, f1=2 * precision * recall / max(precision + recall, 1e-12))


def report(held, model, score_col, cutoff):
    rows = []
    for split in ("early_stop", "calibrate"):
        d = held[held["split"] == split]
        call = d[score_col].to_numpy() >= cutoff
        for name, flag in {"all": "ALL", **CONDITIONS}.items():
            mask = np.ones(len(d), bool) if flag == "ALL" else (~d["extreme_flag"] if flag is None else d[flag]).to_numpy()
            if mask.sum() == 0 or not d["y"].to_numpy()[mask].any():
                continue
            rows.append({"model": model, "split": split, "condition": name, "cutoff": cutoff,
                         **scores(d["y"].to_numpy()[mask], d[score_col].to_numpy()[mask], call[mask])})
    return rows


def run(con, city, fold, variants):
    t0 = time.time()
    train_df, held = load(con, city, fold)
    es = held["split"] == "early_stop"
    print(f"{city} {fold} {LABEL}: {len(train_df):,} train rows ({train_df['y'].mean():.2%} surge), "
          f"{es.sum():,} early-stop and {(held['split'] == 'calibrate').sum():,} calibrate rows "
          f"(load {time.time() - t0:.0f}s)", flush=True)

    # Baselines
    zone_rate = train_df.groupby("zone")["y"].mean()
    held["b_last_hour"] = held["ratio_lag1"].fillna(0)
    held["b_last_week"] = held["ratio_lag168"].fillna(0)
    held["b_zone_rate"] = held["zone"].map(zone_rate).fillna(0)
    rows = []
    for model, col in (("surged_last_hour", "b_last_hour"), ("surged_last_week", "b_last_week"),
                       ("zone_surge_rate", "b_zone_rate")):
        cutoff = 1.5 if col != "b_zone_rate" else best_cutoff(held.loc[es, "y"], held.loc[es, col])
        rows += report(held, model, col, cutoff)

    for variant in variants:
        tag = f"{city}_{fold}_surge150" + ("" if variant == "full" else f"_{variant}")
        t1 = time.time()
        dtrain = lgb.Dataset(matrix(train_df, city, variant), train_df["y"], weight=train_df["weight"])
        dvalid = lgb.Dataset(matrix(held[es], city, variant), held.loc[es, "y"], reference=dtrain)
        booster = lgb.train(PARAMS, dtrain, MAX_ROUNDS, valid_sets=[dvalid], valid_names=["early_stop"],
                            callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, first_metric_only=True),
                                       lgb.log_evaluation(200)])
        col = f"p_{variant}"
        held[col] = booster.predict(matrix(held, city, variant), num_iteration=booster.best_iteration)
        cutoff = best_cutoff(held.loc[es, "y"], held.loc[es, col])
        rows += report(held, f"lightgbm_{variant}", col, cutoff)
        MODELS.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(MODELS / f"{tag}.txt"))
        held[["zone", "hour", "split", "y", col, *FLAGS]].rename(columns={col: "score"}).assign(
            call=lambda d: d["score"] >= cutoff).to_parquet(PROC_ML / f"{tag}_pred.parquet", compression="zstd", index=False)
        imp = pd.DataFrame({"feature": booster.feature_name(), "gain": booster.feature_importance("gain")})
        imp.sort_values("gain", ascending=False).to_csv(RESULTS / f"{tag}_importance.csv", index=False)
        print(f"{tag}: best iteration {booster.best_iteration}, cutoff {cutoff:.3f}, fit {(time.time() - t1) / 60:.1f} min\n"
              "top features by gain:\n" + imp.nlargest(12, "gain").to_string(index=False), flush=True)

    res = pd.DataFrame(rows)
    out = RESULTS / f"{city}_{fold}_surge150_metrics.csv"
    res.to_csv(out, index=False)
    cal = res[res["split"] == "calibrate"]
    print("\ncalibrate, all eligible zone-hours:\n"
          + cal[cal["condition"] == "all"][["model", "n", "surge_rate", "auc_pr", "precision", "recall", "f1", "cutoff"]]
          .round(3).to_string(index=False)
          + "\n\ncalibrate F1 by condition:\n"
          + cal.pivot_table(index="condition", columns="model", values="f1").round(3).to_string()
          + "\n\ncalibrate AUC-PR by condition:\n"
          + cal.pivot_table(index="condition", columns="model", values="auc_pr").round(3).to_string()
          + f"\n-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["nyc"])
    ap.add_argument("--fold", nargs="+", default=["fold3"])
    ap.add_argument("--variant", nargs="+", default=["full", "deployable"], choices=["full", "deployable"])
    a = ap.parse_args()
    con = duckdb.connect()
    for city in a.city:
        for fold in a.fold:
            run(con, city, fold, a.variant)


if __name__ == "__main__":
    main()
