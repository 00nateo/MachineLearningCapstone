#!/usr/bin/env python
"""LightGBM hotspot classifier per city and fold.

Trains on the sampled, weighted training rows (sample.py), early-stops on the fold's early_stop
split (log loss, since the conformal layer needs reasonable probabilities), and reports metrics
on early_stop and calibrate next to the baselines. The test split is predicted and saved for
conformal.py / evaluate.py but deliberately NOT scored here.

Model inputs are an explicit whitelist (FEATURES). The feature table also holds same-hour
columns (pickups, dropoffs, median_*), the label, fold tags and the extreme flags, none of
which may reach the model. Calendar features are derived from `hour` (known in advance).
`event_flag` is an input despite its name: it comes from event schedules, unlike the extreme
component flags.

Variants: `full` uses every whitelisted input. `deployable` drops the inputs nobody has at forecast
time: MTA/CTA ridership (published ~2 weeks / ~3 months late) and event attendance (reported after
the game), and adds expected_attendance (average attendance of the same team's / venue's earlier
events, known in advance). Pickup lags stay (platform-side forecaster); event_flag stays (schedules).

Outputs (suffix `_deployable` for that variant): data/models/{city}_{fold}.txt,
data/processed_ml/{city}_{fold}_pred.parquet (zone, hour, split, hotspot, score, call, flags),
data/results/{city}_{fold}_model.csv and {city}_{fold}_importance.csv.
Usage: python scripts/train.py [--city nyc] [--fold fold3] [--variant full|deployable]
                               [--param num_leaves=127 ...]
"""
import argparse
import os
import time
from pathlib import Path

import duckdb
import holidays
import lightgbm as lgb
import pandas as pd

from baseline import RESULTS, report, top_decile_call

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"
MODELS = ROOT / "data" / "models"
N_ZONES = {"nyc": 263, "chicago": 77}
SUBDIV = {"nyc": "NY", "chicago": "IL"}
CALENDAR = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"]
TABLE_FEATURES = ["pickup_lag1", "pickup_lag24", "pickup_lag168", "pickup_roll24_mean", "pickup_roll24_std",
                  "pickup_roll168_mean", "pickup_roll168_std",
                  "wx_temp_f_lag1", "wx_wind_kt_lag1", "wx_vis_mi_lag1", "nws_warning_active_lag1",
                  "event_flag", "event_attendance", "flight_sched_count", "flight_delay_rate_lag1"]
CITY_FEATURES = {"nyc": ["subway_entries_lag1", "subway_transfers_lag1"], "chicago": ["cta_rides_prevday"]}
NOT_AVAILABLE_LIVE = {"subway_entries_lag1", "subway_transfers_lag1", "cta_rides_prevday", "event_attendance"}
DEPLOYABLE_ONLY = ["expected_attendance"]  # stand-in for event_attendance in the deployable variant
FLAGS = ["extreme_flag", "severe_weather_flag", "nws_warning_flag", "major_event_flag",
         "transit_disruption_flag", "flight_delay_flag"]
PARAMS = dict(objective="binary", metric=["binary_logloss", "average_precision"], learning_rate=0.05,
              num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, seed=42, deterministic=True, verbose=-1,
              num_threads=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
MAX_ROUNDS, EARLY_STOP_ROUNDS = 3000, 100


def add_calendar(df, city):
    h = df["hour"]
    hol = holidays.country_holidays("US", subdiv=SUBDIV[city], years=range(h.dt.year.min(), h.dt.year.max() + 1))
    return df.assign(hour_of_day=h.dt.hour, day_of_week=h.dt.day_of_week, month=h.dt.month,
                     is_weekend=h.dt.day_of_week >= 5, is_holiday=h.dt.date.isin(set(hol)))


def inputs(city, variant):
    cols = ["zone", *CALENDAR, *TABLE_FEATURES, *CITY_FEATURES[city]]
    if variant == "full":
        return cols
    return [c for c in cols if c not in NOT_AVAILABLE_LIVE] + DEPLOYABLE_ONLY


def matrix(df, city, variant):
    X = df[inputs(city, variant)].copy()
    X["zone"] = pd.Categorical(X["zone"], categories=range(1, N_ZONES[city] + 1))
    for c in X.columns.drop("zone"):
        X[c] = X[c].astype("float32")
    return X


def load(con, city, fold):
    # Numeric features come back as DOUBLE so booleans with gaps (NULL) become NaN for LightGBM.
    feats = ", ".join(f"{c}::DOUBLE AS {c}" for c in TABLE_FEATURES + CITY_FEATURES[city] + DEPLOYABLE_ONLY)
    keep = f"zone, hour, hotspot, {feats}, {', '.join(FLAGS)}"
    train = con.execute(f"SELECT {keep}, weight FROM read_parquet('{PROC_ML / f'{city}_{fold}_train.parquet'}')").df()
    held = con.execute(f"""SELECT {keep}, {fold} AS split FROM read_parquet('{PROC_ML / f'{city}_features.parquet'}')
                           WHERE {fold} IN ('early_stop', 'calibrate', 'test') ORDER BY hour, zone""").df()
    return add_calendar(train, city), add_calendar(held, city)


def train(con, city, fold, variant, overrides):
    t0 = time.time()
    tag = f"{city}_{fold}" + ("" if variant == "full" else f"_{variant}")
    train_df, held = load(con, city, fold)
    es = held[held["split"] == "early_stop"]
    dtrain = lgb.Dataset(matrix(train_df, city, variant), train_df["hotspot"], weight=train_df["weight"])
    dvalid = lgb.Dataset(matrix(es, city, variant), es["hotspot"], reference=dtrain)
    booster = lgb.train({**PARAMS, **overrides}, dtrain, MAX_ROUNDS, valid_sets=[dvalid], valid_names=["early_stop"],
                        callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, first_metric_only=True),
                                   lgb.log_evaluation(100)])
    fit_s = time.time() - t0

    held["score"] = booster.predict(matrix(held, city, variant), num_iteration=booster.best_iteration)
    held["call"] = top_decile_call(con, held)
    model_name = "lightgbm" if variant == "full" else f"lightgbm_{variant}"
    rows = [r for split in ("early_stop", "calibrate") for r in report(held[held["split"] == split], model_name, split)]
    res = pd.DataFrame(rows).assign(best_iteration=booster.best_iteration, fit_seconds=round(fit_s),
                                    train_rows=len(train_df), variant=variant, params=str(overrides or "default"))

    MODELS.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODELS / f"{tag}.txt"))
    held[["zone", "hour", "split", "hotspot", "score", "call", *FLAGS]].to_parquet(
        PROC_ML / f"{tag}_pred.parquet", compression="zstd", index=False)
    res.to_csv(RESULTS / f"{tag}_model.csv", index=False)
    imp = pd.DataFrame({"feature": booster.feature_name(), "gain": booster.feature_importance("gain")})
    imp.sort_values("gain", ascending=False).to_csv(RESULTS / f"{tag}_importance.csv", index=False)

    base = pd.read_csv(RESULTS / f"{city}_{fold}_baselines.csv")
    table = pd.concat([base, res[base.columns]])
    print(f"{tag}: {len(train_df):,} train rows, best iteration {booster.best_iteration}, "
          f"fit {fit_s / 60:.1f} min\n"
          + table[table["subset"] == "all"].round(3).to_string(index=False)
          + "\ntop features by gain:\n" + imp.nlargest(10, "gain").to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["nyc"])
    ap.add_argument("--fold", nargs="+", default=["fold3"])
    ap.add_argument("--variant", default="full", choices=["full", "deployable"])
    ap.add_argument("--param", nargs="*", default=[], help="LightGBM overrides, e.g. num_leaves=127")
    a = ap.parse_args()
    overrides = {k: type(PARAMS[k])(v) if k in PARAMS else float(v)
                 for k, v in (p.split("=", 1) for p in a.param)}
    con = duckdb.connect()
    for city in a.city:
        for fold in a.fold:
            train(con, city, fold, a.variant, overrides)


if __name__ == "__main__":
    main()
