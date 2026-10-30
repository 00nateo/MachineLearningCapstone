"""Shared surge rows and lag history for CPU/GPU trainers and phase evaluation."""
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from model_data import (CITY_FEATURES, DEPLOYABLE_ONLY, FLAGS, PROC_ML,
                        TABLE_FEATURES, add_calendar)

LABEL = "surge_150"
VOLUME_FLOOR = 10
SURGE_INPUTS = ["surge_baseline", "ratio_lag1", "surge_lag1", "surge_lag168"]
ENHANCED_INPUTS = ["ratio_lag2", "ratio_lag3", "ratio_lag24", "ratio_lag168", "ratio_mean3",
                   "ratio_change1", "ratio_change2", "pickup_log_growth1", "pickup_mean4wk",
                   "pickup_to_target_baseline", "seasonal_to_target_baseline",
                   "log_pickup_lag1", "log_pickup_lag24", "log_pickup_lag168", "log_surge_baseline",
                   "hour_sin", "hour_cos", "weekday_sin", "weekday_cos"]


def tag(city, fold, model="torch", variant="full", run_name=""):
    name = f"{city}_{fold}_surge150" + ("_torch" if model == "torch" else "")
    return name + ("_deployable" if variant == "deployable" else "") + (f"_{run_name}" if run_name else "")


def load(con, city, fold, proc_ml=PROC_ML, enhanced=False):
    """Same population/predictors as train_surge.py, plus exact prior-hour metadata.

    All lags are formed before eligibility filtering and training-day sampling.
    Existing row-based predictor lags are retained for LightGBM comparability.
    Phase evaluation requires a known label exactly one local clock-hour earlier;
    gaps (including the skipped spring DST hour) become unknown history.
    """
    proc_ml = Path(proc_ml)
    paths = [proc_ml / f"{city}_features.parquet", proc_ml / f"{city}_surge.parquet",
             proc_ml / f"{city}_{fold}_train.parquet"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run features.py, surge_label.py and sample.py first")
    feats = ", ".join(f"f.{c}::DOUBLE AS {c}" for c in TABLE_FEATURES + CITY_FEATURES[city] + DEPLOYABLE_ONLY)
    extra = """ , lag(s.ratio, 2) OVER w AS ratio_lag2,
                   lag(s.ratio, 3) OVER w AS ratio_lag3,
                   lag(s.ratio, 24) OVER w AS ratio_lag24,
                   avg(s.ratio) OVER (w ROWS BETWEEN 3 PRECEDING AND 1 PRECEDING) AS ratio_mean3,
                   lag(f.pickups, 2) OVER w AS pickup_lag2,
                   (lag(f.pickups, 168) OVER w + lag(f.pickups, 336) OVER w +
                    lag(f.pickups, 504) OVER w + lag(f.pickups, 672) OVER w) / 4.0 AS pickup_mean4wk
    """ if enhanced else ""
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE surge_rows AS
        WITH j AS (
            SELECT f.zone, f.hour, f.{fold} AS split, {feats}, {', '.join(f'f.{c}' for c in FLAGS)},
                   s.{LABEL} AS y, s.baseline AS surge_baseline,
                   lag(s.ratio, 1) OVER w AS ratio_lag1,
                   lag(s.{LABEL}::DOUBLE, 1) OVER w AS surge_lag1,
                   lag(s.{LABEL}::DOUBLE, 168) OVER w AS surge_lag168,
                   lag(s.ratio, 168) OVER w AS ratio_lag168,
                   lag(f.hour) OVER w AS previous_hour {extra}
            FROM read_parquet(?) f JOIN read_parquet(?) s USING (zone, hour)
            WINDOW w AS (PARTITION BY f.zone ORDER BY f.hour))
        SELECT *, CASE WHEN previous_hour = hour - INTERVAL 1 HOUR
                       THEN surge_lag1 END AS previous_surge
        FROM j WHERE surge_baseline >= {VOLUME_FLOOR} AND split IS NOT NULL
    """, [str(paths[0]), str(paths[1])])
    train = con.execute("""SELECT r.*, t.weight FROM surge_rows r
                           JOIN read_parquet(?) t USING (zone, hour) WHERE r.split = 'train'""",
                        [str(paths[2])]).df()
    held = con.execute("""SELECT * FROM surge_rows WHERE split IN ('early_stop', 'calibrate', 'test')
                          ORDER BY hour, zone""").df()
    for df in (train, held):
        if df["y"].isna().any():
            raise ValueError("Eligible surge rows have unknown labels; rebuild surge_label.py outputs")
        df["y"] = df["y"].astype(bool)
        df[FLAGS] = df[FLAGS].astype(bool)
    train, held = add_calendar(train, city), add_calendar(held, city)
    return (enhance(train), enhance(held)) if enhanced else (train, held)


def enhance(df):
    """Deterministic transforms of past-only counts/ratios and known calendar fields."""
    for col in ("pickup_lag1", "pickup_lag24", "pickup_lag168", "surge_baseline"):
        df[f"log_{col}"] = np.log1p(df[col].clip(lower=0))
    df["ratio_change1"] = df["ratio_lag1"] - df["ratio_lag2"]
    df["ratio_change2"] = df["ratio_lag2"] - df["ratio_lag3"]
    df["pickup_log_growth1"] = df["log_pickup_lag1"] - np.log1p(df["pickup_lag2"].clip(lower=0))
    df["pickup_to_target_baseline"] = df["pickup_lag1"] / df["surge_baseline"]
    df["seasonal_to_target_baseline"] = df["pickup_mean4wk"] / df["surge_baseline"]
    for prefix, column, period in (("hour", "hour_of_day", 24), ("weekday", "day_of_week", 7)):
        angle = 2 * np.pi * df[column] / period
        df[f"{prefix}_sin"], df[f"{prefix}_cos"] = np.sin(angle), np.cos(angle)
    return df


def best_cutoff(y, score):
    """Maximize F1 on early_stop only, then lock the threshold for later splits."""
    if len(y) == 0 or np.unique(y).size < 2:
        raise ValueError("Cutoff selection requires a nonempty early_stop split with both classes")
    p, r, thresholds = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thresholds[np.argmax(f1)])


def attach_history(con, predictions, city, proc_ml=PROC_ML):
    """Enrich saved LightGBM predictions without retraining or changing their calls."""
    if "previous_surge" in predictions:
        return predictions
    path = Path(proc_ml) / f"{city}_surge.parquet"
    con.register("phase_predictions", predictions[["zone", "hour", "y"]])
    try:
        history = con.execute(f"""
            WITH h AS (
                SELECT zone, hour, {LABEL} AS source_y,
                       CASE WHEN lag(hour) OVER w = hour - INTERVAL 1 HOUR
                            THEN lag({LABEL}::DOUBLE) OVER w END AS previous_surge
                FROM read_parquet(?) WINDOW w AS (PARTITION BY zone ORDER BY hour))
            SELECT p.zone, p.hour, h.source_y, h.previous_surge
            FROM phase_predictions p LEFT JOIN h USING (zone, hour)
        """, [str(path)]).df()
    finally:
        con.unregister("phase_predictions")
    joined = predictions.merge(history, on=["zone", "hour"], how="left", validate="one_to_one", sort=False)
    if joined["source_y"].isna().any() or not (joined["y"] == joined["source_y"]).all():
        raise ValueError("Saved prediction labels differ from the current surge file; regenerate predictions")
    return joined.drop(columns="source_y")
