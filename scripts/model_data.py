"""Shared feature contract and data loading for LightGBM and PyTorch trainers."""
from pathlib import Path

import holidays

ROOT = Path(__file__).resolve().parents[1]
PROC_ML = ROOT / "data" / "processed_ml"
N_ZONES = {"nyc": 263, "chicago": 77}
SUBDIV = {"nyc": "NY", "chicago": "IL"}
CALENDAR = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"]
TABLE_FEATURES = ["pickup_lag1", "pickup_lag24", "pickup_lag168", "pickup_roll24_mean", "pickup_roll24_std",
                  "pickup_roll168_mean", "pickup_roll168_std",
                  "wx_temp_f_lag1", "wx_wind_kt_lag1", "wx_vis_mi_lag1", "nws_warning_active_lag1",
                  "event_flag", "event_attendance", "flight_sched_count", "flight_delay_rate_lag1"]
CITY_FEATURES = {"nyc": ["subway_entries_lag1", "subway_transfers_lag1"], "chicago": ["cta_rides_prevday"]}
NOT_AVAILABLE_LIVE = {"subway_entries_lag1", "subway_transfers_lag1", "cta_rides_prevday", "event_attendance"}
DEPLOYABLE_ONLY = ["expected_attendance"]
FLAGS = ["extreme_flag", "severe_weather_flag", "nws_warning_flag", "major_event_flag",
         "transit_disruption_flag", "flight_delay_flag"]


def artifact_tag(city, fold, model="lightgbm", variant="full"):
    """Preserve existing LightGBM filenames; keep neural experiments separate."""
    return f"{city}_{fold}" + ("_torch" if model == "torch" else "") + (
        "" if variant == "full" else f"_{variant}")


def model_name(model="lightgbm", variant="full"):
    return model + ("" if variant == "full" else f"_{variant}")


def add_calendar(df, city):
    h = df["hour"]
    if df.empty:
        raise ValueError(f"{city}: no rows found in the requested training/held-out splits")
    hol = holidays.country_holidays("US", subdiv=SUBDIV[city], years=range(h.dt.year.min(), h.dt.year.max() + 1))
    return df.assign(hour_of_day=h.dt.hour, day_of_week=h.dt.day_of_week, month=h.dt.month,
                     is_weekend=h.dt.day_of_week >= 5, is_holiday=h.dt.date.isin(set(hol)))


def inputs(city, variant):
    cols = ["zone", *CALENDAR, *TABLE_FEATURES, *CITY_FEATURES[city]]
    if variant == "full":
        return cols
    return [c for c in cols if c not in NOT_AVAILABLE_LIVE] + DEPLOYABLE_ONLY


def load(con, city, fold, variant=None, proc_ml=PROC_ML):
    """Read the existing sample and full held-out hours; never resample held-out rows.

    variant=None keeps the original LightGBM loader's union of feature columns.
    The neural trainer requests only the predictors needed by its selected variant.
    """
    cols = (TABLE_FEATURES + CITY_FEATURES[city] + DEPLOYABLE_ONLY if variant is None else
            [c for c in inputs(city, variant) if c not in ["zone", *CALENDAR]])
    feats = ", ".join(f"{c}::DOUBLE AS {c}" for c in cols)
    keep = f"zone, hour, hotspot, {feats}, {', '.join(FLAGS)}"
    train_path = Path(proc_ml) / f"{city}_{fold}_train.parquet"
    held_path = Path(proc_ml) / f"{city}_features.parquet"
    for path in (train_path, held_path):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found; run features.py and sample.py first")
    train = con.execute(f"SELECT {keep}, weight FROM read_parquet(?)", [str(train_path)]).df()
    held = con.execute(f"""SELECT {keep}, {fold} AS split FROM read_parquet(?)
                           WHERE {fold} IN ('early_stop', 'calibrate', 'test') ORDER BY hour, zone""",
                       [str(held_path)]).df()
    return add_calendar(train, city), add_calendar(held, city)
