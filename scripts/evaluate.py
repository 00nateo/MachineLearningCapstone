#!/usr/bin/env python
"""Extra evaluation of the model and baselines on one split (default: calibrate).

Test stays untouched until the conformal step. Why each metric:
  - Day-block bootstrap 95% intervals for precision / recall / F1, and for the paired LightGBM
    minus baseline F1 gap. A point estimate can't show the gain is real; whole days are
    resampled because zone-hours within a day are strongly correlated.
  - Brier score and a reliability table for LightGBM. The conformal layer is built on these
    probabilities, and training up-weights extreme rows 3x, which can inflate them.
  - Per-condition breakdown with counts. The composite extreme flag mixes citywide and
    zone-level conditions and made extreme hours look easier; each component is scored alone.
  - Month-by-month F1: a simple drift curve across the split.
Outputs: data/results/{city}_{fold}_eval_{ci,calibration,conditions,monthly}.csv

Usage: python scripts/evaluate.py [--city nyc] [--fold fold3] [--split calibrate] [--boot 1000]
                                 [--model lightgbm|torch] [--variant full|deployable]
"""
import argparse

import numpy as np
import pandas as pd

from baseline import PROC_ML, RESULTS, metrics
from model_data import artifact_tag, model_name

FLAGS = ["extreme_flag", "severe_weather_flag", "nws_warning_flag", "major_event_flag",
         "transit_disruption_flag", "flight_delay_flag"]
CONDITIONS = {  # subset name -> flag column (normal = no extreme flag)
    "normal": None, "extreme (any)": "extreme_flag", "severe weather": "severe_weather_flag",
    "NWS warning": "nws_warning_flag", "transit disruption": "transit_disruption_flag",
    "major event": "major_event_flag", "flight delays": "flight_delay_flag",
}
BASELINES = ("avg_4wk", "seasonal_naive")


def load(city, fold, split, model="lightgbm", variant="full"):
    tag = artifact_tag(city, fold, model, variant)
    m = pd.read_parquet(PROC_ML / f"{tag}_pred.parquet")
    m = m[m["split"] == split].assign(model=model_name(model, variant))
    b = pd.read_parquet(PROC_ML / f"{city}_{fold}_baseline_pred.parquet")
    b = b[b["split"] == split].drop(columns="split").merge(m[["zone", "hour", "hotspot", *FLAGS]], on=["zone", "hour"])
    cols = ["model", "zone", "hour", "hotspot", "score", "call", *FLAGS]
    df = pd.concat([m[cols], b[cols]], ignore_index=True)
    df[["hotspot", "call", *FLAGS]] = df[["hotspot", "call", *FLAGS]].astype(bool)
    return df


def prf(tp, fp, fn):
    p, r = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    return p, r, 2 * p * r / np.maximum(p + r, 1e-12)


def bootstrap(df, n_boot, seed=42, model="lightgbm"):
    """Point estimates and day-block bootstrap 95% intervals; gaps use the same resampled days."""
    d = df.assign(day=df["hour"].dt.normalize(), tp=df["call"] & df["hotspot"],
                  fp=df["call"] & ~df["hotspot"], fn=~df["call"] & df["hotspot"])
    daily = d.groupby(["model", "day"])[["tp", "fp", "fn"]].sum().unstack("model")
    idx = np.random.default_rng(seed).integers(0, len(daily), size=(n_boot, len(daily)))
    rows, point_f1, boot_f1 = [], {}, {}
    for name_model in df["model"].unique():
        counts = [daily[(s, name_model)].to_numpy() for s in ("tp", "fp", "fn")]
        points = prf(*(c.sum() for c in counts))
        boots = prf(*(c[idx].sum(axis=1) for c in counts))
        for name, pt, bt in zip(("precision", "recall", "f1"), points, boots):
            rows.append(dict(comparison=name_model, metric=name, estimate=pt,
                             ci_low=np.percentile(bt, 2.5), ci_high=np.percentile(bt, 97.5)))
        point_f1[name_model], boot_f1[name_model] = points[2], boots[2]
    for base in BASELINES:
        gap = boot_f1[model] - boot_f1[base]
        rows.append(dict(comparison=f"{model} minus {base}", metric="f1 gap",
                         estimate=point_f1[model] - point_f1[base],
                         ci_low=np.percentile(gap, 2.5), ci_high=np.percentile(gap, 97.5)))
    return pd.DataFrame(rows), len(daily)


def calibration(df, model="lightgbm"):
    """Reliability table (10 probability bins), Brier score overall and by condition, and ECE."""
    m = df[df["model"] == model]
    y, p = m["hotspot"].to_numpy(float), m["score"].to_numpy()
    tab = (m.assign(bin=pd.cut(m["score"], np.linspace(0, 1, 11), include_lowest=True))
           .groupby("bin", observed=True)
           .agg(n=("hotspot", "size"), mean_predicted=("score", "mean"), observed_rate=("hotspot", "mean"))
           .reset_index())
    ece = (tab["n"] * (tab["mean_predicted"] - tab["observed_rate"]).abs()).sum() / tab["n"].sum()
    brier = {name: float(np.mean((p[mask] - y[mask]) ** 2)) for name, mask in
             (("all", slice(None)), ("normal", ~m["extreme_flag"].to_numpy()), ("extreme", m["extreme_flag"].to_numpy()))}
    return tab, brier, ece


def conditions(df):
    rows = []
    for model, d in df.groupby("model"):
        for name, flag in CONDITIONS.items():
            mask = ~d["extreme_flag"] if flag is None else d[flag]
            s = d[mask]
            if len(s) == 0 or s["hotspot"].nunique() < 2:
                continue
            rows.append({"model": model, "condition": name, "share_of_zone_hours": len(s) / len(d),
                         **metrics(s["hotspot"].to_numpy(), s["score"].to_numpy(), s["call"].to_numpy())})
    return pd.DataFrame(rows)


def monthly(df):
    d = df.assign(month=df["hour"].dt.to_period("M").astype(str), tp=df["call"] & df["hotspot"],
                  fp=df["call"] & ~df["hotspot"], fn=~df["call"] & df["hotspot"])
    g = d.groupby(["model", "month"])[["tp", "fp", "fn"]].sum().reset_index()
    g["precision"], g["recall"], g["f1"] = prf(g["tp"], g["fp"], g["fn"])
    return g.drop(columns=["tp", "fp", "fn"])


def evaluate(city, fold, split, n_boot, model="lightgbm", variant="full"):
    df = load(city, fold, split, model, variant)
    name = model_name(model, variant)
    prefix = RESULTS / f"{artifact_tag(city, fold, model, variant)}_eval"
    ci, days = bootstrap(df, n_boot, model=name)
    tab, brier, ece = calibration(df, name)
    cond, mon = conditions(df), monthly(df)
    ci.to_csv(f"{prefix}_ci.csv", index=False)
    tab.assign(**{f"brier_{k}": v for k, v in brier.items()}, ece=ece).to_csv(f"{prefix}_calibration.csv", index=False)
    cond.to_csv(f"{prefix}_conditions.csv", index=False)
    mon.to_csv(f"{prefix}_monthly.csv", index=False)
    print(f"{city} {fold} {split}: {days} days, {n_boot} day-block bootstrap resamples\n"
          + ci.round(4).to_string(index=False)
          + f"\n\n{name} calibration: Brier {brier['all']:.4f} (normal {brier['normal']:.4f}, "
            f"extreme {brier['extreme']:.4f}), ECE {ece:.4f}\n" + tab.round(3).to_string(index=False)
          + "\n\nBy condition (F1, share of zone-hours):\n"
          + cond.pivot_table(index="condition", columns="model", values="f1").round(3)
                .join(cond[cond["model"] == name].set_index("condition")[["share_of_zone_hours", "hotspot_rate"]].round(3))
                .to_string()
          + "\n\nMonthly F1:\n" + mon.pivot_table(index="month", columns="model", values="f1").round(3).to_string(),
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["nyc"])
    ap.add_argument("--fold", nargs="+", default=["fold3"])
    ap.add_argument("--split", default="calibrate", choices=["early_stop", "calibrate"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--model", choices=["lightgbm", "torch"], default="lightgbm")
    ap.add_argument("--variant", choices=["full", "deployable"], default="full")
    a = ap.parse_args()
    for city in a.city:
        for fold in a.fold:
            evaluate(city, fold, a.split, a.boot, a.model, a.variant)


if __name__ == "__main__":
    main()
