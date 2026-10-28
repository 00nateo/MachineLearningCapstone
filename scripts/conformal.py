#!/usr/bin/env python
"""Split-conformal abstention layer on top of the LightGBM hotspot probabilities.

Week 4 (--mode global): one threshold for every row. Calibration rows score
s = 1 - p(true class). For miscoverage alpha, q = the ceil((n+1)(1-alpha))-th smallest score.
A new row with hotspot probability p gets the prediction set
  {hotspot} if p >= 1-q,   {not} if p <= q,
and abstains when the set is empty (both excluded, happens when q < 0.5) or full (both
included, q >= 0.5). Week 5 adds --mode mondrian: separate thresholds per class x condition.

Where it's measured: test (2025+) stays untouched until the headline run. By default the
fold's calibrate year is split in time: thresholds are fit on its first half (CAL_SPLIT) and
evaluated on the second half, which also carries the within-year drift seen in Week 3.

Metrics per alpha, for all hours, normal vs extreme, and each condition flag: coverage
(overall and per class), abstention rate (empty / full sets), accuracy when confident,
confident-miss rate (share of true hotspots confidently called "not") and precision of
confident hotspot calls. Output: data/results/{tag}_abstention_{mode}.csv

Usage: python scripts/conformal.py [--city nyc] [--fold fold3] [--variant full|deployable]
                                   [--mode global] [--cal-split 2024-07-01]
"""
import argparse
import math

import numpy as np
import pandas as pd

from baseline import PROC_ML, RESULTS
from evaluate import CONDITIONS, FLAGS

ALPHAS = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20)
CAL_SPLIT = "2024-07-01"  # fold 3: fit on calibrate rows before this hour, evaluate on the rest


def conformal_q(scores, alpha):
    """The ceil((n+1)(1-alpha))-th smallest calibration score (inf if n is too small)."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else np.sort(scores)[k - 1]


def prediction_sets(p, q):
    """Boolean arrays: is 'hotspot' / 'not' in each row's set."""
    return p >= 1 - q, p <= q


def summarize(d, has_hot, has_not):
    y = d["hotspot"].to_numpy(bool)
    single = has_hot ^ has_not
    covered = np.where(y, has_hot, has_not)
    conf_hot, conf_not = has_hot & ~has_not, has_not & ~has_hot
    return dict(
        n=len(d), coverage=covered.mean(),
        coverage_hotspot=covered[y].mean() if y.any() else np.nan,
        coverage_not=covered[~y].mean() if (~y).any() else np.nan,
        abstain_rate=1 - single.mean(),
        abstain_empty=(~has_hot & ~has_not).mean(), abstain_full=(has_hot & has_not).mean(),
        accuracy_when_confident=(covered & single).sum() / max(single.sum(), 1),
        confident_miss_rate=(conf_not & y).sum() / max(y.sum(), 1),
        confident_hotspot_precision=(conf_hot & y).sum() / max(conf_hot.sum(), 1))


def run(city, fold, variant, mode, cal_split):
    tag = f"{city}_{fold}" + ("" if variant == "full" else f"_{variant}")
    pred = pd.read_parquet(PROC_ML / f"{tag}_pred.parquet")
    pred = pred[pred["split"] == "calibrate"]
    pred[["hotspot", *FLAGS]] = pred[["hotspot", *FLAGS]].astype(bool)
    cal, ev = pred[pred["hour"] < cal_split], pred[pred["hour"] >= cal_split]
    p_cal = cal["score"].to_numpy()
    s_cal = np.where(cal["hotspot"].to_numpy(), 1 - p_cal, p_cal)
    rows = []
    for alpha in ALPHAS:
        q = conformal_q(s_cal, alpha)
        has_hot, has_not = prediction_sets(ev["score"].to_numpy(), q)
        for name, flag in CONDITIONS.items():
            mask = (~ev["extreme_flag"] if flag is None else ev[flag]).to_numpy()
            rows.append(dict(alpha=alpha, q=q, condition=name,
                             **summarize(ev[mask], has_hot[mask], has_not[mask])))
        rows.append(dict(alpha=alpha, q=q, condition="all", **summarize(ev, has_hot, has_not)))
    res = pd.DataFrame(rows)
    out = RESULTS / f"{tag}_abstention_{mode}.csv"
    res.to_csv(out, index=False)
    show = ["alpha", "q", "coverage", "coverage_hotspot", "abstain_rate", "abstain_empty", "abstain_full",
            "accuracy_when_confident", "confident_miss_rate"]
    print(f"{tag} {mode}: thresholds fit on {len(cal):,} rows before {cal_split}, evaluated on {len(ev):,}\n"
          + res[res["condition"] == "all"][show].round(3).to_string(index=False)
          + "\n\nalpha = 0.05, by condition:\n"
          + res[res["alpha"] == 0.05].set_index("condition")[show[2:]].round(3).to_string()
          + f"\n-> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", nargs="+", default=["nyc"])
    ap.add_argument("--fold", nargs="+", default=["fold3"])
    ap.add_argument("--variant", default="full", choices=["full", "deployable"])
    ap.add_argument("--mode", default="global", choices=["global"])
    ap.add_argument("--cal-split", default=CAL_SPLIT)
    a = ap.parse_args()
    for city in a.city:
        for fold in a.fold:
            run(city, fold, a.variant, a.mode, a.cal_split)


if __name__ == "__main__":
    main()
