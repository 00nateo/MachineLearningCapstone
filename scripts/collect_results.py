#!/usr/bin/env python
"""Collect run outputs into results/ (committed) for the team.

Copies the small metric CSVs from data/results/ (gitignored) into results/<task>/, and writes
what each saved model was trained with (LightGBM params and input list, read from the model
file itself) to results/config/, one row per model in results/runs.csv. results/README.md is
written by hand. Usage: python scripts/collect_results.py
"""
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from sample import DAYS_PER_MONTH, EXTREME_WEIGHT, SEED

ROOT = Path(__file__).resolve().parents[1]
SRC, MODELS, OUT = ROOT / "data" / "results", ROOT / "data" / "models", ROOT / "results"
TASKS = {"surge": "surge", "abstention": "conformal"}  # first match in the file name wins
KEY_PARAMS = ["objective", "learning_rate", "num_leaves", "min_data_in_leaf", "feature_fraction",
              "bagging_fraction", "lambda_l2", "seed"]


def task(name):
    return next((d for k, d in TASKS.items() if k in name), "hotspot_top10")


def read_model(path):
    """Params ([key: value] block) and feature names LightGBM writes into every saved model."""
    text = path.read_text()
    params = dict(re.findall(r"^\[(\w+): (.*)\]$", text[text.rindex("parameters:"):], re.M))
    features = re.search(r"^feature_names=(.*)$", text, re.M).group(1).split()
    return params, features, text.count("\nTree=")


def main():
    csvs = sorted(SRC.glob("*.csv"))
    for f in csvs:
        (OUT / task(f.name)).mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, OUT / task(f.name) / f.name)
    cfg = OUT / "config"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "sampling.json").write_text(json.dumps(dict(
        days_per_month=DAYS_PER_MONTH, extreme_weight=EXTREME_WEIGHT, seed=SEED,
        folds="see MODELING_PLAN.md / label.py"), indent=2) + "\n")
    runs = []
    for m in sorted(MODELS.glob("*.txt")):
        params, features, trees = read_model(m)
        (cfg / f"{m.stem}_params.json").write_text(json.dumps(params, indent=2) + "\n")
        (cfg / f"{m.stem}_features.txt").write_text("\n".join(features) + "\n")
        runs.append(dict(tag=m.stem, trained_at=datetime.fromtimestamp(m.stat().st_mtime).isoformat(" ", "minutes"),
                         trees=trees, n_features=len(features), **{k: params.get(k) for k in KEY_PARAMS}))
    pd.DataFrame(runs).to_csv(OUT / "runs.csv", index=False)
    print(f"{len(runs)} models, {len(csvs)} CSVs -> {OUT}")


if __name__ == "__main__":
    main()
