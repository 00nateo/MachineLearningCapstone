#!/usr/bin/env python
"""Train surge_150 on Apple GPU, and measure surge onset versus continuation.

Same eligible rows and weighted training sample as train_surge.py. Optional enhanced
inputs add past-only momentum, seasonality, log counts and cyclic calendar features.
Uses the MLP/zone embedding, train-only preprocessing and GPU loop from train_torch.py.
Selects global/joint thresholds on early_stop only, protecting pooled precision and
checking a joint rule on later validation hours before accepting it in auto mode.
Test predictions are saved, never scored. Existing CPU surge artifacts are preserved.

Usage: python scripts/train_surge_torch.py --city nyc --device mps
       python scripts/train_surge_torch.py --city nyc --device mps --variant deployable
       python scripts/train_surge_torch.py --city nyc --device mps --run-name wider --hidden 256 128

Artifacts: {city}_{fold}_surge150_torch[_deployable][_{run_name}] in the existing
model, processed_ml and results folders (.pt, _pred.parquet, _metrics.csv,
_phases.csv and _history.csv). Phase metrics include false alarms as well as hits.
"""
import json
import re
import time

import duckdb
import numpy as np
import pandas as pd
import torch

from evaluate_surge import report, saved_report, write_reports
from model_data import FLAGS, inputs
from surge_data import ENHANCED_INPUTS, LABEL, SURGE_INPUTS, VOLUME_FLOOR, best_cutoff, load, tag
from surge_thresholds import apply_policy, select_policy
from train_torch import fit_network, parser as training_parser, predict, select_device, validate_args


def baselines(train_df, held):
    """Match CPU surge baselines and add strict prior-label persistence."""
    es = held["split"] == "early_stop"
    rates = train_df.groupby("zone")["y"].mean()
    zone_rate = held["zone"].map(rates).fillna(0)
    return [
        ("surged_last_hour", held["ratio_lag1"].fillna(0), 1.5),
        ("surged_last_week", held["ratio_lag168"].fillna(0), 1.5),
        ("zone_surge_rate", zone_rate, best_cutoff(held.loc[es, "y"], zone_rate[es])),
        ("persistence_label", held["previous_surge"].fillna(0), 0.5),
    ]


def train(city, fold, args):
    started = time.perf_counter()
    device = select_device(args.device)
    name = tag(city, fold, "torch", args.variant, args.run_name)
    print(f"{name}: {LABEL}, baseline >= {VOLUME_FLOOR}/h, device={device}, "
          f"batch_size={args.batch_size}, cache_device={args.cache_device}", flush=True)
    with duckdb.connect() as con:
        train_df, held = load(con, city, fold, args.data_dir, enhanced=args.features == "enhanced")
    if args.event_context:
        from event_context import attach
        for label, frame in (("train", train_df), ("held", held)):
            enriched = attach(frame, city, args.data_dir, off_venue=args.off_venue)
            print(f"{label}: kept {len(enriched):,}/{len(frame):,} rows after event context/mask", flush=True)
            if label == "train":
                train_df = enriched
            else:
                held = enriched
    es = held["split"] == "early_stop"
    # Validate this before spending compute on training; do not use calibrate/test labels.
    if es.sum() == 0 or held.loc[es, "y"].nunique() < 2:
        raise ValueError("Need both classes in early_stop for surge cutoff selection")
    baseline_scores = baselines(train_df, held)
    extra_features = SURGE_INPUTS + (ENHANCED_INPUTS if args.features == "enhanced" else [])
    if args.event_context:
        from event_context import CONTEXT_FEATURES
        extra_features = extra_features + CONTEXT_FEATURES
    model, prep, state, history = fit_network(train_df, held[es], city, args,
                                             extra_features=extra_features, target="y")
    fit_seconds = time.perf_counter() - started
    train_rows = len(train_df)
    del train_df
    held["score"] = predict(model, held, prep, args.eval_batch_size, device)
    policy, threshold_audit = select_policy(held[es], args.threshold_mode, args.min_precision,
                                             args.threshold_grid_size, args.threshold_min_gain)
    held["call"], held["decision_threshold"] = apply_policy(held, policy)
    cutoff = policy["global_cutoff"]  # compatibility/reference; inference must apply threshold_policy
    config = {k: str(v) if hasattr(v, "__fspath__") else v for k, v in vars(args).items()}
    checkpoint = dict(state, config=config, city=city, fold=fold, variant=args.variant, target="y", label=LABEL,
                      features=inputs(city, args.variant) + extra_features, cutoff=cutoff,
                      threshold_policy=policy, feature_set=args.features,
                      event_context=args.event_context, off_venue=args.off_venue,
                      volume_floor=VOLUME_FLOOR, train_rows=train_rows, fit_seconds=fit_seconds)
    for directory in (args.model_dir, args.results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.model_dir / f"{name}.pt")
    columns = ["zone", "hour", "split", "y", "score", "call", "decision_threshold",
               "previous_hour", "previous_surge", *FLAGS]
    if args.event_context:
        columns += ["venue_excluded"]
    pred = held[columns].copy()
    pred.to_parquet(args.data_dir / f"{name}_pred.parquet", compression="zstd", index=False)
    pd.DataFrame(history).to_csv(args.results_dir / f"{name}_history.csv", index=False)
    threshold_audit.to_csv(args.results_dir / f"{name}_threshold_selection.csv", index=False)
    with (args.results_dir / f"{name}_threshold_policy.json").open("w") as file:
        json.dump(policy, file, indent=2)
    results = [report(pred, f"torch_{args.variant}").assign(
        best_epoch=state["best_epoch"], fit_seconds=fit_seconds, train_rows=train_rows,
        device=str(device), feature_set=args.features, threshold_mode=policy["mode"],
        params=json.dumps(config, sort_keys=True))]
    for baseline_name, score, threshold in baseline_scores:
        results.append(report(pred.assign(score=np.asarray(score)), baseline_name, threshold))
    # Evaluate the saved CPU models on the same exact history groups; never retrain them here.
    if not args.skip_lightgbm and not args.off_venue:
        with duckdb.connect() as con:
            for variant in ("full", "deployable"):
                path = args.data_dir / f"{tag(city, fold, 'lightgbm', variant)}_pred.parquet"
                if path.exists():
                    results.append(saved_report(con, city, fold, "lightgbm", variant, args.data_dir))
            # Keep the old Torch run visible as an ablation/reference when using a run suffix.
            reference = args.data_dir / f"{tag(city, fold, 'torch', args.variant)}_pred.parquet"
            if args.run_name and reference.exists():
                old = saved_report(con, city, fold, "torch", args.variant, args.data_dir)
                old["model"] = f"torch_{args.variant}_reference"
                results.append(old)
    result = pd.concat(results, ignore_index=True)
    write_reports(result, args.results_dir / name)
    print("\nThreshold selection (early_stop only):\n"
          + threshold_audit[["stage", "policy", "precision", "recall", "f1"]].round(4).to_string(index=False), flush=True)
    print(f"Best epoch {state['best_epoch']}; {policy['mode']} policy: onset={policy['onset']:.4f}, "
          f"continuation={policy['continuation']:.4f}, unknown={cutoff:.4f}; "
          f"fit {fit_seconds / 60:.1f} min. Test predicted, not scored.", flush=True)


def parser():
    ap = training_parser()
    ap.description = __doc__
    ap.set_defaults(patience=10)
    ap.add_argument("--run-name", default="improved", help="Artifact suffix; default preserves the original run")
    ap.add_argument("--skip-lightgbm", action="store_true", help="Skip comparisons with existing saved models")
    ap.add_argument("--features", choices=["basic", "enhanced"], default="enhanced")
    ap.add_argument("--event-context", action="store_true", help="Use citywide event categories/windows and named-day flags")
    ap.add_argument("--off-venue", action="store_true", help="Exclude active mapped venue zones from training AND prediction targets; requires --event-context")
    ap.add_argument("--threshold-mode", choices=["global", "joint", "auto"], default="auto")
    ap.add_argument("--min-precision", type=float, default=None,
                    help="Validation precision floor; default joint floor matches global-rule precision")
    ap.add_argument("--threshold-grid-size", type=int, default=201)
    ap.add_argument("--threshold-min-gain", type=float, default=0.001,
                    help="Minimum later-validation pooled F1 gain required to accept a joint rule")
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    validate_args(ap, args)
    if args.off_venue and not args.event_context:
        ap.error("--off-venue requires --event-context")
    if args.event_context and args.run_name in ("", "improved"):
        ap.error("Use a new --run-name, e.g. events_offvenue, to preserve reference predictions")
    if args.min_precision is not None and not 0 < args.min_precision <= 1:
        ap.error("--min-precision must be in (0, 1]")
    if args.threshold_grid_size < 2 or args.threshold_min_gain < 0:
        ap.error("threshold grid size must be >= 2 and minimum gain must be nonnegative")
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        ap.error("--run-name must contain only letters, digits, underscores and hyphens")
    try:
        for city in args.city:
            for fold in args.fold:
                train(city, fold, args)
    except (ValueError, FileNotFoundError) as exc:
        ap.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
