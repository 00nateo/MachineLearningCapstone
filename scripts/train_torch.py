#!/usr/bin/env python
"""Train a tabular neural hotspot classifier on Apple MPS, CUDA, or CPU.

Uses the same sampled rows, weights, predictors, folds and scoring as train.py.
Numeric imputation/scaling is fitted on training rows only. A zone embedding and
two hidden layers replace boosted trees. Test rows are predicted but not scored.

Setup (Python 3.12 recommended): pip install -r requirements-torch.txt
Usage: python scripts/train_torch.py --city nyc --device mps
       python scripts/train_torch.py --city nyc --variant deployable --device mps
       python scripts/train_torch.py --city nyc --device mps --cache-device

Outputs use {city}_{fold}_torch[_deployable], preserving LightGBM artifacts:
  data/models/{tag}.pt                 weights, preprocessing and run configuration
  data/processed_ml/{tag}_pred.parquet same prediction schema as train.py
  data/results/{tag}_{model,history}.csv
--cache-device keeps training/validation tensors on GPU (more memory, fewer transfers).
Otherwise only batches move to GPU. No claim of speed or accuracy over LightGBM is assumed.
"""
import argparse
from dataclasses import dataclass
import json
import os
import random
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:
    raise SystemExit("PyTorch is missing. Use Python 3.12 and pip install -r requirements-torch.txt") from exc

from baseline import RESULTS, report, top_decile_call
from model_data import (FLAGS, N_ZONES, PROC_ML, ROOT, artifact_tag, inputs,
                        load, model_name)


def select_device(requested):
    mps = torch.backends.mps.is_available()
    cuda = torch.cuda.is_available()
    if requested == "auto":
        requested = "mps" if mps else "cuda" if cuda else "cpu"
    if requested == "mps" and not mps:
        raise ValueError("MPS is unavailable. Install an Apple-silicon PyTorch build, or use --device cpu.")
    if requested == "cuda" and not cuda:
        raise ValueError("CUDA is unavailable. Use --device mps on Apple silicon, or --device cpu.")
    return torch.device(requested)


@dataclass
class Preprocessor:
    city: str
    columns: list
    mean: np.ndarray
    scale: np.ndarray
    observed: np.ndarray

    @classmethod
    def fit(cls, df, city, variant):
        columns = [c for c in inputs(city, variant) if c != "zone"]
        # Per-column statistics avoid a full float64 copy of a multi-million-row matrix.
        means, scales, observed = [], [], []
        for col in columns:
            x = df[col].to_numpy(dtype=np.float32, na_value=np.nan)
            x = x[np.isfinite(x)]
            means.append(float(x.mean(dtype=np.float64)) if len(x) else 0.0)
            std = float(x.std(dtype=np.float64)) if len(x) else 0.0
            scales.append(std if std > 1e-6 else 1.0)
            observed.append(bool(len(x)))
        return cls(city, columns, np.array(means, dtype=np.float32), np.array(scales, dtype=np.float32),
                   np.array(observed, dtype=bool))

    def transform(self, df):
        zones = df["zone"].to_numpy(dtype=np.float64, na_value=np.nan)
        if not np.all(np.isfinite(zones) & (zones == np.floor(zones)) &
                      (zones >= 1) & (zones <= N_ZONES[self.city])):
            raise ValueError(f"Zone IDs must be integers in 1..{N_ZONES[self.city]} for {self.city}")
        x = df[self.columns].to_numpy(dtype=np.float32, na_value=np.nan, copy=True)
        # An entirely unobserved training feature has no learned response. Keep it
        # disabled even if it appears later (e.g. MTA ridership in fold 1).
        missing = ~np.isfinite(x) | ~self.observed
        x = np.where(missing, self.mean, x)
        x = (x - self.mean) / self.scale
        # Preserve missingness, including features absent for entire periods/cities.
        x = np.concatenate([x, missing.astype(np.float32)], axis=1)
        if not np.isfinite(x).all():
            raise ValueError("Non-finite values after scaling; check input feature ranges")
        return zones.astype(np.int64), np.ascontiguousarray(x, dtype=np.float32)

    def state(self):
        return dict(city=self.city, columns=self.columns, mean=self.mean.tolist(), scale=self.scale.tolist(),
                    observed=self.observed.tolist())

    @classmethod
    def from_state(cls, state):
        return cls(state["city"], state["columns"], np.array(state["mean"], dtype=np.float32),
                   np.array(state["scale"], dtype=np.float32), np.array(state["observed"], dtype=bool))


class HotspotMLP(nn.Module):
    def __init__(self, n_zones, n_numeric, hidden=(128, 64), embedding_dim=8, dropout=0.1):
        super().__init__()
        self.zone_embedding = nn.Embedding(n_zones + 1, embedding_dim)
        layers, width = [], n_numeric + embedding_dim
        for size in hidden:
            layers.extend([nn.Linear(width, size), nn.ReLU(), nn.Dropout(dropout)])
            width = size
        self.network = nn.Sequential(*layers, nn.Linear(width, 1))

    def forward(self, zone, numeric):
        return self.network(torch.cat([self.zone_embedding(zone), numeric], dim=1)).squeeze(-1)


def tensors(df, prep, device=None, training=False):
    z, x = prep.transform(df)
    values = [torch.from_numpy(z), torch.from_numpy(x)]
    if training:
        y = df["hotspot"].to_numpy(dtype=np.float32, na_value=np.nan)
        w = df["weight"].to_numpy(dtype=np.float32, na_value=np.nan)
        if not np.isin(y, [0, 1]).all() or not (np.isfinite(w) & (w > 0)).all():
            raise ValueError("Training labels must be 0/1 and sample weights must be finite and positive")
        values.extend([torch.from_numpy(y), torch.from_numpy(w)])
    return tuple(t.to(device) if device is not None else t for t in values)


def batches(values, batch_size, device, order=None):
    for start in range(0, len(values[0]), batch_size):
        index = slice(start, start + batch_size) if order is None else order[start:start + batch_size]
        if isinstance(index, torch.Tensor):
            index = index.to(values[0].device)
        yield tuple(t[index].to(device) for t in values)


@torch.inference_mode()
def validation_loss(model, values, labels, batch_size, device):
    model.eval()
    total = torch.zeros((), device=device)
    for i, (zone, x) in enumerate(batches(values, batch_size, device)):
        y = labels[i * batch_size:i * batch_size + len(zone)].to(device)
        total += F.binary_cross_entropy_with_logits(model(zone, x), y, reduction="sum")
    return total.item() / len(labels)


@torch.inference_mode()
def predict(model, df, prep, batch_size, device):
    model.eval()
    scores = np.empty(len(df), dtype=np.float32)
    # Transform one batch at a time; do not materialize all held-out tensors on GPU.
    for start in range(0, len(df), batch_size):
        block = df.iloc[start:start + batch_size]
        zone, x = tensors(block, prep, device=device)
        scores[start:start + len(block)] = torch.sigmoid(model(zone, x)).cpu().numpy()
    return scores


def restore(path, device="cpu"):
    """Return (model, preprocessing, checkpoint) for inference in another process."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    prep = Preprocessor.from_state(checkpoint["preprocessing"])
    model = HotspotMLP(**checkpoint["architecture"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), prep, checkpoint


def train(city, fold, args):
    started = time.perf_counter()
    device = select_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tag = artifact_tag(city, fold, "torch", args.variant)
    print(f"{tag}: device={device}, batch_size={args.batch_size}, cache_device={args.cache_device}", flush=True)
    with duckdb.connect() as con:
        train_df, held = load(con, city, fold, args.variant, args.data_dir)
    es = held[held["split"] == "early_stop"]
    if es.empty or train_df.empty or train_df["hotspot"].nunique() < 2:
        raise ValueError("Need both classes in training and a nonempty early_stop split")
    prep = Preprocessor.fit(train_df, city, args.variant)
    cache = device if args.cache_device else None
    train_values = tensors(train_df, prep, cache, training=True)
    valid_values = tensors(es, prep, cache)
    valid_y = torch.tensor(es["hotspot"].to_numpy(dtype=np.float32), device=cache)
    architecture = dict(n_zones=N_ZONES[city], n_numeric=2 * len(prep.columns),
                        hidden=list(args.hidden), embedding_dim=args.embedding_dim, dropout=args.dropout)
    model = HotspotMLP(**architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed)
    initial_loss = validation_loss(model, valid_values, valid_y, args.eval_batch_size, device)
    print(f"{len(train_df):,} training rows, {len(es):,} validation rows; initial log loss={initial_loss:.6f}", flush=True)
    history = []
    best_loss, best_epoch, best_state = float("inf"), 0, None
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_loss = torch.zeros((), device=device)
        total_weight = torch.zeros((), device=device)
        order = torch.randperm(len(train_df), generator=generator)
        for zone, x, y, w in batches(train_values, args.batch_size, device, order):
            optimizer.zero_grad(set_to_none=True)
            per_row = F.binary_cross_entropy_with_logits(model(zone, x), y, reduction="none")
            numerator, denominator = (per_row * w).sum(), w.sum()
            (numerator / denominator).backward()
            optimizer.step()
            total_loss += numerator.detach()
            total_weight += denominator.detach()
        val_loss = validation_loss(model, valid_values, valid_y, args.eval_batch_size, device)
        train_loss = (total_loss / total_weight).item()
        if not np.isfinite([train_loss, val_loss]).all():
            raise ValueError("Training diverged; lower --lr and inspect input values")
        seconds = time.perf_counter() - epoch_start
        history.append(dict(epoch=epoch, train_logloss=train_loss, early_stop_logloss=val_loss,
                            seconds=seconds, rows_per_second=len(train_df) / seconds))
        print(f"epoch {epoch:03d}: train={train_loss:.6f} early_stop={val_loss:.6f} ({seconds:.1f}s)", flush=True)
        if val_loss < best_loss - args.min_delta:
            best_loss, best_epoch = val_loss, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= args.patience:
            print(f"Early stopping; restoring epoch {best_epoch}.", flush=True)
            break
    fit_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    del train_values, valid_values, valid_y, es
    train_rows = len(train_df)
    del train_df
    name = model_name("torch", args.variant)
    held["score"] = predict(model, held, prep, args.eval_batch_size, device)
    with duckdb.connect() as con:
        held["call"] = top_decile_call(con, held)
    rows = [r for split in ("early_stop", "calibrate")
            if not held[held["split"] == split].empty
            for r in report(held[held["split"] == split], name, split)]
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    res = pd.DataFrame(rows).assign(best_epoch=best_epoch, fit_seconds=round(fit_seconds, 2),
                                    train_rows=train_rows, variant=args.variant, device=str(device),
                                    params=json.dumps(config, sort_keys=True))
    for directory in (args.model_dir, args.results_dir):
        directory.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(state_dict=best_state, preprocessing=prep.state(), architecture=architecture,
                      config=config, city=city, fold=fold, variant=args.variant, features=inputs(city, args.variant),
                      best_epoch=best_epoch, best_logloss=best_loss, initial_logloss=initial_loss,
                      fit_seconds=fit_seconds, torch_version=str(torch.__version__))
    torch.save(checkpoint, args.model_dir / f"{tag}.pt")
    held[["zone", "hour", "split", "hotspot", "score", "call", *FLAGS]].to_parquet(
        args.data_dir / f"{tag}_pred.parquet", compression="zstd", index=False)
    res.to_csv(args.results_dir / f"{tag}_model.csv", index=False)
    pd.DataFrame(history).to_csv(args.results_dir / f"{tag}_history.csv", index=False)
    comparisons = [res]
    for comparison_tag, suffix in ((f"{city}_{fold}", "baselines"),
                                   (artifact_tag(city, fold, variant=args.variant), "model")):
        path = args.results_dir / f"{comparison_tag}_{suffix}.csv"
        if path.exists():
            comparisons.append(pd.read_csv(path))
    table = pd.concat(comparisons, ignore_index=True)
    print(table[(table["subset"] == "all") & (table["split"] == "calibrate")][
        ["model", "auc_pr", "precision", "recall", "f1"]].round(4).to_string(index=False), flush=True)
    print(f"Saved {tag}: best epoch {best_epoch}, fit {fit_seconds / 60:.1f} min. Test predicted, not scored.", flush=True)


def parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--city", nargs="+", choices=list(N_ZONES), default=["nyc"])
    ap.add_argument("--fold", nargs="+", choices=["fold1", "fold2", "fold3"], default=["fold3"])
    ap.add_argument("--variant", choices=["full", "deployable"], default="full")
    ap.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--eval-batch-size", type=int, default=65536)
    ap.add_argument("--hidden", nargs="+", type=int, default=[128, 64])
    ap.add_argument("--embedding-dim", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    ap.add_argument("--cache-device", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=PROC_ML)
    ap.add_argument("--model-dir", type=Path, default=ROOT / "data" / "models")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.eval_batch_size,
           args.embedding_dim, args.threads, *args.hidden) < 1:
        ap.error("epoch, patience, batch, embedding, thread and hidden-layer sizes must be positive")
    if not 0 <= args.dropout < 1 or args.lr <= 0 or args.weight_decay < 0 or args.min_delta < 0:
        ap.error("need 0 <= dropout < 1, lr > 0, weight_decay >= 0 and min_delta >= 0")
    try:
        for city in args.city:
            for fold in args.fold:
                train(city, fold, args)
    except (ValueError, FileNotFoundError) as exc:
        ap.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
