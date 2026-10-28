"""Joint surge decision thresholds, selected entirely within early_stop.

Optimize pooled F1 subject to a pooled precision floor, not separate subgroup
F1s. Unknown history always uses a global cutoff. Auto mode fits candidate rules
on the first half of validation hours, checks F1 AND precision on the later half,
then refits only the chosen kind of rule on all early_stop rows. This is model
selection within an already-used validation year, not independent test evidence.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve


def _check(df):
    if df.empty or df.y.isna().any() or not np.isfinite(df.score).all():
        raise ValueError("Threshold fitting requires known labels and finite scores")
    if not df.score.between(0, 1).all() or df.y.nunique() < 2:
        raise ValueError("Threshold fitting requires probabilities in [0,1] and both classes")


def summary(y, call):
    y, call = np.asarray(y, bool), np.asarray(call, bool)
    tp, called, positive = int((y & call).sum()), int(call.sum()), int(y.sum())
    return dict(n=len(y), positives=positive, predicted_positives=called, tp=tp,
                precision=tp / called if called else 0., recall=tp / positive if positive else 0.,
                f1=2 * tp / (called + positive) if called + positive else 0.)


def global_policy(df, min_precision=None):
    _check(df)
    p, r, t = precision_recall_curve(df.y, df.score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    valid = p[:-1] >= (min_precision or 0.) - 1e-12
    if not valid.any():
        raise ValueError("No validation cutoff meets --min-precision; lower the requested floor")
    index = int(np.argmax(np.where(valid, f1, -1)))
    cutoff = float(t[index])
    return dict(mode="global", global_cutoff=cutoff, onset=cutoff, continuation=cutoff,
                unknown_history=cutoff, min_precision=min_precision)


def apply_policy(df, policy):
    thresholds = np.full(len(df), policy["global_cutoff"], dtype=np.float64)
    for phase, state in (("onset", 0), ("continuation", 1)):
        mask = df.previous_surge.eq(state).fillna(False).to_numpy(bool)
        thresholds[mask] = policy[phase]
    return df.score.to_numpy() >= thresholds, thresholds


def _candidates(y, score, global_cutoff, grid_size):
    """Exact tie-aware counts at a tail-focused grid, including the global rule."""
    y, score = np.asarray(y, bool), np.asarray(score, float)
    if not len(y):
        return np.array([global_cutoff]), np.array([0]), np.array([0])
    unique = np.unique(score)
    if len(unique) > grid_size:
        rates = np.geomspace(1 / len(score), 1., grid_size)
        unique = np.quantile(score, 1 - rates)
    thresholds = np.unique(np.r_[unique, global_cutoff, 1.000001])
    order = np.argsort(score, kind="stable")
    cumulative = np.r_[0, np.cumsum(y[order], dtype=np.int64)]
    below = np.searchsorted(score[order], thresholds, side="left")
    return thresholds, cumulative[-1] - cumulative[below], len(y) - below


def joint_policy(df, min_precision=None, grid_size=201):
    reference = global_policy(df, min_precision)
    base_call, _ = apply_policy(df, reference)
    # By default, protect the precision of the best global F1 cutoff on the fit rows.
    floor = min_precision if min_precision is not None else summary(df.y, base_call)["precision"]
    a = df.previous_surge.eq(0).fillna(False).to_numpy(bool)
    b = df.previous_surge.eq(1).fillna(False).to_numpy(bool)
    unknown = ~(a | b)
    cuts_a, tp_a, n_a = _candidates(df.y[a], df.score[a], reference["global_cutoff"], grid_size)
    cuts_b, tp_b, n_b = _candidates(df.y[b], df.score[b], reference["global_cutoff"], grid_size)
    tp_unknown = int((df.y.to_numpy(bool)[unknown] & base_call[unknown]).sum())
    n_unknown = int(base_call[unknown].sum())
    tp = tp_a[:, None] + tp_b[None, :] + tp_unknown
    called = n_a[:, None] + n_b[None, :] + n_unknown
    precision = tp / np.maximum(called, 1)
    f1 = 2 * tp / np.maximum(called + int(df.y.sum()), 1)
    valid = (called > 0) & (precision >= floor - 1e-12)
    best = np.max(np.where(valid, f1, -1))
    # Resolve equal-F1 rules by higher precision. The global pair is always in the grid.
    tied = valid & np.isclose(f1, best, atol=1e-12, rtol=0)
    i, j = np.unravel_index(np.argmax(np.where(tied, precision, -1)), f1.shape)
    return dict(reference, mode="joint", onset=float(cuts_a[i]), continuation=float(cuts_b[j]),
                min_precision=float(floor))


def select_policy(df, mode="auto", min_precision=None, grid_size=201, min_gain=0.001):
    """Return a final rule and an audit table; never access calibrate/test labels."""
    if set(df.split.unique()) != {"early_stop"}:
        raise ValueError("Threshold selection accepts early_stop rows only")
    _check(df)
    audit = []

    def record(stage, subset, policy):
        call, _ = apply_policy(subset, policy)
        row = dict(stage=stage, policy=policy["mode"], onset_cutoff=policy["onset"],
                   continuation_cutoff=policy["continuation"], global_cutoff=policy["global_cutoff"],
                   **summary(subset.y, call))
        audit.append(row)
        return row

    chosen = mode
    if mode == "auto":
        hours = np.sort(df.hour.unique())
        split_hour = hours[len(hours) // 2]
        fit, check = df[df.hour < split_hour], df[df.hour >= split_hour]
        if fit.y.nunique() < 2 or check.y.nunique() < 2:
            chosen = "global"
        else:
            global_rule = global_policy(fit, min_precision)
            joint_rule = joint_policy(fit, min_precision, grid_size)
            record("fit", fit, global_rule)
            record("fit", fit, joint_rule)
            g = record("check", check, global_rule)
            j = record("check", check, joint_rule)
            precision_target = max(g["precision"], min_precision or 0.)
            chosen = "joint" if (j["f1"] >= g["f1"] + min_gain and
                                 j["precision"] >= precision_target - 1e-12) else "global"
    policy = (joint_policy(df, min_precision, grid_size) if chosen == "joint" else
              global_policy(df, min_precision))
    final = record("refit", df, policy)
    policy["selection"] = dict(requested_mode=mode, selected_mode=chosen,
                               temporal_check=mode == "auto" and len(audit) > 1,
                               min_gain=min_gain, validation_metrics={k: final[k]
                               for k in ("precision", "recall", "f1")})
    return policy, pd.DataFrame(audit)
