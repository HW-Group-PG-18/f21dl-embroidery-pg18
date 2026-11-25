"""Metrics, group-level bootstrap intervals, and McNemar's test.

With six test groups, resampling images would pretend we have 230 independent samples. The
bootstrap here resamples whole groups instead, which is the honest unit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "roc_auc"]


def classification_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "n": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc": np.nan,
    }
    if y_score is not None and len(np.unique(y_true)) == 2:
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except ValueError:  # pragma: no cover
            out["roc_auc"] = np.nan
    return out


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def group_bootstrap_ci(
    df: pd.DataFrame,
    pred_col: str = "pred",
    prob_col: str = "prob",
    label_col: str = "label_id",
    group_col: str = "group_id",
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CI resampling *groups* (the independent unit), not images."""
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    idx_by_group = {g: np.flatnonzero((df[group_col] == g).to_numpy()) for g in groups}
    draws: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_boot):
        picked = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        sub = df.iloc[idx]
        m = classification_metrics(sub[label_col], sub[pred_col], sub[prob_col])
        for k in METRIC_NAMES:
            draws[k].append(m[k])
    out = {}
    for k, vals in draws.items():
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = (np.nan, np.nan)
        else:
            out[k] = (float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2))))
    return out


def mcnemar(y_true, pred_a, pred_b) -> dict[str, float]:
    """Exact McNemar test on paired predictions (same images, two models/conditions)."""
    y_true = np.asarray(y_true)
    a_ok = np.asarray(pred_a) == y_true
    b_ok = np.asarray(pred_b) == y_true
    b01 = int(np.sum(a_ok & ~b_ok))
    b10 = int(np.sum(~a_ok & b_ok))
    if b01 + b10 == 0:
        return {"n_discordant": 0, "p_value": 1.0, "a_only_correct": b01, "b_only_correct": b10}
    res = stats.binomtest(b01, b01 + b10, 0.5)
    return {
        "n_discordant": b01 + b10,
        "p_value": float(res.pvalue),
        "a_only_correct": b01,
        "b_only_correct": b10,
    }


def best_threshold(y_true, y_score, grid: int = 200) -> tuple[float, float]:
    """Threshold that maximises F1 on the given scores (used on validation only).

    Balanced class weights shift predicted probabilities, so a fixed 0.5 cut is unfair to
    some models. Picking the cut on validation keeps the test split untouched.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")
    lo, hi = float(np.min(y_score)), float(np.max(y_score))
    if hi <= lo:
        return 0.5, f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(lo, hi, grid):
        f = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_t, best_f1 = float(t), float(f)
    return best_t, best_f1


def format_table(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Markdown table with consistent float formatting (for the report)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else float_fmt.format(v))
    header = "| " + " | ".join(map(str, out.columns)) + " |"
    sep = "|" + "|".join(["---"] * len(out.columns)) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in out.itertuples(index=False)]
    return "\n".join([header, sep, *rows])
