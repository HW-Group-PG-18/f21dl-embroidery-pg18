"""Is the high accuracy real, or is it recognising the garment?

The main results use a group split, so no defect is seen twice. But all images still come
from four pieces of fabric, and normal/abnormal photos of the same piece were taken in
different sessions. A model can therefore learn "this fabric under this light is normal"
and score well without learning anything about defects.

This module measures that. Leave-one-sample-out: train on three pieces of fabric, test on
the fourth, which the model has never seen. Compare that with the group split. A large gap
means the group-split numbers describe within-garment separation, not general detection.

It also prints brightness and contrast per class and view, because a lighting difference
between the normal and abnormal sessions is the most likely shortcut.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config, TABLES_DIR
from .data import view_case_frame
from .evaluation import classification_metrics
from .features import build_feature_table

K_BEST = 40


def _model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(partial(mutual_info_classif, random_state=42), k=K_BEST)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])


def leave_one_sample_out(data: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for sample in sorted(data["sample"].unique()):
        tr = data[data["sample"] != sample]
        te = data[data["sample"] == sample]
        est = _model().fit(tr[cols], tr.label_id)
        prob = est.predict_proba(te[cols])[:, 1]
        met = classification_metrics(te.label_id, (prob >= 0.5).astype(int), prob)
        rows.append({"scheme": f"leave_out_{sample}", "n_train": len(tr), "n_test": len(te), **met})

    # Same-sample floor: does the model work at all when the garment is familiar?
    tr = data[data.split == "train"]
    te = data[data.split == "test"]
    est = _model().fit(tr[cols], tr.label_id)
    prob = est.predict_proba(te[cols])[:, 1]
    rows.append({"scheme": "group_split (same garments)", "n_train": len(tr), "n_test": len(te),
                 **classification_metrics(te.label_id, (prob >= 0.5).astype(int), prob)})
    return pd.DataFrame(rows)


def lighting_by_class(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    stats = pd.read_csv(TABLES_DIR / "pixel_stats.csv")
    merged = stats.merge(df[["image_id", "label", "view", "sample"]], on="image_id")
    out = (merged.groupby(["view", "label"])[["brightness", "contrast", "sharpness"]]
           .mean().round(3).reset_index())
    out.to_csv(TABLES_DIR / "lighting_by_class.csv", index=False)
    return out


def run(cfg: Config, df: pd.DataFrame) -> pd.DataFrame:
    feats = build_feature_table(df, size=cfg.feature_size)
    data = df.merge(feats, on="image_id")
    cols = [c for c in feats.columns if c != "image_id"]
    loso = leave_one_sample_out(data, cols)

    # Also break it down by view, since the straight view is where the perfect scores appeared
    extra = []
    for case in ["straight_only", "straight_45", "straight_60"]:
        sub = view_case_frame(data, case)
        for sample in sorted(sub["sample"].unique()):
            tr = sub[sub["sample"] != sample]
            te = sub[sub["sample"] == sample]
            if tr.label_id.nunique() < 2 or te.label_id.nunique() < 2:
                continue
            est = _model().fit(tr[cols], tr.label_id)
            prob = est.predict_proba(te[cols])[:, 1]
            met = classification_metrics(te.label_id, (prob >= 0.5).astype(int), prob)
            extra.append({"scheme": f"loso_{sample}", "case": case, "n_train": len(tr),
                          "n_test": len(te), **met})
    out = pd.concat([loso.assign(case="all_views"), pd.DataFrame(extra)], ignore_index=True)
    out.to_csv(TABLES_DIR / "cross_sample.csv", index=False)
    lighting_by_class(cfg, df)
    return out
