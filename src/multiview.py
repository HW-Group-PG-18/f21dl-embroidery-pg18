"""H1 and H2.

H1 is checked twice on purpose. First on the full test set of each condition, which is what
a deployed system would see, but the conditions contain different images. Then on the 90
degree test images only, which appear in all three conditions, so the training data is the
only thing that changes. McNemar's test on those paired predictions.

H2 uses one model per angle and a train-angle by test-angle matrix.
"""
from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Config, FIGURES_DIR, TABLES_DIR, VIEW_CASES, VIEW_LABELS, VIEWS
from .evaluation import classification_metrics, group_bootstrap_ci, mcnemar
from .features import build_feature_table

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
CASES = list(VIEW_CASES)


def _load_predictions() -> pd.DataFrame:
    frames = []
    for name in ["baselines_predictions.csv", "mlp_predictions.csv", "cnn_predictions.csv"]:
        f = TABLES_DIR / name
        if f.exists():
            frames.append(pd.read_csv(f))
    if not frames:
        raise FileNotFoundError("Run the modelling stages before the multiview stage")
    return pd.concat(frames, ignore_index=True)


def _metrics_row(df: pd.DataFrame, **extra) -> dict:
    met = classification_metrics(df.label_id, df.pred, df.prob)
    ci = group_bootstrap_ci(df, n_boot=400)
    return {**extra, **met, "acc_lo": ci["accuracy"][0], "acc_hi": ci["accuracy"][1],
            "f1_lo": ci["f1"][0], "f1_hi": ci["f1"][1],
            "auc_lo": ci["roc_auc"][0], "auc_hi": ci["roc_auc"][1]}


def h1_analysis(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full-set and paired (common-image) comparison across view conditions."""
    full_rows, paired_rows, tests = [], [], []

    # Images present in all three conditions (the 90 degree test images)
    sets = [set(preds[preds.case == c].image_id) for c in CASES]
    common = set.intersection(*sets)

    for (model, case), grp in preds.groupby(["model", "case"]):
        full_rows.append(_metrics_row(grp, model=model, case=case,
                                      views="+".join(VIEW_CASES[case])))
        paired = grp[grp.image_id.isin(common)]
        paired_rows.append(_metrics_row(paired, model=model, case=case, n_common=len(common)))

    full = pd.DataFrame(full_rows)
    paired = pd.DataFrame(paired_rows)

    # McNemar: straight-only vs each multi-view condition, per model, on common images
    base = "straight_only"
    for model in sorted(preds.model.unique()):
        a = (preds[(preds.model == model) & (preds.case == base)]
             .drop_duplicates("image_id").set_index("image_id"))
        a = a[a.index.isin(common)]
        if a.empty:
            continue
        for case in CASES[1:]:
            b = (preds[(preds.model == model) & (preds.case == case)]
                 .drop_duplicates("image_id").set_index("image_id"))
            idx = a.index.intersection(b.index)
            if len(idx) == 0:
                continue
            res = mcnemar(a.loc[idx, "label_id"], a.loc[idx, "pred"], b.loc[idx, "pred"])
            tests.append({"model": model, "comparison": f"{base} vs {case}", "n": len(idx), **res})

    full.to_csv(TABLES_DIR / "h1_view_conditions.csv", index=False)
    paired.to_csv(TABLES_DIR / "h1_paired_common.csv", index=False)
    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(TABLES_DIR / "h1_mcnemar.csv", index=False)

    # ---- figure: accuracy with CI, paired subset ----
    keep = [m for m in ["decision_tree", "naive_bayes", "logistic_regression", "perceptron",
                        "knn", "random_forest", "gradient_boosting", "mlp", "cnn"]
            if m in paired.model.unique()]
    width = 0.26
    fig, ax = plt.subplots(figsize=(0.95 * len(keep) + 2.2, 3.4))
    for i, case in enumerate(CASES):
        sub = paired[paired.case == case].set_index("model").loc[keep]
        x = np.arange(len(keep)) + (i - 1) * width
        ax.bar(x, sub.accuracy, width, label="+".join(VIEW_CASES[case]),
               yerr=[sub.accuracy - sub.acc_lo, sub.acc_hi - sub.accuracy],
               capsize=2, alpha=0.9)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="chance")
    ax.set_xticks(range(len(keep)), [k.replace("_", " ") for k in keep], rotation=30, ha="right")
    ax.set_ylabel("accuracy on 90° test images")
    ax.set_ylim(0, 1.02)
    ax.set_title("H1: effect of adding a second view to training (paired images, 95% group bootstrap CI)")
    ax.legend(frameon=False, fontsize=7, ncol=3, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "h1_multi_view.png", bbox_inches="tight")
    plt.close(fig)
    return full, paired, tests_df


def _fit_view_model(X, y, groups):
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("select", SelectKBest(partial(mutual_info_classif, random_state=42), k=40)),
        ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear")),
    ])
    grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    cv = StratifiedGroupKFold(n_splits=max(2, min(4, len(np.unique(groups)))),
                              shuffle=True, random_state=42)
    return GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1).fit(X, y, groups=groups)


def h2_analysis(cfg: Config, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-view models and a train-view x test-view transfer matrix."""
    feats = build_feature_table(df, size=cfg.feature_size)
    data = df.merge(feats, on="image_id")
    cols = [c for c in feats.columns if c != "image_id"]

    own_rows, matrix_rows = [], []
    for train_view in VIEWS:
        tr = data[(data.split == "train") & (data.view == train_view)]
        est = _fit_view_model(tr[cols].to_numpy(), tr.label_id.to_numpy(), tr.group_id)
        for test_view in VIEWS:
            te = data[(data.split == "test") & (data.view == test_view)]
            prob = est.predict_proba(te[cols].to_numpy())[:, 1]
            met = classification_metrics(te.label_id.to_numpy(), (prob >= 0.5).astype(int), prob)
            matrix_rows.append({"train_view": train_view, "test_view": test_view,
                                "n_test": len(te), **met})
            if train_view == test_view:
                ci = group_bootstrap_ci(
                    pd.DataFrame({"label_id": te.label_id, "pred": (prob >= 0.5).astype(int),
                                  "prob": prob, "group_id": te.group_id}), n_boot=400)
                own_rows.append({"view": train_view, "n_train": len(tr), "n_test": len(te),
                                 "cv_f1": float(est.best_score_), **met,
                                 "acc_lo": ci["accuracy"][0], "acc_hi": ci["accuracy"][1]})

    own = pd.DataFrame(own_rows)
    matrix = pd.DataFrame(matrix_rows)
    own.to_csv(TABLES_DIR / "h2_per_view.csv", index=False)
    matrix.to_csv(TABLES_DIR / "h2_cross_view.csv", index=False)

    # ---- figures ----
    auc_mat = matrix.pivot(index="train_view", columns="test_view", values="roc_auc").reindex(
        index=VIEWS, columns=VIEWS)
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    im = ax.imshow(auc_mat.to_numpy(), cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(VIEWS)), [VIEW_LABELS[v] for v in VIEWS])
    ax.set_yticks(range(len(VIEWS)), [VIEW_LABELS[v] for v in VIEWS])
    ax.set_xlabel("tested on")
    ax.set_ylabel("trained on")
    ax.grid(False)
    for i in range(len(VIEWS)):
        for j in range(len(VIEWS)):
            ax.text(j, i, f"{auc_mat.iat[i, j]:.3f}", ha="center", va="center",
                    color="white" if auc_mat.iat[i, j] < 0.8 else "black")
    ax.set_title("H2: cross-view transfer (ROC-AUC)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "h2_cross_view.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.bar([VIEW_LABELS[v] for v in own.view], own.f1)
    for i, r in own.iterrows():
        ax.text(i, r.f1 + 0.02, f"{r.f1:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 (own-view model)")
    ax.set_title("H2: single-view performance by angle")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "h2_per_view.png", bbox_inches="tight")
    plt.close(fig)
    return own, matrix


def run(cfg: Config, df: pd.DataFrame) -> dict:
    preds = _load_predictions()
    full, paired, tests = h1_analysis(preds)
    own, matrix = h2_analysis(cfg, df)

    summary = {
        "h1_full": full.to_dict(orient="records"),
        "h1_paired": paired.to_dict(orient="records"),
        "h1_mcnemar": tests.to_dict(orient="records"),
        "h2_per_view": own.to_dict(orient="records"),
        "h2_cross_view": matrix.to_dict(orient="records"),
    }
    (TABLES_DIR / "multiview_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
