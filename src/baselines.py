"""The classical learners for R3, all on the same splits.

Seven models: decision tree, Gaussian naive Bayes, logistic regression, perceptron, k-NN,
random forest and histogram gradient boosting. Tuning uses StratifiedGroupKFold inside the
training split with F1 as the criterion, so a defect group never spans the fold
boundary, and the test split is scored once per model per view condition.

Every model gets the same front end: standardise, keep the 40 features with the highest
mutual information (fitted inside each fold), then the classifier.
"""
from __future__ import annotations

import json
from functools import partial

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import Config, MODELS_DIR, TABLES_DIR, VIEW_CASES
from .data import view_case_frame
from .evaluation import best_threshold, classification_metrics, group_bootstrap_ci
from .features import build_feature_table

K_BEST = 40
RANDOM_STATE = 42


def model_zoo() -> dict[str, tuple[object, dict]]:
    """name -> (estimator, hyper-parameter grid)."""
    return {
        "decision_tree": (
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [3, 5, 7, 10, None],
                "clf__min_samples_leaf": [1, 5, 10],
            },
        ),
        "naive_bayes": (
            GaussianNB(),
            {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear"),
            {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        ),
        "perceptron": (
            Perceptron(max_iter=3000, class_weight="balanced", random_state=RANDOM_STATE),
            {"clf__alpha": [1e-4, 1e-3, 1e-2]},
        ),
        "knn": (
            KNeighborsClassifier(),
            {"clf__n_neighbors": [3, 5, 11, 21], "clf__weights": ["uniform", "distance"]},
        ),
        "random_forest": (
            RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample",
                                   random_state=RANDOM_STATE, n_jobs=-1),
            {"clf__max_depth": [None, 10], "clf__min_samples_leaf": [1, 3]},
        ),
        "gradient_boosting": (
            HistGradientBoostingClassifier(random_state=RANDOM_STATE,
                                           class_weight="balanced",
                                           early_stopping=True, validation_fraction=0.15),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [3, 5], "clf__max_iter": [200, 400]},
        ),
    }


def make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("select", SelectKBest(partial(mutual_info_classif, random_state=RANDOM_STATE), k=K_BEST)),
            ("clf", estimator),
        ]
    )


def scores(est, X) -> np.ndarray:
    """Probability of the abnormal class, or a decision-function fallback (perceptron)."""
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        d = est.decision_function(X)
        return expit(d if d.ndim == 1 else d[:, 1])
    return est.predict(X).astype(float)  # pragma: no cover


def run(cfg: Config, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Train every learner in every view condition; write metrics and predictions."""
    feats = build_feature_table(df, size=cfg.feature_size)
    data = df.merge(feats, on="image_id", how="left")
    feature_cols = [c for c in feats.columns if c != "image_id"]

    all_metrics, all_preds = [], []
    for case in VIEW_CASES:
        sub = view_case_frame(data, case)
        train = sub[sub.split == "train"]
        val = sub[sub.split == "val"]
        test = sub[sub.split == "test"]
        if verbose:
            print(f"  case {case}: train={len(train)} test={len(test)}")
        for name, (estimator, grid) in model_zoo().items():
            # Stratified group folds keep both classes in each fold. Plain GroupKFold left
            # some straight-only folds single-class, which made the AUC score NaN and the
            # model selection meaningless.
            search = GridSearchCV(
                make_pipeline(estimator), grid,
                cv=StratifiedGroupKFold(
                    n_splits=max(2, min(cfg.cv_folds, train.group_id.nunique())),
                    shuffle=True, random_state=cfg.seed),
                # F1, not ROC-AUC: with 90 degree images only, some folds hold no normal
                # images, where AUC is undefined but F1 is not.
                scoring="f1", n_jobs=-1, refit=True,
            )
            search.fit(train[feature_cols].to_numpy(), train.label_id.to_numpy(),
                       groups=train.group_id)
            est = search.best_estimator_
            prob = scores(est, test[feature_cols].to_numpy())
            # Calibrate the decision threshold on validation, not on test
            thr, val_f1 = best_threshold(val.label_id.to_numpy(),
                                         scores(est, val[feature_cols].to_numpy()))
            pred = (prob >= thr).astype(int)
            met = classification_metrics(test.label_id.to_numpy(), pred, prob)
            ci = group_bootstrap_ci(
                pd.DataFrame({"label_id": test.label_id.to_numpy(), "pred": pred,
                              "prob": prob, "group_id": test.group_id.to_numpy()}),
                n_boot=400, seed=cfg.seed,
            )
            all_metrics.append(
                {
                    "case": case, "model": name, "family": "classical",
                    "cv_f1": float(search.best_score_),
                    "threshold": round(float(thr), 4),
                    "val_f1_at_threshold": round(float(val_f1), 3),
                    "best_params": json.dumps({k.replace("clf__", ""): v
                                               for k, v in search.best_params_.items()}),
                    **met,
                    "f1_lo": ci["f1"][0], "f1_hi": ci["f1"][1],
                    "auc_lo": ci["roc_auc"][0], "auc_hi": ci["roc_auc"][1],
                }
            )
            all_preds.append(
                test[["image_id", "group_id", "view", "label_id", "split"]].assign(
                    case=case, model=name, family="classical", pred=pred, prob=prob
                )
            )
            if verbose:
                print(f"    {name:22s} acc={met['accuracy']:.3f} f1={met['f1']:.3f} "
                      f"auc={met['roc_auc']:.3f} cv_f1={search.best_score_:.3f}")

    metrics = pd.DataFrame(all_metrics)
    preds = pd.concat(all_preds, ignore_index=True)
    metrics.to_csv(TABLES_DIR / "baselines_metrics.csv", index=False)
    preds.to_csv(TABLES_DIR / "baselines_predictions.csv", index=False)
    return metrics
