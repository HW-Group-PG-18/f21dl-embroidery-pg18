"""Collects the generated tables into one markdown file for the report.

Nothing here computes results; it just reads the CSVs and lays them out so we can copy from
one place instead of hunting through results/tables/.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import FIGURES_DIR, TABLES_DIR, VIEW_CASES
from .evaluation import format_table

OUT = TABLES_DIR.parent / "report_tables.md"


def _read(name: str) -> pd.DataFrame | None:
    f = TABLES_DIR / name
    return pd.read_csv(f) if f.exists() else None


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body.strip()}\n"


def build() -> Path:
    parts = ["# Generated report tables",
             "\nAll tables below are produced by `python -m src.run_all`; do not edit by hand.\n"]

    # ---- dataset ----
    ds = _read("dataset_summary.csv")
    if ds is not None:
        parts.append(_section("Dataset composition (R2)", format_table(ds)))

    pv = _read("partition_by_view.csv")
    if pv is not None:
        wide = pv.pivot_table(index="split", columns="view", values=["images", "normal"],
                              aggfunc="sum", fill_value=0)
        wide.columns = [f"{a}_{b}" for a, b in wide.columns]
        wide = wide.reset_index().sort_values("split", key=lambda s: s.map(
            {"train": 0, "val": 1, "test": 2}))
        parts.append(_section("Canonical group-level partition", format_table(wide)))

    qc = _read("qc_report.csv")
    if qc is not None:
        summary = pd.DataFrame([{
            "images": len(qc),
            "missing_files": int((~qc["exists"]).sum()),
            "unreadable": int((~qc["readable"].astype(bool)).sum()),
            "exact_duplicates": int(qc["exact_duplicate"].sum()),
            "distinct_resolutions": int(qc[["width", "height"]].drop_duplicates().shape[0]),
            "min_size": f"{int(qc.width.min())}x{int(qc.height.min())}",
            "max_size": f"{int(qc.width.max())}x{int(qc.height.max())}",
        }])
        parts.append(_section("Quality control (R2)", format_table(summary)))

    gs = _read("group_structure.csv")
    if gs is not None:
        parts.append(_section(
            "Group structure: 35 sample x defect-type groups, 4 physical samples",
            f"- groups: **{len(gs)}**\n- groups containing both labels: **{int(gs.mixed_label.sum())}**\n"
            f"- images in the largest group: **{int(gs.images.max())}**",
        ))

    sp = _read("shortcut_probe.csv")
    if sp is not None:
        parts.append(_section(
            "Shortcut probe: is the label predictable from lighting alone? (R2 caveat)",
            format_table(sp[["features", "n_features", "accuracy", "f1", "roc_auc"]]),
        ))

    fs = _read("feature_selection.csv")
    if fs is not None:
        top = fs[fs.kept_topk].head(15)
        parts.append(_section("Top-15 selected features (R2)",
                              format_table(top[["feature", "mutual_info", "rf_importance"]])))

    cl = _read("cluster_metrics.csv")
    if cl is not None:
        best_km = cl[cl.algorithm == "kmeans"].sort_values("silhouette", ascending=False).head(2)
        best_gm = cl[cl.algorithm == "gmm_em"].sort_values("bic").head(2)
        parts.append(_section("Clustering (R2): best k-means (silhouette) / best GMM (BIC)",
                              format_table(pd.concat([best_km, best_gm]))))
    cc = _read("cluster_composition.csv")
    if cc is not None:
        parts.append(_section("What the k-means clusters contain", format_table(cc)))

    # ---- classical ----
    bl = _read("baselines_metrics.csv")
    if bl is not None:
        cols = ["case", "model", "cv_f1", "accuracy", "precision", "recall", "f1",
                "balanced_accuracy", "roc_auc", "f1_lo", "f1_hi"]
        for case in VIEW_CASES:
            sub = bl[bl.case == case].sort_values("f1", ascending=False)[cols]
            parts.append(_section(f"Classical learners (R3) - {case}",
                                  "".join(["+".join(VIEW_CASES[case]), "\n\n", format_table(sub)])))

    # ---- neural ----
    mlp = _read("mlp_metrics.csv")
    cnn = _read("cnn_metrics.csv")
    neural_frames = [f for f in (mlp, cnn) if f is not None]
    if neural_frames:
        nn = pd.concat(neural_frames, ignore_index=True)
        cols = ["case", "model", "accuracy", "precision", "recall", "f1", "balanced_accuracy",
                "roc_auc", "f1_lo", "f1_hi", "best_params"]
        parts.append(_section("Neural networks (R4)", format_table(nn[cols])))

    # ---- generative ----
    vae = _read("vae_anomaly.csv")
    if vae is not None:
        parts.append(_section("Generative anomaly detection: VAE (trained on normals only)",
                              format_table(vae[["model", "n", "accuracy", "precision", "recall",
                                                "f1", "roc_auc", "threshold"]])))

    # ---- generalisation check ----
    cs = _read("cross_sample.csv")
    if cs is not None:
        cols = ["case", "scheme", "n_train", "n_test", "accuracy", "f1", "roc_auc"]
        parts.append(_section("Generalisation check: leave-one-sample-out vs the group split",
                              format_table(cs[cols])))
    lb = _read("lighting_by_class.csv")
    if lb is not None:
        parts.append(_section("Mean pixel statistics per view and class (shortcut check)",
                              format_table(lb)))

    # ---- H1 / H2 ----
    h1 = _read("h1_paired_common.csv")
    if h1 is not None:
        cols = ["case", "model", "n_common", "accuracy", "f1", "roc_auc", "acc_lo", "acc_hi"]
        parts.append(_section("H1 - paired comparison on identical 90° test images",
                              format_table(h1[cols].sort_values(["model", "case"]))))
    mc = _read("h1_mcnemar.csv")
    if mc is not None and len(mc):
        parts.append(_section("H1 - McNemar exact test (straight-only vs multi-view)",
                              format_table(mc)))
    h2 = _read("h2_per_view.csv")
    if h2 is not None:
        parts.append(_section("H2 - single-view models", format_table(h2[["view", "n_train", "n_test", "cv_f1", "accuracy", "f1", "roc_auc"]])))
    h2x = _read("h2_cross_view.csv")
    if h2x is not None:
        wide = h2x.pivot(index="train_view", columns="test_view", values="roc_auc")
        wide.columns = [f"test_{c}" for c in wide.columns]
        parts.append(_section("H2 - cross-view transfer (ROC-AUC)",
                              format_table(wide.reset_index())))

    figs = sorted(p.name for p in FIGURES_DIR.glob("*.png"))
    parts.append(_section("Figures", "\n".join(f"- `figures/{f}`" for f in figs)))

    OUT.write_text("\n".join(parts))
    return OUT


def headline_numbers() -> dict:
    """Compact dict of the numbers the abstract/discussion will quote."""
    out: dict[str, object] = {}
    bl = _read("baselines_metrics.csv")
    if bl is not None:
        best = bl.sort_values("f1", ascending=False).iloc[0]
        out["best_classical"] = {"model": best.model, "case": best.case,
                                 "f1": round(float(best.f1), 3), "auc": round(float(best.roc_auc), 3)}
    for name, key in [("mlp_metrics.csv", "mlp"), ("cnn_metrics.csv", "cnn")]:
        df = _read(name)
        if df is not None:
            out[key] = {r.case: {"f1": round(float(r.f1), 3), "auc": round(float(r.roc_auc), 3)}
                        for r in df.itertuples()}
    vae = _read("vae_anomaly.csv")
    if vae is not None:
        out["vae"] = {"auc": round(float(vae.auc.iloc[0]), 3), "f1": round(float(vae.f1.iloc[0]), 3)}
    h1 = _read("h1_paired_common.csv")
    if h1 is not None:
        out["h1_delta"] = {
            f"{r.model}:{r.case}": round(float(r.accuracy), 3) for r in h1.itertuples()
        }
    (TABLES_DIR / "headline_numbers.json").write_text(json.dumps(out, indent=2))
    return out
