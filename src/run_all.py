"""Runs the stages in order.

    python -m src.run_all --dataset ../Vina-Embroidery
    python -m src.run_all --stages eda,clustering
    python -m src.run_all --stages cnn --force-features

Each stage writes its own outputs under results/ and nothing else.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from . import baselines, clustering, cross_sample, eda, generative, multiview, neural, report_assets
from .config import Config, RESULTS_DIR, TABLES_DIR, ensure_dirs
from .data import load_prepared, prepare
from .features import build_feature_table

STAGES = ["data", "features", "eda", "clustering", "baselines", "mlp", "cnn", "vae",
          "crosssample", "multiview", "report"]


class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=None, help="path to the Vina-Embroidery folder")
    parser.add_argument("--stages", default="all", help="comma separated stages, or 'all'")
    parser.add_argument("--force-features", action="store_true", help="recompute features")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    stages = STAGES if args.stages == "all" else [s.strip() for s in args.stages.split(",")]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        print(f"unknown stages: {unknown}", file=sys.stderr)
        return 2

    ensure_dirs()
    cfg = Config.load()
    if args.dataset:
        cfg.dataset = str(Path(args.dataset).expanduser().resolve())
    if stages != ["report"] and not Path(cfg.dataset).exists():
        print(f"Dataset not found: {cfg.dataset}\nPass --dataset <path to Vina-Embroidery>",
              file=sys.stderr)
        return 2
    cfg.save()

    log = Logger(RESULTS_DIR / "run.log")
    log(f"\n=== run {time.strftime('%Y-%m-%d %H:%M:%S')} | dataset={cfg.dataset} ===")

    df: pd.DataFrame | None = None

    def frame() -> pd.DataFrame:
        nonlocal df
        if df is None:
            df = load_prepared(cfg)
            if "split" not in df.columns:
                df, _ = prepare(cfg)
        return df

    for stage in stages:
        t0 = time.time()
        log(f"\n--- stage: {stage} ---")
        try:
            if stage == "data":
                nonlocal_df, part = prepare(cfg)
                df = nonlocal_df
                log(f"    {len(df)} images | {df.group_id.nunique()} groups | "
                    f"test={int((df.split == 'test').sum())} val={int((df.split == 'val').sum())}")
            elif stage == "features":
                feats = build_feature_table(frame(), size=cfg.feature_size,
                                            force=args.force_features)
                log(f"    feature matrix: {feats.shape[0]} x {feats.shape[1] - 1}")
            elif stage == "eda":
                probe = eda.run(frame())
                log("    " + probe.to_string(index=False).replace("\n", "\n    "))
            elif stage == "clustering":
                m = clustering.run(frame())
                log(f"    best k-means silhouette={m[m.algorithm == 'kmeans'].silhouette.max():.3f}"
                    f" | best ARI={m.ari_vs_label.max():.3f}")
            elif stage == "baselines":
                m = baselines.run(cfg, frame(), verbose=not args.quiet)
                best = m.sort_values("f1", ascending=False).iloc[0]
                log(f"    best: {best.model} ({best.case}) f1={best.f1:.3f} auc={best.roc_auc:.3f}")
            elif stage == "mlp":
                neural.run_mlp(cfg, frame(), verbose=not args.quiet)
            elif stage == "cnn":
                neural.run_cnn(cfg, frame(), verbose=not args.quiet)
            elif stage == "vae":
                generative.run(cfg, frame(), verbose=not args.quiet)
            elif stage == "crosssample":
                cs = cross_sample.run(cfg, frame())
                log("    " + cs[["case", "scheme", "accuracy", "f1", "roc_auc"]]
                    .to_string(index=False).replace("\n", "\n    "))
            elif stage == "multiview":
                multiview.run(cfg, frame())
            elif stage == "report":
                out = report_assets.build()
                numbers = report_assets.headline_numbers()
                log(f"    wrote {out}")
                log(f"    headline numbers: {numbers}")
        except Exception as exc:  # keep going: report the failing stage clearly
            log(f"    STAGE FAILED: {type(exc).__name__}: {exc}")
            raise
        log(f"    ({time.time() - t0:.1f}s)")

    log("\n=== run complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
