"""Loading the manifest, checking the data, and building the train/val/test split.

A few things worth knowing before reading this file:

* manifest.csv is the source of truth for path, angle and label.
* Photos of the same sample and defect type share a group_id. Those frames are seconds
  apart, so they are almost the same picture. We split on groups, otherwise the test set
  is full of photos the model has effectively already seen.
* The splits/ folder that came with the dataset is checked in audit_supplied_splits() but
  not used. Its group-to-split assignment moves between view cases (see that function).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import Config, TABLES_DIR, VIEW_CASES, VIEWS

LABEL_MAP = {"normal": 0, "abnormal": 1}


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def dataset_root(dataset: str | Path) -> Path:
    root = Path(dataset)
    if (root / "raw" / "vai_theu" / "manifest.csv").exists():
        return root
    for cand in root.rglob("manifest.csv"):
        if cand.parent.name == "vai_theu":
            return cand.parents[2]
    raise FileNotFoundError(f"Could not find raw/vai_theu/manifest.csv under {root}")


def load_manifest(dataset: str | Path) -> pd.DataFrame:
    """Load the manifest, add absolute paths and integer labels."""
    root = dataset_root(dataset)
    df = pd.read_csv(root / "raw" / "vai_theu" / "manifest.csv")
    df = df.rename(columns={"angle": "view"})
    df["view"] = df["view"].astype(str)
    df["label_id"] = df["label"].map(LABEL_MAP).astype(int)
    df["path"] = df["rel_path"].apply(lambda p: str(root / "raw" / "vai_theu" / p))
    df["sample"] = df["sample_id"].astype(str)
    # Normal images in the 45/60 folders reuse the defect-type folder name, so a group can
    # contain both labels (see docs/DATA_NOTES.md).  Record that instead of hiding it.
    grp = df.groupby("group_id")["label"]
    df["group_n_labels"] = df["group_id"].map(grp.nunique())
    df["group_is_mixed"] = (df["group_n_labels"] > 1).astype(int)
    return df


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for view in VIEWS:
        sub = df[df.view == view]
        rows.append(
            {
                "view": view,
                "images": len(sub),
                "normal": int((sub.label_id == 0).sum()),
                "abnormal": int((sub.label_id == 1).sum()),
                "samples": sub["sample"].nunique(),
                "groups": sub.group_id.nunique(),
            }
        )
    rows.append(
        {
            "view": "all",
            "images": len(df),
            "normal": int((df.label_id == 0).sum()),
            "abnormal": int((df.label_id == 1).sum()),
            "samples": df["sample"].nunique(),
            "groups": df.group_id.nunique(),
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Quality control
# --------------------------------------------------------------------------------------
def _md5(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def quality_control(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check existence, decodability, resolution, exact duplicates, group/label structure."""
    valid, rows = [], []
    hashes: dict[str, list[str]] = {}
    for _, r in df.iterrows():
        rec = {
            "image_id": r.image_id,
            "view": r.view,
            "label": r.label,
            "group_id": r.group_id,
            "sample": r["sample"],
            "exists": Path(r.path).exists(),
            "readable": False,
            "width": np.nan,
            "height": np.nan,
            "format": None,
            "bytes": np.nan,
            "md5": None,
        }
        if rec["exists"]:
            rec["bytes"] = Path(r.path).stat().st_size
            try:
                with Image.open(r.path) as im:
                    rec["width"], rec["height"] = im.size
                    rec["format"] = im.format
                    im.verify()
                rec["readable"] = True
                rec["md5"] = _md5(r.path)
                hashes.setdefault(rec["md5"], []).append(r.rel_path)
            except Exception:  # pragma: no cover - corrupt file branch
                rec["readable"] = False
        if rec["readable"]:
            valid.append(r.name)
        rows.append(rec)
    qc = pd.DataFrame(rows)

    dup_mask = qc["md5"].notna() & qc["md5"].duplicated(keep="first")
    qc["exact_duplicate"] = dup_mask
    qc.to_csv(TABLES_DIR / "qc_report.csv", index=False)

    structure = (
        df.groupby(["group_id", "sample"])
        .agg(
            images=("image_id", "size"),
            normal=("label_id", lambda s: int((s == 0).sum())),
            abnormal=("label_id", lambda s: int((s == 1).sum())),
            views=("view", lambda s: ",".join(sorted(set(s)))),
        )
        .reset_index()
    )
    structure["mixed_label"] = (structure.normal > 0) & (structure.abnormal > 0)
    return qc, structure


def audit_supplied_splits(dataset: str | Path) -> pd.DataFrame:
    """Quantify inconsistencies in the dataset's own split CSVs.

    The supplied splits define group -> split independently per view case, so a group can
    be *test* in ``straight_only`` and *train* in ``straight_60``.  Comparing metrics
    across those files therefore mixes the view effect with a split effect.
    """
    root = dataset_root(dataset)
    base = root / "splits" / "vai_theu"
    rows = []
    per_case: dict[str, pd.Series] = {}
    for case in sorted(p.name for p in base.iterdir() if p.is_dir()):
        f = base / case / "split.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        assign = s.groupby("group_id")["split"].first()
        per_case[case] = assign
        rows.append(
            {
                "case": case,
                "images": len(s),
                "groups": s.group_id.nunique(),
                "test_groups": s[s.split == "test"].group_id.nunique(),
                "test_normal_frac": round(
                    float((s[s.split == "test"].label == "normal").mean()), 3
                ),
            }
        )
    out = pd.DataFrame(rows)
    if len(per_case) > 1:
        cases = list(per_case)
        inconsistent = 0
        shared = set().union(*[set(v.index) for v in per_case.values()])
        for g in shared:
            vals = {c: per_case[c].get(g) for c in cases if g in per_case[c].index}
            if len(set(vals.values())) > 1:
                inconsistent += 1
        out.attrs["inconsistent_groups_across_cases"] = inconsistent
        out["groups_reassigned_across_cases"] = [
            "" for _ in range(len(out))
        ]  # filled below for readability
        out.loc[out.index[0], "groups_reassigned_across_cases"] = (
            f"{inconsistent}/{len(shared)} groups change split between view cases"
        )
    return out


# --------------------------------------------------------------------------------------
# Canonical partition
# --------------------------------------------------------------------------------------
@dataclass
class Partition:
    assignments: dict[str, str]        # group_id -> train/val/test
    seed: int
    stats: pd.DataFrame


def _group_table(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby("group_id")
        .agg(
            sample=("sample", "first"),
            images=("image_id", "size"),
            normal=("label_id", lambda s: int((s == 0).sum())),
            abnormal=("label_id", lambda s: int((s == 1).sum())),
            views=("view", lambda s: tuple(sorted(set(s)))),
        )
        .reset_index()
    )
    g["has_all_views"] = g.views.apply(lambda v: set(v) == set(VIEWS))
    g["normal_frac"] = g.normal / g.images
    return g


def _per_view_class_counts(df: pd.DataFrame, groups: set[str], min_per_class: int) -> bool:
    """Every evaluated view must contain both classes, otherwise accuracy is undefined."""
    sub = df[df.group_id.isin(groups)]
    for view in VIEWS:
        v = sub[sub.view == view]
        if len(v) == 0:
            continue
        if (v.label_id == 0).sum() < min_per_class or (v.label_id == 1).sum() < min_per_class:
            return False
    return True


def build_partition(
    df: pd.DataFrame,
    test_groups: int = 6,
    val_groups: int = 6,
    n_candidates: int = 4000,
    seed: int = 42,
    min_per_view_class: int = 8,
) -> Partition:
    """Choose the split.

    Constraints, all about composition and none about model scores:

    1. no group in two splits;
    2. test and validation each hold a normal-only group, groups from at least two samples,
       and a group with all three views. The normal-only group is needed because normal 90
       degree photos only exist in the four normal-only groups, so a test set without one has
       no straight-on negatives and per-view accuracy is undefined;
    3. every view has at least min_per_view_class images of each class in test and val;
    4. test and val class balance close to the dataset prior.

    We try seeded permutations and keep the cheapest. Nothing here looks at how any model
    performs, which is the point.
    """
    g = _group_table(df)
    g["kind"] = np.where(g.abnormal == 0, "normal_only", np.where(g.normal == 0, "abnormal_only", "mixed"))
    n_required = test_groups + val_groups
    if len(g) < n_required + 5:
        raise ValueError("Not enough groups for the requested split sizes")

    target_norm = float((df.label_id == 0).mean())
    best: tuple[float, dict[str, str], int] | None = None

    for s in range(n_candidates):
        rng = np.random.default_rng(seed + s)
        order = rng.permutation(len(g))
        test = g.iloc[order[:test_groups]]
        val = g.iloc[order[test_groups:n_required]]
        for split in (test, val):
            kinds = set(split.kind)
            if "normal_only" not in kinds or split["sample"].nunique() < 2:
                break
            if not _per_view_class_counts(df, set(split.group_id), min_per_view_class):
                break
            if not split.has_all_views.any():
                break
        else:
            train_groups = set(g.group_id) - set(test.group_id) - set(val.group_id)
            if not _per_view_class_counts(df, train_groups, min_per_view_class):
                continue
            cost = (
                abs(test.normal_frac.mean() - target_norm)
                + abs(val.normal_frac.mean() - target_norm)
                + abs(test.normal_frac.mean() - val.normal_frac.mean())
            )
            if best is None or cost < best[0]:
                assign = {gid: "train" for gid in g.group_id}
                assign.update({gid: "test" for gid in test.group_id})
                assign.update({gid: "val" for gid in val.group_id})
                best = (float(cost), assign, seed + s)

    if best is None:  # pragma: no cover
        raise RuntimeError("No valid partition found - relax min_per_view_class")

    cost, assign, used_seed = best

    stats = g.copy()
    stats["split"] = stats.group_id.map(assign)
    stats["cost"] = cost
    stats["candidate_seed"] = used_seed
    return Partition(assignments=assign, seed=used_seed, stats=stats)


def attach_partition(df: pd.DataFrame, part: Partition) -> pd.DataFrame:
    out = df.copy()
    out["split"] = out.group_id.map(part.assignments)
    if out["split"].isna().any():
        missing = sorted(out.loc[out.split.isna(), "group_id"].unique())
        raise ValueError(f"Groups without a split assignment: {missing}")
    return out


def view_case_frame(df: pd.DataFrame, case: str) -> pd.DataFrame:
    """Rows belonging to one view condition (angles limited, splits unchanged)."""
    views = VIEW_CASES[case]
    return df[df.view.isin(views)].copy()


def prepare(cfg: Config, save: bool = True) -> tuple[pd.DataFrame, Partition]:
    """Full data preparation: load, QC, partition, persist artefacts."""
    df = load_manifest(cfg.dataset)
    qc, structure = quality_control(df)
    part = build_partition(
        df, test_groups=cfg.test_groups, val_groups=cfg.val_groups, seed=cfg.seed
    )
    df = attach_partition(df, part)

    summary = dataset_summary(df)
    split_summary = (
        df.groupby(["split", "view"])
        .agg(images=("image_id", "size"), normal=("label_id", lambda s: int((s == 0).sum())))
        .reset_index()
    )
    if save:
        summary.to_csv(TABLES_DIR / "dataset_summary.csv", index=False)
        split_summary.to_csv(TABLES_DIR / "partition_by_view.csv", index=False)
        structure.to_csv(TABLES_DIR / "group_structure.csv", index=False)
        df.drop(columns=["path"]).to_csv(TABLES_DIR / "manifest_with_split.csv", index=False)
        try:
            audit_supplied_splits(cfg.dataset).to_csv(
                TABLES_DIR / "supplied_split_audit.csv", index=False
            )
        except Exception as exc:  # pragma: no cover - dataset layout differences
            (TABLES_DIR / "supplied_split_audit.txt").write_text(f"audit failed: {exc}\n")
        (TABLES_DIR / "partition.json").write_text(
            json.dumps(
                {
                    "seed": part.seed,
                    "assignments": part.assignments,
                    "test_groups": cfg.test_groups,
                    "val_groups": cfg.val_groups,
                },
                indent=2,
            )
        )
    return df, part


def load_prepared(cfg: Config) -> pd.DataFrame:
    """Reload a prepared manifest (with ``split``) from the cached table."""
    cached = TABLES_DIR / "manifest_with_split.csv"
    if cached.exists():
        df = pd.read_csv(cached)
        df["path"] = df["rel_path"].apply(
            lambda p: str(dataset_root(cfg.dataset) / "raw" / "vai_theu" / p)
        )
        df["view"] = df["view"].astype(str)
        return df
    df, _ = prepare(cfg, save=True)
    return df
