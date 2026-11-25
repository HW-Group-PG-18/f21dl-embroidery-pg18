"""Settings used by every stage: paths, seeds, image sizes, epochs.

Keeping them in one place means a run is reproducible: change the seed or the image size
here and every module picks it up. The pipeline writes the values it used to config.yaml,
so a result can always be traced back to the settings behind it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

def _find_default_dataset() -> Path:
    candidates = [
        REPO_ROOT / "Vina-Embroidery",
        REPO_ROOT.parent / "Vina-Embroidery",
        REPO_ROOT / "data" / "Vina-Embroidery",
        REPO_ROOT.parent / "data" / "Vina-Embroidery",
    ]
    for c in candidates:
        if c.exists():
            return c
    return REPO_ROOT / "Vina-Embroidery"

DEFAULT_DATASET = _find_default_dataset()  # auto-detected or use --dataset for custom location
RESULTS_DIR = REPO_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = RESULTS_DIR / "models"
FEATURES_DIR = RESULTS_DIR / "features"

VIEWS = ["straight", "45", "60"]
VIEW_LABELS = {"straight": "90° (straight)", "45": "45°", "60": "60°"}
# The three view conditions used to test H1 (pitch deck / coursework R2-R4).
VIEW_CASES = {
    "straight_only": ["straight"],
    "straight_45": ["straight", "45"],
    "straight_60": ["straight", "60"],
}


@dataclass
class Config:
    """Pipeline configuration."""

    dataset: str = str(DEFAULT_DATASET)
    seed: int = 42
    img_size: int = 160          # CNN input size (square crop)
    feature_size: int = 128      # size used for handcrafted features
    test_groups: int = 6         # group-level test size (must have all three views)
    val_groups: int = 6
    cv_folds: int = 5            # GroupKFold folds inside the training split
    bootstrap: int = 1000        # group-level bootstrap resamples for CIs
    n_jobs: int = -1
    cnn_epochs: int = 30
    cnn_patience: int = 6
    vae_epochs: int = 40
    mlp_hidden: tuple = (128, 64)

    def save(self) -> None:
        cfg = asdict(self)
        cfg.pop("dataset")  # machine-specific, so it never goes into the shared config file
        (REPO_ROOT / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        path = path or (REPO_ROOT / "config.yaml")
        if not path.exists():
            return cls()
        data: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def ensure_dirs() -> None:
    for d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, FEATURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
