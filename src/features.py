"""Turning each image into a fixed-length vector of numbers, plus feature selection.

The classical models in R3 (decision tree, naive Bayes, linear model, perceptron, k-NN)
cannot take pixels, so they need a vector. We build one out of colour, texture, edges,
frequency content and a rough lighting map, then keep the useful columns.

Families:
  colour     RGB and HSV means/spreads, hue and saturation histograms
  grey level intensity stats, 32-bin histogram entropy, Laplacian variance as sharpness
  texture    GLCM properties at three distances, LBP histogram
  edges      Sobel magnitude, Canny edge density, gradient direction histogram
  frequency  radial FFT band energies (stitch regularity shows up here)
  spatial    brightness of the four quadrants (catches lighting gradients)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from skimage import color, feature, filters
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from tqdm import tqdm

from .config import FEATURES_DIR, TABLES_DIR

GLCM_DISTANCES = (1, 2, 4)
GLCM_PROPS = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation")
LBP_BINS = 10
ORIENT_BINS = 8


def _glcm_features(gray8: np.ndarray) -> tuple[list[float], list[str]]:
    q = (gray8 // 32).astype(np.uint8)  # 8 grey levels keeps the GLCM tiny and stable
    m = graycomatrix(q, distances=list(GLCM_DISTANCES), angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                     levels=8, symmetric=True, normed=True)
    vals, names = [], []
    for prop in GLCM_PROPS:
        arr = graycoprops(m, prop).mean(axis=1)  # average over the 4 directions
        for d, v in zip(GLCM_DISTANCES, arr):
            vals.append(float(v))
            names.append(f"glcm_{prop}_d{d}")
    return vals, names


def extract_features(path: str, size: int = 128) -> tuple[np.ndarray, list[str]]:
    """Extract one fixed-length descriptor from an image."""
    with Image.open(path) as im:
        rgb = im.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.asarray(rgb, dtype=np.float32) / 255.0
    gray = color.rgb2gray(arr)
    hsv = color.rgb2hsv(arr)
    gray8 = (gray * 255).astype(np.uint8)

    vals: list[float] = []
    names: list[str] = []

    # --- colour -------------------------------------------------------------------
    for i, ch in enumerate("RGB"):
        vals += [float(arr[..., i].mean()), float(arr[..., i].std())]
        names += [f"rgb_{ch}_mean", f"rgb_{ch}_std"]
    for i, ch in enumerate("HSV"):
        vals += [float(hsv[..., i].mean()), float(hsv[..., i].std())]
        names += [f"hsv_{ch}_mean", f"hsv_{ch}_std"]
    for i, ch in enumerate("HSV"):
        hist, _ = np.histogram(hsv[..., i], bins=8, range=(0.0, 1.0), density=True)
        vals += hist.tolist()
        names += [f"hsv_{ch}_hist{b}" for b in range(8)]

    # --- grey level statistics ----------------------------------------------------
    g = gray.ravel()
    vals += [
        float(g.mean()), float(g.std()),
        float(stats.skew(g)), float(stats.kurtosis(g)),
        float(stats.entropy(np.histogram(g, bins=32, range=(0, 1))[0] + 1e-9)),
        float(filters.laplace(gray).var()),
    ]
    names += ["gray_mean", "gray_std", "gray_skew", "gray_kurtosis", "gray_entropy", "sharpness"]

    # --- texture ------------------------------------------------------------------
    gv, gn = _glcm_features(gray8)
    vals += gv
    names += gn
    lbp = local_binary_pattern(gray8, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    vals += hist.tolist()
    names += [f"lbp_{b}" for b in range(LBP_BINS)]

    # --- edges --------------------------------------------------------------------
    sx, sy = filters.sobel_h(gray), filters.sobel_v(gray)
    mag = np.hypot(sx, sy)
    vals += [float(mag.mean()), float(mag.std()), float(feature.canny(gray, sigma=1.5).mean())]
    names += ["sobel_mean", "sobel_std", "edge_density"]
    ang = (np.arctan2(sy, sx) % np.pi) / np.pi  # undirected gradient orientation
    oh, _ = np.histogram(ang, bins=ORIENT_BINS, range=(0, 1), weights=mag)
    oh = oh / (oh.sum() + 1e-9)
    vals += oh.tolist()
    names += [f"orient_hist{b}" for b in range(ORIENT_BINS)]

    # --- frequency ----------------------------------------------------------------
    f = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - h / 2, xx - w / 2) / (min(h, w) / 2)
    for lo, hi in [(0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]:
        mask = (r >= lo) & (r < hi)
        vals.append(float(f[mask].mean() / (f.mean() + 1e-9)))
        names.append(f"fft_band_{lo}_{hi}")
    vals.append(float(f[r >= 0.5].sum() / (f.sum() + 1e-9)))
    names.append("fft_high_ratio")

    # --- spatial lighting ---------------------------------------------------------
    half = size // 2
    quads = [gray[:half, :half], gray[:half, half:], gray[half:, :half], gray[half:, half:]]
    vals += [float(q.mean()) for q in quads]
    names += [f"quad_mean_{i}" for i in range(4)]
    vals.append(float(np.std([q.mean() for q in quads])))
    names.append("quad_mean_spread")

    return np.asarray(vals, dtype=np.float32), names


def build_feature_table(df: pd.DataFrame, size: int = 128, force: bool = False) -> pd.DataFrame:
    """Extract features for every manifest row and cache them to ``features.npz``."""
    cache = FEATURES_DIR / f"features_{size}.npz"
    if cache.exists() and not force:
        arr = np.load(cache, allow_pickle=True)
        if list(arr["image_id"]) == list(df.image_id):
            return pd.DataFrame(arr["X"], columns=list(arr["names"])).assign(image_id=df.image_id.values)

    X, names = [], None
    for p in tqdm(df.path, desc="features", unit="img"):
        v, names = extract_features(p, size=size)
        X.append(v)
    X = np.vstack(X)
    np.savez_compressed(cache, X=X, names=np.array(names), image_id=df.image_id.values)
    return pd.DataFrame(X, columns=names).assign(image_id=df.image_id.values)


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    corr_threshold: float = 0.95,
    top_k: int = 40,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Variance, correlation and mutual-information based feature selection.

    Fit on training data only.  Returns (X_selected, kept_names, scoring_table).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif

    df = pd.DataFrame(X, columns=names)
    df["_y"] = y
    Xdf = df.drop(columns="_y")

    # 1) drop constant / near-constant columns
    std = Xdf.std(axis=0)
    keep = [c for c in Xdf.columns if std[c] > 1e-8]
    # 2) drop one of each highly-correlated pair (keep the one with higher MI)
    mi = pd.Series(mutual_info_classif(Xdf[keep], y, random_state=42), index=keep)
    order = mi.sort_values(ascending=False).index.tolist()
    dropped: set[str] = set()
    corr = Xdf[order].corr().abs()
    for i, c in enumerate(order):
        if c in dropped:
            continue
        for c2 in order[i + 1 :]:
            if c2 in dropped:
                continue
            if corr.loc[c, c2] > corr_threshold:
                dropped.add(c2)
    keep = [c for c in order if c not in dropped]

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Xdf[keep], y)
    importance = pd.Series(rf.feature_importances_, index=keep)
    table = pd.DataFrame(
        {
            "feature": keep,
            "mutual_info": [float(mi[c]) for c in keep],
            "rf_importance": [float(importance[c]) for c in keep],
        }
    ).sort_values("rf_importance", ascending=False)
    table["kept_topk"] = table.feature.isin(table.feature.head(top_k))
    table["dropped_by_correlation"] = False
    dropped_df = pd.DataFrame(
        {"feature": sorted(dropped), "mutual_info": [float(mi[c]) for c in sorted(dropped)],
         "rf_importance": np.nan, "kept_topk": False, "dropped_by_correlation": True}
    )
    table = pd.concat([table, dropped_df], ignore_index=True)
    table.to_csv(TABLES_DIR / "feature_selection.csv", index=False)

    kept = table.loc[table.kept_topk, "feature"].tolist()
    return Xdf[kept].to_numpy(np.float32), kept, table
