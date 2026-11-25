"""Exploratory plots and the lighting shortcut test.

What it makes: class and view counts, example images, brightness/contrast/sharpness by
class, a PCA view of the feature space, a feature correlation map, and the shortcut probe.

The probe matters. If a classifier fed only brightness and contrast can predict the label,
then some of the accuracy in the later stages is about exposure rather than embroidery. We
would rather find that out here than have a marker find it in the discussion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .config import FIGURES_DIR, TABLES_DIR, VIEW_LABELS, VIEWS
from .evaluation import classification_metrics
from .features import build_feature_table, select_features

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25})
COLORS = {"normal": "#2a9d8f", "abnormal": "#e76f51"}


def pixel_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap per-image statistics used for the EDA plots and the shortcut probe."""
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="pixel stats", unit="img"):
        with Image.open(r.path) as im:
            g = np.asarray(im.convert("L").resize((128, 128)), dtype=np.float32) / 255.0
            rgb = np.asarray(im.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        lap = np.abs(4 * g[1:-1, 1:-1] - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:])
        rows.append(
            {
                "image_id": r.image_id,
                "brightness": float(g.mean()),
                "contrast": float(g.std()),
                "sharpness": float(lap.mean()),
                "r_mean": float(rgb[..., 0].mean()),
                "g_mean": float(rgb[..., 1].mean()),
                "b_mean": float(rgb[..., 2].mean()),
                "saturation": float(rgb.max(axis=2).mean() - rgb.min(axis=2).mean()),
            }
        )
    return pd.DataFrame(rows)


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)


def plot_composition(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))

    ct = pd.crosstab(df.view, df.label).reindex(VIEWS)
    ct[["normal", "abnormal"]].plot(kind="bar", stacked=True, ax=axes[0],
                                     color=[COLORS["normal"], COLORS["abnormal"]])
    axes[0].set_title("Images per view and class")
    axes[0].set_xlabel("view")
    axes[0].set_ylabel("images")
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(frameon=False, fontsize=8)

    per_sample = df.pivot_table(index="sample", columns="view", values="image_id",
                                aggfunc="count").reindex(columns=VIEWS).fillna(0)
    im = axes[1].imshow(per_sample.to_numpy(), cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(VIEWS)), [VIEW_LABELS[v] for v in VIEWS])
    axes[1].set_yticks(range(len(per_sample)), per_sample.index)
    axes[1].set_title("Images per physical sample and view")
    axes[1].grid(False)
    for i in range(per_sample.shape[0]):
        for j in range(per_sample.shape[1]):
            axes[1].text(j, i, int(per_sample.iat[i, j]), ha="center", va="center",
                         color="white", fontsize=8)
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.03)

    gs = df.groupby("group_id").label.nunique()
    axes[2].bar(["1 label", "2 labels"], [int((gs == 1).sum()), int((gs == 2).sum())],
                color=["#8ecae6", "#fb8500"])
    axes[2].set_title("Groups by number of image labels\n(group = sample + defect type)")
    axes[2].set_ylabel("groups")
    _save(fig, "eda_composition.png")


def plot_examples(df: pd.DataFrame, n_per_cell: int = 2) -> None:
    fig, axes = plt.subplots(len(VIEWS) * 2, n_per_cell, figsize=(2.1 * n_per_cell, 6.4))
    for i, view in enumerate(VIEWS):
        for j, label in enumerate(["normal", "abnormal"]):
            sub = df[(df.view == view) & (df.label == label)].sample(
                min(n_per_cell, len(df[(df.view == view) & (df.label == label)])), random_state=1
            )
            for k in range(n_per_cell):
                ax = axes[i * 2 + j, k]
                ax.axis("off")
                if k < len(sub):
                    with Image.open(sub.iloc[k].path) as im:
                        ax.imshow(im.convert("RGB").resize((160, 160)))
                if k == 0:
                    ax.set_title(f"{VIEW_LABELS[view]} · {label}", fontsize=7, loc="left")
    _save(fig, "eda_examples.png")


def plot_pixel_stats(stats: pd.DataFrame, df: pd.DataFrame) -> None:
    merged = stats.merge(df[["image_id", "label", "view"]], on="image_id")
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))
    for ax, col, title in zip(
        axes,
        ["brightness", "contrast", "sharpness"],
        ["Mean intensity", "Intensity spread (contrast)", "High-frequency energy (sharpness)"],
    ):
        data = [merged.loc[merged.label == lab, col].to_numpy() for lab in ["normal", "abnormal"]]
        bp = ax.boxplot(data, tick_labels=["normal", "abnormal"], patch_artist=True, widths=0.55,
                        medianprops={"color": "black"})
        for patch, lab in zip(bp["boxes"], ["normal", "abnormal"]):
            patch.set_facecolor(COLORS[lab])
            patch.set_alpha(0.75)
        ax.set_title(title)
    _save(fig, "eda_pixel_stats.png")


def plot_pca(features: pd.DataFrame, df: pd.DataFrame) -> None:
    X = features.drop(columns="image_id").to_numpy(np.float32)
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=0).fit(X)
    p = pca.transform(X)
    evr = pca.explained_variance_ratio_ * 100
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), sharex=True, sharey=True)
    for lab in ["normal", "abnormal"]:
        m = (df.label == lab).to_numpy()
        axes[0].scatter(p[m, 0], p[m, 1], s=7, alpha=0.65, label=lab, c=COLORS[lab], linewidths=0)
    axes[0].set_title("coloured by class", fontsize=8)
    axes[0].legend(frameon=False, markerscale=1.6, fontsize=7)
    markers = {"straight": "o", "45": "^", "60": "s"}
    for view in VIEWS:
        m = (df.view == view).to_numpy()
        axes[1].scatter(p[m, 0], p[m, 1], s=7, alpha=0.6, label=VIEW_LABELS[view],
                        marker=markers[view], linewidths=0)
    axes[1].set_title("coloured by viewing angle", fontsize=8)
    axes[1].legend(frameon=False, markerscale=1.6, fontsize=7)
    for ax in axes:
        ax.set_xlabel(f"PC 1 ({evr[0]:.0f}%)", fontsize=8)
        ax.set_ylabel(f"PC 2 ({evr[1]:.0f}%)", fontsize=8)
        ax.tick_params(labelsize=7)
    _save(fig, "eda_pca.png")


def plot_correlation(features: pd.DataFrame, top: int = 18) -> None:
    from sklearn.ensemble import RandomForestClassifier

    X = features.drop(columns="image_id")
    y = features.attrs.get("y")
    if y is None:
        return
    rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(X, y)
    keep = pd.Series(rf.feature_importances_, index=X.columns).nlargest(top).index
    corr = X[keep].corr()
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)), keep, rotation=90, fontsize=6)
    ax.set_yticks(range(len(keep)), keep, fontsize=6)
    ax.grid(False)
    ax.set_title(f"Correlation between the {top} most important features")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    _save(fig, "eda_feature_correlation.png")


def shortcut_probe(df: pd.DataFrame, features: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Compare a lighting-only classifier with the full descriptor, same group CV."""
    merged = stats.merge(df[["image_id", "label_id", "group_id", "view", "split"]], on="image_id")
    light_cols = ["brightness", "contrast", "sharpness", "r_mean", "g_mean", "b_mean", "saturation"]
    X_light = merged[light_cols].to_numpy()
    full = features.merge(df[["image_id", "label_id", "group_id", "view", "split"]], on="image_id")
    X_full = full.drop(columns=["image_id", "label_id", "group_id", "view", "split"]).to_numpy()

    rows = []
    for name, X, frame in [("lighting_stats_only", X_light, merged), ("full_descriptor", X_full, full)]:
        y = frame.label_id.to_numpy()
        n_groups = frame.loc[frame.split == "train", "group_id"].nunique()
        cv = GroupKFold(n_splits=min(5, n_groups))
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        prob = cross_val_predict(pipe, X, y, cv=cv, groups=frame.group_id, method="predict_proba")[:, 1]
        m = classification_metrics(y, (prob >= 0.5).astype(int), prob)
        rows.append({"features": name, "n_features": X.shape[1], **m})
    table = pd.DataFrame(rows)
    table.to_csv(TABLES_DIR / "shortcut_probe.csv", index=False)
    return table


def run(df: pd.DataFrame) -> pd.DataFrame:
    stats = pixel_stats(df)
    stats.to_csv(TABLES_DIR / "pixel_stats.csv", index=False)
    features = build_feature_table(df, size=128)
    features.attrs["y"] = df.label_id.to_numpy()

    plot_composition(df)
    plot_examples(df)
    plot_pixel_stats(stats, df)
    plot_pca(features, df)
    plot_correlation(features)
    # Feature evaluation for the report, scored on the training split only.
    train = (df.split == "train").to_numpy()
    X = features.drop(columns="image_id").to_numpy(np.float32)
    select_features(X[train], df.label_id.to_numpy()[train], list(features.columns.drop("image_id")))
    probe = shortcut_probe(df, features, stats)
    return probe
