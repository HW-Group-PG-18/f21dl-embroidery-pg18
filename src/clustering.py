"""Clustering the feature space, with no labels involved while fitting.

Two algorithms from the syllabus: k-means and a Gaussian mixture fitted by EM. Internal
indices (silhouette, Davies-Bouldin) say whether the clusters are tight. ARI and NMI against
the normal/abnormal label are there only to interpret what the clusters contain; they never
influence the fit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import FIGURES_DIR, TABLES_DIR, VIEW_LABELS, VIEWS
from .features import build_feature_table

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})


def _embed(X: np.ndarray, n_components: int = 2) -> np.ndarray:
    return PCA(n_components=n_components, random_state=0).fit_transform(X)


def run(df: pd.DataFrame) -> pd.DataFrame:
    feats = build_feature_table(df, size=128)
    X = StandardScaler().fit_transform(feats.drop(columns="image_id").to_numpy(np.float32))
    y = df.label_id.to_numpy()
    Z = _embed(X)

    rows = []
    km_models, gm_models = {}, {}
    for k in range(2, 9):
        km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
        km_models[k] = km
        rows.append(
            {
                "algorithm": "kmeans",
                "k": k,
                "silhouette": silhouette_score(X, km.labels_),
                "davies_bouldin": davies_bouldin_score(X, km.labels_),
                "ari_vs_label": adjusted_rand_score(y, km.labels_),
                "nmi_vs_label": normalized_mutual_info_score(y, km.labels_),
                "bic": np.nan,
            }
        )
        gm = GaussianMixture(n_components=k, covariance_type="full", random_state=0,
                             n_init=3, max_iter=300).fit(X)
        gm_models[k] = gm
        labels = gm.predict(X)
        rows.append(
            {
                "algorithm": "gmm_em",
                "k": k,
                "silhouette": silhouette_score(X, labels),
                "davies_bouldin": davies_bouldin_score(X, labels),
                "ari_vs_label": adjusted_rand_score(y, labels),
                "nmi_vs_label": normalized_mutual_info_score(y, labels),
                "bic": gm.bic(X),
            }
        )
    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(TABLES_DIR / "cluster_metrics.csv", index=False)

    # Chosen solutions: k-means by silhouette, GMM by BIC (standard selection rules).
    k_km = int(metrics[metrics.algorithm == "kmeans"].sort_values("silhouette", ascending=False).k.iloc[0])
    k_gm = int(metrics[metrics.algorithm == "gmm_em"].sort_values("bic").k.iloc[0])
    km_labels = km_models[k_km].labels_
    gm_labels = gm_models[k_gm].predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for algo, ax in [(("kmeans"), axes[0]), ("gmm_em", axes[1])]:
        sub = metrics[metrics.algorithm == algo]
        ax.plot(sub.k, sub.silhouette, marker="o", label="silhouette")
        ax.plot(sub.k, sub.ari_vs_label, marker="s", label="ARI vs label")
        ax.plot(sub.k, sub.nmi_vs_label, marker="^", label="NMI vs label")
        ax.set_xlabel("k")
        ax.set_title(algo.replace("_", " "))
        ax.legend(frameon=False, fontsize=7)
    axes[0].set_ylabel("score")
    for labels, ax, title in [
        (km_labels, axes[2], f"k-means (k={k_km})"),
    ]:
        ax.scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=8, linewidths=0)
        ax.set_title(f"PCA space, {title} clusters")
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cluster_selection.png", bbox_inches="tight")
    plt.close(fig)

    # Cluster composition: what do the clusters actually contain?
    comp = pd.DataFrame({"cluster": km_labels, "label": np.where(y == 1, "abnormal", "normal"),
                         "view": df.view.to_numpy(), "group": df.group_id.to_numpy()})
    comp_table = pd.crosstab(comp.cluster, comp.label)
    comp_table["n"] = comp_table.sum(axis=1)
    comp_table["normal_frac"] = (comp_table.get("normal", 0) / comp_table["n"]).round(3)
    comp_table.to_csv(TABLES_DIR / "cluster_composition.csv")

    view_table = pd.crosstab(comp.cluster, comp.view).reindex(columns=VIEWS).fillna(0).astype(int)
    view_table.to_csv(TABLES_DIR / "cluster_view_composition.csv")

    # Row-level assignment for error analysis in the report.
    pd.DataFrame({"image_id": df.image_id, "cluster": km_labels, "gmm_cluster": gm_labels,
                  "label_id": y, "view": df.view}).to_csv(TABLES_DIR / "cluster_assignments.csv", index=False)
    return metrics
