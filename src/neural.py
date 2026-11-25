"""The two neural networks for R4, plus Grad-CAM.

MLP: trained on the same 88 features as the classical models, so if it beats logistic
regression the gain is non-linearity, not a better input.

CNN: four conv blocks on 128x128 crops, augmentation, class weights, early stopping on
validation AUC. Grad-CAM afterwards so we can see what it is actually looking at.
"""
from __future__ import annotations

import json
import time
from functools import partial

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config, FIGURES_DIR, MODELS_DIR, TABLES_DIR, VIEW_CASES
from .data import view_case_frame
from .evaluation import best_threshold, classification_metrics, confusion, group_bootstrap_ci
from .features import build_feature_table

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
DEVICE = torch.device("cpu")  # dataset and model are small; CPU keeps the run reproducible


# ======================================================================================
# MLP on handcrafted features
# ======================================================================================
MLP_GRID = [
    {"hidden": (64,), "alpha": 1e-3},
    {"hidden": (128, 64), "alpha": 1e-3},
    {"hidden": (128, 64), "alpha": 1e-2},
    {"hidden": (256, 128), "alpha": 1e-3},
]


def run_mlp(cfg: Config, df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    feats = build_feature_table(df, size=cfg.feature_size)
    data = df.merge(feats, on="image_id", how="left")
    cols = [c for c in feats.columns if c != "image_id"]

    rows, preds, curves = [], [], {}
    for case in VIEW_CASES:
        sub = view_case_frame(data, case)
        tr, va, te = (sub[sub.split == s] for s in ("train", "val", "test"))
        Xtr, ytr = tr[cols].to_numpy(), tr.label_id.to_numpy()
        Xva, yva = va[cols].to_numpy(), va.label_id.to_numpy()
        Xte, yte = te[cols].to_numpy(), te.label_id.to_numpy()

        best = None
        for cfgm in MLP_GRID:
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("select", SelectKBest(partial(mutual_info_classif, random_state=cfg.seed), k=40)),
                ("clf", MLPClassifier(hidden_layer_sizes=cfgm["hidden"], alpha=cfgm["alpha"],
                                      learning_rate_init=1e-3, max_iter=800,
                                      early_stopping=True, n_iter_no_change=25,
                                      validation_fraction=0.15, random_state=cfg.seed)),
            ])
            pipe.fit(Xtr, ytr)
            va_auc = roc_auc_score(yva, pipe.predict_proba(Xva)[:, 1])
            if best is None or va_auc > best[0]:
                best = (va_auc, cfgm, pipe)
            if verbose:
                print(f"    mlp {cfgm} val_auc={va_auc:.3f}")
        _, cfgm, pipe = best
        prob = pipe.predict_proba(Xte)[:, 1]
        thr, val_f1 = best_threshold(yva, pipe.predict_proba(Xva)[:, 1])
        pred = (prob >= thr).astype(int)
        met = classification_metrics(yte, pred, prob)
        ci = group_bootstrap_ci(pd.DataFrame({"label_id": yte, "pred": pred, "prob": prob,
                                              "group_id": te.group_id.to_numpy()}),
                                n_boot=400, seed=cfg.seed)
        rows.append({"case": case, "model": "mlp", "family": "neural",
                     "cv_auc": float(best[0]), "best_params": json.dumps(cfgm),
                     "threshold": round(float(thr), 4),
                     "val_f1_at_threshold": round(float(val_f1), 3), **met,
                     "f1_lo": ci["f1"][0], "f1_hi": ci["f1"][1],
                     "auc_lo": ci["roc_auc"][0], "auc_hi": ci["roc_auc"][1]})
        preds.append(te[["image_id", "group_id", "view", "label_id", "split"]].assign(
            case=case, model="mlp", family="neural", pred=pred, prob=prob))
        curves[case] = list(getattr(pipe.named_steps["clf"], "loss_curve_", []))
        if verbose:
            print(f"  mlp {case}: acc={met['accuracy']:.3f} f1={met['f1']:.3f} auc={met['roc_auc']:.3f}")

    metrics = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    metrics.to_csv(TABLES_DIR / "mlp_metrics.csv", index=False)
    pred_df.to_csv(TABLES_DIR / "mlp_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for case, loss in curves.items():
        if loss:
            ax.plot(range(1, len(loss) + 1), loss, label=case)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("MLP learning curves")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mlp_training.png", bbox_inches="tight")
    plt.close(fig)
    return metrics, pred_df


# ======================================================================================
# CNN on raw images
# ======================================================================================
class EmbroideryCNN(nn.Module):
    """Small VGG-style CNN sized for ~1k images and CPU training."""

    def __init__(self, width: int = 32, dropout: float = 0.3):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(block(3, width), block(width, width * 2),
                                      block(width * 2, width * 4), block(width * 4, width * 4))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(width * 4, 2))

    def forward(self, x):
        x = self.features(x)
        self.last_conv = x  # kept for Grad-CAM
        return self.classifier(self.pool(x).flatten(1))


class ImageDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths, self.labels, self.transform = list(paths), list(labels), transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
            return self.transform(img), int(self.labels[i])


def _transforms(size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return train_tf, eval_tf


def _predict(model, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(DEVICE))
            probs.append(torch.softmax(logits, dim=1)[:, 1].numpy())
            ys.append(y.numpy())
    return np.concatenate(probs), np.concatenate(ys)


# Fine-tuning grid. Each setting costs a full training run (~8 min per case on a CPU laptop),
# so the grid stays small; the winner is picked on validation ROC-AUC, never on test.
CNN_GRID = [{"lr": 1e-3, "dropout": 0.3}, {"lr": 3e-4, "dropout": 0.3}]


def train_cnn(cfg: Config, case: str, sub: pd.DataFrame, size: int, verbose: bool = True,
              lr: float = 1e-3, dropout: float = 0.3):
    tr, va, te = (sub[sub.split == s] for s in ("train", "val", "test"))
    train_tf, eval_tf = _transforms(size)
    ds_tr = ImageDataset(tr.path, tr.label_id, train_tf)
    ds_va = ImageDataset(va.path, va.label_id, eval_tf)
    ds_te = ImageDataset(te.path, te.label_id, eval_tf)
    dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=64, num_workers=0)
    dl_te = DataLoader(ds_te, batch_size=64, num_workers=0)

    torch.manual_seed(cfg.seed)  # weights, shuffling and augmentation all draw from this
    model = EmbroideryCNN(dropout=dropout).to(DEVICE)
    counts = np.bincount(tr.label_id.to_numpy(), minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.cnn_epochs)

    history = {"epoch": [], "train_loss": [], "val_auc": [], "val_f1": []}
    best_auc, best_state, patience = -np.inf, None, 0
    for epoch in range(1, cfg.cnn_epochs + 1):
        model.train()
        losses = []
        for x, y in dl_tr:
            opt.zero_grad()
            loss = criterion(model(x.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        sched.step()
        vp, vy = _predict(model, dl_va)
        va_auc = roc_auc_score(vy, vp) if len(np.unique(vy)) == 2 else np.nan
        vf1 = classification_metrics(vy, (vp >= 0.5).astype(int), vp)["f1"]
        history["epoch"].append(epoch)
        history["train_loss"].append(float(np.mean(losses)))
        history["val_auc"].append(float(va_auc))
        history["val_f1"].append(float(vf1))
        if verbose:
            print(f"    cnn {case} epoch {epoch:2d} loss={np.mean(losses):.3f} val_auc={va_auc:.3f}")
        if va_auc > best_auc:
            best_auc, patience = va_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.cnn_patience:
                if verbose:
                    print("    early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    tp, ty = _predict(model, dl_te)
    vp, vy = _predict(model, dl_va)
    thr, val_f1 = best_threshold(vy, vp)
    tpred = (tp >= thr).astype(int)
    met = classification_metrics(ty, tpred, tp)
    ci = group_bootstrap_ci(pd.DataFrame({"label_id": ty, "pred": tpred, "prob": tp,
                                          "group_id": te.group_id.to_numpy()}),
                            n_boot=400, seed=cfg.seed)
    return {
        "model": model, "metrics": {**met, "cv_auc": float(best_auc),
                                    "threshold": round(float(thr), 4),
                                    "val_f1_at_threshold": round(float(val_f1), 3),
                                    "f1_lo": ci["f1"][0], "f1_hi": ci["f1"][1],
                                    "auc_lo": ci["roc_auc"][0], "auc_hi": ci["roc_auc"][1]},
        "history": history, "test": (te.reset_index(drop=True), tp, tpred),
        "loader": dl_te, "transform": eval_tf,
    }


def grad_cam(model: EmbroideryCNN, image_tensor, target: int = 1) -> np.ndarray:
    """Grad-CAM over the last convolutional block."""
    model.eval()
    x = image_tensor.unsqueeze(0).to(DEVICE).requires_grad_(False)
    feats = model.features(x)
    feats.retain_grad()
    logits = model.classifier(model.pool(feats).flatten(1))
    model.zero_grad()
    logits[0, target].backward()
    grads = feats.grad.detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1)).squeeze().detach().numpy()
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-9)


def plot_cnn_assets(results: dict[str, dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    for case, r in results.items():
        h = r["history"]
        axes[0].plot(h["epoch"], h["train_loss"], label=case)
        axes[1].plot(h["epoch"], h["val_auc"], label=case)
    axes[0].set_title("CNN training loss")
    axes[0].set_xlabel("epoch")
    axes[1].set_title("CNN validation ROC-AUC")
    axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cnn_training.png", bbox_inches="tight")
    plt.close(fig)

    # Confusion matrices per view condition
    fig, axes = plt.subplots(1, len(results), figsize=(3.1 * len(results), 2.8))
    for ax, (case, r) in zip(np.atleast_1d(axes), results.items()):
        te, tp, tpred = r["test"]
        cm = confusion(te.label_id.to_numpy(), tpred)
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        ax.set_xticks([0, 1], ["normal", "abnormal"])
        ax.set_yticks([0, 1], ["normal", "abnormal"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("actual")
        ax.set_title(case)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cnn_confusion.png", bbox_inches="tight")
    plt.close(fig)


def grad_cam_grid(case: str, r: dict, n: int = 6) -> None:
    te, tp, tpred = r["test"]
    model = r["model"]
    idx = np.linspace(0, len(te) - 1, n, dtype=int)
    fig, axes = plt.subplots(n, 3, figsize=(6.4, 2.0 * n))
    for row, i in enumerate(idx):
        with Image.open(te.path.iloc[i]) as im:
            img = im.convert("RGB")
            tensor = r["transform"](img)
        cam = grad_cam(model, tensor)
        cam_img = np.asarray(Image.fromarray((cam * 255).astype(np.uint8)).resize((128, 128))) / 255.0
        rgb = np.asarray(img.resize((128, 128)), dtype=np.float32) / 255.0
        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"actual: {'abnormal' if te.label_id.iloc[i] else 'normal'}",
                               fontsize=7, loc="left")
        axes[row, 1].imshow(cam_img, cmap="inferno")
        axes[row, 1].set_title(f"pred={tp[i]:.2f}", fontsize=7, loc="left")
        axes[row, 2].imshow(rgb)
        axes[row, 2].imshow(cam_img, cmap="inferno", alpha=0.45)
        axes[row, 2].set_title("overlay", fontsize=7, loc="left")
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle(f"Grad-CAM, {case}", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"cnn_gradcam_{case}.png", bbox_inches="tight")
    plt.close(fig)


def run_cnn(cfg: Config, df: pd.DataFrame, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    size = cfg.img_size
    rows, preds, results = [], [], {}
    for case in VIEW_CASES:
        t0 = time.time()
        sub = view_case_frame(df, case)
        runs = [(p, train_cnn(cfg, case, sub, size, verbose=verbose, **p)) for p in CNN_GRID]
        params, r = max(runs, key=lambda pr: pr[1]["metrics"]["cv_auc"])
        torch.save({"state_dict": r["model"].state_dict(), "size": size, "case": case, **params},
                   MODELS_DIR / f"cnn_{case}.pt")
        results[case] = r
        te, tp, tpred = r["test"]
        tried = {f"lr={p['lr']},dropout={p['dropout']}": round(x["metrics"]["cv_auc"], 4)
                 for p, x in runs}
        rows.append({"case": case, "model": "cnn", "family": "neural",
                     "best_params": json.dumps({"width": 32, "size": size, "batch": 32, **params,
                                                "val_auc_per_setting": tried}),
                     **r["metrics"]})
        preds.append(te[["image_id", "group_id", "view", "label_id", "split"]].assign(
            case=case, model="cnn", family="neural", pred=tpred, prob=tp))
        print(f"  cnn {case} finished in {time.time() - t0:.0f}s  "
              f"acc={r['metrics']['accuracy']:.3f} auc={r['metrics']['roc_auc']:.3f}")

    metrics = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    metrics.to_csv(TABLES_DIR / "cnn_metrics.csv", index=False)
    pred_df.to_csv(TABLES_DIR / "cnn_predictions.csv", index=False)
    plot_cnn_assets(results)
    grad_cam_grid("straight_45", results["straight_45"])
    return metrics, pred_df
