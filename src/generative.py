"""Generative modelling, which the brief asks for alongside the discriminative models.

The VAE here is trained only on normal training images, so it never sees a defect. At test
time the reconstruction error becomes the anomaly score and the threshold comes from the
validation normals. Gaussian mixture (in clustering.py) is the other generative model.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import Config, FIGURES_DIR, MODELS_DIR, TABLES_DIR
from .evaluation import classification_metrics, confusion

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
DEVICE = torch.device("cpu")


class VAEDataset(Dataset):
    def __init__(self, paths, size: int = 64):
        self.paths = list(paths)
        self.tf = transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), 0


class VAE(nn.Module):
    def __init__(self, latent: int = 16, width: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, width, 4, 2, 1), nn.ReLU(True),          # 32
            nn.Conv2d(width, width * 2, 4, 2, 1), nn.ReLU(True),  # 16
            nn.Conv2d(width * 2, width * 4, 4, 2, 1), nn.ReLU(True),  # 8
        )
        self.fc_mu = nn.Linear(width * 4 * 8 * 8, latent)
        self.fc_logvar = nn.Linear(width * 4 * 8 * 8, latent)
        self.fc_dec = nn.Linear(latent, width * 4 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(width * 4, width * 2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(width * 2, width, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(width, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        # sample only while training; scoring uses the mean so the anomaly score is stable
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        h = self.fc_dec(z).view(-1, 128, 8, 8)
        return self.dec(h), mu, logvar


def _score(model, loader) -> tuple[np.ndarray, np.ndarray]:
    """Per-image reconstruction MSE and the reconstructions themselves."""
    model.eval()
    scores, recons, origs = [], [], []
    with torch.no_grad():
        for x, _ in loader:
            recon, mu, _ = model(x)
            err = ((x - recon) ** 2).mean(dim=(1, 2, 3))
            scores.append(err.numpy())
            recons.append(recon.numpy())
            origs.append(x.numpy())
    return np.concatenate(scores), np.concatenate(recons)


def run(cfg: Config, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    torch.manual_seed(cfg.seed)
    size = 64
    tr = df[(df.split == "train") & (df.label_id == 0)]
    va = df[(df.split == "val") & (df.label_id == 0)]
    te = df[df.split == "test"]

    dl_tr = DataLoader(VAEDataset(tr.path), batch_size=64, shuffle=True)
    dl_va = DataLoader(VAEDataset(va.path), batch_size=64)
    dl_te = DataLoader(VAEDataset(te.path), batch_size=64)

    model = VAE().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    for epoch in range(1, cfg.vae_epochs + 1):
        model.train()
        losses = []
        for x, _ in dl_tr:
            opt.zero_grad()
            recon, mu, logvar = model(x)
            rec = ((x - recon) ** 2).mean()
            kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = rec + 0.5 * kld
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if verbose and epoch % 10 == 0:
            print(f"    vae epoch {epoch:2d} loss={history[-1]:.4f}")

    torch.save(model.state_dict(), MODELS_DIR / "vae.pt")

    val_scores, _ = _score(model, dl_va)
    threshold = float(np.percentile(val_scores, 95))
    test_scores, recons = _score(model, dl_te)
    y = te.label_id.to_numpy()
    pred = (test_scores >= threshold).astype(int)
    met = classification_metrics(y, pred, test_scores)
    cm = confusion(y, pred)

    table = pd.DataFrame([
        {"model": "vae_anomaly", **met, "threshold": threshold,
         "threshold_rule": "95th percentile of validation normal reconstruction error",
         "auc": roc_auc_score(y, test_scores)},
    ])
    table.to_csv(TABLES_DIR / "vae_anomaly.csv", index=False)
    pd.DataFrame({"image_id": te.image_id, "group_id": te.group_id, "view": te.view,
                  "label_id": y, "score": test_scores, "pred": pred}).to_csv(
        TABLES_DIR / "vae_scores.csv", index=False)

    # --- figures -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.0))
    axes[0].plot(range(1, len(history) + 1), history)
    axes[0].set_title("VAE training loss")
    axes[0].set_xlabel("epoch")
    for lab, name in [(0, "normal"), (1, "abnormal")]:
        axes[1].hist(test_scores[y == lab], bins=25, alpha=0.65, label=name, density=True)
    axes[1].axvline(threshold, color="k", ls="--", lw=1, label="threshold (val 95%)")
    axes[1].set_title("Reconstruction error on test images")
    axes[1].set_xlabel("mean squared reconstruction error")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vae_anomaly.png", bbox_inches="tight")
    plt.close(fig)

    # reconstruction examples
    order = np.argsort(test_scores)[::-1]
    pick = np.concatenate([order[:4], order[len(order) // 2 - 2: len(order) // 2 + 2]])
    fig, axes = plt.subplots(2, len(pick), figsize=(1.7 * len(pick), 3.6))
    for j, i in enumerate(pick):
        x = recons[i]
        orig = None
        with Image.open(te.path.iloc[i]) as im:
            orig = np.asarray(im.convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
        axes[0, j].imshow(orig)
        axes[1, j].imshow(np.clip(x.transpose(1, 2, 0), 0, 1))
        axes[0, j].set_title(f"score {test_scores[i]:.3f}", fontsize=6)
        for ax in (axes[0, j], axes[1, j]):
            ax.axis("off")
            ax.grid(False)
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstruction")
    fig.suptitle("VAE reconstructions, highest-error test images first", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "vae_reconstructions.png", bbox_inches="tight")
    plt.close(fig)
    return table
