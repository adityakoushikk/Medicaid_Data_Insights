"""Lightning Module wrapping the Autoencoder for training and scoring."""

from __future__ import annotations

import logging
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as L
from hydra.utils import instantiate
from omegaconf import DictConfig

log = logging.getLogger(__name__)


class AnomalyLitModule(L.LightningModule):
    """Trains the Autoencoder and computes per-provider reconstruction error."""

    def __init__(self, net, optimizer_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(ignore=["net"])
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        recon = self.net(x)
        loss = F.mse_loss(recon, x)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        recon = self.net(x)
        loss = F.mse_loss(recon, x)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return instantiate(self.hparams.optimizer_cfg, params=self.net.parameters())

    @torch.no_grad()
    def compute_anomaly_scores(
        self, X: np.ndarray, batch_size: int = 512
    ) -> np.ndarray:
        """Mean absolute reconstruction error per provider (higher = more anomalous).

        Runs in batches to handle large datasets without OOM.
        """
        self.eval()
        device = next(self.net.parameters()).device
        X_t = torch.tensor(X, dtype=torch.float32)
        errors = []
        for start in range(0, len(X_t), batch_size):
            chunk = X_t[start: start + batch_size].to(device)
            recon = self.net(chunk)
            err = torch.mean(torch.abs(chunk - recon), dim=1)
            errors.append(err.cpu().numpy())
        return np.concatenate(errors)
