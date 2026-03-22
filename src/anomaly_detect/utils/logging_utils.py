"""WANDB hyperparameter logging helpers."""

from __future__ import annotations

import logging
from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def log_hyperparameters(object_dict: dict) -> None:
    """Compile config + model stats and log to all active loggers (e.g. WANDB)."""
    cfg = object_dict.get("cfg")
    model = object_dict.get("model")
    trainer = object_dict.get("trainer")

    if cfg is None or trainer is None:
        return

    hparams = OmegaConf.to_container(cfg, resolve=True)

    if model is not None:
        try:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            hparams["model/total_params"] = total
            hparams["model/trainable_params"] = trainable
        except Exception:
            pass

    for logger in trainer.loggers:
        if hasattr(logger, "log_hyperparams"):
            logger.log_hyperparams(hparams)
