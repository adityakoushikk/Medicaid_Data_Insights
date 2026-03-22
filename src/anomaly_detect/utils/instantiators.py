"""Hydra instantiation helpers."""

from __future__ import annotations

import logging
from omegaconf import DictConfig
from hydra.utils import instantiate

log = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list:
    """Instantiate Lightning callbacks from a config dict keyed by name."""
    callbacks = []
    if not callbacks_cfg:
        return callbacks
    for name, cb_conf in callbacks_cfg.items():
        if cb_conf is not None and "_target_" in cb_conf:
            cb = instantiate(cb_conf)
            callbacks.append(cb)
            log.info(f"Instantiated callback <{name}>: {type(cb).__name__}")
    return callbacks
