"""Unsupervised splitter: train and score on all data, no held-out test set."""

from __future__ import annotations

import numpy as np
from anomaly_detect.data.splitters.base_splitter import BaseSplitter


class UnsupervisedSplitter(BaseSplitter):
    """Train and score on all providers; no held-out test set.

    Parameters
    ----------
    val_frac : float
        Fraction of data held out for reconstruction-loss monitoring and
        early stopping.  0.0 disables val loop and early stopping.
    seed : int
        RNG seed for reproducible val selection.
    """

    def __init__(self, val_frac: float = 0.1, seed: int = 42):
        self.val_frac = val_frac
        self.seed = seed

    def split(
        self,
        n_samples: int,
        y: np.ndarray = None,
        X: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_idx = np.arange(n_samples)

        if self.val_frac > 0:
            n_val = max(1, int(n_samples * self.val_frac))
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n_samples)
            val_idx = perm[:n_val]
            train_idx = perm[n_val:]
            return train_idx, val_idx, all_idx

        return all_idx, np.array([], dtype=int), all_idx
