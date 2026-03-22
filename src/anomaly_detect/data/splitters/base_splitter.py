"""Abstract base class for data splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseSplitter(ABC):
    @abstractmethod
    def split(
        self,
        n_samples: int,
        y: np.ndarray = None,
        X: np.ndarray = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (train_idx, val_idx, test_idx) as 1-D integer arrays."""
