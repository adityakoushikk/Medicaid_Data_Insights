"""PyTorch Dataset for anomaly detection."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class AnomalyDataset(Dataset):
    """Holds a feature matrix X, integer labels y, and string NPI IDs.

    __getitem__ returns (X[i], y[i]).  NPIs are accessible via dataset.npis[i]
    (string array — not returned by __getitem__ to avoid DataLoader collation issues).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, npis: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.npis = (
            npis if npis is not None else np.arange(len(X)).astype(str)
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]
