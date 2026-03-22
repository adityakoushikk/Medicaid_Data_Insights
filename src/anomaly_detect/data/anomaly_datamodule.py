"""Lightning DataModule for Medicaid provider anomaly detection.

Pipeline
--------
1. _get_or_compute_features()  →  run create_provider_level_from_month.py (cached)
2. _clean_features()           →  drop high-NaN cols, drop remaining NaN rows
3. _select_features()          →  AUROC filter + optional variance/corr dedup
4. setup()                     →  scale, split, build AnomalyDataset + numpy arrays
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import lightning.pytorch as L
import torch
from torch.utils.data import DataLoader, Subset

from anomaly_detect.data.anomaly_dataset import AnomalyDataset
from provider_level_runner import run_provider_level  # scripts/ on PYTHONPATH via run_anomaly.sh

log = logging.getLogger(__name__)

# Columns that are never model inputs regardless of config
_ALWAYS_EXCLUDE = frozenset([
    "label", "excldate",
    "billing_provider_npi",
    "insufficient_history_flag",
    "cohort_label",
    "months_observed", "first_month", "last_month",
    "span_months", "missing_months_within_span",
    "fraction_months_observed", "mean_gap_months", "max_gap_months",
])


class AnomalyDataModule(L.LightningDataModule):
    """DataModule that wraps the full provider-level feature pipeline."""

    def __init__(
        self,
        provider_month_csv: str,
        provider_level_csv: str,
        provider_level_script: str,
        splitter,
        min_months: int = 6,
        date_cutoff: str = "2024-12-31",
        no_filter: bool = False,
        provider_level_features: Optional[dict] = None,
        nan_drop_threshold: float = 0.05,
        feature_selection: Optional[dict] = None,
        exclude_cols: Optional[List[str]] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        scaler: str = "standardize",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["splitter"])
        self.splitter = splitter
        self.feature_selection = feature_selection or {"auroc_threshold": 0.65}
        self.exclude_cols = _ALWAYS_EXCLUDE | set(exclude_cols or [])

        # Populated during setup()
        self._dataset: Optional[AnomalyDataset] = None
        self._train_idx: Optional[np.ndarray] = None
        self._val_idx: Optional[np.ndarray] = None
        self._test_idx: Optional[np.ndarray] = None
        self._scaler = None

        self.feature_names: Optional[List[str]] = None
        self.auroc_df: Optional[pd.DataFrame] = None

        # Full scaled arrays for post-training scoring (set during setup)
        self.X_all_np: Optional[np.ndarray] = None
        self.y_all_np: Optional[np.ndarray] = None
        self.npis_all: Optional[np.ndarray] = None

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def n_features(self) -> int:
        if self._dataset is None:
            raise RuntimeError("Call setup() first.")
        return self._dataset.X.shape[1]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_compute_features(self) -> pd.DataFrame:
        out = Path(self.hparams.provider_level_csv)
        log.info(f"Running provider_level script → {out}")
        run_provider_level(
            input_csv=self.hparams.provider_month_csv,
            output_csv=str(out),
            provider_level_script=self.hparams.provider_level_script,
            min_months=self.hparams.min_months,
            date_cutoff=self.hparams.date_cutoff,
            no_filter=self.hparams.no_filter,
            provider_level_features=self.hparams.provider_level_features,
        )
        return pd.read_csv(out, low_memory=False)

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop high-NaN columns then drop remaining NaN rows (from feature cols only)."""
        threshold = self.hparams.nan_drop_threshold
        nan_frac = df.isna().mean()
        feature_cols = [c for c in df.columns if c not in self.exclude_cols]
        cols_to_drop = [
            c for c in feature_cols if nan_frac.get(c, 0) > threshold
        ]
        if cols_to_drop:
            log.info(
                f"Dropping {len(cols_to_drop)} columns with >{threshold:.0%} NaN. "
                f"First 5: {cols_to_drop[:5]}"
            )
            df = df.drop(columns=cols_to_drop)

        before = len(df)
        feature_cols_remaining = [c for c in df.columns if c not in self.exclude_cols]
        df = df.dropna(subset=feature_cols_remaining)
        log.info(f"After NaN row-drop: {len(df):,} rows (dropped {before - len(df):,})")
        return df.reset_index(drop=True)

    def _select_features(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        """Select features by univariate AUROC threshold.

        If auroc_threshold is null in config, all features are kept and AUROC
        is still computed (stored in auroc_df) but not used for filtering.
        """
        threshold = self.feature_selection.get("auroc_threshold")
        if threshold is None:
            log.info(f"Feature selection disabled — keeping all {len(X.columns)} features.")
            return list(X.columns)

        rows = []
        for feat in X.columns:
            series = X[feat].values
            valid_mask = ~np.isnan(series)
            xv, yv = series[valid_mask], y[valid_mask]
            if len(np.unique(yv)) < 2:
                continue
            try:
                auc = roc_auc_score(yv, xv)
                auc = max(auc, 1 - auc)
                rows.append({"feature": feat, "auroc": round(float(auc), 4)})
            except Exception:
                continue
        self.auroc_df = (
            pd.DataFrame(rows)
            .sort_values("auroc", ascending=False)
            .reset_index(drop=True)
        )
        selected = self.auroc_df[self.auroc_df["auroc"] >= threshold]["feature"].tolist()
        log.info(f"AUROC >= {threshold}: {len(selected)} / {len(X.columns)} features selected.")
        return selected

    def _fit_scale(self, X: np.ndarray, train_idx: np.ndarray) -> np.ndarray:
        scaler_type = self.hparams.scaler
        if scaler_type == "standardize":
            self._scaler = StandardScaler()
        elif scaler_type == "minmax":
            self._scaler = MinMaxScaler()
        else:
            return X
        X_fit = X[train_idx] if len(train_idx) > 0 else X
        self._scaler.fit(X_fit)
        return self._scaler.transform(X).astype(np.float32)

    # ── Lightning DataModule interface ────────────────────────────────────────

    def setup(self, stage: str = None) -> None:
        df = self._get_or_compute_features()
        df = self._clean_features(df)

        # Extract metadata columns
        npis = (
            df["billing_provider_npi"].values.astype(str)
            if "billing_provider_npi" in df.columns
            else np.arange(len(df)).astype(str)
        )
        y = (
            df["label"].fillna(0).values.astype(np.int64)
            if "label" in df.columns
            else np.zeros(len(df), dtype=np.int64)
        )

        # Build feature matrix
        feature_cols = [c for c in df.columns if c not in self.exclude_cols]
        X_df = df[feature_cols]

        # Feature selection
        self.feature_names = self._select_features(X_df, y)
        X_df = X_df[self.feature_names]
        X = X_df.values.astype(np.float32)

        # Split
        train_idx, val_idx, test_idx = self.splitter.split(len(X), y=y, X=X)
        self._train_idx = train_idx
        self._val_idx = val_idx
        self._test_idx = test_idx

        # Scale (fit on train only)
        X_scaled = self._fit_scale(X, train_idx)

        # Build dataset (full)
        self._dataset = AnomalyDataset(X_scaled, y, npis)

        # Full arrays for post-training scoring
        self.X_all_np = X_scaled
        self.y_all_np = y
        self.npis_all = npis

        _tr = train_idx if len(train_idx) else np.arange(len(X))
        _te = test_idx if len(test_idx) else np.arange(len(X))

        log.info(
            f"Setup complete | {len(X):,} providers | {self.n_features} features | "
            f"train={len(_tr):,} test={len(_te):,} val={len(val_idx):,}"
        )

    def _make_loader(self, idx: np.ndarray, shuffle: bool) -> DataLoader:
        subset = (
            Subset(self._dataset, idx)
            if len(idx) < len(self._dataset)
            else self._dataset
        )
        return DataLoader(
            subset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def _label0_idx(self, idx: np.ndarray) -> np.ndarray:
        """Filter index array to label=0 (normal) providers only."""
        neg_mask = self._dataset.y.numpy() == 0
        return idx[neg_mask[idx]]

    def train_dataloader(self) -> DataLoader:
        idx = self._train_idx if len(self._train_idx) else np.arange(len(self._dataset))
        return self._make_loader(self._label0_idx(idx), shuffle=True)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self._val_idx is None or len(self._val_idx) == 0:
            return None
        return self._make_loader(self._label0_idx(self._val_idx), shuffle=False)

    def test_dataloader(self) -> DataLoader:
        idx = self._test_idx if len(self._test_idx) else np.arange(len(self._dataset))
        return self._make_loader(idx, shuffle=False)
