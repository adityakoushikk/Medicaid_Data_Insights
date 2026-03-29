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
        provider_level_script: str,
        splitter,
        provider_level_csv: Optional[str] = None,
        output_dir: Optional[str] = None,
        min_months: int = 6,
        no_filter: bool = False,
        quick_features: bool = False,
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
        self.feature_selection = feature_selection or {"auroc_top_n": 50}
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
        if self.hparams.provider_level_csv:
            out = Path(self.hparams.provider_level_csv)
            log.info(f"Loading existing provider_level: {out}")
            return pd.read_csv(out, low_memory=False)

        if self.hparams.output_dir:
            out = Path(self.hparams.output_dir) / "provider_level.csv"
        else:
            src = Path(self.hparams.provider_month_csv)
            out = src.parent / f"provider_level_{src.stem}.csv"

        log.info(f"Computing provider_level → {out}")
        run_provider_level(
            input_csv=self.hparams.provider_month_csv,
            output_csv=str(out),
            provider_level_script=self.hparams.provider_level_script,
            min_months=self.hparams.min_months,
            no_filter=self.hparams.no_filter,
            quick_features=self.hparams.quick_features,
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

    def _select_features_unsupervised(self, X: pd.DataFrame) -> List[str]:
        """Drop near-constant then correlation-prune. No labels required."""
        const_threshold = self.feature_selection.get("const_threshold", 0.01)
        corr_threshold  = self.feature_selection.get("corr_threshold", 0.9)

        std = X.std(skipna=True)
        drop_const = std[std < const_threshold].index.tolist()
        X = X.drop(columns=drop_const)
        log.info(f"Unsupervised: dropped {len(drop_const)} near-constant features (std < {const_threshold}) → {X.shape[1]} remain")

        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_corr = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X = X.drop(columns=drop_corr)
        log.info(f"Unsupervised: dropped {len(drop_corr)} correlated features (r > {corr_threshold}) → {X.shape[1]} remain")

        return list(X.columns)

    # ── Group definitions for grouped feature selection ───────────────────────
    _FEATURE_GROUPS = {
        "A": ["paid_t", "claims_t", "hcpcs_count_t", "beneficiaries_proxy_t"],
        "B": ["paid_per_claim_t", "claims_per_beneficiary_proxy_t", "paid_per_beneficiary_proxy_t"],
        "C": ["top_code_paid_share", "top_3_code_paid_share", "hcpcs_entropy", "hcpcs_hhi"],
        "D": ["top_code_claim_share_t", "top_3_code_claim_share_t", "hcpcs_claim_entropy_t", "hcpcs_claim_hhi_t"],
        "E": ["top_code_beneficiary_share_t", "top_3_code_beneficiary_share_t", "hcpcs_beneficiary_entropy_t", "hcpcs_beneficiary_hhi_t"],
        "F": ["top_code_paid_minus_claim_share_t", "top_code_paid_minus_beneficiary_share_t", "hcpcs_paid_hhi_minus_claim_hhi_t"],
    }

    def _assign_feature_groups(self, features: List[str]) -> dict:
        """Map each feature name to a group label. Longest source prefix wins."""
        # Build sorted lookup: longest monthly col name first to prevent prefix theft
        lookup = []
        for group, cols in self._FEATURE_GROUPS.items():
            for col in cols:
                lookup.append((col, group))
        lookup.sort(key=lambda x: len(x[0]), reverse=True)

        assignment = {}
        for feat in features:
            for col, group in lookup:
                if feat.startswith(col):
                    assignment[feat] = group
                    break
        return assignment  # unmatched features are omitted

    def _select_features_grouped(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        """Per-group pipeline: const filter → corr filter → top-N by AUROC."""
        top_n = self.feature_selection.get("top_n_per_group", 5)
        const_threshold = self.feature_selection.get("const_threshold", 0.05)
        corr_threshold = self.feature_selection.get("corr_threshold", 0.9)
        has_labels = len(np.unique(y)) >= 2

        assignment = self._assign_feature_groups(list(X.columns))
        groups: dict[str, List[str]] = {}
        for feat, grp in assignment.items():
            groups.setdefault(grp, []).append(feat)

        all_rows = []
        selected = []

        for grp in sorted(groups.keys()):
            feats = groups[grp]
            Xg = X[feats]

            # Constant filter
            std = Xg.std(skipna=True)
            Xg = Xg.loc[:, std >= const_threshold]
            if Xg.empty:
                continue

            # Correlation filter
            corr = Xg.corr().abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            drop_corr = [c for c in upper.columns if any(upper[c] > corr_threshold)]
            Xg = Xg.drop(columns=drop_corr)
            if Xg.empty:
                continue

            survivors = list(Xg.columns)

            if not has_labels:
                # No labels — take first top_n survivors
                chosen = survivors[:top_n]
                for f in chosen:
                    all_rows.append({"feature": f, "auroc": float("nan"), "group": grp})
                selected.extend(chosen)
                continue

            # AUROC ranking
            rows = []
            for feat in survivors:
                series = Xg[feat].values
                valid_mask = ~np.isnan(series)
                xv, yv = series[valid_mask], y[valid_mask]
                if len(np.unique(yv)) < 2:
                    continue
                try:
                    auc = roc_auc_score(yv, xv)
                    auc = max(auc, 1 - auc)
                    rows.append({"feature": feat, "auroc": round(float(auc), 4), "group": grp})
                except Exception:
                    continue

            rows.sort(key=lambda r: r["auroc"], reverse=True)
            chosen = [r["feature"] for r in rows[:top_n]]
            all_rows.extend(rows)
            selected.extend(chosen)
            log.info(f"Group {grp}: {len(chosen)}/{len(survivors)} features selected (top {top_n} by AUROC).")

        self.auroc_df = (
            pd.DataFrame(all_rows)
            .sort_values(["group", "auroc"], ascending=[True, False])
            .reset_index(drop=True)
        )
        log.info(f"Grouped selection: {len(selected)} features across {len(groups)} groups.")
        return selected

    def _select_features(self, X: pd.DataFrame, y: np.ndarray) -> List[str]:
        """Dispatch to the configured feature selection method."""
        use_unsupervised = self.feature_selection.get("unsupervised", False)
        has_labels = len(np.unique(y)) >= 2
        method = self.feature_selection.get("method", "auroc")

        if use_unsupervised or not has_labels:
            if not has_labels and not use_unsupervised:
                log.warning("No positive labels found — falling back to unsupervised feature selection.")
            return self._select_features_unsupervised(X)

        if method == "demo":
            return self._select_features_grouped(X, y)

        # Default: flat top-N by AUROC
        top_n = self.feature_selection.get("auroc_top_n")
        if top_n is None:
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
        n = min(top_n, len(self.auroc_df))
        selected = self.auroc_df["feature"].head(n).tolist()
        log.info(f"Top {n} features by AUROC selected (out of {len(X.columns)}).")
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
        if self._dataset is not None:
            return  # already set up; Lightning calls this again inside trainer.fit()
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
