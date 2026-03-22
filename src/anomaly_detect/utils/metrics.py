"""Lift metrics for anomaly detection evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Tuple

LIFT_PERCENTILES = [0.5, 1, 2, 3, 5, 10]


def compute_lift_at_percentiles(
    scores: np.ndarray,
    labels: np.ndarray,
    npis: np.ndarray,
    percentiles: List[float] = LIFT_PERCENTILES,
) -> Tuple[dict, pd.DataFrame]:
    """Compute lift at top X% of anomaly scores (higher score = more anomalous).

    Returns
    -------
    metrics : dict
        Keys: base_rate, n_providers, n_positive,
              lift_top_{p}pct, precision_top_{p}pct, hits_top_{p}pct for each p.
    results_df : pd.DataFrame
        All providers sorted by anomaly_score descending, with label column.
    """
    n = len(scores)
    base_rate = float(np.mean(labels))

    sort_idx = np.argsort(scores)[::-1]
    sorted_scores = scores[sort_idx]
    sorted_labels = labels[sort_idx]
    sorted_npis = npis[sort_idx]

    metrics: dict = {
        "base_rate": round(base_rate, 6),
        "n_providers": n,
        "n_positive": int(labels.sum()),
    }

    for pct in percentiles:
        k = max(1, int(n * pct / 100))
        hits = int(sorted_labels[:k].sum())
        precision_k = hits / k
        lift = precision_k / base_rate if base_rate > 0 else float("nan")
        pct_str = str(pct).replace(".", "_")
        metrics[f"hits_top_{pct_str}pct"] = hits
        metrics[f"precision_top_{pct_str}pct"] = round(precision_k, 4)
        metrics[f"lift_top_{pct_str}pct"] = round(lift, 4)

    results_df = pd.DataFrame({
        "billing_provider_npi": sorted_npis,
        "label": sorted_labels,
        "anomaly_score": sorted_scores,
    }).reset_index(drop=True)

    return metrics, results_df


def build_lift_table(metrics: dict, percentiles: List[float]) -> pd.DataFrame:
    """Return lift table as a DataFrame (used for saving and printing)."""
    base_rate = metrics.get("base_rate", float("nan"))
    n_tot = metrics.get("n_providers", 1)
    rows = []
    for pct in percentiles:
        pct_str = str(pct).replace(".", "_")
        k = max(1, int(n_tot * pct / 100))
        hits = metrics.get(f"hits_top_{pct_str}pct", 0)
        prec = metrics.get(f"precision_top_{pct_str}pct", float("nan"))
        lift = metrics.get(f"lift_top_{pct_str}pct", float("nan"))
        expected_hits = round(k * base_rate, 2)
        rows.append({
            "top_pct": pct,
            "k": k,
            "hits": hits,
            "expected_hits_random": expected_hits,
            "precision": prec,
            "lift": lift,
        })
    return pd.DataFrame(rows)


def print_lift_table(metrics: dict, percentiles: List[float]) -> None:
    base_rate = metrics.get("base_rate", float("nan"))
    n_pos = metrics.get("n_positive", "?")
    n_tot = metrics.get("n_providers", "?")
    print(f"\nBase rate (LEIE prevalence): {base_rate:.4f}  ({n_pos} positives / {n_tot} total)")
    print(f"{'Top %':>8}  {'k':>6}  {'Hits':>6}  {'Expected':>10}  {'Precision':>10}  {'Lift':>8}")
    print("-" * 60)
    df = build_lift_table(metrics, percentiles)
    for _, row in df.iterrows():
        print(
            f"{row.top_pct:>7.1f}%  {int(row.k):>6}  {int(row.hits):>6}"
            f"  {row.expected_hits_random:>10.1f}  {row.precision:>10.4f}  {row.lift:>8.2f}x"
        )
