"""Build provider-level anomaly detection feature table from a provider-month CSV.

Design: column-driven, family-driven.
    provider-level features = (numeric monthly column) × (reusable feature family)

Each feature family is a pure function:
    (col, values, [gaps,] [cfg]) → flat dict of feature_name → value

The main loop applies each eligible family to every numeric monthly column present
in the input.  Adding a new monthly signal requires only adding it to one of the
eligibility sets below — no new logic blocks.

Key tuning knobs live in config.yaml under provider_level_features.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats

try:
    import ruptures as rpt
    RUPTURES_AVAILABLE = True
except ImportError:
    RUPTURES_AVAILABLE = False


# ── Config ─────────────────────────────────────────────────────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


#overridden at runtime when --config is passed from the Hydra pipeline.
_DEFAULT_CONFIG: dict = {
    "provider_level_features": {
        "min_months_observed": 6,
        "pct_growth_prior_floor": 10.0,
        "rolling_flags": {
            "window_obs": 3,
            "robust_z_threshold": 2.5,
            "last_k_observed": 6,
        },
        "changepoints": {
            "penalty": 2.0,
            "model": "rbf",
            "min_size": 3,
            "min_obs": 6,
        },
    },
}


def _load_config(path: Path = _CONFIG_PATH) -> dict:
    if not path.is_file():
        return _DEFAULT_CONFIG
    with open(path) as f:
        return yaml.safe_load(f)


_cfg_root = _load_config()
_cfg      = _cfg_root["provider_level_features"]

DEFAULT_OUTPUT_CSV = "provider_level.csv"
MIN_MONTHS_DEFAULT = _cfg["min_months_observed"]


# ── Column eligibility ─────────────────────────────────────────────────────────
# All numeric monthly columns to process, in output order.
# Columns absent from the input CSV are silently skipped (null blocks emitted).
_GROUP_A = ["paid_t", "claims_t", "hcpcs_count_t", "beneficiaries_proxy_t"]
_GROUP_B = ["paid_per_claim_t", "claims_per_beneficiary_proxy_t", "paid_per_beneficiary_proxy_t"]
_GROUP_C = ["top_code_paid_share", "top_3_code_paid_share", "hcpcs_entropy", "hcpcs_hhi"]
_GROUP_D = ["top_code_claim_share_t", "top_3_code_claim_share_t",
            "hcpcs_claim_entropy_t", "hcpcs_claim_hhi_t"]
_GROUP_E = ["top_code_beneficiary_share_t", "top_3_code_beneficiary_share_t",
            "hcpcs_beneficiary_entropy_t", "hcpcs_beneficiary_hhi_t"]
_GROUP_F = ["top_code_paid_minus_claim_share_t", "top_code_paid_minus_beneficiary_share_t",
            "hcpcs_paid_hhi_minus_claim_hhi_t"]

ALL_NUMERIC_COLUMNS: list[str] = _GROUP_A + _GROUP_B + _GROUP_C + _GROUP_D + _GROUP_E + _GROUP_F

# x_sum only for columns where summing across months is meaningful
ADDITIVE_COLUMNS: frozenset[str] = frozenset(_GROUP_A)

# Percent-growth only for positive-valued, non-bounded signals
PCT_GROWTH_COLUMNS: frozenset[str] = frozenset(_GROUP_A + _GROUP_B)

# Ratio spike features (max/median, max/mean) are suppressed for signed columns
SIGNED_OR_MISMATCH_COLUMNS: frozenset[str] = frozenset(_GROUP_F)

# Hardcoded per-family minimum obs (not in config — not worth tuning)
_MIN_OBS_SUMMARY    = 1
_MIN_OBS_CHANGE     = 2   # diffs, spikes, pct_growth
_MIN_OBS_SLOPE      = 3   # linregress needs ≥ 3 distinct points
# changepoints: governed by cfg["changepoints"]["min_obs"]

# Numerical floors (hardcoded — stable, no need to tune)
_RATIO_FLOOR   = 1e-8
_CV_MEAN_FLOOR = 1e-8


# ── Helpers ────────────────────────────────────────────────────────────────────
def _gap_months(months: pd.Series) -> np.ndarray:
    """Calendar-month gaps between consecutive observed months.
    Uses datetime64[M] arithmetic; exact regardless of day-of-month.
    E.g. [Jan-2023, Apr-2023, May-2023] → [3, 1]
    """
    m = months.to_numpy(dtype="datetime64[M]").astype(int)
    return np.diff(m).astype(float)


def _valid_obs_with_gaps(prov: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """Non-NaN values of `col` sorted by month, and inter-observation calendar gaps."""
    mask  = prov[col].notna()
    valid = prov[mask].sort_values("month")
    values = valid[col].to_numpy(dtype=float)
    gaps   = _gap_months(valid["month"]) if len(values) > 1 else np.array([], dtype=float)
    return values, gaps


def _safe_gaps(gaps: np.ndarray) -> np.ndarray:
    """Clamp to ≥ 1.0 to prevent divide-by-zero."""
    return np.where(gaps <= 0, 1.0, gaps)


def _mad(x: np.ndarray) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))))


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    """Robust z of each row against prior `window` OBSERVED rows (no look-ahead).
    Returns NaN until a full window is available.
    """
    values = series.to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    for i in range(window, len(values)):
        win = values[i - window: i]
        med = np.nanmedian(win)
        mad = float(np.nanmedian(np.abs(win - med)))
        if np.isnan(mad) or mad == 0:
            continue
        result[i] = (values[i] - med) / (1.4826 * mad)
    return pd.Series(result, index=series.index)


# ── Feature families ───────────────────────────────────────────────────────────

def family_history_support(prov: pd.DataFrame) -> dict:
    """History and panel-support features — computed ONCE per provider, not per column."""
    months = prov["month"].sort_values().dropna().reset_index(drop=True)
    n = len(months)
    if n == 0:
        return {
            "months_observed": 0, "first_month": None, "last_month": None,
            "span_months": np.nan, "missing_months_within_span": np.nan,
            "fraction_months_observed": np.nan,
            "mean_gap_months": np.nan, "max_gap_months": np.nan,
        }
    m_int = months.to_numpy(dtype="datetime64[M]").astype(int)
    span  = int(m_int[-1] - m_int[0]) + 1
    gaps  = np.diff(m_int).astype(float) if n > 1 else np.array([], dtype=float)
    return {
        "months_observed":            n,
        "first_month":                str(months.iloc[0])[:7],
        "last_month":                 str(months.iloc[-1])[:7],
        "span_months":                span,
        "missing_months_within_span": span - n,
        "fraction_months_observed":   float(n / span) if span > 0 else np.nan,
        "mean_gap_months":            float(gaps.mean()) if len(gaps) else np.nan,
        "max_gap_months":             float(gaps.max())  if len(gaps) else np.nan,
    }


def family_summary(col: str, values: np.ndarray) -> dict:
    """Summary statistics for one monthly signal.  Applies to ALL numeric columns.
    x_sum emitted only for ADDITIVE_COLUMNS.
    """
    if len(values) < _MIN_OBS_SUMMARY:
        out = {f"{col}_{s}": np.nan
               for s in ("mean", "median", "std", "min", "max", "q25", "q75", "iqr")}
        if col in ADDITIVE_COLUMNS:
            out[f"{col}_sum"] = np.nan
        return out
    q25, q75 = np.percentile(values, [25, 75])
    out = {
        f"{col}_mean":   float(np.mean(values)),
        f"{col}_median": float(np.median(values)),
        f"{col}_std":    float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        f"{col}_min":    float(np.min(values)),
        f"{col}_max":    float(np.max(values)),
        f"{col}_q25":    float(q25),
        f"{col}_q75":    float(q75),
        f"{col}_iqr":    float(q75 - q25),
    }
    if col in ADDITIVE_COLUMNS:
        out[f"{col}_sum"] = float(np.sum(values))
    return out


def family_gap_aware_change(col: str, values: np.ndarray, gaps: np.ndarray) -> dict:
    """Gap-aware change and variability features.  Applies to ALL numeric columns.

    Change metrics divide each consecutive diff by its calendar-month gap so that
    a 3-month gap counts the same per month as a 1-month gap.
    """
    n  = len(values)
    slope = np.nan
    if n >= _MIN_OBS_SLOPE:
        s, *_ = stats.linregress(np.arange(n, dtype=float), values)
        slope = float(s)

    mean_v = float(np.mean(values)) if n >= 1 else np.nan
    std_v  = float(np.std(values, ddof=1)) if n >= 2 else np.nan
    cv = float(std_v / mean_v) if (n >= 2 and not np.isnan(std_v) and abs(mean_v) > _CV_MEAN_FLOOR) else np.nan

    if n >= _MIN_OBS_CHANGE and len(gaps) > 0:
        sg          = _safe_gaps(gaps)
        abs_monthly = np.abs(np.diff(values)) / sg
        mean_abs   = float(np.mean(abs_monthly))
        median_abs = float(np.median(abs_monthly))
        max_abs    = float(np.max(abs_monthly))
    else:
        mean_abs = median_abs = max_abs = np.nan

    return {
        f"{col}_slope":                         slope,
        f"{col}_mad":                           _mad(values),
        f"{col}_cv":                            cv,
        f"{col}_mean_abs_monthlyized_change":   mean_abs,
        f"{col}_median_abs_monthlyized_change": median_abs,
        f"{col}_max_abs_monthlyized_change":    max_abs,
    }


def family_spike(col: str, values: np.ndarray, gaps: np.ndarray, is_signed: bool) -> dict:
    """Spike, drop, and extreme-value features.  Applies to ALL numeric columns.

    Ratio features (spike_ratio_max_to_*) are NaN for SIGNED_OR_MISMATCH_COLUMNS.
    """
    if len(values) < _MIN_OBS_CHANGE:
        return {
            f"{col}_spike_ratio_max_to_median":    np.nan,
            f"{col}_spike_ratio_max_to_mean":      np.nan,
            f"{col}_largest_monthlyized_increase": np.nan,
            f"{col}_largest_monthlyized_decrease": np.nan,
            f"{col}_max_abs_deviation_from_median": np.nan,
        }

    median_v = float(np.median(values))
    mean_v   = float(np.mean(values))
    max_v    = float(np.max(values))

    ratio_med  = (float(max_v / median_v) if not is_signed and abs(median_v) > _RATIO_FLOOR else np.nan)
    ratio_mean = (float(max_v / mean_v)   if not is_signed and abs(mean_v)   > _RATIO_FLOOR else np.nan)
    max_dev    = float(np.max(np.abs(values - median_v)))

    if len(gaps) > 0:
        sg          = _safe_gaps(gaps)
        monthlyized = np.diff(values) / sg
        pos = monthlyized[monthlyized > 0]
        neg = monthlyized[monthlyized < 0]
        largest_inc = float(pos.max()) if len(pos) else 0.0
        largest_dec = float(-neg.min()) if len(neg) else 0.0
    else:
        largest_inc = largest_dec = np.nan

    return {
        f"{col}_spike_ratio_max_to_median":    ratio_med,
        f"{col}_spike_ratio_max_to_mean":      ratio_mean,
        f"{col}_largest_monthlyized_increase": largest_inc,
        f"{col}_largest_monthlyized_decrease": largest_dec,
        f"{col}_max_abs_deviation_from_median": max_dev,
    }


def family_pct_growth(col: str, values: np.ndarray, gaps: np.ndarray, prior_floor: float) -> dict:
    """Percent-growth features.  Only applied to PCT_GROWTH_COLUMNS.

    Pairs where the prior value is below prior_floor are skipped to avoid
    near-zero blow-up.  NaN propagates naturally — no fillna(0).
    """
    empty = {
        f"{col}_max_monthlyized_pct_growth":   np.nan,
        f"{col}_largest_monthlyized_pct_drop": np.nan,
    }
    if len(values) < _MIN_OBS_CHANGE or len(gaps) == 0:
        return empty
    sg      = _safe_gaps(gaps)
    prior   = values[:-1]
    nxt     = values[1:]
    gate    = prior >= prior_floor
    if not gate.any():
        return empty
    pct     = (nxt[gate] - prior[gate]) / prior[gate] / sg[gate]
    drops   = -pct[pct < 0]
    return {
        f"{col}_max_monthlyized_pct_growth":   float(np.max(pct)),
        f"{col}_largest_monthlyized_pct_drop": float(drops.max()) if len(drops) else np.nan,
    }


def family_changepoint(col: str, values: np.ndarray, cfg_cp: dict) -> dict:
    """PELT changepoint detection.  Applies to ALL numeric columns.

    Operates on the observed sequence only; calendar gaps are not modelled.
    NaN = detection could not run (ruptures unavailable or insufficient obs).
    No fillna(0) — NaN propagates to downstream models.
    """
    empty = {
        f"{col}_changepoint_count":            np.nan,
        f"{col}_largest_level_shift":          np.nan,
        f"{col}_largest_relative_level_shift": np.nan,
    }
    if not RUPTURES_AVAILABLE or len(values) < cfg_cp["min_obs"]:
        return empty
    try:
        model = rpt.Pelt(
            model=cfg_cp["model"], min_size=cfg_cp["min_size"], jump=1
        ).fit(values.reshape(-1, 1))
        bps = [b for b in model.predict(pen=cfg_cp["penalty"]) if b < len(values)]
        largest_abs = largest_rel = 0.0
        if bps:
            segs = [0] + bps + [len(values)]
            for i in range(len(segs) - 2):
                pre  = float(np.median(values[segs[i]:segs[i + 1]]))
                post = float(np.median(values[segs[i + 1]:segs[i + 2]]))
                a = abs(post - pre)
                largest_abs = max(largest_abs, a)
                largest_rel = max(largest_rel, a / max(abs(pre), 1.0))
        return {
            f"{col}_changepoint_count":            len(bps),
            f"{col}_largest_level_shift":          largest_abs,
            f"{col}_largest_relative_level_shift": largest_rel,
        }
    except Exception:
        return empty


def family_flag(col: str, values: np.ndarray, cfg_flags: dict) -> dict:
    """Rolling robust-z anomaly flags.  Applies to ALL numeric columns.

    Rolling window is over OBSERVED rows (values already non-NaN), not calendar
    months.  This matches the original behavior.
    """
    window    = cfg_flags["window_obs"]
    threshold = cfg_flags["robust_z_threshold"]
    last_k    = cfg_flags["last_k_observed"]

    rz      = _rolling_robust_z(pd.Series(values), window)
    flagged = rz.abs() > threshold

    max_consec = cur = 0
    for f in flagged:
        if f:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    n         = len(values)
    evaluable = max(n - window, 0)
    recent    = int(flagged.iloc[-last_k:].sum()) if evaluable >= last_k else np.nan
    max_rz    = float(rz.abs().max()) if evaluable > 0 else np.nan

    return {
        f"{col}_max_abs_robust_z":           max_rz,
        f"{col}_num_flagged_months":          int(flagged.sum()),
        f"{col}_fraction_flagged_months":     float(flagged.sum() / evaluable) if evaluable > 0 else np.nan,
        f"{col}_num_flagged_last_k_obs":      recent,
        f"{col}_max_consecutive_flagged_obs": max_consec,
    }


def _null_block(col: str) -> dict:
    """All-NaN feature block for a column that is absent from the input or has no valid obs."""
    out: dict = {}
    for s in ("mean", "median", "std", "min", "max", "q25", "q75", "iqr"):
        out[f"{col}_{s}"] = np.nan
    if col in ADDITIVE_COLUMNS:
        out[f"{col}_sum"] = np.nan
    for s in ("slope", "mad", "cv",
              "mean_abs_monthlyized_change", "median_abs_monthlyized_change",
              "max_abs_monthlyized_change"):
        out[f"{col}_{s}"] = np.nan
    for s in ("spike_ratio_max_to_median", "spike_ratio_max_to_mean",
              "largest_monthlyized_increase", "largest_monthlyized_decrease",
              "max_abs_deviation_from_median"):
        out[f"{col}_{s}"] = np.nan
    if col in PCT_GROWTH_COLUMNS:
        out[f"{col}_max_monthlyized_pct_growth"]   = np.nan
        out[f"{col}_largest_monthlyized_pct_drop"] = np.nan
    for s in ("changepoint_count", "largest_level_shift", "largest_relative_level_shift"):
        out[f"{col}_{s}"] = np.nan
    for s in ("max_abs_robust_z", "num_flagged_months", "fraction_flagged_months",
              "num_flagged_last_k_obs", "max_consecutive_flagged_obs"):
        out[f"{col}_{s}"] = np.nan
    return out


# ── Main collapse ──────────────────────────────────────────────────────────────
def build_provider_level(
    df: pd.DataFrame,
    min_months: int = MIN_MONTHS_DEFAULT,
) -> pd.DataFrame:
    """One row per provider.

    For each provider:
      1. Compute history/support features once.
      2. For each numeric monthly column in ALL_NUMERIC_COLUMNS that exists in df:
         apply summary, gap_aware_change, spike, pct_growth (if eligible),
         changepoint, and flag families.
      3. Emit insufficient_history_flag.

    Columns missing from df produce all-NaN blocks (via _null_block).
    NaN is never imputed with 0 — undefined features stay NaN.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.sort_values(["billing_provider_npi", "month"])

    cfg_cp      = _cfg["changepoints"]
    cfg_flags   = _cfg["rolling_flags"]
    prior_floor = _cfg["pct_growth_prior_floor"]

    rows: list[dict] = []
    for npi, prov in df.groupby("billing_provider_npi", sort=False):
        prov = prov.sort_values("month").reset_index(drop=True)
        row: dict = {"billing_provider_npi": npi}

        # ── 1. History support (once per provider) ──────────────────────────
        row.update(family_history_support(prov))

        # ── 2. Feature families per numeric monthly column ──────────────────
        for col in ALL_NUMERIC_COLUMNS:
            if col not in prov.columns:
                row.update(_null_block(col))
                continue

            values, gaps = _valid_obs_with_gaps(prov, col)

            if len(values) == 0:
                row.update(_null_block(col))
                continue

            is_signed = col in SIGNED_OR_MISMATCH_COLUMNS

            row.update(family_summary(col, values))
            row.update(family_gap_aware_change(col, values, gaps))
            row.update(family_spike(col, values, gaps, is_signed))
            if col in PCT_GROWTH_COLUMNS:
                row.update(family_pct_growth(col, values, gaps, prior_floor))
            row.update(family_changepoint(col, values, cfg_cp))
            row.update(family_flag(col, values, cfg_flags))

        row["insufficient_history_flag"] = int(row["months_observed"] < min_months)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)



# ── I/O ────────────────────────────────────────────────────────────────────────
def load_provider_month(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provider-month CSV not found: {path}")
    df = pd.read_csv(p, low_memory=False)
    required = {"billing_provider_npi", "month", "paid_t"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return df


def run(
    input_csv: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    min_months: int = MIN_MONTHS_DEFAULT,
    filter_output: bool = True,
) -> pd.DataFrame:
    """Build provider-level features and save CSV.

    NaN is never imputed with 0 — undefined features propagate to the output.
    The only hard filter applied is providers whose median paid_t == 0
    (spike_ratio_max_to_median is NaN), which are degenerate for paid-based detection.
    Date range filtering must be applied upstream in create_provider_month_dataset.py.
    """
    if not RUPTURES_AVAILABLE:
        print("Warning: ruptures not installed — changepoint features will be NaN.\n"
              "Install with: pip install ruptures", file=sys.stderr)

    df = load_provider_month(input_csv)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    provider_level = build_provider_level(df, min_months=min_months)

    n_insuf = int(provider_level["insufficient_history_flag"].sum())
    print(f"Providers built: {len(provider_level):,}  "
          f"({n_insuf:,} with months_observed < {min_months} → insufficient_history_flag=1)")

    if filter_output:
        provider_level = provider_level[provider_level["insufficient_history_flag"] == 0].copy()
        print(f"After history filter: {len(provider_level):,} providers retained.")

    # Drop providers where median paid_t == 0 — degenerate for paid-based anomaly detection
    spike_col = "paid_t_spike_ratio_max_to_median"
    if spike_col in provider_level.columns:
        before = len(provider_level)
        provider_level = provider_level[provider_level[spike_col].notna()].copy()
        print(f"Dropped {before - len(provider_level):,} providers with median paid_t = 0.")

    # ── Condense cohort_label (first value per provider) ─────────────────────
    if "cohort_label" in df.columns:
        npi_cohort = df.groupby("billing_provider_npi")["cohort_label"].first().reset_index()
        provider_level = provider_level.merge(npi_cohort, on="billing_provider_npi", how="left")
        cols = ["billing_provider_npi", "cohort_label"] + [
            c for c in provider_level.columns
            if c not in ("billing_provider_npi", "cohort_label")
        ]
        provider_level = provider_level[cols]

    # ── Condense label (max per provider — 1 if ever excluded) ───────────────
    if "label" in df.columns:
        npi_label = df.groupby("billing_provider_npi")["label"].max().reset_index()
        provider_level = provider_level.merge(npi_label, on="billing_provider_npi", how="left")
        print(f"Labels condensed: {int((provider_level['label'] == 1).sum()):,} providers with label=1.")
    else:
        provider_level["label"] = float("nan")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider_level.to_csv(out_path, index=False)
    print(f"Output CSV:   {out_path.resolve()}")
    print(f"Output shape: {provider_level.shape}")
    return provider_level


def main():
    parser = argparse.ArgumentParser(
        description="Build provider-level anomaly detection features from a provider-month CSV."
    )
    parser.add_argument("input_csv",
                        help="Path to provider-month CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).")
    parser.add_argument("--min-months", type=int, default=MIN_MONTHS_DEFAULT,
                        help=f"Threshold for insufficient_history_flag (default: {MIN_MONTHS_DEFAULT}).")
    parser.add_argument("--no-filter", action="store_true", default=False,
                        help="Keep all providers in output regardless of months_observed.")
    parser.add_argument("--config", default=None,
                        help="Override path to config.yaml (default: scripts/config.yaml). "
                             "Useful when called from the Hydra pipeline with a temp config.")
    args = parser.parse_args()

    # Reload globals from an alternate config if provided.
    if args.config:
        global _cfg_root, _cfg
        _cfg_root = _load_config(Path(args.config))
        _cfg = _cfg_root["provider_level_features"]

    provider_level = run(
        args.input_csv, args.output,
        args.min_months,
        filter_output=not args.no_filter,
    )
    print(f"\nColumns ({len(provider_level.columns)}): {list(provider_level.columns)[:10]} ...")
    print(f"\nHead:\n{provider_level.head()}")
    print(f"\nMissing value counts (top 20):\n{provider_level.isna().sum().sort_values(ascending=False).head(20)}")


if __name__ == "__main__":
    main()
