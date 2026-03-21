"""Build provider-level anomaly detection feature table from a provider-month CSV."""

import argparse
import json
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

# ── CONFIG — loaded from config.yaml next to this script ──────────────────────
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

def _load_config(path: Path = _CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)

_cfg = _load_config()

DEFAULT_OUTPUT_CSV     = "provider_level.csv"
DEFAULT_DICTIONARY_CSV = "provider_level_data_dictionary.csv"
MIN_MONTHS_DEFAULT     = _cfg["min_months_default"]
DATE_CUTOFF            = _cfg["date_cutoff"]
ROBUST_Z_WINDOW        = _cfg["robust_z_window"]
ROBUST_Z_THRESHOLD     = _cfg["robust_z_threshold"]
FLAG_RECENT_MONTHS     = _cfg["flag_recent_months"]
MOM_MIN_DENOMINATOR    = _cfg["mom_min_denominator"]
HIGH_ENTROPY_CHANGE    = _cfg["high_entropy_change"]
HIGH_HHI_CHANGE        = _cfg["high_hhi_change"]
HIGH_CONCENTRATION_HHI = _cfg["high_concentration_hhi"]
TOP3_HIGH_THRESHOLD    = _cfg["top3_high_threshold"]
CHANGEPOINT_PENALTY    = _cfg["changepoint_penalty"]
CHANGEPOINT_MODEL      = _cfg["changepoint_model"]
CHANGEPOINT_MIN_SIZE   = _cfg["changepoint_min_size"]
CHANGEPOINT_MIN_OBS    = _cfg["changepoint_min_obs"]


# ── Helpers ────────────────────────────────────────────────────────────────────
def _gap_months(months: pd.Series) -> np.ndarray:
    """Calendar-month gaps between consecutive observed months.
    Uses datetime64[M] arithmetic; exact regardless of day-of-month.
    Example: [Jan-2023, Apr-2023, May-2023] → [3, 1]
    """
    m = months.to_numpy(dtype="datetime64[M]").astype(int)
    return np.diff(m).astype(float)


def _valid_obs_with_gaps(prov: pd.DataFrame, col: str = "paid_t") -> tuple[np.ndarray, np.ndarray]:
    """Non-NaN values of `col` (sorted by month) and their inter-observation calendar gaps."""
    mask = ~prov[col].isna()
    valid = prov[mask].sort_values("month")
    values = valid[col].to_numpy(dtype=float)
    gaps = _gap_months(valid["month"]) if len(values) > 1 else np.array([], dtype=float)
    return values, gaps


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (NaN-safe)."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(np.median(np.abs(x - np.median(x))))


def _rolling_robust_z(series: pd.Series, window: int) -> pd.Series:
    """Robust z-score of each row against the prior `window` OBSERVED rows (no look-ahead).
    Baseline is prior-N-observed-rows, NOT prior-N-calendar-months.
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


def _linear_slope(x: np.ndarray) -> float:
    """OLS slope per observed step (not per calendar month — spacing is irregular)."""
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    slope, *_ = stats.linregress(np.arange(len(x), dtype=float), x)
    return float(slope)


def _nan_safe_frac(arr: np.ndarray, condition: np.ndarray) -> float:
    """Fraction of non-NaN entries satisfying `condition`.
    Excludes NaN positions from both numerator and denominator.
    """
    valid = ~np.isnan(arr)
    if not valid.any():
        return np.nan
    return float(np.mean(condition[valid]))


# ── Feature groups ─────────────────────────────────────────────────────────────
def _has_valid(arr: np.ndarray) -> bool:
    """True if arr has at least one non-NaN value."""
    return bool(np.any(~np.isnan(arr)))


def compute_volume_scale_features(prov: pd.DataFrame) -> dict:
    paid = prov["paid_t"].to_numpy(dtype=float)
    v = _has_valid(paid)
    return {
        "months_active": int(prov["month"].nunique()),
        "sum_paid":      float(np.nansum(paid)),
        "mean_paid":     float(np.nanmean(paid))   if v else np.nan,
        "median_paid":   float(np.nanmedian(paid)) if v else np.nan,
        "max_paid":      float(np.nanmax(paid))    if v else np.nan,
    }


def compute_unit_economics_features(prov: pd.DataFrame) -> dict:
    """Paid-per-claim summaries.
    claim_weighted_paid_per_claim = sum(paid_t)/sum(claims_t); more robust than
    the unweighted mean because high-volume months contribute proportionally.
    """
    ppc = prov["paid_per_claim_t"].to_numpy(dtype=float)

    if "claims_t" in prov.columns:
        valid = prov["claims_t"] > 0
        cwppc = float(prov.loc[valid, "paid_t"].sum() / prov.loc[valid, "claims_t"].sum()) if valid.any() else np.nan
    else:
        cwppc = np.nan

    return {
        "mean_paid_per_claim":           float(np.nanmean(ppc))        if len(ppc) else np.nan,
        "median_paid_per_claim":         float(np.nanmedian(ppc))      if len(ppc) else np.nan,
        "std_paid_per_claim":            float(np.nanstd(ppc, ddof=1)) if len(ppc) > 1 else np.nan,
        "claim_weighted_paid_per_claim": cwppc,
    }


def compute_volatility_features(prov: pd.DataFrame) -> dict:
    """Volatility of paid_t.
    mean_abs_monthlyized_change_paid normalizes each consecutive diff by its
    calendar gap so a jump across 3 months counts the same per month as a 1-month jump.
    """
    paid = prov["paid_t"].to_numpy(dtype=float)
    clean = paid[~np.isnan(paid)]
    mean_p = float(np.nanmean(clean)) if len(clean) else np.nan
    std_p  = float(np.nanstd(clean, ddof=1)) if len(clean) > 1 else np.nan
    cv = float(std_p / mean_p) if (mean_p and mean_p != 0 and not np.isnan(std_p)) else np.nan

    values, gaps = _valid_obs_with_gaps(prov)
    if len(values) > 1 and len(gaps):
        safe_gaps = np.where(gaps <= 0, 1.0, gaps)
        mean_abs_monthlyized = float(np.mean(np.abs(np.diff(values)) / safe_gaps))
    else:
        mean_abs_monthlyized = np.nan

    return {
        "cv_paid":                          cv,
        "mad_paid":                         _mad(paid),
        "mean_abs_monthlyized_change_paid": mean_abs_monthlyized,
    }


def compute_spike_drop_features(prov: pd.DataFrame) -> dict:
    """Spike, drop, and growth features; all change metrics are gap-aware (monthlyized).
    max_monthlyized_paid_growth: pct growth rate per month; pairs where prior < MOM_MIN_DENOMINATOR are skipped.
    largest_monthlyized_paid_drop = max(-adj_diff_i for negative monthlyized diffs)
    """
    paid = prov["paid_t"].to_numpy(dtype=float)
    clean = paid[~np.isnan(paid)]

    if len(clean) == 0:
        return {
            "spike_ratio_paid":              np.nan,
            "max_monthlyized_paid_growth":   np.nan,
            "largest_monthlyized_paid_drop": np.nan,
        }

    median_p = float(np.nanmedian(clean))
    spike_ratio = float(np.nanmax(clean) / median_p) if median_p > 0 else np.nan

    values, gaps = _valid_obs_with_gaps(prov)
    if len(values) < 2 or not len(gaps):
        return {
            "spike_ratio_paid":              spike_ratio,
            "max_monthlyized_paid_growth":   np.nan,
            "largest_monthlyized_paid_drop": np.nan,
        }

    safe_gaps = np.where(gaps <= 0, 1.0, gaps)
    adj_diffs = np.diff(values) / safe_gaps

    drops = -adj_diffs[adj_diffs < 0]
    largest_monthlyized_drop = float(drops.max()) if len(drops) else 0.0

    prior = values[:-1]
    gate  = prior >= MOM_MIN_DENOMINATOR
    if gate.any():
        adj_pct = (values[1:][gate] - prior[gate]) / prior[gate] / safe_gaps[gate]
        max_growth = float(np.max(adj_pct))
    else:
        max_growth = np.nan

    return {
        "spike_ratio_paid":              spike_ratio,
        "max_monthlyized_paid_growth":   max_growth,
        "largest_monthlyized_paid_drop": largest_monthlyized_drop,
    }


def compute_changepoint_features(prov: pd.DataFrame) -> dict:
    """PELT changepoint detection on the ordered observed paid_t sequence.
    Operates on observed rows only; calendar gaps between rows are not modelled.
    largest_level_shift_paid = abs(median(seg_{j+1}) - median(seg_j)) in dollars.
    largest_relative_level_shift_paid = above / max(|pre_median|, 1).
    """
    empty = {
        "changepoint_count_paid":            np.nan,
        "largest_level_shift_paid":          np.nan,
        "largest_relative_level_shift_paid": np.nan,
    }

    if not RUPTURES_AVAILABLE:
        return empty

    values, _ = _valid_obs_with_gaps(prov)
    if len(values) < CHANGEPOINT_MIN_OBS:
        return empty

    try:
        model       = rpt.Pelt(model=CHANGEPOINT_MODEL, min_size=CHANGEPOINT_MIN_SIZE, jump=1).fit(values.reshape(-1, 1))
        breakpoints = [b for b in model.predict(pen=CHANGEPOINT_PENALTY) if b < len(values)]
        count       = len(breakpoints)

        largest_abs_shift = 0.0
        largest_rel_shift = 0.0
        if breakpoints:
            segs = [0] + breakpoints + [len(values)]
            abs_shifts, rel_shifts = [], []
            for i in range(len(segs) - 2):
                pre_med  = float(np.median(values[segs[i]:segs[i + 1]]))
                post_med = float(np.median(values[segs[i + 1]:segs[i + 2]]))
                a = abs(post_med - pre_med)
                abs_shifts.append(a)
                rel_shifts.append(a / max(abs(pre_med), 1.0))
            largest_abs_shift = float(max(abs_shifts))
            largest_rel_shift = float(max(rel_shifts))
    except Exception:
        return empty

    return {
        "changepoint_count_paid":            count,
        "largest_level_shift_paid":          largest_abs_shift,
        "largest_relative_level_shift_paid": largest_rel_shift,
    }


def compute_flagged_month_features(prov: pd.DataFrame) -> dict:
    """Flag anomalous months via rolling robust z-score.
    Baseline is prior ROBUST_Z_WINDOW OBSERVED rows, not calendar months.
    All counts refer to observed-row positions, not calendar continuity.
    """
    paid     = prov["paid_t"].reset_index(drop=True)
    robust_z = _rolling_robust_z(paid, ROBUST_Z_WINDOW)
    flagged  = robust_z.abs() > ROBUST_Z_THRESHOLD

    max_consec, cur = 0, 0
    for f in flagged:
        if f:
            cur += 1
            max_consec = max(max_consec, cur)
        else:
            cur = 0

    n = len(flagged)
    # Denominator is only the rows that were actually evaluated (past the warmup window).
    evaluable = max(n - ROBUST_Z_WINDOW, 0)
    # num_flagged_last_6_obs is NaN when fewer than FLAG_RECENT_MONTHS rows were evaluated,
    # so it is not compared against providers with fuller history.
    recent = int(flagged.iloc[-FLAG_RECENT_MONTHS:].sum()) if evaluable >= FLAG_RECENT_MONTHS else np.nan

    return {
        "num_flagged_months":          int(flagged.sum()),
        "fraction_flagged_months":     float(flagged.sum() / evaluable) if evaluable > 0 else np.nan,
        "num_flagged_last_6_obs":      recent,
        "max_consecutive_flagged_obs": max_consec,
    }


def compute_code_mix_summary_features(prov: pd.DataFrame) -> dict:
    """Code-mix summaries. Entropy/HHI change features are gap-aware (monthlyized).
    Fraction features exclude NaN months from both numerator and denominator.
    """
    out: dict = {}

    # ── top-code paid share ────────────────────────────────────────────────────
    if "top_code_paid_share" in prov.columns:
        tcps = prov["top_code_paid_share"].to_numpy(dtype=float)
        out["mean_top_code_paid_share"] = float(np.nanmean(tcps)) if _has_valid(tcps) else np.nan
        out["max_top_code_paid_share"]  = float(np.nanmax(tcps))  if _has_valid(tcps) else np.nan
    else:
        out["mean_top_code_paid_share"] = np.nan
        out["max_top_code_paid_share"]  = np.nan

    # ── top-3 paid share ──────────────────────────────────────────────────────
    if "top_3_code_paid_share" in prov.columns:
        t3 = prov["top_3_code_paid_share"].to_numpy(dtype=float)
        out["mean_top_3_code_paid_share"]       = float(np.nanmean(t3)) if _has_valid(t3) else np.nan
        out["max_top_3_code_paid_share"]        = float(np.nanmax(t3))  if _has_valid(t3) else np.nan
        out["fraction_months_top3_above_80pct"] = _nan_safe_frac(t3, t3 > TOP3_HIGH_THRESHOLD)
    else:
        out["mean_top_3_code_paid_share"]       = np.nan
        out["max_top_3_code_paid_share"]        = np.nan
        out["fraction_months_top3_above_80pct"] = np.nan

    # ── entropy ────────────────────────────────────────────────────────────────
    if "hcpcs_entropy" in prov.columns:
        ent = prov["hcpcs_entropy"].to_numpy(dtype=float)
        out["min_hcpcs_entropy"]  = float(np.nanmin(ent))  if _has_valid(ent) else np.nan
        out["mean_hcpcs_entropy"] = float(np.nanmean(ent)) if _has_valid(ent) else np.nan

        ent_vals, gaps = _valid_obs_with_gaps(prov, col="hcpcs_entropy")
        if len(ent_vals) > 1 and len(gaps):
            adj = np.abs(np.diff(ent_vals)) / np.where(gaps <= 0, 1.0, gaps)
            out["max_monthlyized_entropy_change"]  = float(np.max(adj))
            out["mean_monthlyized_entropy_change"] = float(np.mean(adj))
            out["num_months_high_entropy_change"]  = int(np.sum(adj > HIGH_ENTROPY_CHANGE))
        else:
            out["max_monthlyized_entropy_change"]  = np.nan
            out["mean_monthlyized_entropy_change"] = np.nan
            out["num_months_high_entropy_change"]  = 0
    else:
        out.update({k: np.nan for k in (
            "min_hcpcs_entropy", "mean_hcpcs_entropy",
            "max_monthlyized_entropy_change", "mean_monthlyized_entropy_change",
        )})
        out["num_months_high_entropy_change"] = 0

    # ── HHI ────────────────────────────────────────────────────────────────────
    if "hcpcs_hhi" in prov.columns:
        hhi = prov["hcpcs_hhi"].to_numpy(dtype=float)
        out["mean_hcpcs_hhi"]                    = float(np.nanmean(hhi)) if _has_valid(hhi) else np.nan
        out["max_hcpcs_hhi"]                     = float(np.nanmax(hhi))  if _has_valid(hhi) else np.nan
        out["fraction_months_high_concentration"] = _nan_safe_frac(hhi, hhi > HIGH_CONCENTRATION_HHI)

        hhi_vals, gaps = _valid_obs_with_gaps(prov, col="hcpcs_hhi")
        if len(hhi_vals) > 1 and len(gaps):
            adj = np.abs(np.diff(hhi_vals)) / np.where(gaps <= 0, 1.0, gaps)
            out["max_monthlyized_hhi_change"]  = float(np.max(adj))
            out["mean_monthlyized_hhi_change"] = float(np.mean(adj))
            out["num_months_high_hhi_change"]  = int(np.sum(adj > HIGH_HHI_CHANGE))
        else:
            out["max_monthlyized_hhi_change"]  = np.nan
            out["mean_monthlyized_hhi_change"] = np.nan
            out["num_months_high_hhi_change"]  = 0
    else:
        out.update({k: np.nan for k in (
            "mean_hcpcs_hhi", "max_hcpcs_hhi", "fraction_months_high_concentration",
            "max_monthlyized_hhi_change", "mean_monthlyized_hhi_change",
        )})
        out["num_months_high_hhi_change"] = 0

    return out


def compute_code_diversity_features(prov: pd.DataFrame) -> dict:
    """hcpcs_count_trend is slope per observed step (not per calendar month)."""
    if "hcpcs_count_t" not in prov.columns:
        return {k: np.nan for k in (
            "mean_hcpcs_count", "max_hcpcs_count", "std_hcpcs_count",
            "hcpcs_count_trend", "fraction_months_single_code",
        )}

    hc = prov["hcpcs_count_t"].to_numpy(dtype=float)
    return {
        "mean_hcpcs_count":            float(np.nanmean(hc)),
        "max_hcpcs_count":             float(np.nanmax(hc))         if len(hc) else np.nan,
        "std_hcpcs_count":             float(np.nanstd(hc, ddof=1)) if len(hc) > 1 else np.nan,
        "hcpcs_count_trend":           _linear_slope(hc),
        "fraction_months_single_code": _nan_safe_frac(hc, np.round(hc).astype(int) == 1),
    }


# ── Main collapse ──────────────────────────────────────────────────────────────
def build_provider_level(df: pd.DataFrame, min_months: int = MIN_MONTHS_DEFAULT) -> pd.DataFrame:
    """One row per provider with all feature groups.
    All providers are kept; insufficient_history_flag=1 when months_active < min_months.
    """
    df = df.copy()
    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    df = df.sort_values(["billing_provider_npi", "month"])

    rows = []
    for npi, prov in df.groupby("billing_provider_npi", sort=False):
        prov = prov.sort_values("month").reset_index(drop=True)
        row: dict = {"billing_provider_npi": npi}
        row.update(compute_volume_scale_features(prov))
        row.update(compute_unit_economics_features(prov))
        row.update(compute_volatility_features(prov))
        row.update(compute_spike_drop_features(prov))
        row.update(compute_changepoint_features(prov))
        row.update(compute_flagged_month_features(prov))
        row.update(compute_code_mix_summary_features(prov))
        row.update(compute_code_diversity_features(prov))
        row["insufficient_history_flag"] = int(row["months_active"] < min_months)
        rows.append(row)

    return pd.DataFrame(rows).reset_index(drop=True)


# ── Data dictionary ───────────────────────────────────────────────────────────
_DICT_SOURCE = Path(__file__).parent / "provider_level_data_dictionary.json"


def build_data_dictionary(df: pd.DataFrame, source: Path = _DICT_SOURCE) -> list[dict]:
    """Load dictionary from JSON, validate against df columns, return in df column order."""
    with open(source, encoding="utf-8") as f:
        entries = json.load(f)
    dict_cols   = {e["column_name"] for e in entries}
    actual_cols = set(df.columns)
    missing_in_dict = actual_cols - dict_cols
    extra_in_dict   = dict_cols - actual_cols
    if missing_in_dict:
        raise ValueError(f"Data dictionary missing entries for columns: {missing_in_dict}")
    if extra_in_dict:
        raise ValueError(f"Data dictionary has entries for columns not in output: {extra_in_dict}")
    col_order = {col: i for i, col in enumerate(df.columns)}
    return sorted(entries, key=lambda e: col_order[e["column_name"]])


# ── I/O ────────────────────────────────────────────────────────────────────────
def load_provider_month(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provider-month CSV not found: {path}")
    df = pd.read_csv(p, low_memory=False)
    required = {"billing_provider_npi", "month", "paid_t", "claims_t", "paid_per_claim_t"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing required columns: {missing}")
    return df


def run(
    input_csv: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    dictionary_csv: str = DEFAULT_DICTIONARY_CSV,
    dictionary_json: Path = _DICT_SOURCE,
    min_months: int = MIN_MONTHS_DEFAULT,
    date_cutoff: str = DATE_CUTOFF,
    filter_output: bool = True,
) -> pd.DataFrame:
    """Build features for ALL providers, then optionally filter output to months_active >= min_months.
    Saves the provider-level CSV and a human-readable data dictionary CSV.
    """
    if not RUPTURES_AVAILABLE:
        print("Warning: ruptures not installed — changepoint features will be NaN.\n"
              "Install with: pip install ruptures", file=sys.stderr)

    df = load_provider_month(input_csv)
    df["month"] = pd.to_datetime(df["month"], errors="coerce")

    cutoff = pd.Timestamp(date_cutoff)
    before = len(df)
    df = df[df["month"] <= cutoff]
    print(f"Date cutoff {date_cutoff}: {before - len(df):,} rows removed, {len(df):,} remaining.")

    provider_level = build_provider_level(df, min_months=min_months)

    n_insuf = int(provider_level["insufficient_history_flag"].sum())
    print(f"Providers built: {len(provider_level):,}  "
          f"({n_insuf:,} with months_active < {min_months} → insufficient_history_flag=1)")

    if filter_output:
        provider_level = provider_level[provider_level["insufficient_history_flag"] == 0].copy()
        print(f"After output filter: {len(provider_level):,} providers retained.")

    # Drop providers where median paid == 0 (spike_ratio_paid NaN) — not useful for paid-based detection
    before = len(provider_level)
    provider_level = provider_level[provider_level["spike_ratio_paid"].notna()].copy()
    print(f"Dropped {before - len(provider_level):,} providers with median paid = 0.")

    # Impute changepoint shift columns with 0 (no changepoint detected = no shift)
    provider_level["largest_level_shift_paid"] = provider_level["largest_level_shift_paid"].fillna(0)
    provider_level["largest_relative_level_shift_paid"] = provider_level["largest_relative_level_shift_paid"].fillna(0)

    # Impute variation features with 0 where NaN (all-identical values → no variation)
    for col in ["cv_paid", "std_paid_per_claim", "max_monthlyized_paid_growth"]:
        if col in provider_level.columns:
            provider_level[col] = provider_level[col].fillna(0)

    # Condense label from provider-month (all months share the same label per provider)
    if "label" in df.columns:
        npi_label = df.groupby("billing_provider_npi")["label"].max().reset_index()
        provider_level = provider_level.merge(npi_label, on="billing_provider_npi", how="left")
        n_positive = int((provider_level["label"] == 1).sum())
        print(f"Labels condensed: {n_positive:,} providers with label=1.")
    else:
        provider_level["label"] = float("nan")

    # Save CSV
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider_level.to_csv(out_path, index=False)
    print(f"Output CSV:       {out_path.resolve()}")

    # Save data dictionary CSV (JSON source lives statically in scripts/)
    data_dict = build_data_dictionary(provider_level, source=Path(dictionary_json))
    csv_path = Path(dictionary_csv)
    if not csv_path.is_absolute():
        csv_path = out_path.parent / csv_path
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data_dict).to_csv(csv_path, index=False)
    print(f"Dictionary CSV:   {csv_path.resolve()}")

    return provider_level


def main():
    parser = argparse.ArgumentParser(
        description="Build provider-level anomaly detection features from a provider-month CSV."
    )
    parser.add_argument("input_csv", help="Path to provider-month CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_CSV,
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).")
    parser.add_argument("--dictionary-csv", default=DEFAULT_DICTIONARY_CSV,
                        help=f"Output path for data dictionary CSV (default: {DEFAULT_DICTIONARY_CSV}).")
    parser.add_argument("--dictionary-json", default=str(_DICT_SOURCE),
                        help=f"Path to data dictionary JSON source (default: {_DICT_SOURCE}).")
    parser.add_argument("--min-months", type=int, default=MIN_MONTHS_DEFAULT,
                        help=f"Threshold for insufficient_history_flag (default: {MIN_MONTHS_DEFAULT}).")
    parser.add_argument("--date-cutoff", default=DATE_CUTOFF,
                        help=f"Exclude months after this date (default: {DATE_CUTOFF}).")
    parser.add_argument("--no-filter", action="store_true", default=False,
                        help="Keep all providers in output regardless of months_active.")
    args = parser.parse_args()

    provider_level = run(
        args.input_csv, args.output, args.dictionary_csv, args.dictionary_json,
        args.min_months, args.date_cutoff,
        filter_output=not args.no_filter,
    )

    print(f"\nprovider_level shape: {provider_level.shape}")
    print(f"Columns:              {list(provider_level.columns)}")
    print(f"\nHead:\n{provider_level.head()}")
    print(f"\nMissing value counts:\n{provider_level.isna().sum()}")
    print(f"\nOutput: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
