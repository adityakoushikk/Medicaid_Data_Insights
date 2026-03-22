"""Build provider-month feature table from raw Medicaid billing data.

Input must be pre-filtered to a SINGLE cohort. The script raises an error
if multiple distinct cohort_label values are detected in the input.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


COLUMN_MAP = {
    "BILLING_PROVIDER_NPI_NUM": "billing_provider_npi",
    "CLAIM_FROM_MONTH": "month",
    "HCPCS_CODE": "hcpcs_code",
    "TOTAL_PAID": "paid_amount",
    "TOTAL_CLAIMS": "claims_count",
    "TOTAL_UNIQUE_BENEFICIARIES": "beneficiary_count",
}

# If True: one row per (provider, month) for every month in [min, max] per provider.
# If False: only rows where there is at least one billing row.
BUILD_BALANCED_PANEL = False

DEFAULT_OUTPUT_CSV = "provider_month.csv"


def load_raw_data(input_path: str) -> pd.DataFrame:
    """Load raw billing CSV and apply column mapping."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(path, low_memory=False)
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if not rename:
        raise ValueError(
            "None of the COLUMN_MAP keys found in CSV. Check column names. "
            f"CSV columns: {list(df.columns)[:20]}..."
        )
    df = df.rename(columns=rename)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Type conversion and month parsing."""
    required = {"billing_provider_npi", "month", "hcpcs_code", "paid_amount", "claims_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"After mapping, missing columns: {missing}")

    out = df.copy()

    for col in ["paid_amount", "claims_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "beneficiary_count" in out.columns:
        out["beneficiary_count"] = pd.to_numeric(out["beneficiary_count"], errors="coerce")

    month = out["month"]
    if pd.api.types.is_numeric_dtype(month):
        month = month.dropna().astype(int).astype(str)
        out["month"] = pd.to_datetime(month + "01", format="%Y%m%d", errors="coerce")
    elif month.dtype == object or month.dtype.name == "string":
        out["month"] = pd.to_datetime(month.astype(str), format="%Y-%m", errors="coerce")
    else:
        out["month"] = pd.to_datetime(month, errors="coerce")

    out = out.dropna(subset=["billing_provider_npi", "month"])
    out["billing_provider_npi"] = out["billing_provider_npi"].astype(str).str.strip()
    out = out[out["billing_provider_npi"].str.len() > 0]
    return out


def build_provider_month_panel(df: pd.DataFrame, build_balanced: bool) -> pd.DataFrame:
    active_pairs = df[["billing_provider_npi", "month"]].drop_duplicates()

    if not build_balanced:
        return active_pairs.copy()

    month_range = df["month"].min(), df["month"].max()
    if pd.isna(month_range[0]) or pd.isna(month_range[1]):
        return active_pairs.copy()

    months = pd.date_range(start=month_range[0], end=month_range[1], freq="MS")
    providers = df["billing_provider_npi"].unique()
    full_index = pd.MultiIndex.from_product(
        [providers, months], names=["billing_provider_npi", "month"]
    )
    return full_index.to_frame(index=False)


def compute_core_monthly_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (billing_provider_npi, month) with paid_t, claims_t, hcpcs_count_t."""
    core = df.groupby(["billing_provider_npi", "month"], dropna=False).agg(
        paid_amount=("paid_amount", "sum"),
        claims_count=("claims_count", "sum"),
        hcpcs_code=("hcpcs_code", "nunique"),
    )
    core = core.rename(columns={
        "paid_amount": "paid_t",
        "claims_count": "claims_t",
        "hcpcs_code": "hcpcs_count_t",
    })
    return core.reset_index()


def compute_code_level_totals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (billing_provider_npi, month, hcpcs_code).

    NOTE: code_bene is a PROXY — sums TOTAL_UNIQUE_BENEFICIARIES at the
    provider × HCPCS × month grain, which can double-count beneficiaries who
    appear under multiple HCPCS codes.
    """
    agg_dict: dict = {
        "paid_amount": ("paid_amount", "sum"),
        "claims_count": ("claims_count", "sum"),
    }
    if "beneficiary_count" in df.columns:
        agg_dict["beneficiary_count"] = ("beneficiary_count", "sum")

    code_agg = (
        df.groupby(["billing_provider_npi", "month", "hcpcs_code"], dropna=False)
        .agg(**agg_dict)
        .reset_index()
    )
    rename_map = {"paid_amount": "code_paid", "claims_count": "code_claims"}
    if "beneficiary_count" in code_agg.columns:
        rename_map["beneficiary_count"] = "code_bene"
    code_agg = code_agg.rename(columns=rename_map)
    return code_agg


def compute_ratio_features(provider_month_df: pd.DataFrame) -> pd.DataFrame:
    """Add paid_per_claim_t."""
    out = provider_month_df.copy()
    out["paid_per_claim_t"] = np.where(
        out["claims_t"] != 0, out["paid_t"] / out["claims_t"], np.nan
    )
    return out


def _merge_to_panel(provider_month_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    return provider_month_df.merge(features_df, on=["billing_provider_npi", "month"], how="left")


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy -sum(p*log(p))."""
    p = np.asarray(probs).ravel()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _hhi(probs: np.ndarray) -> float:
    """Herfindahl-Hirschman index sum(p^2)."""
    p = np.asarray(probs).ravel()
    return float(np.sum(p ** 2))


def compute_beneficiary_proxy_features(
    df: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """Add beneficiaries_proxy_t, claims_per_beneficiary_proxy_t, paid_per_beneficiary_proxy_t.

    PROXY features — sums beneficiary_count across HCPCS codes, which can double-count
    beneficiaries appearing under multiple codes.
    """
    BENE_COL = "beneficiary_count"

    if BENE_COL not in df.columns:
        print(
            f"Warning: source column '{BENE_COL}' not found in raw data — "
            "beneficiary_proxy features will be NaN.",
            file=sys.stderr,
        )
        out = provider_month_df.copy()
        out["beneficiaries_proxy_t"] = np.nan
        out["claims_per_beneficiary_proxy_t"] = np.nan
        out["paid_per_beneficiary_proxy_t"] = np.nan
        return out

    bene_agg = (
        df.groupby(["billing_provider_npi", "month"], dropna=False)[BENE_COL]
        .sum()
        .reset_index()
        .rename(columns={BENE_COL: "beneficiaries_proxy_t"})
    )

    out = _merge_to_panel(provider_month_df, bene_agg)
    valid = out["beneficiaries_proxy_t"] > 0
    out["claims_per_beneficiary_proxy_t"] = np.where(
        valid, out["claims_t"] / out["beneficiaries_proxy_t"], np.nan
    )
    out["paid_per_beneficiary_proxy_t"] = np.where(
        valid, out["paid_t"] / out["beneficiaries_proxy_t"], np.nan
    )
    return out


def compute_code_mix_features(
    code_level: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """Add paid-based code-mix features."""
    grp_keys = ["billing_provider_npi", "month"]
    g = code_level.groupby(grp_keys)

    paid_total = g["code_paid"].sum()
    top_code_paid_share = (g["code_paid"].max() / paid_total).where(paid_total > 0)

    def _top3_sum(x):
        n = min(3, len(x))
        return float(np.partition(x.values, -n)[-n:].sum())

    top_3_code_paid_share = (g["code_paid"].apply(_top3_sum) / paid_total).where(paid_total > 0)

    paid_tot_t = code_level.groupby(grp_keys)["code_paid"].transform("sum")
    p_paid = code_level["code_paid"].div(paid_tot_t.replace(0, np.nan)).fillna(0.0)
    cl = code_level.assign(_p_paid=p_paid)
    hcpcs_entropy = cl.groupby(grp_keys)["_p_paid"].apply(_entropy)
    hcpcs_hhi = cl.groupby(grp_keys)["_p_paid"].apply(_hhi)

    mix_df = pd.DataFrame({
        "top_code_paid_share": top_code_paid_share,
        "top_3_code_paid_share": top_3_code_paid_share,
        "hcpcs_entropy": hcpcs_entropy,
        "hcpcs_hhi": hcpcs_hhi,
    }).reset_index()

    return _merge_to_panel(provider_month_df, mix_df)


def compute_claim_code_mix_features(
    code_level: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """Add claim-based code-mix features."""
    grp_keys = ["billing_provider_npi", "month"]
    g = code_level.groupby(grp_keys)

    claim_total = g["code_claims"].sum()
    top_code_claim_share = (g["code_claims"].max() / claim_total).where(claim_total > 0)

    def _top3_sum(x):
        n = min(3, len(x))
        return float(np.partition(x.values, -n)[-n:].sum())

    top_3_code_claim_share = (g["code_claims"].apply(_top3_sum) / claim_total).where(claim_total > 0)

    claim_tot_t = code_level.groupby(grp_keys)["code_claims"].transform("sum")
    p_claim = code_level["code_claims"].div(claim_tot_t.replace(0, np.nan)).fillna(0.0)
    cl = code_level.assign(_p_claim=p_claim)
    hcpcs_claim_entropy = cl.groupby(grp_keys)["_p_claim"].apply(_entropy)
    hcpcs_claim_hhi = cl.groupby(grp_keys)["_p_claim"].apply(_hhi)

    mix_df = pd.DataFrame({
        "top_code_claim_share_t": top_code_claim_share,
        "top_3_code_claim_share_t": top_3_code_claim_share,
        "hcpcs_claim_entropy_t": hcpcs_claim_entropy,
        "hcpcs_claim_hhi_t": hcpcs_claim_hhi,
    }).reset_index()

    return _merge_to_panel(provider_month_df, mix_df)


def compute_beneficiary_code_mix_features(
    code_level: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """Add beneficiary-proxy code-mix features.

    PROXY FEATURES — shares computed from code_bene, which can double-count
    beneficiaries appearing under multiple HCPCS codes.
    """
    BENE_COL = "code_bene"
    grp_keys = ["billing_provider_npi", "month"]
    _NEW_COLS = (
        "top_code_beneficiary_share_t",
        "top_3_code_beneficiary_share_t",
        "hcpcs_beneficiary_entropy_t",
        "hcpcs_beneficiary_hhi_t",
    )

    if BENE_COL not in code_level.columns:
        print(
            f"Warning: source column '{BENE_COL}' not found in code-level data — "
            "beneficiary code-mix features will be NaN.",
            file=sys.stderr,
        )
        out = provider_month_df.copy()
        for col in _NEW_COLS:
            out[col] = np.nan
        return out

    g = code_level.groupby(grp_keys)
    bene_total = g[BENE_COL].sum()
    top_code_bene_share = (g[BENE_COL].max() / bene_total).where(bene_total > 0)

    def _top3_sum(x):
        n = min(3, len(x))
        return float(np.partition(x.values, -n)[-n:].sum())

    top_3_code_bene_share = (g[BENE_COL].apply(_top3_sum) / bene_total).where(bene_total > 0)

    bene_tot_t = code_level.groupby(grp_keys)[BENE_COL].transform("sum")
    p_bene = code_level[BENE_COL].div(bene_tot_t.replace(0, np.nan)).fillna(0.0)
    cl = code_level.assign(_p_bene=p_bene)
    hcpcs_bene_entropy = cl.groupby(grp_keys)["_p_bene"].apply(_entropy)
    hcpcs_bene_hhi = cl.groupby(grp_keys)["_p_bene"].apply(_hhi)

    mix_df = pd.DataFrame({
        "top_code_beneficiary_share_t": top_code_bene_share,
        "top_3_code_beneficiary_share_t": top_3_code_bene_share,
        "hcpcs_beneficiary_entropy_t": hcpcs_bene_entropy,
        "hcpcs_beneficiary_hhi_t": hcpcs_bene_hhi,
    }).reset_index()

    return _merge_to_panel(provider_month_df, mix_df)


def compute_mismatch_features(provider_month_df: pd.DataFrame) -> pd.DataFrame:
    """Add paid-vs-claim and paid-vs-beneficiary mismatch features."""
    out = provider_month_df.copy()
    out["top_code_paid_minus_claim_share_t"] = (
        out["top_code_paid_share"] - out["top_code_claim_share_t"]
    )
    out["top_code_paid_minus_beneficiary_share_t"] = (
        out["top_code_paid_share"] - out["top_code_beneficiary_share_t"]
    )
    out["hcpcs_paid_hhi_minus_claim_hhi_t"] = out["hcpcs_hhi"] - out["hcpcs_claim_hhi_t"]
    return out


def build_provider_month_df(
    raw_df: pd.DataFrame,
    panel: pd.DataFrame,
    core: pd.DataFrame,
    code_level: pd.DataFrame,
) -> pd.DataFrame:
    out = _merge_to_panel(panel, core)
    out = compute_ratio_features(out)
    out = compute_code_mix_features(code_level, out)
    out = compute_claim_code_mix_features(code_level, out)
    out = compute_beneficiary_code_mix_features(code_level, out)
    out = compute_beneficiary_proxy_features(raw_df, out)
    out = compute_mismatch_features(out)
    out = out.sort_values(["billing_provider_npi", "month"]).reset_index(drop=True)
    return out


def run(
    input_csv: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    labels_csv: str | None = None,
) -> pd.DataFrame:
    """Load, clean, build panel, compute features, save. Returns provider_month_df."""
    df = load_raw_data(input_csv)
    df = clean_raw_data(df)

    # Enforce single cohort
    if "cohort_label" in df.columns:
        unique_cohorts = df["cohort_label"].dropna().unique()
        if len(unique_cohorts) > 1:
            raise ValueError(
                f"Input contains {len(unique_cohorts)} distinct cohort_labels: "
                f"{unique_cohorts.tolist()}. Filter to a single cohort before running."
            )

    panel = build_provider_month_panel(df, BUILD_BALANCED_PANEL)
    core = compute_core_monthly_aggregates(df)
    code_level = compute_code_level_totals(df)
    provider_month_df = build_provider_month_df(df, panel, core, code_level)

    if labels_csv is not None:
        labels = pd.read_csv(labels_csv, dtype={"npi": str})
        provider_month_df["billing_provider_npi"] = provider_month_df["billing_provider_npi"].astype(str)
        provider_month_df = provider_month_df.merge(
            labels.rename(columns={"npi": "billing_provider_npi"}),
            on="billing_provider_npi", how="left"
        )
        n_positive = int((provider_month_df["label"] == 1).sum())
        print(f"Labels joined: {n_positive:,} provider-month rows with label=1.")

        has_excldate = provider_month_df["excldate"].notna()
        excl_dt = pd.to_datetime(provider_month_df.loc[has_excldate, "excldate"], format="%Y%m%d", errors="coerce")
        month_dt = pd.to_datetime(provider_month_df.loc[has_excldate, "month"])
        drop_mask = has_excldate.copy()
        drop_mask[has_excldate] = month_dt > excl_dt
        before = len(provider_month_df)
        provider_month_df = provider_month_df[~drop_mask].reset_index(drop=True)
        print(f"Dropped {before - len(provider_month_df):,} provider-month rows after exclusion date.")
    else:
        provider_month_df["label"] = float("nan")
        provider_month_df["excldate"] = float("nan")

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    provider_month_df.to_csv(out_path, index=False)
    print(f"Output: {out_path.resolve()}  shape: {provider_month_df.shape}")
    return provider_month_df


def main():
    parser = argparse.ArgumentParser(
        description="Build provider-month feature table from raw Medicaid billing CSV. "
                    "Input must be pre-filtered to a single cohort."
    )
    parser.add_argument("input_csv", help="Path to raw billing CSV (single cohort).")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Path to provider_labels.csv. If provided, joins label and excldate onto every provider-month row.",
    )
    args = parser.parse_args()
    provider_month_df = run(args.input_csv, args.output, labels_csv=args.labels_csv)
    print("Shape:", provider_month_df.shape)
    print("Columns:", list(provider_month_df.columns))
    print("\nHead:\n", provider_month_df.head())
    print("\nMissing value counts:\n", provider_month_df.isna().sum())


if __name__ == "__main__":
    main()
