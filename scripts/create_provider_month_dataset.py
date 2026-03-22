"""Build provider-month feature table from raw Medicaid billing data.

Exactly one cohort must be specified via --cohort-csv and --cohorts.
Raises an error if zero or more than one cohort is passed.
The cohort_label column is NOT written to the output.
"""

import argparse
import sys
import tempfile
from pathlib import Path

import duckdb
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


def _raw_npi_column() -> str:
    for raw, standard in COLUMN_MAP.items():
        if standard == "billing_provider_npi":
            return raw
    raise ValueError("COLUMN_MAP has no entry for billing_provider_npi")


def filter_raw_to_cohort(
    raw_csv_path: str,
    cohort_csv_path: str,
    cohort: str,
    output_path: str,
) -> None:
    """Use DuckDB to keep only rows whose provider is in the specified single cohort."""
    raw_path = Path(raw_csv_path)
    cohort_path = Path(cohort_csv_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw billing CSV not found: {raw_csv_path}")
    if not cohort_path.is_file():
        raise FileNotFoundError(f"Cohort CSV not found: {cohort_csv_path}")

    raw_npi_col = _raw_npi_column()
    npi_col_safe = f'"{raw_npi_col}"'
    esc_cohort = str(cohort).replace("'", "''")

    cohort_path_safe = str(cohort_path.resolve()).replace("'", "''")
    raw_path_safe = str(raw_path.resolve()).replace("'", "''")
    out_path_safe = str(Path(output_path).resolve()).replace("'", "''")

    con = duckdb.connect()
    try:
        con.execute("PRAGMA progress_bar = true;")
    except Exception:
        pass

    con.execute(
        f"CREATE TEMP VIEW cohort_npis AS "
        f"SELECT DISTINCT \"npi\" FROM read_csv_auto('{cohort_path_safe}') "
        f"WHERE CAST(cohort_label AS VARCHAR) = '{esc_cohort}' "
        f"   OR CAST(cohort AS VARCHAR) = '{esc_cohort}'"
    )
    con.execute(
        f"""
        COPY (
            SELECT r.*
            FROM read_csv_auto('{raw_path_safe}') r
            INNER JOIN cohort_npis c
            ON CAST(r.{npi_col_safe} AS VARCHAR) = CAST(c."npi" AS VARCHAR)
        ) TO '{out_path_safe}' (HEADER, DELIMITER ',')
        """
    )
    con.close()


def load_raw_data(input_path: str) -> pd.DataFrame:
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
    return df.rename(columns=rename)


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
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
    core = df.groupby(["billing_provider_npi", "month"], dropna=False).agg(
        paid_amount=("paid_amount", "sum"),
        claims_count=("claims_count", "sum"),
        hcpcs_code=("hcpcs_code", "nunique"),
    )
    return core.rename(columns={
        "paid_amount": "paid_t",
        "claims_count": "claims_t",
        "hcpcs_code": "hcpcs_count_t",
    }).reset_index()


def compute_code_level_totals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (billing_provider_npi, month, hcpcs_code).

    NOTE: code_bene is a PROXY — sums TOTAL_UNIQUE_BENEFICIARIES at the
    provider × HCPCS × month grain, which can double-count beneficiaries
    appearing under multiple HCPCS codes.
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
    return code_agg.rename(columns=rename_map)


def compute_ratio_features(provider_month_df: pd.DataFrame) -> pd.DataFrame:
    out = provider_month_df.copy()
    out["paid_per_claim_t"] = np.where(
        out["claims_t"] != 0, out["paid_t"] / out["claims_t"], np.nan
    )
    return out


def _merge_to_panel(provider_month_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    return provider_month_df.merge(features_df, on=["billing_provider_npi", "month"], how="left")


def _entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs).ravel()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _hhi(probs: np.ndarray) -> float:
    p = np.asarray(probs).ravel()
    return float(np.sum(p ** 2))


def compute_beneficiary_proxy_features(
    df: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    BENE_COL = "beneficiary_count"
    if BENE_COL not in df.columns:
        print(f"Warning: '{BENE_COL}' not found — beneficiary_proxy features will be NaN.", file=sys.stderr)
        out = provider_month_df.copy()
        out["beneficiaries_proxy_t"] = np.nan
        out["claims_per_beneficiary_proxy_t"] = np.nan
        out["paid_per_beneficiary_proxy_t"] = np.nan
        return out
    bene_agg = (
        df.groupby(["billing_provider_npi", "month"], dropna=False)[BENE_COL]
        .sum().reset_index()
        .rename(columns={BENE_COL: "beneficiaries_proxy_t"})
    )
    out = _merge_to_panel(provider_month_df, bene_agg)
    valid = out["beneficiaries_proxy_t"] > 0
    out["claims_per_beneficiary_proxy_t"] = np.where(valid, out["claims_t"] / out["beneficiaries_proxy_t"], np.nan)
    out["paid_per_beneficiary_proxy_t"] = np.where(valid, out["paid_t"] / out["beneficiaries_proxy_t"], np.nan)
    return out


def compute_code_mix_features(code_level: pd.DataFrame, provider_month_df: pd.DataFrame) -> pd.DataFrame:
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
    mix_df = pd.DataFrame({
        "top_code_paid_share": top_code_paid_share,
        "top_3_code_paid_share": top_3_code_paid_share,
        "hcpcs_entropy": cl.groupby(grp_keys)["_p_paid"].apply(_entropy),
        "hcpcs_hhi": cl.groupby(grp_keys)["_p_paid"].apply(_hhi),
    }).reset_index()
    return _merge_to_panel(provider_month_df, mix_df)


def compute_claim_code_mix_features(code_level: pd.DataFrame, provider_month_df: pd.DataFrame) -> pd.DataFrame:
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
    mix_df = pd.DataFrame({
        "top_code_claim_share_t": top_code_claim_share,
        "top_3_code_claim_share_t": top_3_code_claim_share,
        "hcpcs_claim_entropy_t": cl.groupby(grp_keys)["_p_claim"].apply(_entropy),
        "hcpcs_claim_hhi_t": cl.groupby(grp_keys)["_p_claim"].apply(_hhi),
    }).reset_index()
    return _merge_to_panel(provider_month_df, mix_df)


def compute_beneficiary_code_mix_features(code_level: pd.DataFrame, provider_month_df: pd.DataFrame) -> pd.DataFrame:
    BENE_COL = "code_bene"
    grp_keys = ["billing_provider_npi", "month"]
    _NEW_COLS = (
        "top_code_beneficiary_share_t", "top_3_code_beneficiary_share_t",
        "hcpcs_beneficiary_entropy_t", "hcpcs_beneficiary_hhi_t",
    )
    if BENE_COL not in code_level.columns:
        print(f"Warning: '{BENE_COL}' not found — beneficiary code-mix features will be NaN.", file=sys.stderr)
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
    mix_df = pd.DataFrame({
        "top_code_beneficiary_share_t": top_code_bene_share,
        "top_3_code_beneficiary_share_t": top_3_code_bene_share,
        "hcpcs_beneficiary_entropy_t": cl.groupby(grp_keys)["_p_bene"].apply(_entropy),
        "hcpcs_beneficiary_hhi_t": cl.groupby(grp_keys)["_p_bene"].apply(_hhi),
    }).reset_index()
    return _merge_to_panel(provider_month_df, mix_df)


def compute_mismatch_features(provider_month_df: pd.DataFrame) -> pd.DataFrame:
    out = provider_month_df.copy()
    out["top_code_paid_minus_claim_share_t"] = out["top_code_paid_share"] - out["top_code_claim_share_t"]
    out["top_code_paid_minus_beneficiary_share_t"] = out["top_code_paid_share"] - out["top_code_beneficiary_share_t"]
    out["hcpcs_paid_hhi_minus_claim_hhi_t"] = out["hcpcs_hhi"] - out["hcpcs_claim_hhi_t"]
    return out


def build_provider_month_df(
    raw_df: pd.DataFrame, panel: pd.DataFrame, core: pd.DataFrame, code_level: pd.DataFrame,
) -> pd.DataFrame:
    out = _merge_to_panel(panel, core)
    out = compute_ratio_features(out)
    out = compute_code_mix_features(code_level, out)
    out = compute_claim_code_mix_features(code_level, out)
    out = compute_beneficiary_code_mix_features(code_level, out)
    out = compute_beneficiary_proxy_features(raw_df, out)
    out = compute_mismatch_features(out)
    return out.sort_values(["billing_provider_npi", "month"]).reset_index(drop=True)


def run(
    input_csv: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    cohort_csv: str | None = None,
    cohort: str | None = None,
    labels_csv: str | None = None,
) -> pd.DataFrame:
    """Load, filter to single cohort, build features, save. Returns provider_month_df."""
    if cohort_csv is None or cohort is None:
        raise ValueError(
            "Both --cohort-csv and --cohort are required. "
            "Run build_provider_cohorts.py first to generate the cohort CSV, "
            "then pass a single cohort label (e.g. NY_individual)."
        )

    # Filter raw billing CSV to the single cohort via DuckDB
    temp_path = tempfile.NamedTemporaryFile(prefix="provider_month_cohort_", suffix=".csv", delete=False).name
    print(f"Filtering raw CSV to cohort '{cohort}'...")
    filter_raw_to_cohort(input_csv, cohort_csv, cohort, temp_path)

    try:
        df = load_raw_data(temp_path)
    finally:
        Path(temp_path).unlink(missing_ok=True)

    df = clean_raw_data(df)
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
        description="Build provider-month feature table for a single cohort."
    )
    parser.add_argument("input_csv", help="Path to raw billing CSV.")
    parser.add_argument(
        "--cohort-csv", required=True,
        help="Path to provider_cohorts.csv (from build_provider_cohorts.py).",
    )
    parser.add_argument(
        "--cohort", required=True,
        help="Single cohort to process, e.g. NY_individual or CA_organization.",
    )
    parser.add_argument(
        "--output", default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--labels-csv", default=None,
        help="Path to provider_labels.csv. If provided, joins label and excldate.",
    )
    args = parser.parse_args()
    run(args.input_csv, args.output, args.cohort_csv, args.cohort, args.labels_csv)


if __name__ == "__main__":
    main()
