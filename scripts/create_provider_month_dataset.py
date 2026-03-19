"""Build provider-month feature table from raw Medicaid billing data."""

import argparse
import json
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

# -----------------------------------------------------------------------------
# CONFIG: Panel
# -----------------------------------------------------------------------------
# If True: one row per (provider, month) for every month in [min, max] per provider.
# If False: only rows where there is at least one billing row (default; report as-is).
BUILD_BALANCED_PANEL = False

# Output paths (defaults; can override via CLI)
DEFAULT_OUTPUT_CSV = "provider_month.csv"
DEFAULT_DICTIONARY_CSV = "provider_month_data_dictionary.csv"

# Data dictionary source: editable JSON (one object per column with keys below).
# Edit this file to change definitions without touching code.
DEFAULT_DATA_DICTIONARY_JSON = "provider_month_data_dictionary.json"
DICTIONARY_ENTRY_KEYS = (
    "column_name",
    "feature_group",
    "definition",
    "formula_or_logic",
    "source_columns",
    "grain",
    "data_type",
    "missing_value_logic",
    "notes",
)


def _raw_npi_column() -> str:
    """Return the raw CSV column name for billing provider NPI (for DuckDB cohort filter)."""
    for raw, standard in COLUMN_MAP.items():
        if standard == "billing_provider_npi":
            return raw
    raise ValueError("COLUMN_MAP has no entry for billing_provider_npi")


def filter_raw_to_cohort(
    raw_csv_path: str,
    cohort_csv_path: str,
    cohorts: list[str] | None,
    output_path: str,
) -> None:
    """
    Use DuckDB to keep only rows in raw billing CSV whose provider is in the cohort CSV.
    If cohorts is non-empty, restrict to those cohort IDs or cohort_labels (e.g. '1' or 'NM_organization').
    Writes filtered rows to output_path (same schema as raw).
    """
    raw_path = Path(raw_csv_path)
    cohort_path = Path(cohort_csv_path)
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw billing CSV not found: {raw_csv_path}")
    if not cohort_path.is_file():
        raise FileNotFoundError(f"Cohort CSV not found: {cohort_csv_path}")

    raw_npi_col = _raw_npi_column()
    con = duckdb.connect()
    try:
        con.execute("PRAGMA progress_bar = true;")
    except Exception:
        pass

    # Quote column name if it contains special chars (DuckDB identifiers)
    npi_col_safe = f'"{raw_npi_col}"' if '"' not in raw_npi_col else raw_npi_col

    if cohorts:
        esc = lambda s: str(s).replace("'", "''")
        placeholders = ", ".join(f"'{esc(c)}'" for c in cohorts)
        cohort_filter = f"(CAST(cohort AS VARCHAR) IN ({placeholders}) OR cohort_label IN ({placeholders}))"
    else:
        cohort_filter = "1=1"

    # DuckDB CREATE VIEW and COPY TO don't support ? parameters; embed paths with escaped quotes.
    cohort_path_safe = str(cohort_path.resolve()).replace("'", "''")
    raw_path_safe = str(raw_path.resolve()).replace("'", "''")
    out_path_safe = str(Path(output_path).resolve()).replace("'", "''")

    # Cohort CSV uses billing_provider_npi (same as raw). No "npi" column.
    cohort_npi_safe = '"billing_provider_npi"'

    con.execute(
        f"CREATE TEMP VIEW cohort_filter_npis AS SELECT * FROM read_csv_auto('{cohort_path_safe}')"
    )
    con.execute(
        f"CREATE TEMP VIEW cohort_npis AS SELECT DISTINCT {cohort_npi_safe} FROM cohort_filter_npis WHERE {cohort_filter}"
    )
    con.execute(
        f"""
        COPY (
            SELECT r.*
            FROM read_csv_auto('{raw_path_safe}') r
            INNER JOIN cohort_npis c
            ON CAST(r.{npi_col_safe} AS VARCHAR) = CAST(c.{cohort_npi_safe} AS VARCHAR)
        ) TO '{out_path_safe}' (HEADER, DELIMITER ',')
        """
    )
    con.close()


def load_raw_data(input_path: str) -> pd.DataFrame:
    """Load raw billing CSV and apply column mapping."""
    path = Path(input_path)
    if not path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    df = pd.read_csv(path, low_memory=False)
    # Apply mapping for columns that exist
    rename = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    if not rename:
        raise ValueError(
            "None of the COLUMN_MAP keys found in CSV. Check column names. "
            f"CSV columns: {list(df.columns)[:20]}..."
        )
    df = df.rename(columns=rename)
    return df


def clean_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Type conversion and month parsing.
    Expects standard names: billing_provider_npi, month, hcpcs_code,
    paid_amount, claims_count, beneficiary_count.
    """
    required = {"billing_provider_npi", "month", "hcpcs_code", "paid_amount", "claims_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"After mapping, missing columns: {missing}")

    out = df.copy()

    # Coerce numeric
    for col in ["paid_amount", "claims_count"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if "beneficiary_count" in out.columns:
        out["beneficiary_count"] = pd.to_numeric(out["beneficiary_count"], errors="coerce")

    # Month: support YYYY-MM, YYYYMM, or already datetime
    month = out["month"]
    if pd.api.types.is_numeric_dtype(month):
        month = month.dropna().astype(int).astype(str)
        out["month"] = pd.to_datetime(month + "01", format="%Y%m%d", errors="coerce")
    elif month.dtype == object or month.dtype.name == "string":
        out["month"] = pd.to_datetime(month.astype(str), format="%Y-%m", errors="coerce")
    else:
        out["month"] = pd.to_datetime(month, errors="coerce")

    # Drop rows where we cannot determine month or provider
    out = out.dropna(subset=["billing_provider_npi", "month"])
    out["billing_provider_npi"] = out["billing_provider_npi"].astype(str).str.strip()
    out = out[out["billing_provider_npi"].str.len() > 0]
    return out


def build_provider_month_panel(df: pd.DataFrame, build_balanced: bool) -> pd.DataFrame:
    """
    Build the (billing_provider_npi, month) panel.
    If not build_balanced: one row per (provider, month) with at least one billing row.
    If build_balanced: one row per (provider, month) for every month in [min, max] per provider.
    """
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
    """
    One row per (billing_provider_npi, month) with:
    paid_t, claims_t, beneficiaries_proxy_t, hcpcs_count_t.
    """
    agg = {
        "paid_amount": "sum",
        "claims_count": "sum",
        "hcpcs_code": "nunique",
    }
    if "beneficiary_count" in df.columns:
        agg["beneficiary_count"] = "sum"
    core = df.groupby(["billing_provider_npi", "month"], dropna=False).agg(agg)
    core = core.rename(
        columns={
            "paid_amount": "paid_t",
            "claims_count": "claims_t",
            "beneficiary_count": "beneficiaries_proxy_t",
            "hcpcs_code": "hcpcs_count_t",
        }
    )
    if "beneficiaries_proxy_t" not in core.columns:
        core["beneficiaries_proxy_t"] = np.nan
    return core.reset_index()


def compute_code_level_totals(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (billing_provider_npi, month, hcpcs_code) with code-level paid and claims."""
    code_agg = (
        df.groupby(["billing_provider_npi", "month", "hcpcs_code"], dropna=False)
        .agg(paid_amount=("paid_amount", "sum"), claims_count=("claims_count", "sum"))
        .reset_index()
    )
    code_agg = code_agg.rename(
        columns={"paid_amount": "code_paid", "claims_count": "code_claims"}
    )
    return code_agg


def compute_ratio_features(provider_month_df: pd.DataFrame) -> pd.DataFrame:
    """Add paid_per_claim_t, claims_per_beneficiaries_proxy_t, paid_per_beneficiaries_proxy_t."""
    out = provider_month_df.copy()
    out["paid_per_claim_t"] = np.where(
        out["claims_t"] != 0, out["paid_t"] / out["claims_t"], np.nan
    )
    out["claims_per_beneficiaries_proxy_t"] = np.where(
        out["beneficiaries_proxy_t"] != 0,
        out["claims_t"] / out["beneficiaries_proxy_t"],
        np.nan,
    )
    out["paid_per_beneficiaries_proxy_t"] = np.where(
        out["beneficiaries_proxy_t"] != 0,
        out["paid_t"] / out["beneficiaries_proxy_t"],
        np.nan,
    )
    return out


def _merge_to_panel(provider_month_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join features_df onto provider_month_df on (billing_provider_npi, month)."""
    return provider_month_df.merge(features_df, on=["billing_provider_npi", "month"], how="left")


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy -sum(p*log(p)); p must be non-negative and sum to 1."""
    p = np.asarray(probs).ravel()
    p = p[p > 0]
    if len(p) <= 1:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def _hhi(probs: np.ndarray) -> float:
    """Herfindahl-Hirschman index sum(p^2)."""
    p = np.asarray(probs).ravel()
    return float(np.sum(p ** 2))


def _compute_col_stats(g, code_level: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Compute distribution stats for one code-level column; returns a MultiIndexed DataFrame."""
    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)

    stats = g[col].agg([
        ("mean", "mean"), ("median", "median"), ("max", "max"), ("min", "min"),
        ("std", "std"), ("iqr", lambda x: q3(x) - q1(x)),
        ("mad", lambda x: (x - x.median()).abs().median()),
        ("skew", "skew"), ("kurt", "kurt"),
    ]).rename(columns=lambda c: (
        f"{c}_{prefix}_single_code" if c in ("max", "min") else f"{c}_{prefix}_per_code"
    ))
    stats[f"{prefix}_code_range"] = (
        stats[f"max_{prefix}_single_code"] - stats[f"min_{prefix}_single_code"]
    )
    stats[f"{prefix}_code_energy"] = g[col].apply(lambda x: (x ** 2).sum())
    above_mean = (
        code_level.assign(
            _mean=code_level.groupby(["billing_provider_npi", "month"])[col].transform("mean")
        )
        .query(f"{col} > _mean")
        .groupby(["billing_provider_npi", "month"])
        .size()
        .reindex(stats.index)
        .fillna(0)
        .astype(int)
    )
    stats[f"num_codes_above_mean_{prefix}"] = above_mean
    return stats


def compute_code_mix_features(
    code_level: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add top_code_paid_share, top_code_claim_share, top_3 variants, hcpcs_entropy,
    hcpcs_hhi, top_hcpcs_code_t. Merge back to provider_month_df.
    """
    grp_keys = ["billing_provider_npi", "month"]
    g = code_level.groupby(grp_keys)

    paid_total = g["code_paid"].sum()
    claims_total = g["code_claims"].sum()

    top_code_paid_share = (g["code_paid"].max() / paid_total).where(paid_total > 0)
    top_code_claim_share = (g["code_claims"].max() / claims_total).where(claims_total > 0)

    def _top3_sum(x):
        n = min(3, len(x))
        return float(np.partition(x.values, -n)[-n:].sum())

    top_3_code_paid_share = (g["code_paid"].apply(_top3_sum) / paid_total).where(paid_total > 0)
    top_3_code_claim_share = (g["code_claims"].apply(_top3_sum) / claims_total).where(claims_total > 0)

    paid_tot_t = code_level.groupby(grp_keys)["code_paid"].transform("sum")
    p_paid = code_level["code_paid"].div(paid_tot_t.replace(0, np.nan)).fillna(0.0)
    cl = code_level.assign(_p_paid=p_paid)
    hcpcs_entropy = cl.groupby(grp_keys)["_p_paid"].apply(_entropy)
    hcpcs_hhi = cl.groupby(grp_keys)["_p_paid"].apply(_hhi)

    top_idx = g["code_paid"].idxmax()
    top_hcpcs = (
        code_level.loc[top_idx.values, "hcpcs_code"]
        .astype(str)
        .set_axis(top_idx.index)
        .where(paid_total > 0, "")
    )

    mix_df = pd.DataFrame({
        "top_code_paid_share": top_code_paid_share,
        "top_code_claim_share": top_code_claim_share,
        "top_3_code_paid_share": top_3_code_paid_share,
        "top_3_code_claim_share": top_3_code_claim_share,
        "hcpcs_entropy": hcpcs_entropy,
        "hcpcs_hhi": hcpcs_hhi,
        "top_hcpcs_code_t": top_hcpcs,
    }).reset_index()

    return _merge_to_panel(provider_month_df, mix_df)


def compute_code_distribution_features(
    code_level: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add within-month code-level stats: mean/median/std/min/max/range/iqr/mad/skew/kurt
    for paid and claims, plus num_codes_above_mean_*, paid_code_energy, claims_code_energy.
    """
    g = code_level.groupby(["billing_provider_npi", "month"])
    paid_stats = _compute_col_stats(g, code_level, "code_paid", "paid")
    claims_stats = _compute_col_stats(g, code_level, "code_claims", "claims")
    dist_df = paid_stats.join(claims_stats).reset_index()
    return _merge_to_panel(provider_month_df, dist_df)


def build_provider_month_df(
    raw_df: pd.DataFrame,
    panel: pd.DataFrame,
    core: pd.DataFrame,
    code_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble provider_month_df: merge panel with core, then add ratio,
    code-mix, and code-distribution features.
    """
    out = _merge_to_panel(panel, core)

    # Ratios
    out = compute_ratio_features(out)
    # Code-mix
    out = compute_code_mix_features(code_level, out)
    # Code distribution
    out = compute_code_distribution_features(code_level, out)

    # Sort for reproducibility
    out = out.sort_values(["billing_provider_npi", "month"]).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Data dictionary: loaded from external JSON (editable without changing code)
# -----------------------------------------------------------------------------
def load_data_dictionary_metadata(json_path: str) -> list[dict]:
    """
    Load data dictionary entries from a JSON file.
    Expected: array of objects with keys in DICTIONARY_ENTRY_KEYS.
    """
    path = Path(json_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Data dictionary JSON not found: {json_path}. "
            "Create it or pass a valid --dictionary-json path."
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"Data dictionary JSON must be a JSON array of objects; got {type(data).__name__}"
        )
    for i, entry in enumerate(data):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Data dictionary entry at index {i} must be an object; got {type(entry).__name__}"
            )
        missing_keys = set(DICTIONARY_ENTRY_KEYS) - set(entry)
        if missing_keys:
            raise ValueError(
                f"Data dictionary entry for column '{entry.get('column_name', '?')}' "
                f"missing keys: {missing_keys}"
            )
    return data


def build_data_dictionary(
    provider_month_df: pd.DataFrame, dictionary_json_path: str
) -> pd.DataFrame:
    """
    Build provider_month_data_dictionary from external JSON; validate column set match.
    Rows are reordered to match provider_month_df column order when writing.
    """
    meta = load_data_dictionary_metadata(dictionary_json_path)
    dict_df = pd.DataFrame(meta)
    dict_cols = set(dict_df["column_name"])
    actual_cols = set(provider_month_df.columns)
    missing_in_dict = actual_cols - dict_cols
    extra_in_dict = dict_cols - actual_cols
    if missing_in_dict:
        raise ValueError(
            f"Data dictionary missing columns in provider_month_df: {missing_in_dict}"
        )
    if extra_in_dict:
        raise ValueError(
            f"Data dictionary has columns not in provider_month_df: {extra_in_dict}"
        )
    # Reorder to match table column order
    order = [c for c in provider_month_df.columns if c in dict_df["column_name"].values]
    dict_df = dict_df.set_index("column_name").loc[order].reset_index()
    return dict_df


def save_outputs(
    provider_month_df: pd.DataFrame,
    provider_month_data_dictionary: pd.DataFrame,
    output_csv: str,
    dictionary_csv: str,
) -> None:
    """Write provider_month_df and data dictionary to CSV."""
    out_path = Path(output_csv)
    dict_path = Path(dictionary_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    provider_month_df.to_csv(out_path, index=False)
    provider_month_data_dictionary.to_csv(dict_path, index=False)


def run(
    input_csv: str,
    output_csv: str = DEFAULT_OUTPUT_CSV,
    dictionary_csv: str = DEFAULT_DICTIONARY_CSV,
    dictionary_json: str = DEFAULT_DATA_DICTIONARY_JSON,
    cohort_csv: str | None = None,
    cohorts: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, build panel, compute features, build dictionary from JSON, save. Returns (provider_month_df, data_dictionary)."""
    data_path = input_csv
    temp_path = None
    if cohort_csv:
        temp_path = tempfile.NamedTemporaryFile(
            prefix="create_datasets_cohort_", suffix=".csv", delete=False
        ).name
        filter_raw_to_cohort(input_csv, cohort_csv, cohorts or [], temp_path)
        data_path = temp_path
    try:
        df = load_raw_data(data_path)
    finally:
        if temp_path and Path(temp_path).is_file():
            Path(temp_path).unlink(missing_ok=True)
    df = clean_raw_data(df)
    panel = build_provider_month_panel(df, BUILD_BALANCED_PANEL)
    core = compute_core_monthly_aggregates(df)
    code_level = compute_code_level_totals(df)
    provider_month_df = build_provider_month_df(df, panel, core, code_level)
    provider_month_data_dictionary = build_data_dictionary(
        provider_month_df, dictionary_json
    )
    save_outputs(provider_month_df, provider_month_data_dictionary, output_csv, dictionary_csv)
    return provider_month_df, provider_month_data_dictionary


def main():
    parser = argparse.ArgumentParser(
        description="Build provider-month feature table from raw Medicaid billing CSV."
    )
    parser.add_argument("input_csv", help="Path to raw billing CSV.")
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output path for provider-month CSV (default: {DEFAULT_OUTPUT_CSV}).",
    )
    parser.add_argument(
        "--dictionary",
        default=DEFAULT_DICTIONARY_CSV,
        help=f"Output path for data dictionary CSV (default: {DEFAULT_DICTIONARY_CSV}).",
    )
    parser.add_argument(
        "--dictionary-json",
        default=DEFAULT_DATA_DICTIONARY_JSON,
        help=f"Path to editable data dictionary JSON (default: {DEFAULT_DATA_DICTIONARY_JSON}).",
    )
    parser.add_argument(
        "--cohort-csv",
        metavar="PATH",
        help="Path to cohort CSV (from build_provider_cohorts). If set, only providers in this file are used.",
    )
    parser.add_argument(
        "--cohorts",
        nargs="*",
        metavar="COHORT",
        help="Restrict to these cohorts. Each value is either a cohort ID (number, e.g. 1) or a cohort_label (e.g. NM_organization). Examples: --cohorts 1  or  --cohorts NM_organization  or  --cohorts 1 2 CA_individual. Only used when --cohort-csv is set; if omitted, all providers in the cohort file are kept.",
    )
    args = parser.parse_args()
    provider_month_df, provider_month_data_dictionary = run(
        args.input_csv,
        args.output,
        args.dictionary,
        args.dictionary_json,
        cohort_csv=args.cohort_csv,
        cohorts=args.cohorts if args.cohorts else None,
    )
    print("provider_month_df shape:", provider_month_df.shape)
    print("provider_month_df columns:", list(provider_month_df.columns))
    print("\nprovider_month_df head():\n", provider_month_df.head())
    print("\nMissing value counts by column:")
    print(provider_month_df.isna().sum())
    print("\nprovider_month_df CSV:", Path(args.output).resolve())
    print("provider_month_data_dictionary CSV:", Path(args.dictionary).resolve())
    print("provider_month_data_dictionary JSON (source):", Path(args.dictionary_json).resolve())
    print("Data dictionary row count:", len(provider_month_data_dictionary))


if __name__ == "__main__":
    main()
