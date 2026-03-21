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

    cohort_npi_safe = '"npi"'

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
    """One row per (billing_provider_npi, month, hcpcs_code) with code-level paid, claims,
    and beneficiary count (code_bene).

    NOTE: code_bene is a PROXY — it sums TOTAL_UNIQUE_BENEFICIARIES at the
    provider × HCPCS × month grain, which can double-count beneficiaries who
    appear under multiple HCPCS codes.  If beneficiary_count is absent from
    df, code_bene will not be present in the output.
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


def compute_beneficiary_proxy_features(
    df: pd.DataFrame, provider_month_df: pd.DataFrame
) -> pd.DataFrame:
    """Add beneficiary-proxy columns to the provider-month table.

    IMPORTANT — these are PROXY features, not true unique-beneficiary counts.
    The raw data contains beneficiary counts at the provider × HCPCS × month grain.
    Summing across HCPCS codes to reach provider-month level can double-count the
    same beneficiary who appears under multiple codes.  All three output columns
    carry the suffix ``_proxy`` to make this explicit.

    Columns added
    -------------
    beneficiaries_proxy_t
        Sum of beneficiary_count across all (provider, HCPCS, month) rows.
        Over-counts shared beneficiaries; use only as a relative proxy.
    claims_per_beneficiary_proxy_t
        claims_t / beneficiaries_proxy_t.  NaN when beneficiaries_proxy_t <= 0.
    paid_per_beneficiary_proxy_t
        paid_t / beneficiaries_proxy_t.  NaN when beneficiaries_proxy_t <= 0.

    If beneficiary_count is absent from the raw data, all three columns are NaN
    and a warning is printed.
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

    # Ratio features: NaN-safe — zero or missing proxy denominator → NaN
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
    """Add paid-based code-mix features: top_code_paid_share, top_3_code_paid_share,
    hcpcs_entropy, hcpcs_hhi."""
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
    """Add claim-based code-mix features: top_code_claim_share_t, top_3_code_claim_share_t,
    hcpcs_claim_entropy_t, hcpcs_claim_hhi_t.

    Shares are computed using TOTAL_CLAIMS (code_claims) within each provider-month.
    """
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
    """Add beneficiary-proxy code-mix features: top_code_beneficiary_share_t,
    top_3_code_beneficiary_share_t, hcpcs_beneficiary_entropy_t, hcpcs_beneficiary_hhi_t.

    IMPORTANT — PROXY FEATURES: shares are computed from code_bene, which is the
    sum of TOTAL_UNIQUE_BENEFICIARIES at the provider × HCPCS × month grain.
    Summing across HCPCS codes can double-count a beneficiary who appears under
    multiple codes.  All output columns carry the ``_beneficiary_`` infix to make
    this explicit.  If beneficiary_count is absent from the raw data, all four
    columns are NaN and a warning is printed.
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
    """Add paid-vs-claim and paid-vs-beneficiary mismatch features.

    Requires top_code_paid_share, top_code_claim_share_t, top_code_beneficiary_share_t,
    hcpcs_hhi, and hcpcs_claim_hhi_t to already be present.  Results inherit NaN
    from any missing upstream column.
    """
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
    """
    Assemble provider_month_df: merge panel with core, then add ratio,
    code-mix, and beneficiary-proxy features.
    """
    out = _merge_to_panel(panel, core)
    out = compute_ratio_features(out)
    out = compute_code_mix_features(code_level, out)            # paid-based (existing)
    out = compute_claim_code_mix_features(code_level, out)      # claim-based (new)
    out = compute_beneficiary_code_mix_features(code_level, out)  # bene-proxy code-mix (new)
    out = compute_beneficiary_proxy_features(raw_df, out)       # bene-proxy totals (existing)
    out = compute_mismatch_features(out)                        # paid-vs-claim/bene diffs (new)

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
    labels_csv: str | None = None,
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

    # Join cohort_label from cohort CSV so each row is tagged with its cohort
    if cohort_csv is not None:
        cohort_df = pd.read_csv(cohort_csv, dtype={"npi": str})[["npi", "cohort_label"]].drop_duplicates("npi")
        provider_month_df["billing_provider_npi"] = provider_month_df["billing_provider_npi"].astype(str)
        provider_month_df = provider_month_df.merge(
            cohort_df.rename(columns={"npi": "billing_provider_npi"}),
            on="billing_provider_npi", how="left"
        )
        # Move cohort_label to second column
        cols = ["billing_provider_npi", "cohort_label"] + [c for c in provider_month_df.columns if c not in ("billing_provider_npi", "cohort_label")]
        provider_month_df = provider_month_df[cols]
        print(f"Cohort labels joined: {provider_month_df['cohort_label'].nunique()} distinct cohorts.")
    else:
        provider_month_df.insert(1, "cohort_label", None)

    # Join label and excldate from LEIE labels file (one row per NPI)
    if labels_csv is not None:
        labels = pd.read_csv(labels_csv, dtype={"npi": str})
        provider_month_df["billing_provider_npi"] = provider_month_df["billing_provider_npi"].astype(str)
        provider_month_df = provider_month_df.merge(
            labels.rename(columns={"npi": "billing_provider_npi"}),
            on="billing_provider_npi", how="left"
        )
        n_positive = int((provider_month_df["label"] == 1).sum())
        print(f"Labels joined: {n_positive:,} provider-month rows with label=1.")

        # Drop months after exclusion date for LEIE providers
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
    parser.add_argument(
        "--labels-csv",
        default=None,
        help="Path to provider_labels.csv from build_labels.py. If provided, joins label and excldate onto every provider-month row.",
    )
    args = parser.parse_args()
    provider_month_df, provider_month_data_dictionary = run(
        args.input_csv,
        args.output,
        args.dictionary,
        args.dictionary_json,
        cohort_csv=args.cohort_csv,
        cohorts=args.cohorts if args.cohorts else None,
        labels_csv=args.labels_csv,
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
