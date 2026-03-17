# Medicaid Provider Anomaly Detection

## Goal

Detect temporally anomalous billing behavior among Medicaid providers. A provider is anomalous if their billing patterns — volume, unit economics, code mix, volatility — deviate significantly from others in the same cohort (same state + entity type). The pipeline produces a provider-level feature table suitable for unsupervised anomaly detection (e.g. Isolation Forest, DBSCAN, LOF).

## Data Sources

| File | Size | Description |
|---|---|---|
| `data/datasets/medicaid-provider-spending.csv` | ~10 GB | Raw Medicaid billing data. One row per (provider, month, HCPCS code). Columns: `BILLING_PROVIDER_NPI_NUM`, `CLAIM_FROM_MONTH`, `HCPCS_CODE`, `TOTAL_PAID`, `TOTAL_CLAIMS`, `TOTAL_UNIQUE_BENEFICIARIES`. |
| `data/datasets/npidata_pfile_20050523-20260308.csv` | ~10 GB | NPPES provider registry. Contains NPI, entity type (individual vs organization), state, and up to 15 taxonomy codes with primary taxonomy switch flags. |

## Pipeline

```
medicaid-provider-spending.csv  ──┐
                                  ├──► build_provider_cohorts.py ──► provider_cohorts.csv
npidata_pfile_*.csv  ─────────────┘

medicaid-provider-spending.csv  ──┐
                                  ├──► create_provider_month_dataset.py ──► provider_month_<cohort>.csv
provider_cohorts.csv  ────────────┘                                         provider_month_data_dictionary.csv

provider_month_<cohort>.csv ───────► create_provider_level_from_month.py ──► provider_level.csv
```

### Step 1 — Build cohorts (`build_provider_cohorts.py`)

Joins Medicaid NPIs against NPPES to assign each provider a cohort label based on `(state, entity_type)`. Drops providers with invalid/unknown states, missing entity types, or unknown taxonomy.

**Inputs:**
- `--nppes_csv` — NPPES registry CSV
- `--medicaid_csv` — raw Medicaid billing CSV (used only to get the set of active NPIs)

**Output:** `provider_cohorts.csv` — 3 columns: `npi`, `cohort_label` (e.g. `TX_individual`), `cohort` (integer ID)

---

### Step 2 — Build provider-month features (`create_provider_month_dataset.py`)

Aggregates raw billing rows to one row per `(provider, month)`. Computes features across 3 groups:

- **Core aggregates** — `paid_t`, `claims_t`, `beneficiaries_proxy_t`, `hcpcs_count_t`
- **Unit economics** — `paid_per_claim_t`, `claims_per_beneficiaries_proxy_t`, `paid_per_beneficiaries_proxy_t`
- **Code-mix** — entropy, HHI, top code paid/claim share, top-3 shares, top HCPCS code
- **Code distribution** — mean/median/std/min/max/IQR/MAD/skew/kurtosis of paid and claims across codes within a month, energy, num codes above mean

Only observed `(provider, month)` pairs are included — months with no billing are not zero-filled.

**Inputs:**
- `input_csv` (positional) — raw Medicaid billing CSV (or a cohort-filtered subset)
- `--cohort-csv` — cohort mapping CSV from step 1
- `--cohorts` — filter to specific cohort labels or IDs (requires `--cohort-csv`)
- `--dictionary-json` — editable JSON file defining column metadata (source of truth for the dictionary)

**Outputs:**
- `--output` — provider-month feature CSV
- `--dictionary` — data dictionary CSV (generated from `--dictionary-json`)

---

### Step 3 — Build provider-level features (`create_provider_level_from_month.py`)

Collapses the provider-month table to one row per provider. All change/diff features are **gap-aware**: consecutive observed months may not be adjacent calendar months, so differences are divided by the calendar-month gap between rows to produce monthlyized rates.

Feature groups:
- **Volume & scale** — totals and averages of paid, claims, beneficiaries over time
- **Unit economics** — mean/std of per-claim and per-beneficiary rates
- **Volatility / instability** — std, CV, and MoM change rates for paid and claims
- **Spike & drop behavior** — magnitude and frequency of sudden increases/decreases
- **Structural breaks (PELT)** — number and location of changepoints in billing series (requires `ruptures`)
- **Flagged month persistence** — how often a provider has months flagged as anomalous
- **Code-mix summaries** — all-time averages of entropy, HHI, top code share
- **Code diversity** — unique codes, turnover rate, specialization stability

Providers with fewer than `--min-months` observed months get `insufficient_history_flag=1`. By default these are dropped from the output; use `--no-filter` to keep them.

**Input:** provider-month CSV from step 2

**Output:** `--output` — one row per provider, all features

---

## Scripts

```
scripts/
├── build_provider_cohorts.py          # Step 1: NPI → cohort mapping
├── create_provider_month_dataset.py   # Step 2: raw billing → provider-month features
└── create_provider_level_from_month.py  # Step 3: provider-month → provider-level features
```

---

## Missing Data & Edge Case Handling

### `build_provider_cohorts.py`

All filtering happens in SQL before any data leaves DuckDB.

| Situation | Behaviour |
|---|---|
| State field is empty, `NULL`, or `'N/A'` | Row dropped |
| State is a full name (e.g. `"NEW MEXICO"`) | Normalized to 2-letter code via `STATE_NORMALIZE` lookup |
| State is unrecognized or `ZZ` (foreign) | Normalizes to `NULL` → row dropped |
| Entity type not `'1'` or `'2'` | Row dropped |
| No taxonomy slot has Primary Switch = `'Y'` | Falls back to first non-null taxonomy code across slots 1–15 |
| All 15 taxonomy slots are null | Assigned `'UNKNOWN'` → row dropped |
| NPI in Medicaid but not in NPPES | Dropped (INNER JOIN) |
| NPI in NPPES but not in Medicaid | Dropped (INNER JOIN) |

---

### `create_provider_month_dataset.py`

Filtering happens in pandas after loading. Invalid values are coerced to `NaN` rather than dropping rows, so partial data is preserved.

| Situation | Behaviour |
|---|---|
| `CLAIM_FROM_MONTH` cannot be parsed as a date | Coerced to `NaT` → row dropped |
| `BILLING_PROVIDER_NPI_NUM` is null or empty string | Row dropped |
| `TOTAL_PAID` or `TOTAL_CLAIMS` is non-numeric | Coerced to `NaN`; row kept, `NaN` propagates into aggregates |
| `TOTAL_UNIQUE_BENEFICIARIES` column missing entirely | `beneficiaries_proxy_t` set to `NaN` for all rows |
| `claims_t = 0` in a month | `paid_per_claim_t = NaN` |
| `beneficiaries_proxy_t = 0` in a month | Both beneficiary ratio features = `NaN` |
| `paid_total = 0` for a (provider, month) in code-mix | `top_code_paid_share`, `top_3_code_paid_share`, `hcpcs_entropy`, `hcpcs_hhi` = `NaN`; `top_hcpcs_code_t = ""` |
| Only 1 code billed in a month | Distribution features with `ddof=1` (std) = `NaN`; others compute normally |
| `--cohorts` passed without `--cohort-csv` | `--cohorts` is silently ignored; all providers are kept |

---

### `create_provider_level_from_month.py`

All features are built for every provider before any filtering. `NaN` is returned rather than raising exceptions for edge cases.

| Situation | Behaviour |
|---|---|
| Months after `DATE_CUTOFF` | Dropped before feature computation |
| Provider has fewer than `--min-months` observed months | Features still computed; `insufficient_history_flag = 1`; dropped from output by default (use `--no-filter` to keep) |
| Only 1 observed month | All diff/change/volatility features = `NaN` (require ≥ 2 observations) |
| All `paid_t` values are `NaN` | Volume features = `NaN`; `sum_paid = 0.0` (via `nansum`) |
| Calendar gap between consecutive months ≤ 0 | Gap clamped to 1.0 to prevent division by zero in monthlyized rates |
| Prior `paid_t < MOM_MIN_DENOMINATOR` (default $10) | MoM growth rate = `NaN` to avoid near-zero denominator blow-up |
| `median_paid = 0` | `spike_ratio_paid = NaN` |
| `cv_paid` when mean = 0 or std is `NaN` | `cv_paid = NaN` |
| Rolling robust z-score: fewer than `ROBUST_Z_WINDOW` (6) prior rows | `NaN` until window fills; flagged-month features count only where z-score is non-NaN |
| Rolling robust z-score: MAD of baseline window = 0 (constant signal) | Z-score skipped for that row → `NaN` |
| `ruptures` not installed | All changepoint features = `NaN`; warning printed to stderr |
| Fewer than 6 valid `paid_t` observations | Changepoint features = `NaN` (PELT minimum) |
| PELT raises an exception | Changepoint features = `NaN` (caught silently) |
| Optional columns absent from input (e.g. `hcpcs_entropy`, `top_code_paid_share`) | Corresponding features = `NaN`; script does not error |

## Run Scripts

```
run_build_provider_cohorts.sh   # Runs step 1 for the full dataset
run_nm_organization.sh          # Runs step 2 for NM_organization cohort only
```

## Outputs

All outputs go to `data/outputs/`:

| File | Produced by | Description |
|---|---|---|
| `provider_cohorts.csv` | step 1 | NPI → cohort mapping |
| `provider_month_NM_organization.csv` | step 2 | Provider-month features for NM orgs |
| `provider_month_data_dictionary.csv` | step 2 | Column definitions for provider-month table |
| `provider_month_data_dictionary.json` | hand-edited | Source of truth for column definitions |

## Dependencies

```
pip install -r requirements.txt
```

Key packages: `duckdb` (large CSV processing), `pandas`, `numpy`, `scipy`, `ruptures` (changepoint detection), `scikit-learn`, `kmodes`.
