#!/usr/bin/env bash
# Build the full provider-month pipeline for a single cohort.
#
#   Step 1 — build_labels.py          → provider_labels.csv
#   Step 2 — build_provider_cohorts.py → provider_cohorts.csv
#   Step 3 — create_provider_month_dataset.py (single cohort) → provider_month.csv
#
# Usage:
#   ./run_provider_month.sh NY_individual
#   ./run_provider_month.sh CA_organization
#
# Required inputs (override via environment if paths differ):
#   MEDICAID_CSV  raw Medicaid billing CSV
#   LEIE_CSV      LEIE exclusion list CSV
#   NPPES_CSV     NPPES provider registry CSV

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

# ── Cohort (required) ─────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  echo "Usage: $0 <cohort_label>  e.g. NY_individual or CA_organization"
  echo "Run build_provider_cohorts.py first to see available cohort labels."
  exit 1
fi
COHORT="$1"

# ── Input paths ───────────────────────────────────────────────────────────────
MEDICAID_CSV="${MEDICAID_CSV:-$REPO/data/datasets/medicaid-provider-spending.csv}"
LEIE_CSV="${LEIE_CSV:-$REPO/data/datasets/LEIElabels.csv}"
NPPES_CSV="${NPPES_CSV:-$REPO/data/datasets/nppes.csv}"

# ── Output paths ──────────────────────────────────────────────────────────────
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"
COHORTS_CSV="$REPO/data/outputs/provider_cohorts.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_month_${COHORT}.csv"

# ── Step 1: Labels ────────────────────────────────────────────────────────────
echo "=== Step 1/3: Building labels ==="
python "$REPO/scripts/build_labels.py" \
  --leie_csv "$LEIE_CSV" \
  --medicaid_csv "$MEDICAID_CSV" \
  --output_csv "$LABELS_CSV"

# ── Step 2: Cohorts ───────────────────────────────────────────────────────────
echo "=== Step 2/3: Building provider cohorts ==="
python "$REPO/scripts/build_provider_cohorts.py" \
  --nppes_csv "$NPPES_CSV" \
  --medicaid_csv "$MEDICAID_CSV" \
  --output_csv "$COHORTS_CSV"

# ── Step 3: Provider-month features (single cohort) ──────────────────────────
echo "=== Step 3/3: Building provider-month features for cohort '$COHORT' ==="
python "$REPO/scripts/create_provider_month_dataset.py" "$MEDICAID_CSV" \
  --cohort-csv "$COHORTS_CSV" \
  --cohort "$COHORT" \
  --labels-csv "$LABELS_CSV" \
  --output "$OUTPUT_CSV"

echo ""
echo "Done. Outputs:"
echo "  Labels:         $LABELS_CSV"
echo "  Cohorts:        $COHORTS_CSV"
echo "  Provider-month: $OUTPUT_CSV"
