#!/usr/bin/env bash
# Build the full provider-month pipeline in three steps:
#
#   Step 1 — build_labels.py
#             LEIE CSV + Medicaid billing CSV → provider_labels.csv
#
#   Step 2 — build_provider_cohorts.py
#             NPPES CSV + Medicaid billing CSV → provider_cohorts.csv
#
#   Step 3 — create_provider_month_dataset.py
#             Medicaid billing CSV + provider_labels.csv → provider_month.csv
#
# Required inputs (edit paths below or override via environment):
#   MEDICAID_CSV  raw Medicaid billing CSV
#   LEIE_CSV      LEIE exclusion list CSV
#   NPPES_CSV     NPPES provider registry CSV
#
# Usage:
#   ./run_provider_month.sh
#   MEDICAID_CSV=/path/to/billing.csv ./run_provider_month.sh

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

# ── Input paths ───────────────────────────────────────────────────────────────
MEDICAID_CSV="${MEDICAID_CSV:-$REPO/data/datasets/medicaid-provider-spending.csv}"
LEIE_CSV="${LEIE_CSV:-$REPO/data/datasets/leie.csv}"
NPPES_CSV="${NPPES_CSV:-$REPO/data/datasets/nppes.csv}"

# ── Output paths ──────────────────────────────────────────────────────────────
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"
COHORTS_CSV="$REPO/data/outputs/provider_cohorts.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_month.csv"

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

# ── Step 3: Provider-month features ──────────────────────────────────────────
echo "=== Step 3/3: Building provider-month features ==="
python "$REPO/scripts/create_provider_month_dataset.py" "$MEDICAID_CSV" \
  --labels-csv "$LABELS_CSV" \
  --output "$OUTPUT_CSV"

echo ""
echo "Done. Outputs:"
echo "  Labels:         $LABELS_CSV"
echo "  Cohorts:        $COHORTS_CSV"
echo "  Provider-month: $OUTPUT_CSV"
