#!/usr/bin/env bash
# One-time setup: build provider labels and cohort assignments.
# Run this once before running run_provider_month.sh for any cohort.
#
# Usage:
#   ./build_cohorts.sh
#
# Required inputs (override via environment if paths differ):
#   MEDICAID_CSV  raw Medicaid billing CSV
#   LEIE_CSV      LEIE exclusion list CSV
#   NPPES_CSV     NPPES provider registry CSV

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

# ── Input paths ───────────────────────────────────────────────────────────────
MEDICAID_CSV="${MEDICAID_CSV:-$REPO/data/datasets/medicaid-provider-spending.csv}"
LEIE_CSV="${LEIE_CSV:-$REPO/data/datasets/LEIElabels.csv}"
NPPES_CSV="${NPPES_CSV:-$REPO/data/datasets/nppes.csv}"

# ── Output paths ──────────────────────────────────────────────────────────────
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"
COHORTS_CSV="$REPO/data/outputs/provider_cohorts.csv"

# ── Step 1: Labels ────────────────────────────────────────────────────────────
echo "=== Step 1/2: Building labels ==="
python "$REPO/scripts/build_labels.py" \
  --leie_csv "$LEIE_CSV" \
  --medicaid_csv "$MEDICAID_CSV" \
  --output_csv "$LABELS_CSV"

# ── Step 2: Cohorts ───────────────────────────────────────────────────────────
echo "=== Step 2/2: Building provider cohorts ==="
python "$REPO/scripts/build_provider_cohorts.py" \
  --nppes_csv "$NPPES_CSV" \
  --medicaid_csv "$MEDICAID_CSV" \
  --output_csv "$COHORTS_CSV"

echo ""
echo "Done. Outputs:"
echo "  Labels:  $LABELS_CSV"
echo "  Cohorts: $COHORTS_CSV"
