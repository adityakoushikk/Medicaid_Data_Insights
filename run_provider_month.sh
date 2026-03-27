#!/usr/bin/env bash
# Build provider-month features for a single cohort.
# Requires build_cohorts.sh to have been run first.
#
# Usage:
#   ./run_provider_month.sh NY_individual
#   ./run_provider_month.sh CA_organization [date_start] [date_end]
#
# Required inputs (override via environment if paths differ):
#   MEDICAID_CSV  raw Medicaid billing CSV

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"

# ── Args ──────────────────────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
  echo "Usage: $0 <cohort_label> [date_start] [date_end]"
  echo "  e.g. $0 NY_individual 2020-01-01 2024-12-31"
  echo "Run build_cohorts.sh first to generate cohorts and labels."
  exit 1
fi
COHORT="$1"
DATE_START="${2:-}"
DATE_END="${3:-}"

# ── Input paths ───────────────────────────────────────────────────────────────
MEDICAID_CSV="${MEDICAID_CSV:-$REPO/data/datasets/medicaid-provider-spending.csv}"
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"
COHORTS_CSV="$REPO/data/outputs/provider_cohorts.csv"

# ── Validate prereqs ──────────────────────────────────────────────────────────
if [ ! -f "$COHORTS_CSV" ] || [ ! -f "$LABELS_CSV" ]; then
  echo "Error: $COHORTS_CSV or $LABELS_CSV not found. Run build_cohorts.sh first."
  exit 1
fi

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_CSV="$REPO/data/outputs/provider_month_${COHORT}.csv"

# ── Build provider-month features ─────────────────────────────────────────────
echo "=== Building provider-month features for cohort '$COHORT' ==="
CMD=(python "$REPO/scripts/create_provider_month_dataset.py" "$MEDICAID_CSV"
  --cohort-csv "$COHORTS_CSV"
  --cohort "$COHORT"
  --labels-csv "$LABELS_CSV"
  --output "$OUTPUT_CSV")

[ -n "$DATE_START" ] && CMD+=(--date-start "$DATE_START")
[ -n "$DATE_END" ]   && CMD+=(--date-end   "$DATE_END")

"${CMD[@]}"

echo ""
echo "Done. Output: $OUTPUT_CSV"
