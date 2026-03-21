#!/usr/bin/env bash
# Generate provider-month table for a given cohort.
# Usage: ./run_provider_month.sh [COHORT_LABEL]
# Default cohort: NY_individual

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
COHORT="${1:-NY_individual}"

RAW_CSV="$REPO/data/datasets/medicaid-provider-spending.csv"
COHORT_CSV="$REPO/data/outputs/provider_cohorts.csv"
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_month_${COHORT}.csv"
DICTIONARY_CSV="$REPO/data/outputs/provider_month_data_dictionary.csv"
DICTIONARY_JSON="$REPO/provider_month_data_dictionary.json"

python "$REPO/scripts/create_provider_month_dataset.py" "$RAW_CSV" \
  --cohort-csv "$COHORT_CSV" \
  --cohorts "$COHORT" \
  --labels-csv "$LABELS_CSV" \
  --output "$OUTPUT_CSV" \
  --dictionary "$DICTIONARY_CSV" \
  --dictionary-json "$DICTIONARY_JSON"

echo "Done: $OUTPUT_CSV"
