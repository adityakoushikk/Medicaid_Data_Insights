#!/usr/bin/env bash
# Build provider-level anomaly detection feature table.

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
INPUT_CSV="$REPO/data/outputs/provider_month_NM_organization.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_level.csv"
DICTIONARY_CSV="$REPO/data/outputs/provider_level_data_dictionary.csv"
DICTIONARY_JSON="$REPO/provider_level_data_dictionary.json"
LABELS_CSV="$REPO/data/outputs/provider_labels.csv"

python "$REPO/scripts/create_provider_level_from_month.py" "$INPUT_CSV" \
  --output "$OUTPUT_CSV" \
  --dictionary-csv "$DICTIONARY_CSV" \
  --dictionary-json "$DICTIONARY_JSON" \
  --labels-csv "$LABELS_CSV"

echo "Done: $OUTPUT_CSV"
