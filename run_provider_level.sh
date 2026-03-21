#!/usr/bin/env bash
# Build provider-level anomaly detection feature table.

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
INPUT_CSV="$REPO/data/outputs/provider_month_NY_individual.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_level.csv"
DICTIONARY_CSV="$REPO/data/outputs/provider_level_data_dictionary.csv"
DICTIONARY_JSON="$REPO/provider_level_data_dictionary.json"

python "$REPO/scripts/create_provider_level_from_month.py" "$INPUT_CSV" \
  --output "$OUTPUT_CSV" \
  --dictionary-csv "$DICTIONARY_CSV" \
  --dictionary-json "$DICTIONARY_JSON"

echo "Done: $OUTPUT_CSV"
