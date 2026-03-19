#!/usr/bin/env bash
# Build fraud labels from LEIE exclusion data + raw Medicaid billing data.

set -e

REPO="$(cd "$(dirname "$0")" && pwd)"
LEIE_CSV="$REPO/data/datasets/LEIElabels.csv"
MEDICAID_CSV="$REPO/data/datasets/medicaid-provider-spending.csv"
OUTPUT_CSV="$REPO/data/outputs/provider_labels.csv"

python "$REPO/scripts/build_labels.py" \
  --leie_csv "$LEIE_CSV" \
  --medicaid_csv "$MEDICAID_CSV" \
  --output_csv "$OUTPUT_CSV"

echo "Done: $OUTPUT_CSV"
