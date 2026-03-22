#!/usr/bin/env bash
# Run anomaly detection training with Hydra.
#
# Examples:
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv model=isolation_forest
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv model=autoencoder data.train_on_negatives=true
#   ./run_anomaly.sh data.provider_month_csv=data/outputs/provider_month.csv data/splitter=stratified_kfold model=logistic_regression

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"

python -m anomaly_detect.train "$@"
