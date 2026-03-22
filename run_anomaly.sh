#!/usr/bin/env bash
# Run anomaly detection training (autoencoder) with Hydra.

set -e
REPO="$(cd "$(dirname "$0")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO}"
export PYTHONPATH="${REPO}/src:${REPO}/scripts:${PYTHONPATH:-}"

python -m anomaly_detect.train "$@"
