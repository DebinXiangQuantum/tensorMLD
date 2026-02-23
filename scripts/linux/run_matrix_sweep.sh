#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/linux/run_matrix_sweep.sh [.venv-mps-linux] [matrix_dir] [config_path]
#
# Expected matrix_dir layout:
#   <matrix_dir>/<case_name>/H.npy
#   <matrix_dir>/<case_name>/logical.npy
#   <matrix_dir>/<case_name>/noise.npy   (optional)

ENV_PATH="${1:-.venv-mps-linux}"
MATRIX_DIR="${2:-experiments/data/cases}"
CONFIG_PATH="${3:-experiments/configs/qldpc_six_codes.yaml}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSPACE}"

if [[ ! -f "${ENV_PATH}/bin/activate" ]]; then
  echo "[run][error] env not found: ${ENV_PATH}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

python experiments/run_decoder_comparison.py \
  --config "${CONFIG_PATH}" \
  --output-dir experiments/results \
  --matrix-dir "${MATRIX_DIR}" \
  --device cuda \
  --allow-cpu-fallback

LATEST_JSON="$(ls -t experiments/results/decoder_comparison_*.json | head -n1)"
python experiments/summarize_decoder_comparison.py \
  --input "${LATEST_JSON}" \
  --output-md experiments/results/decoder_comparison_summary_matrix.md

echo "[run] latest json: ${LATEST_JSON}"
echo "[run] summary    : experiments/results/decoder_comparison_summary_matrix.md"
