#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/linux/run_decoder_sweep.sh [.venv-mps-linux] [config_path]

ENV_PATH="${1:-.venv-mps-linux}"
CONFIG_PATH="${2:-experiments/configs/qldpc_six_codes.yaml}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cd "${WORKSPACE}"

if [[ ! -f "${ENV_PATH}/bin/activate" ]]; then
  echo "[run][error] env not found: ${ENV_PATH}"
  echo "[run] run setup first: bash scripts/linux/setup_cudaq_mps_env.sh ${ENV_PATH}"
  exit 1
fi

# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

OUT_DIR="experiments/results"
MATRIX_DIR="experiments/data/cases"

python experiments/run_decoder_comparison.py \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUT_DIR}" \
  --matrix-dir "${MATRIX_DIR}" \
  --device cuda \
  --dtype float32 \
  --bond-dim 16 \
  --low-threshold 0.4 \
  --high-threshold 0.6 \
  --max-rounds 16 \
  --warmup 5 \
  --latency-repeats 50 \
  --gpu-index 0 \
  --gpu-sample-interval-ms 50 \
  --allow-cpu-fallback

LATEST_JSON="$(ls -t ${OUT_DIR}/decoder_comparison_*.json | head -n1)"
python experiments/summarize_decoder_comparison.py \
  --input "${LATEST_JSON}" \
  --output-md "${OUT_DIR}/decoder_comparison_summary.md"

echo "[run] latest json: ${LATEST_JSON}"
echo "[run] summary    : ${OUT_DIR}/decoder_comparison_summary.md"
