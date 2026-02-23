#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/linux/run_builtin_sweep.sh [.venv-mps-linux]

ENV_PATH="${1:-.venv-mps-linux}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSPACE}"

if [[ ! -f "${ENV_PATH}/bin/activate" ]]; then
  echo "[run][error] env not found: ${ENV_PATH}"
  exit 1
fi
# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

python experiments/run_decoder_comparison.py \
  --config experiments/configs/code_sweep.yaml \
  --output-dir experiments/results \
  --matrix-dir /tmp/nonexistent_matrix_cases \
  --device cuda \
  --allow-cpu-fallback

LATEST_JSON="$(ls -t experiments/results/decoder_comparison_*.json | head -n1)"
python experiments/summarize_decoder_comparison.py \
  --input "${LATEST_JSON}" \
  --output-md experiments/results/decoder_comparison_summary_builtin.md

echo "[run] latest json: ${LATEST_JSON}"
echo "[run] summary    : experiments/results/decoder_comparison_summary_builtin.md"
