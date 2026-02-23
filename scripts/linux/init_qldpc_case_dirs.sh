#!/usr/bin/env bash
set -euo pipefail

# Initialize folder layout for six qLDPC benchmark codes.
# Usage:
#   bash scripts/linux/init_qldpc_case_dirs.sh [base_dir]

BASE_DIR="${1:-experiments/data/cases}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSPACE}"

CASES=(
  "bb_18_4_4"
  "bb_60_8_4"
  "bb_72_12_6"
  "tb_25_3_4"
  "tb_30_6_4"
  "tb_48_4_8"
)

mkdir -p "${BASE_DIR}"
for c in "${CASES[@]}"; do
  d="${BASE_DIR}/${c}"
  mkdir -p "${d}"
  cat > "${d}/README.md" <<EOF
# ${c}

Required files in this folder:
- H.npy           shape: (num_checks, num_errors)
- logical.npy     shape: (num_logicals, num_errors) or (num_errors,)

Optional:
- noise.npy       shape: (num_errors,)
- meta.json       additional metadata
EOF
done

echo "[init] created qLDPC case directories under: ${BASE_DIR}"
echo "[init] place matrix files, then run:"
echo "       bash scripts/linux/run_decoder_sweep.sh .venv-mps-linux experiments/configs/qldpc_six_codes.yaml"
