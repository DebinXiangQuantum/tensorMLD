#!/usr/bin/env bash
set -euo pipefail

# Initialize folder layout for qLDPC benchmark matrix cases.
# BB codes are generated from experiments/codes/codes.py and do not need files.
# TB code tb_25_3_4 and tb_30_6_4 are loaded from
# experiments/codes/gnd_data/ldpc_* and do not need matrix files.
# tb_48_4_8 currently uses matrix files.
# Usage:
#   bash scripts/linux/init_qldpc_case_dirs.sh [base_dir]

BASE_DIR="${1:-experiments/data/cases}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSPACE}"

CASES=(
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
echo "[init] place tb_48_4_8 matrix files, then run:"
echo "       bash scripts/linux/run_decoder_sweep.sh .venv-mps-linux experiments/configs/qldpc_six_codes.yaml"
