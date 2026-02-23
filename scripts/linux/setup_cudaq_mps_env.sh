#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/linux/setup_cudaq_mps_env.sh [.venv-mps-linux]

ENV_PATH="${1:-.venv-mps-linux}"
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "[setup] workspace: ${WORKSPACE}"
echo "[setup] env path : ${ENV_PATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup][error] uv not found. Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[setup][warn] nvidia-smi not found. GPU profiling fields may be unavailable."
else
  echo "[setup] nvidia-smi:"
  nvidia-smi || true
fi

cd "${WORKSPACE}"
uv venv "${ENV_PATH}" --python 3.12

# shellcheck disable=SC1090
source "${ENV_PATH}/bin/activate"

python -m pip install -U pip setuptools wheel

# CUDA-Q QEC + decoder dependencies + profiling dependencies.
python -m pip install -i https://pypi.org/simple \
  cudaq-qec \
  numpy scipy quimb opt_einsum cotengra \
  pynvml pyyaml psutil tqdm

# Patch installed cudaq_qec package with local decoder sources.
python scripts/linux/patch_cudaq_qec_plugins.py --workspace "${WORKSPACE}"

echo "[setup] verification:"
python - <<'PY'
import cudaq_qec as qec
print("cudaq_qec imported from:", qec.__file__)
if hasattr(qec, "get_available_decoders"):
    print("available decoders (subset):")
    decoders = qec.get_available_decoders()
    for name in sorted(decoders):
        if "tensor_network" in name:
            print(" -", name)
else:
    print("get_available_decoders() not exposed; will validate by construction in run script.")
print("available codes (subset):")
codes = qec.get_available_codes()
for name in sorted(codes):
    if name in {"steane", "surface_code", "repetition"}:
        print(" -", name)
PY

echo "[setup] done"
echo "[setup] activate with: source ${ENV_PATH}/bin/activate"
echo "[setup] next:"
echo "  bash scripts/linux/init_qldpc_case_dirs.sh experiments/data/cases"
echo "  bash scripts/linux/run_decoder_sweep.sh ${ENV_PATH} experiments/configs/qldpc_six_codes.yaml"
