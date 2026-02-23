#!/bin/bash
# ============================================================================ #
# Run script for Erasure Error Tensor Network Decoder tests                    #
# ============================================================================ #

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================"
echo "Erasure Error Tensor Network Decoder"
echo "========================================"

# Check if uv is installed
if command -v uv &> /dev/null; then
    echo "[INFO] Using uv for environment management"

    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "[INFO] Creating virtual environment..."
        uv venv
    fi

    # Activate venv
    source .venv/bin/activate

    # Install dependencies
    echo "[INFO] Installing dependencies..."
    uv pip install -e . --quiet

else
    echo "[WARNING] uv not found, using pip"

    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        echo "[INFO] Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate venv
    source .venv/bin/activate

    # Install dependencies
    pip install -e . --quiet
fi

# Parse command line arguments
RUN_UNIT_TESTS=false
RUN_BENCHMARK=false
RUN_COMPARISON=false
RUN_PLOTS=false
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            RUN_UNIT_TESTS=true
            shift
            ;;
        --benchmark)
            RUN_BENCHMARK=true
            shift
            ;;
        --comparison)
            RUN_COMPARISON=true
            shift
            ;;
        --plots)
            RUN_PLOTS=true
            shift
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--unit] [--benchmark] [--comparison] [--plots] [--all]"
            exit 1
            ;;
    esac
done

# Default to running all if no arguments
if [ "$RUN_UNIT_TESTS" = false ] && [ "$RUN_BENCHMARK" = false ] && \
   [ "$RUN_COMPARISON" = false ] && [ "$RUN_PLOTS" = false ] && [ "$RUN_ALL" = false ]; then
    RUN_ALL=true
fi

if [ "$RUN_ALL" = true ]; then
    RUN_UNIT_TESTS=true
    RUN_BENCHMARK=true
    RUN_COMPARISON=true
    RUN_PLOTS=true
fi

# Run tests
if [ "$RUN_UNIT_TESTS" = true ]; then
    echo ""
    echo "========================================"
    echo "Running Unit Tests"
    echo "========================================"
    python -m pytest tests/test_erasure_decoder.py -v -s
fi

if [ "$RUN_BENCHMARK" = true ]; then
    echo ""
    echo "========================================"
    echo "Running Benchmark Tests"
    echo "========================================"
    python tests/test_benchmark.py
fi

if [ "$RUN_COMPARISON" = true ]; then
    echo ""
    echo "========================================"
    echo "Running Decoder Comparison"
    echo "========================================"
    python tests/decoder_comparison.py
fi

if [ "$RUN_PLOTS" = true ]; then
    echo ""
    echo "========================================"
    echo "Generating Plots"
    echo "========================================"
    python plots/plot_results.py
fi

echo ""
echo "========================================"
echo "All tasks completed!"
echo "========================================"
echo "Results are in: $SCRIPT_DIR/results/"
echo "Plots are in: $SCRIPT_DIR/plots/"
