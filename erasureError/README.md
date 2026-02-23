# Erasure Error Tensor Network Decoder

This package extends the tensor network MLD (Maximum Likelihood Decoding) approach to support erasure errors in quantum error correction.

## Mathematical Background

### Erasure Errors
Unlike depolarizing errors (where error location is unknown), erasure errors are "heralded" - the error location is known, but the Pauli type (X or Z) is unknown.

### Key Mathematical Extension
For erasure errors, we set:
- Error probability `p = 0.5` (maximum uncertainty)
- Weight `β = (1/2) * ln((1-p)/p) = 0`
- Tensor becomes `[0.5, 0.5]` (identity-like, "broken bond")

This corresponds to "broken bonds" in the spin-glass model, where there is zero energy penalty for assigning an error to an erased qubit.

## Project Structure

```
erasureError/
├── src/                    # Source code
│   ├── __init__.py
│   ├── erasure_tensor_network_decoder.py  # Main decoder class
│   ├── noise_models.py     # Erasure-aware noise models
│   ├── simple_decoders.py  # BP/OSD/Relay comparison wrappers
│   └── qldpc_cases.py      # Six qLDPC benchmark case loader
├── tests/                  # Test code
│   ├── test_erasure_decoder.py    # Unit tests
│   ├── test_benchmark.py          # Tensor decoder benchmark (qLDPC)
│   └── decoder_comparison.py      # Multi-decoder comparison (qLDPC)
├── plots/                  # Plotting scripts
│   └── plot_results.py     # Generate result plots
├── data/cases/             # qLDPC case matrices (H.npy/logical.npy)
├── results/                # JSON result files
├── pyproject.toml          # Project configuration
└── README.md               # This file
```

## Installation

### Using uv (Recommended)

```bash
# Navigate to the erasureError directory
cd erasureError

# Create virtual environment with uv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
uv pip install -e .

# For full features (including cudaq-qec and stim):
uv pip install -e ".[all]"
```

### Using pip

```bash
cd erasureError
pip install -e .
```

## Usage

### 1. Run Unit Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_erasure_decoder.py -v -s
```

### 2. Prepare qLDPC Cases

Put case matrices under `data/cases/<case_name>/`:

- `H.npy`
- `logical.npy`

Required case names:

- `bb_18_4_4`
- `bb_60_8_4`
- `bb_72_12_6`
- `tb_25_3_4`
- `tb_30_6_4`
- `tb_48_4_8`

### 3. Run Benchmark Tests

```bash
# Run tensor decoder benchmark on qLDPC cases
python tests/test_benchmark.py
```

### 4. Run Decoder Comparison

```bash
# Compare tensor MLD with BP+OSD / Min-Sum BP / Sequential Relay BP
python tests/decoder_comparison.py
```

### 5. Generate Plots

```bash
# Generate plots from results
python plots/plot_results.py
```

## Example Usage in Code

```python
import numpy as np
from src.erasure_tensor_network_decoder import ErasureTensorNetworkDecoder

# Define a simple repetition code
H = np.array([
    [1, 1, 0],
    [0, 1, 1]
], dtype=np.float64)

logical_obs = np.array([[1, 1, 1]], dtype=np.float64)

# Error probabilities for each qubit
error_probs = [0.01, 0.01, 0.01]

# Erasure mask: first qubit is erased
erasure_mask = [True, False, False]

# Create decoder
decoder = ErasureTensorNetworkDecoder(
    H=H,
    logical_obs=logical_obs,
    error_probabilities=error_probs,
    erasure_mask=erasure_mask,
    debug=True
)

# Decode a syndrome
syndrome = [0.0, 0.0]  # No triggered checks
result = decoder.decode(syndrome)

print(f"Logical error probability: {result['logical_error_prob']:.4f}")
```

## Results

Results are saved as JSON files in the `results/` directory:
- `benchmark_results_*.json`: Benchmark test results
- `decoder_comparison_*.json`: Decoder comparison results

## Plots

After running tests, plots can be generated in the `plots/` directory:
- `ler_vs_error_*.png`: Logical error rate vs physical error rate
- `erasure_effect_*.png`: Effect of erasure rate on LER
- `decoder_comparison_summary.png`: Summary comparison across codes
- `verification_summary.png`: Fraction of points satisfying `LER < p`

## Supported Codes

Currently tested with:
- BB [[18,4,4]]
- BB [[60,8,4]]
- BB [[72,12,6]]
- TB [[25,3,4]]
- TB [[30,6,4]]
- TB [[48,4,8]]

## Dependencies

Core:
- Python >= 3.11
- numpy >= 1.24.0
- quimb >= 1.8.0
- autoray >= 0.6.0
- matplotlib >= 3.7.0

Optional:
- cudaq-qec: For nv-qldpc-decoder comparison
- stim: For circuit-level noise simulation

## References

1. TensorMLD Paper: Maximum Likelihood Decoding using Tensor Networks
2. NVIDIA CUDA-Q QEC Documentation
3. Erasure Error in Neutral Atom Systems
