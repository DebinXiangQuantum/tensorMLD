# Erasure Error Tensor Network Decoder

This package extends the tensor network MLD (Maximum Likelihood Decoding) approach to support erasure errors in quantum error correction.

## Mathematical Background

### Erasure Errors
Unlike depolarizing errors (where error location is unknown), erasure errors are "heralded" - the error location is known, but the Pauli type (X or Z) is unknown.

### Key Mathematical Extension
For erasure errors, we set:
- Error probability `p = 0.5` (maximum uncertainty)
- Weight `beta = (1/2) * ln((1-p)/p) = 0`
- Tensor becomes `[0.5, 0.5]` (identity-like, "broken bond")

This corresponds to "broken bonds" in the spin-glass model, where there is zero energy penalty for assigning an error to an erased qubit.

## Project Structure

```
erasureError/
├── src/                              # Source code
│   ├── __init__.py
│   ├── erasure_tensor_network_decoder.py  # Main tensor network MLD decoder
│   └── simple_decoders.py            # LDPC decoder wrappers (BP+OSD, Min-Sum BP)
├── codes/                            # Code construction
│   ├── codes.py                      # CSS code class, BB/HP code generators
│   ├── gnd_ldpc_codes.py             # GND file loaders for BB/TB codes
│   └── utils.py                      # Linear algebra utilities
├── tests/                            # Benchmark tests
│   ├── test_erasure_decoder.py       # Unit tests
│   ├── decoder_comparison.py         # Repetition/Surface code comparison
│   ├── bb_code_benchmark.py          # BB code benchmark
│   └── comprehensive_benchmark.py    # Full qLDPC benchmark (BB/TB/HP)
├── plots/                            # Plotting scripts
│   ├── plot_results.py               # Basic result plots
│   ├── plot_bb_results.py            # BB code specific plots
│   └── plot_comprehensive_results.py # Comprehensive benchmark plots
├── results/                          # JSON result files
├── .venv/                            # Python virtual environment
└── README.md                         # This file
```

## Installation

### Step 1: Create Virtual Environment

```bash
cd erasureError

# Create virtual environment
python3 -m venv .venv

# Install pip in venv
.venv/bin/python -m ensurepip --upgrade
```

### Step 2: Install Dependencies

```bash
# Install core dependencies
.venv/bin/python -m pip install numpy quimb autoray matplotlib scipy

# Install ldpc package for BP decoders
.venv/bin/python -m pip install ldpc

# Install torch for loading GND code files (BB/TB codes)
.venv/bin/python -m pip install torch
```

### Step 3: Verify Installation

```bash
.venv/bin/python -c "import numpy, quimb, ldpc, torch; print('All dependencies installed!')"
```

## Running Benchmarks

### 1. Quick Test - Repetition and Surface Codes

```bash
.venv/bin/python tests/decoder_comparison.py
```

This tests tensor network MLD against BP+OSD and Min-Sum BP on:
- Repetition codes (d=5, 7, 9)
- Surface code (d=3)

### 3. Comprehensive qLDPC Benchmark (Recommended)

```bash
.venv/bin/python tests/comprehensive_benchmark.py
```

Benchmarks all codes with 500 shots:

| Code Type | Code | Parameters |
|-----------|------|------------|
| BB | BB_18_4_4 | [[18, 4, 4]] |
| BB | BB_60_8_4 | [[60, 8, 4]] |
| BB | BB_72_12_6 | [[72, 12, 6]] |
| TB | TB_25_3_4 | [[25, 3, 4]] |
| TB | TB_30_6_4 | [[30, 6, 4]] |
| HP | HP_50 | [[50, 2, 4]] |
| HP | HP_98 | [[98, 2, 4]] |

### 4. Generate Plots

```bash
# For comprehensive benchmark results
.venv/bin/python plots/plot_comprehensive_results.py

```

## Example Usage

```python
import numpy as np
import sys
sys.path.insert(0, 'src')

from erasure_tensor_network_decoder import ErasureTensorNetworkDecoder

# Define a simple repetition code
H = np.array([
    [1, 1, 0],
    [0, 1, 1]
], dtype=np.float64)

logical_obs = np.array([[1, 1, 1]], dtype=np.float64)

# Error probabilities for each qubit
error_probs = [0.01, 0.01, 0.01]

# Erasure mask: first qubit is erased (location known, type unknown)
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
syndrome = [0.0, 0.0]
result = decoder.decode(syndrome)

print(f"Logical error probability: {result['logical_error_prob']:.4f}")
```

## Decoders Compared

| Decoder | Description | Speed |
|---------|-------------|-------|
| `tensor_network_mld` | Exact MLD via tensor network contraction | Slow |
| `bp_osd` | Belief Propagation + Ordered Statistics Decoding | Fast |
| `min_sum_bp` | Min-Sum Belief Propagation | Fast |

## Results

Results are saved as JSON in `results/`:
- `comprehensive_benchmark_*.json` - Full qLDPC benchmark results
- `decoder_comparison_*.json` - Basic comparison results

## Generated Plots

After running benchmarks, plots are saved in `plots/`:

| Plot | Description |
|------|-------------|
| `bb_codes_comparison.png` | BB codes LER vs error rate (3 erasure levels) |
| `tb_codes_comparison.png` | TB codes LER vs error rate |
| `hp_codes_comparison.png` | HP codes LER vs error rate |
| `decoder_performance_summary.png` | Fraction of points where LER < p |
| `erasure_impact_analysis.png` | Impact of erasure rate on LER |
| `code_comparison_fixed_params.png` | All codes at fixed error rate |

## Key Results

1. **Tensor Network MLD** achieves the best performance, especially at low error rates
2. **BP+OSD** is the second-best decoder with much faster runtime
3. Error correction is effective (LER < p) below threshold (~1-3% depending on code)
4. Erasure errors increase LER but the decoder handles them correctly by setting p=0.5

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.24 | Numerical operations |
| quimb | >= 1.8 | Tensor network contraction |
| autoray | >= 0.6 | Backend-agnostic array operations |
| matplotlib | >= 3.7 | Plotting |
| ldpc | >= 2.0 | BP+OSD and Min-Sum BP decoders |
| torch | >= 2.0 | Loading GND code files |
| scipy | >= 1.10 | Sparse matrix operations |

## References

1. Bravyi et al., "Maximum Likelihood Decoding using Tensor Networks"
2. IBM Gross Code Paper: Bivariate Bicycle Codes
3. Voss et al., "Trivariate Bicycle Codes"
4. Roffe et al., "LDPC: Python tools for low density parity check codes"
