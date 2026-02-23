# MPS Decoder Benchmarks

This folder contains benchmark helpers for the compressed MPS decoder:

- `benchmark_tensor_network_mps_decoder.py`
  - Builds `tensor_network_mps_decoder`
  - Prints offline compression stats
  - Runs online decode and latency benchmark

## Quick Start

```bash
python cudaqx/libs/qec/python/benchmarks/benchmark_tensor_network_mps_decoder.py \
  --bond-dim 16 \
  --repeats 100 \
  --verbose
```

You can also pass input matrices:

```bash
python cudaqx/libs/qec/python/benchmarks/benchmark_tensor_network_mps_decoder.py \
  --h-path /path/to/H.npy \
  --logical-path /path/to/logical.npy \
  --noise-path /path/to/noise.npy \
  --bond-dim 16
```

Output is a JSON summary with:

- model shape and runtime config
- offline contraction/compression statistics
- single-shot decode result
- latency statistics (`avg_ms`, `min_ms`, `max_ms`)
