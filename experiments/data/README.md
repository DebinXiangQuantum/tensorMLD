# Matrix Case Layout

Put external code matrices (e.g., BB/TB codes) under:

`experiments/data/cases/<case_name>/`

Required files:

- `H.npy`: parity-check matrix, shape `(num_checks, num_errors)`
- `logical.npy`: logical observable matrix, shape `(num_logicals, num_errors)` or `(num_errors,)`

Optional files:

- `noise.npy`: per-qubit physical error probabilities, shape `(num_errors,)`
- `meta.json`: extra case metadata (merged into case config)

Example:

```text
experiments/data/cases/bb_72_12_6/
  H.npy
  logical.npy
  noise.npy
  meta.json
```

`experiments/run_decoder_comparison.py` auto-discovers every subfolder matching this layout.

For the default qLDPC six-code benchmark, initialize folders with:

```bash
bash scripts/linux/init_qldpc_case_dirs.sh experiments/data/cases
```

BB codes (`bb_18_4_4`, `bb_60_8_4`, `bb_72_12_6`) are generated directly from
`experiments/codes/codes.py` in config `experiments/configs/qldpc_six_codes.yaml`.

You need matrix folders mainly for TB codes:

- `tb_25_3_4`
- `tb_30_6_4`
- `tb_48_4_8`
