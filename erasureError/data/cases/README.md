# qLDPC Case Layout

Place benchmark matrices under:

`data/cases/<case_name>/`

Required files per case:

- `H.npy`: parity-check matrix, shape `(num_checks, num_errors)`
- `logical.npy`: logical observable matrix, shape `(num_logicals, num_errors)` or `(num_errors,)`

Example:

```text
data/cases/bb_72_12_6/
  H.npy
  logical.npy
```

Default six qLDPC benchmark cases:

- `bb_18_4_4`
- `bb_60_8_4`
- `bb_72_12_6`
- `tb_25_3_4`
- `tb_30_6_4`
- `tb_48_4_8`

After placing matrices, run:

```bash
python tests/decoder_comparison.py
python plots/plot_results.py
```
