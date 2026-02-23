# Tensor MLD Linux+CUDA 运行说明

本说明覆盖以下目标：

1. 在 Linux + NVIDIA CUDA 环境安装并运行本项目解码器。  
2. 批量测试 qLDPC 码（BB/TB 六码）。  
3. 对比 `tensor_network_mps_decoder` 与 `tensor_network_decoder`（exact MLD）。  
4. 统计 GPU 显存占用与利用率。

---

## 1. 环境要求

- OS: Linux (x86_64)
- Python: 3.12
- GPU: NVIDIA (建议 A100/H100 或同级)
- 驱动/CUDA: `nvidia-smi` 可用
- 工具: `uv`, `bash`

---

## 2. 一键安装

在仓库根目录执行：

```bash
bash scripts/linux/setup_cudaq_mps_env.sh .venv-mps-linux
```

该脚本会执行：

- 创建 `uv` 虚拟环境
- 安装 `cudaq-qec`、`quimb`、`pynvml` 等依赖
- 将当前工作区的解码器源码覆盖到已安装 `cudaq_qec` 包中：
  - `tensor_network_mps_decoder.py`
  - `mps_decoder_core.py`

---

## 3. qLDPC 六码基准（默认）

先初始化六码目录：

```bash
bash scripts/linux/init_qldpc_case_dirs.sh experiments/data/cases
```

然后把每个码的 `H.npy`、`logical.npy`（可选 `noise.npy`）放入对应目录：

- `tb_25_3_4`
- `tb_30_6_4`
- `tb_48_4_8`

说明：

- BB 三个码（`bb_18_4_4`、`bb_60_8_4`、`bb_72_12_6`）由
  `experiments/codes/codes.py` 自动构造，无需手工矩阵文件。
- TB 三个码默认从上述目录读取矩阵文件。
  若 TB 文件缺失，当前配置会自动跳过 TB case。

再执行：

```bash
bash scripts/linux/run_decoder_sweep.sh .venv-mps-linux experiments/configs/qldpc_six_codes.yaml
```

输出：

- 详细 JSON：`experiments/results/decoder_comparison_*.json`
- 汇总 CSV：`experiments/results/decoder_comparison_*.csv`
- Markdown 汇总：`experiments/results/decoder_comparison_summary.md`

---

## 4. 仅跑内置码（可选）

```bash
bash scripts/linux/run_builtin_sweep.sh .venv-mps-linux
```

默认覆盖：

- `steane`
- `repetition` 距离 3/5
- `surface_code` 距离 3/5

配置文件位于 `experiments/configs/code_sweep.yaml`，可自行追加更多内置码参数组合。

---

## 5. 跑外部矩阵码（自定义 qLDPC）

将你的码数据放到：

```text
experiments/data/cases/<case_name>/
  H.npy
  logical.npy
  noise.npy   # 可选
  meta.json   # 可选
```

然后执行：

```bash
bash scripts/linux/run_matrix_sweep.sh .venv-mps-linux experiments/data/cases experiments/configs/qldpc_six_codes.yaml
```

脚本会按配置文件中的 case 路径运行，并自动扫描 `experiments/data/cases/*` 里的可发现 case。

---

## 6. 对比指标说明

每个 case / 噪声点输出以下核心指标：

- `exact_total_ms`: exact decoder 全部样本总时延
- `mps_total_ms`: MPS decoder 全部样本总时延
- `speedup_mps_over_exact`: 加速比（exact / mps）
- `mean_abs_prob_diff`: 两解码器输出逻辑概率差的平均绝对值
- `max_abs_prob_diff`: 最大绝对差
- `decision_agreement`: 以 `p>=0.5` 判决时的一致率

---

## 7. GPU 资源分析（显存 + 利用率）

脚本使用 `pynvml` 周期采样（默认 50ms），输出：

- `mem_used_peak_mb`, `mem_used_avg_mb`
- `gpu_util_peak`, `gpu_util_avg`
- `mem_util_peak`, `mem_util_avg`

并记录初始化阶段快照差值：

- `exact_init_mem_delta_mb`
- `mps_init_mem_delta_mb`

可用于比较两种解码器的显存开销差异。

---

## 8. 直接运行对比脚本（高级参数）

```bash
source .venv-mps-linux/bin/activate

python experiments/run_decoder_comparison.py \
  --config experiments/configs/qldpc_six_codes.yaml \
  --output-dir experiments/results \
  --matrix-dir experiments/data/cases \
  --shots 256 \
  --noise-values 0.005,0.01,0.02 \
  --bond-dim 16 \
  --low-threshold 0.4 \
  --high-threshold 0.6 \
  --max-rounds 16 \
  --device cuda \
  --dtype float32 \
  --gpu-index 0 \
  --gpu-sample-interval-ms 50 \
  --allow-cpu-fallback
```

仅跑某些 case（正则）：

```bash
python experiments/run_decoder_comparison.py \
  --config experiments/configs/qldpc_six_codes.yaml \
  --case-filter "bb_72|tb_48" \
  --allow-cpu-fallback
```

---

## 9. 结果汇总

```bash
python experiments/summarize_decoder_comparison.py \
  --input experiments/results/decoder_comparison_YYYYMMDDTHHMMSSZ.json \
  --output-md experiments/results/decoder_comparison_summary.md
```

---

## 10. 常见问题

- `ModuleNotFoundError: cudaq_qec`
  - 说明环境未安装成功，重新执行 setup 脚本。
- `tensor_network_mps_decoder` 未生效
  - 确认已执行 `scripts/linux/patch_cudaq_qec_plugins.py`。
- GPU 字段为 `NA`
  - 通常是 `pynvml` 不可用、无 NVIDIA 驱动、或运行在 CPU fallback 路径。
