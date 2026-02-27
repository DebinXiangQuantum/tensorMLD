# QLDPC FPGA Cycle Simulator 使用说明

本目录实现的是 `algo.tex` 对应的 QLDPC 解码流程的 FPGA cycle-level 近似仿真器。

入口脚本（仓库根目录）：

```bash
python3 fpga_cycle_sim.py [args...]
```

等价入口（包方式）：

```bash
python3 -m qldpc_sim.cli [args...]
```

## 1. 快速示例

随机码 + 随机 syndrome：

```bash
python3 fpga_cycle_sim.py --m 128 --k 32 --chi 16 --parallel-bits 8 --num-codes 20
```

使用真实输入文件：

```bash
python3 fpga_cycle_sim.py \
  --m 128 --k 32 --chi 16 \
  --matrix-file sample_matrix.json \
  --syndrome-file sample_syndrome.txt \
  --num-codes 1
```

输出完整 JSON：

```bash
python3 fpga_cycle_sim.py ... --json
```

估算 U50 资源（LUT/FF/DSP/BRAM）：

```bash
python3 fpga_cycle_sim.py ... --estimate-resource
```

## 2. 参数说明

### 2.1 问题规模与数据生成

- `--m`：syndrome 长度 M（也是 syndrome-side matrix 数量）。
- `--k`：logical bit 数量 K。
- `--chi`：张量 bond 维度。
- `--matrix-density`：随机生成 `matrix_list` 的稀疏密度（仅在未传 `--matrix-file` 时生效）。
- `--logical-density`：随机生成 logical tensor 的密度（仅在未传 `--matrix-file` 或文件里未给 logical_tensors 时生效）。
- `--seed`：基础随机种子。
- `--num-codes`：随机码实例数量；每个实例使用 `seed + idx * 1009`。
- `--syndrome-one-prob`：随机 syndrome 时 bit=1 的概率（仅在未传 `--syndrome/--syndrome-file` 时生效）。
- `--syndrome`：直接传 syndrome 位串（如 `101001...`）。
- `--syndrome-file`：从文件读 syndrome。
- `--matrix-file`：从 JSON 文件读 `matrix_list` 和可选 `logical_tensors`。

### 2.2 并行架构参数（直接影响 cycle）

- `--parallel-bits`：Step2 每批并行处理的 unknown logical bits 数量 N。
- `--selector-width`：syndrome 扫描带宽（每 cycle 扫多少位）。
- `--spmm-pe`：单个 SpMM engine 的并行 MAC lane 数。
- `--spmm-read-bw`：SpMM 读带宽（按 nnz 计）。
- `--spmm-write-bw`：SpMM 写带宽（按 nnz 计）。
- `--spmm-pipeline`：每次 SpMM 固定流水线开销。
- `--spmm-engines`：可并行执行 SpMM/收缩的 engine 数（用于树收缩调度）。
- `--contract-engines`：Step2 中每个 bit/λ 收缩任务的并行 engine 数。
- `--add-bw`：`T0 + T1` 加法吞吐。
- `--add-engines`：Step2 加法并行 engine 数。
- `--chain-select-bw`：构造条件链时的选择带宽（mux 开销）。
- `--trace-bw`：trace 计算带宽。
- `--prob-latency`：条件概率计算延迟。
- `--judge-latency`：判决器延迟。

### 2.3 输出与资源估计

- `--json`：打印完整结果 JSON（包含 breakdown）。
- `--estimate-resource`：启用硬件资源估算。
- `--data-bits`、`--dsp-per-mac`、`--lut-per-mac`、`--ff-per-mac`：资源模型参数，只影响资源估算，不影响 cycle。

## 3. 哪些参数会影响 cycle

### 3.1 强影响（核心）

- `m`：影响 syndrome 扫描成本；也影响可被选中的矩阵总数上限。
- `k`：影响 Step2 每轮任务规模（每个 unknown bit 需要两次条件收缩 λ=0/1，链长约为 K）。
- `chi`：影响矩阵维度与 nnz/MAC 规模，通常显著增大 SpMM 成本。
- `matrix-density`、`logical-density`：通过 nnz 改变 SpMM 的读写与 MAC 数。
- `syndrome`（或 `syndrome-one-prob`）：决定选中矩阵数量与组成，直接影响 Step1 预收缩开销。

### 3.2 架构吞吐/并行参数（直接改变总 cycle）

- `parallel-bits`：每轮处理 unknown bit 的批大小。
- `selector-width`：`selection_scan_cycles = ceil(M / selector_width)`。
- `spmm-pe`、`spmm-read-bw`、`spmm-write-bw`、`spmm-pipeline`：单次 SpMM 周期模型。
- `spmm-engines`、`contract-engines`、`add-engines`：并行调度后的 makespan。
- `add-bw`、`chain-select-bw`、`trace-bw`、`prob-latency`、`judge-latency`：Step2 各子阶段延迟。

### 3.3 仅间接影响或不影响 cycle

- `seed`、`num-codes`：改变实例/统计数量，不改变单个固定实例的公式。
- `--json`：只影响输出格式。
- `--estimate-resource` 及其模型参数：只影响资源估算结果，不改变仿真 cycle。

## 4. cycle 统计项说明（breakdown）

- `selection_scan_cycles`：扫描 syndrome 找 `1` 的周期。
- `selection_fetch_cycles`：按选中矩阵 nnz 估算的数据读取周期。
- `step1_contract_cycles`：Step1 中 syndrome 选中矩阵树收缩周期。
- `step1_attach_cycles`：`T0` 与首个 logical tensor (`0/1`) 连接周期。
- `step2_sum_cycles`：每轮 unknown 位先做 `T0 + T1` 的并行加法周期。
- `step2_batch_cycles`：每轮按 batch 执行条件收缩 + trace + 判决的周期。
- `step2_force_cycles`：若一轮后无进展，强制判决 unknown 的补充周期。
- `spmm_calls`：SpMM 调用次数。
- `spmm_macs`：估算 MAC 总数。

## 5. 输入文件格式

### 5.1 syndrome 文件

支持两类：

- 纯文本：`101100...`（长度必须等于 `--m`）
- JSON：
  - `{"syndrome":[0,1,1,...]}` 或
  - `{"bitstring":"1011..."}` 或
  - 直接数组 `[0,1,1,...]`

### 5.2 matrix 文件（JSON）

最小字段：

- `chi`：必须与 `--chi` 一致
- `matrix_list`：长度必须等于 `--m`

可选字段：

- `logical_tensors`：长度必须等于 `--k`，每个 logical 位需要两张矩阵（0/1）
- 传入 `--matrix-file` 后，`--num-codes` 只会重复仿真同一套矩阵（用于重复统计），不会自动生成新 code 结构。

`matrix_list` 每个矩阵可写为：

- `[[row, col, val], ...]` 或
- `{"entries": [[row,col,val], ...]}` 或
- `{"coo": [[row,col,val], ...]}`

`logical_tensors` 每项可写为：

- `{"zero": [...], "one": [...]}`（或 `t0/t1`, `0/1` 键）
- 或 `[mat0, mat1]`

可参考仓库根目录的 `sample_matrix.json` 与 `sample_syndrome.txt`。

## 6. 常用命令模板

固定真实输入、单次测量：

```bash
python3 fpga_cycle_sim.py \
  --m 128 --k 32 --chi 16 \
  --matrix-file your_matrix.json \
  --syndrome-file your_syndrome.json \
  --num-codes 1 --json
```

扫架构参数（示例：并行度）：

```bash
python3 fpga_cycle_sim.py ... --parallel-bits 4
python3 fpga_cycle_sim.py ... --parallel-bits 8
python3 fpga_cycle_sim.py ... --parallel-bits 16
```

带资源估算：

```bash
python3 fpga_cycle_sim.py ... --estimate-resource
```
