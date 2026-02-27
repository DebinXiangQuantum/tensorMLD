#!/bin/bash
# Run multi-decoder benchmark on all 9 GND codes.
# Usage: tmux new -d -s bench 'bash experiments/tests/run_benchmark.sh'
set -e
cd ~/tensorMLD
source .venv-mps/bin/activate

LOG_DIR=experiments/results
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/benchmark_$(date +%Y%m%d_%H%M%S).log"

echo "=== Multi-decoder benchmark ===" | tee "$LOGFILE"
echo "Start: $(date)" | tee -a "$LOGFILE"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

python -u -c "
import sys, json, time
sys.path.insert(0, '.')
from experiments.tests.test_multi_decoder_benchmark import run_benchmark

t0 = time.time()
results = run_benchmark(
    shots=500,
    noise_p=0.01,
    chi=32,
    seed=2026,
    skip_missing=False,
    parallel_workers=1,
    verbose=True,
)
elapsed = time.time() - t0

out_path = 'experiments/results/multi_decoder_benchmark.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)

print()
print('='*60)
print(f'Total time: {elapsed:.1f}s')
print(f'Results saved to {out_path}')
print()
print('=== Summary ===')
for r in results:
    print(f'{r[\"code\"]} [[{r[\"N\"]},{r[\"K\"]},{r[\"D\"]}]]:')
    for dname, dinfo in r['decoders'].items():
        if isinstance(dinfo, dict) and dinfo.get('status') == 'ok':
            print(f'  {dname}: block_LER={dinfo[\"logical_error_rate\"]:.4f}  ({dinfo[\"per_shot_ms\"]:.3f} ms/shot)')
        else:
            s = dinfo.get('status','?') if isinstance(dinfo,dict) else '?'
            print(f'  {dname}: {s}')
" 2>&1 | tee -a "$LOGFILE"

echo "" | tee -a "$LOGFILE"
echo "End: $(date)" | tee -a "$LOGFILE"
echo "Log saved to: $LOGFILE"
