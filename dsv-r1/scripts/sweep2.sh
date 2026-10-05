#!/bin/bash
# Per-conc server restart with --max-running-requests=conc, num-prompts=conc*8.
D=/shared_nfs/kk/dsr1-mtp-70k
R=$D/run2; mkdir -p $R
PORT=${PORT:-30000}
cd $D
for c in ${CONCS:-4 8 16 32 64}; do
  MRR=$c setsid ./launch.sh > $R/server_c${c}.log 2>&1 &
  SP=$!
  for i in $(seq 1 90); do
    sleep 10
    rg -q 'ready to roll' $R/server_c${c}.log && break
    if rg -q 'Initialization failed|Traceback|error:' $R/server_c${c}.log || ! kill -0 $SP 2>/dev/null; then
      echo "c=$c server_failed"; kill -- -$SP 2>/dev/null; continue 2
    fi
  done
  rg -o 'max_total_num_tokens=[0-9]+|max_running_requests=[0-9]+' $R/server_c${c}.log | tr '\n' ' '; echo
  python3 -m sglang.bench_serving --backend sglang --port $PORT \
    --model /shared_nfs/deepseek-ai/DeepSeek-R1-0528 \
    --dataset-name random --random-input-len 70000 --random-output-len 300 \
    --random-range-ratio 1.0 \
    --max-concurrency $c --num-prompts $((c * 8)) \
    --warmup-requests 2 \
    --output-file $R/c${c}.jsonl > $R/c${c}.log 2>&1
  echo "c=$c exit=$?"
  kill -- -$SP 2>/dev/null
  for i in $(seq 1 30); do
    sleep 5
    [ "$(rocm-smi --showmemuse | rg 'VRAM%' | rg -vc ': 0$')" = "0" ] && break
  done
done
echo SWEEP_DONE
