#!/bin/bash
# Fixed-length ISL=70000 OSL=300, conc 4..64. Per-case log + jsonl; prints only summary lines.
D=/shared_nfs/kk/dsr1-mtp-70k
PORT=${PORT:-30000}
for c in ${CONCS:-4 8 16 32 64}; do
  python3 -m sglang.bench_serving --backend sglang --port $PORT \
    --model /shared_nfs/deepseek-ai/DeepSeek-R1-0528 \
    --dataset-name random --random-input-len 70000 --random-output-len 300 \
    --random-range-ratio 1.0 \
    --max-concurrency $c --num-prompts $((c * 4)) \
    --warmup-requests 2 \
    --output-file $D/c${c}.jsonl > $D/c${c}.log 2>&1
  echo "c=$c exit=$?"
done
