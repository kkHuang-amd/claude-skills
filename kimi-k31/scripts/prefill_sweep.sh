#!/bin/bash
# Prefill TTFT sweep (c1, 4 prompts per ISL, output 1 token) against a server on :30000.
# Usage: [ISLS="4096 8192 ..."] prefill_sweep.sh <tag> -> $K31_OUT/results/prefill/<tag>/summary.txt
set -u
source /workspace/claude-skills/kimi-k31/env.sh
cd /tmp
TAG=${1:?tag}; R=$K31_OUT/results/prefill/$TAG; mkdir -p "$R"
B="python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --model $K31_MODEL --tokenizer $K31_MODEL --dataset-name random --random-range-ratio 1.0"
flush() { curl -s -m 120 -X POST http://127.0.0.1:30000/flush_cache > /dev/null; sleep 2; }
$B --seed 999 --random-input-len 1024 --random-output-len 1 --num-prompts 4 --max-concurrency 1 > "$R/warmup.log" 2>&1
: > "$R/summary.txt"
for isl in ${ISLS:-4096 8192 16384 32768 131072}; do
  flush
  $B --seed $((5000 + isl % 997)) --random-input-len $isl --random-output-len 1 --num-prompts 4 \
     --max-concurrency 1 --output-file "$R/isl$isl.jsonl" > "$R/isl$isl.log" 2>&1
  python3 - "$R/isl$isl.log" "$isl" >> "$R/summary.txt" <<'EOF'
import re, sys
t = open(sys.argv[1]).read()
g = lambda k: (re.search(k + r"[^:]*:\s+([0-9.]+)", t) or [None, "nan"])[1]
print(f"isl{sys.argv[2]:>7}  TTFT mean {g('Mean TTFT')} ms  median {g('Median TTFT')} ms  in {g('Input token throughput')} tok/s")
EOF
done
cat "$R/summary.txt"
