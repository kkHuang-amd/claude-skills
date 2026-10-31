#!/bin/bash
# Fixed A/B bench for OPT_TRACKER items. Server on :30000 must be up.
# Usage: ab_bench.sh <run-tag>   -> $K31_OUT/results/ab/<run-tag>/summary.txt
# Cases: decode 8192/1024 at c4, c16, c48 (n = 4c) + prefill 8192/1 c1 x4.
# Each case flushes cache and uses a fixed per-case seed, so two runs see identical prompts.
set -u
source /workspace/claude-skills/kimi-k31/env.sh
# cwd inside sglang/benchmark shadows HF `datasets` with sglang.benchmark.datasets
cd /tmp
TAG=${1:?run-tag}; R=$K31_OUT/results/ab/$TAG; mkdir -p "$R"
B="python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --model $K31_MODEL --tokenizer $K31_MODEL --dataset-name random --random-range-ratio 1.0"
flush() { curl -s -m 120 -X POST http://127.0.0.1:30000/flush_cache > /dev/null; sleep 2; }

$B --seed 999 --random-input-len 1024 --random-output-len 64 --num-prompts 8 --max-concurrency 8 > "$R/warmup.log" 2>&1
: > "$R/summary.txt"
for c in ${AB_CONC:-4 16 48}; do
  flush
  $B --seed $((3000+c)) --random-input-len 8192 --random-output-len 1024 --num-prompts $((4*c)) \
     --max-concurrency $c --output-file "$R/decode_c$c.jsonl" > "$R/decode_c$c.log" 2>&1
  python3 - "$R/decode_c$c.log" "decode_c$c" >> "$R/summary.txt" <<'EOF'
import re,sys
t=open(sys.argv[1]).read()
g=lambda k: (re.search(k+r"[^:]*:\s+([0-9.]+)",t) or [None,"nan"])[1]
print(f"{sys.argv[2]:12s} TPOT {g('Mean TPOT')} ms  P90ITL {g('P90 ITL')} ms  TTFT {g('Mean TTFT')} ms  out {g('Output token throughput')} tok/s  acc {g('Accept length')}")
EOF
done
flush
$B --seed 3100 --random-input-len 8192 --random-output-len 1 --num-prompts 4 --max-concurrency 1 \
   --output-file "$R/prefill_8k.jsonl" > "$R/prefill_8k.log" 2>&1
python3 - "$R/prefill_8k.log" >> "$R/summary.txt" <<'EOF'
import re,sys
t=open(sys.argv[1]).read()
m=re.search(r"Mean TTFT[^:]*:\s+([0-9.]+)",t); print(f"prefill_8k   TTFT {m[1] if m else 'nan'} ms")
EOF
cat "$R/summary.txt"
