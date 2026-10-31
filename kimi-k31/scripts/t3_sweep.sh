#!/bin/bash
# T3 perf sweep against a running server on :30000. Usage: t3_sweep.sh <run-tag>
# Full per-case logs go to $K31_OUT/results/<run-tag>/; only summary lines are printed.
set -u
source /workspace/claude-skills/kimi-k31/env.sh
cd /tmp
TAG=${1:?run-tag}; R=$K31_OUT/results/$TAG; mkdir -p "$R"
B="python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 \
  --model $K31_MODEL --tokenizer $K31_MODEL --dataset-name random --random-range-ratio 1.0"
# random dataset repeats prompts for a fixed seed -> radix/HiCache hits poison TTFT/prefill.
# Every case flushes the cache and uses its own seed.
flush() { curl -s -m 120 -X POST http://127.0.0.1:30000/flush_cache > /dev/null; sleep 2; }
SEED=1000
PAT='Successful requests|Benchmark duration|Output token throughput|Total token throughput|Mean TTFT|P99 TTFT|Mean TPOT|P90 ITL|Mean ITL|Accept length'

# warmup (first-request JIT)
$B --seed 999 --random-input-len 1024 --random-output-len 64 --num-prompts 8 --max-concurrency 8 > "$R/warmup.log" 2>&1

for c in ${DECODE_CONC:-1 4 8 16 32 48}; do
  n=$(( c*4 > 8 ? c*4 : 8 )); flush; SEED=$((SEED+1))
  $B --seed $SEED --random-input-len 8192 --random-output-len 1024 --num-prompts $n --max-concurrency $c \
     --output-file "$R/decode_c$c.jsonl" > "$R/decode_c$c.log" 2>&1
  echo "== decode c$c exit=$?"; rg "$PAT" "$R/decode_c$c.log" | cut -c1-120
done

for isl in ${PREFILL_ISL:-1024 8192 32768 131072}; do
  flush; SEED=$((SEED+1))
  $B --seed $SEED --random-input-len $isl --random-output-len 1 --num-prompts 4 --max-concurrency 1 \
     --output-file "$R/prefill_isl$isl.jsonl" > "$R/prefill_isl$isl.log" 2>&1
  echo "== prefill isl$isl exit=$?"; rg 'Mean TTFT|Median TTFT|Input token throughput' "$R/prefill_isl$isl.log" | cut -c1-120
done
