#!/bin/bash
# Late-prefill -> short-decode trace, one file per rank (kimi-k3 SGLANG_TRACE_CAPTURE_METHOD).
# Usage: capture_trace.sh <tag> <server_log> [conc] [isl] [osl] [start_after_prefill_batches] [decode_s]
# Server must be launched with SGLANG_PROFILE_V2=0 SGLANG_PROFILE_WITH_STACK=false
# SGLANG_PROFILE_RECORD_SHAPES=false.
set -u
source /workspace/claude-skills/kimi-k31/env.sh
cd /tmp
TAG=$1; SLOG=$2; C=${3:-16}; ISL=${4:-8192}; OSL=${5:-1024}; K=${6:-5}; DS=${7:-2}
T=$K31_OUT/traces/$TAG; mkdir -p "$T"; U=http://127.0.0.1:30000

until rg -q 'ready to roll' "$SLOG"; do rg -q 'Traceback|Initialization failed' "$SLOG" && { echo SERVER-FAILED; exit 1; }; sleep 15; done
echo "server ready"

python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --model $K31_MODEL \
  --dataset-name random --random-range-ratio 1.0 --seed 1999 --random-input-len 1024 --random-output-len 64 \
  --num-prompts 8 --max-concurrency 8 > "$T/warmup.log" 2>&1

base_p=$(rg -c 'TP0\] Prefill batch' "$SLOG"); base_d=$(rg -c 'TP0\] Decode batch' "$SLOG")
python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port 30000 --model $K31_MODEL \
  --dataset-name random --random-range-ratio 1.0 --seed 2001 --random-input-len $ISL --random-output-len $OSL \
  --num-prompts $C --max-concurrency $C --output-file "$T/bench.jsonl" > "$T/bench.log" 2>&1 &
BP=$!

until [ $(( $(rg -c 'TP0\] Prefill batch' "$SLOG") - base_p )) -ge $K ]; do sleep 0.2; done
curl -s -X POST $U/start_profile -H 'Content-Type: application/json' -d "{\"output_dir\":\"$T\",\"activities\":[\"CPU\",\"GPU\"],\"with_stack\":false,\"record_shapes\":false,\"profile_by_stage\":false,\"merge_profiles\":false,\"profile_prefix\":\"$TAG\"}" > "$T/start.resp"
echo "profile started after $K prefill batches"
until [ $(( $(rg -c 'TP0\] Decode batch' "$SLOG") - base_d )) -ge 1 ]; do sleep 0.2; done
sleep $DS
curl -s -X POST $U/stop_profile > "$T/stop.resp"; echo "profile stopped"
wait $BP; echo "bench exit=$?"
rg 'Mean TPOT|Mean TTFT|Output token throughput' "$T/bench.log" | cut -c1-100

sleep 20
n=0; bad=0
for f in "$T"/*.trace.json.gz; do
  [ -f "$f" ] || continue; n=$((n+1))
  gzip -t "$f" 2>/dev/null || { echo "BAD gzip $f"; bad=$((bad+1)); }
  s=$(stat -c %s "$f"); [ $s -gt 524288000 ] && { echo "OVER 500MiB $f"; bad=$((bad+1)); }
done
echo "traces=$n bad=$bad"; du -sh "$T" | cut -f1
