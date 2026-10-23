#!/usr/bin/env bash
# P0 attribution on MI455: one server (MAX_BS=64), torch-profile 20 decode steps at each concurrency.
# Usage: mi455_p0_profile.sh [run-tag]   (concurrencies via CONCS, default "2 64")
set -u
TAG=${1:-p0_profile_$(hostname -s)_$(date +%m%d_%H%M)}
MODEL=${MODEL:-/workspace/proxy_work/proxy_12l}
CONCS=${CONCS:-"2 64"}
OUT=/shared_nfs/kk/results/proxy_12l/$TAG
PORT=18030
mkdir -p "$OUT"; cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 SGLANG_USE_AITER=1 PYTHONUNBUFFERED=1
export SGLANG_TORCH_PROFILER_DIR=$OUT

python3 -m sglang.launch_server --model-path "$MODEL" \
  --tp-size 8 --ep-size ${EP:-8} --kv-cache-dtype fp8_e4m3 \
  --context-length 32768 --mem-fraction-static 0.80 \
  --max-running-requests 64 --cuda-graph-max-bs-decode 64 \
  --chunked-prefill-size 8192 --attention-backend triton \
  --watchdog-timeout 3600 --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
SPID=$!
echo "server pid=$SPID out=$OUT"
for i in $(seq 1 360); do
  curl -sf -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; grep -E 'Error|Traceback' "$OUT/server.log" | tail -10 | cut -c1-200; exit 1; }
  sleep 10
done
echo "server ready after ~$((i*10))s"

for C in $CONCS; do
  python3 -m sglang.bench_serving --backend sglang \
    --host 127.0.0.1 --port $PORT --tokenizer "$MODEL" \
    --dataset-name random-ids --tokenize-prompt \
    --random-input-len 1024 --random-output-len 1024 --random-range-ratio 1.0 \
    --num-prompts $((C*4)) --max-concurrency $C --warmup-requests $C \
    --request-rate inf > "$OUT/bench_c$C.log" 2>&1 &
  BPID=$!
  sleep ${PROFILE_DELAY:-45}
  # Needs sglang PR #44292 on ROCm 10.1 / torch 2.12: without it the schedulers inherit
  # ROCPROFILER_REGISTER_LIBRARY and the trace has no GPU kernels.
  curl -s -m 600 -X POST http://127.0.0.1:$PORT/start_profile -H 'Content-Type: application/json' \
    -d "{\"output_dir\": \"$OUT/c$C\", \"num_steps\": 20, \"activities\": [\"CPU\", \"GPU\"], \"profile_prefix\": \"c$C\"}" \
    | head -c 300; echo
  wait $BPID
  echo "c=$C bench exit=$?"
  grep -E 'Successful requests|Output token throughput|Median ITL' "$OUT/bench_c$C.log" | cut -c1-120
done

kill $SPID; wait $SPID 2>/dev/null
echo "server stopped"
ls "$OUT"/c*/ 2>/dev/null | head -20
