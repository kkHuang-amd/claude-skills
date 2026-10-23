#!/usr/bin/env bash
# One (shape, concurrency) point of the MI455 fixed-seq-len sweep (see proxy-moe/MI455_RUN.md).
# Starts a server sized for CONC, runs bench_serving, kills the server.
# Usage: mi455_point.sh <1k1k|8k1k|70k300> <conc> [run-tag]
set -u
SHAPE=$1; CONC=$2; TAG=${3:-mi455_$(hostname -s)_$(date +%m%d)}
MODEL=${MODEL:-/workspace/proxy_work/proxy_12l}
OUT=/shared_nfs/kk/results/proxy_12l/$TAG
PORT=18030
case $SHAPE in
  1k1k)   ISL=1024;  OSL=1024; CTX=32768;  NP=$((CONC*8)) ;;
  8k1k)   ISL=8192;  OSL=1024; CTX=32768;  NP=$((CONC*8)) ;;
  70k300) ISL=70000; OSL=300;  CTX=131072; NP=$((CONC*4)) ;;
  *) echo "bad shape $SHAPE"; exit 2 ;;
esac
mkdir -p "$OUT"; cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 SGLANG_USE_AITER=1 PYTHONUNBUFFERED=1
NAME=isl$ISL-osl$OSL-c$CONC

python3 -m sglang.launch_server --model-path "$MODEL" \
  --tp-size 8 --ep-size ${EP:-8} --kv-cache-dtype fp8_e4m3 \
  --context-length $CTX --mem-fraction-static 0.80 \
  --max-running-requests $CONC --cuda-graph-max-bs-decode $CONC \
  --chunked-prefill-size 8192 --attention-backend triton \
  --watchdog-timeout 3600 --host 127.0.0.1 --port $PORT > "$OUT/$NAME.server.log" 2>&1 &
SPID=$!
echo "server pid=$SPID log=$OUT/$NAME.server.log"
for i in $(seq 1 360); do
  curl -sf -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; grep -E 'Error|Traceback' "$OUT/$NAME.server.log" | tail -10 | cut -c1-200; exit 1; }
  sleep 10
done
echo "server ready after ~$((i*10))s"
if [ "${SMOKE:-0}" = 1 ]; then  # package correctness check: finite logprobs, short/long/concurrent requests
  python3 ${PKG:-/workspace/proxy_moe_sglang}/tools/smoke_test.py --port $PORT --out "$OUT/$NAME.smoke.json" > "$OUT/$NAME.smoke.log" 2>&1
  echo "smoke exit=$? passed=$(python3 -c "import json;print(json.load(open('$OUT/$NAME.smoke.json')).get('passed'))" 2>&1 | tail -1)"
fi

python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port $PORT --tokenizer "$MODEL" \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
  --num-prompts $NP --max-concurrency $CONC --warmup-requests $((CONC*2)) \
  --request-rate inf \
  --output-file "$OUT/$NAME.jsonl" --output-details > "$OUT/$NAME.bench.log" 2>&1
echo "bench exit=$?"
grep -E 'Successful requests|Output token throughput|Median TTFT|Median ITL|Median TPOT' "$OUT/$NAME.bench.log" | cut -c1-120

kill $SPID; wait $SPID 2>/dev/null
echo "server stopped"
