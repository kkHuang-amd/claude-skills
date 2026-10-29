#!/usr/bin/env bash
# --enable-aiter-allreduce-fusion A/B (h13-17 1029): arms alternate fusion on/off per EP. Based on gr_ab_1028.sh.
# One server start per arm, runs every CONCS point on it + smoke. Usage: ar_fusion_ab_1029.sh [run-tag]
set -u
TAG=${1:-allon_tp8_$(hostname -s)_$(date +%m%d)}
MODEL=/workspace/proxy_work/proxy_12l
OUT=/shared_nfs/kk/results/proxy_12l/$TAG
PORT=18030
CONCS=${CONCS:-"2 16"}
EPS=${EPS:-"1 8"}
ISL=8192; OSL=1024; CTX=32768
MAXC=$(echo $CONCS | tr ' ' '\n' | sort -n | tail -1)
mkdir -p "$OUT"; cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 PYTHONUNBUFFERED=1
set -a; . ${ENVF:-/workspace/proxy_work/allon_tp8.env}; set +a
TILES="AITER_TDM_TILE_M=128 AITER_TDM_TILE_N=256 AITER_TDM_TILE_K=128 AITER_TDM_NUM_BUFFERS=4 \
AITER_TDM_TILE_M2=128 AITER_TDM_TILE_N2=256 AITER_TDM_TILE_K2=256 AITER_TDM_NUM_BUFFERS2=2"

run_arm() {  # $1=EP $2=arm-name $3=extra server args
  local EP=$1 ARM=$2 SARGS=$3 EXTRA="" CSV="" NAME
  [ "$EP" = 1 ] && CSV="AITER_CONFIG_GROUPED_FMOE=/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv"
  NAME=ep${EP}_$ARM
  echo "== $NAME $(date +%H:%M:%S)"
  env $CSV $EXTRA python3 -m sglang.launch_server --model-path "$MODEL" \
    --tp-size 8 --ep-size $EP --kv-cache-dtype fp8_e4m3 \
    --context-length $CTX --mem-fraction-static 0.80 \
    --max-running-requests $MAXC --cuda-graph-max-bs-decode $MAXC \
    --chunked-prefill-size 8192 --attention-backend ${ATTN:-aiter} ${EXTRA_ARGS:---page-size 64} \
    $SARGS --watchdog-timeout 3600 --host 127.0.0.1 --port $PORT > "$OUT/$NAME.server.log" 2>&1 &
  local SPID=$! i
  echo "server pid=$SPID"
  for i in $(seq 1 360); do
    curl -sf -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null && break
    kill -0 $SPID 2>/dev/null || { echo "server died"; grep -E 'Error|Traceback' "$OUT/$NAME.server.log" | tail -5 | cut -c1-200; return 1; }
    sleep 10
  done
  echo "ready ~$((i*10))s"
  python3 /workspace/proxy_moe_sglang/tools/smoke_test.py --port $PORT --out "$OUT/$NAME.smoke.json" > "$OUT/$NAME.smoke.log" 2>&1
  echo "smoke passed=$(python3 -c "import json;print(json.load(open('$OUT/$NAME.smoke.json')).get('passed'))" 2>&1 | tail -1)"
  for C in $CONCS; do
    python3 -m sglang.bench_serving --backend sglang --host 127.0.0.1 --port $PORT --tokenizer "$MODEL" \
      --dataset-name random-ids --tokenize-prompt \
      --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
      --num-prompts $((C*8)) --max-concurrency $C --warmup-requests $((C*2)) --request-rate inf \
      --output-file "$OUT/$NAME.c$C.jsonl" > "$OUT/$NAME.c$C.bench.log" 2>&1
    echo "c=$C exit=$? $(grep -E 'Successful requests|Mean TTFT|Median TTFT|Median ITL|Output token throughput' "$OUT/$NAME.c$C.bench.log" | tr -s ' ' | cut -d: -f2 | tr '\n' ' ')"
  done
  kill $SPID; wait $SPID 2>/dev/null; sleep 15
}

echo "out=$OUT"
for EP in $EPS; do
  for ARM in ${ARMS:-on1 off1 on2 off2}; do
    case $ARM in on*) run_arm $EP arf_$ARM "--enable-aiter-allreduce-fusion" ;; *) run_arm $EP arf_$ARM "" ;; esac
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
