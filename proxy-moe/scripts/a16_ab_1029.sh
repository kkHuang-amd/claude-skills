#!/usr/bin/env bash
# a16w16 decode-config A/B (h13-17 1029): arms alternate base (the 3 tuned gluon a16w16 jsons moved aside) / tuned.
# Based on gr_ab_1028.sh. A trap always restores the jsons.
# One server start per arm, runs every CONCS point + smoke. Usage: a16_ab_1029.sh [run-tag]  (EPS=1 CONCS="2 16" default)
set -u
TAG=${1:-allon_tp8_$(hostname -s)_$(date +%m%d)}
MODEL=/workspace/proxy_work/proxy_12l
OUT=/shared_nfs/kk/results/proxy_12l/$TAG
PORT=18030
CONCS=${CONCS:-"2 16"}
EPS=${EPS:-"1"}
ISL=${ISL:-1024}; OSL=${OSL:-1024}; CTX=32768
MAXC=$(echo $CONCS | tr ' ' '\n' | sort -n | tail -1)
mkdir -p "$OUT"; cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 PYTHONUNBUFFERED=1
set -a; . ${ENVF:-/workspace/proxy_work/allon_tp8.env}; set +a
CFG=/sgl-workspace/aiter/aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16
JS="GEMM-A16W16-N=2304-K=16384.json GEMM-A16W16-N=16384-K=2048.json GEMM-A16W16-N=25256-K=16384.json"
BK=/tmp/a16_ab_backup; mkdir -p $BK
restore() { for j in $JS; do [ -f $BK/$j ] && mv $BK/$j $CFG/$j; done; }
trap restore EXIT
V3=/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv; V4=/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v4.csv
base_on() { for j in $JS; do mv $CFG/$j $BK/$j; done; }

run_arm() {  # $1=EP $2=arm-name $3=extra env
  local EP=$1 ARM=$2 EXTRA=$3 CSV="" NAME
  [ "$EP" = 1 ] && CSV="AITER_CONFIG_GROUPED_FMOE=${CSVF:-/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv}"
  NAME=ep${EP}_$ARM
  echo "== $NAME $(date +%H:%M:%S)"
  env $CSV $EXTRA python3 -m sglang.launch_server --model-path "$MODEL" \
    --tp-size 8 --ep-size $EP --kv-cache-dtype fp8_e4m3 \
    --context-length $CTX --mem-fraction-static 0.80 \
    --max-running-requests $MAXC --cuda-graph-max-bs-decode $MAXC \
    --chunked-prefill-size 8192 --attention-backend ${ATTN:-aiter} ${EXTRA_ARGS:---page-size 64} \
    --watchdog-timeout 3600 --host 127.0.0.1 --port $PORT > "$OUT/$NAME.server.log" 2>&1 &
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
  for ARM in ${ARMS:-base1 a161 all1 base2 a162 all2}; do
    # base = v3 CSV + DEFAULT a16w16; a16 = v3 + tuned a16w16 jsons; all = EP1 CSV v4 (tile_m 16) + tuned jsons
    case $ARM in base*) base_on; CSVF=$V3;; a16*) restore; CSVF=$V3;; all*) restore; CSVF=$V4;; esac
    echo "arm $ARM jsons present: $(ls $CFG | grep -cE '2304-K=16384|16384-K=2048|25256-K=16384')"
    run_arm $EP $ARM ""
    restore
  done
done
echo "ALL DONE $(date +%H:%M:%S)"
