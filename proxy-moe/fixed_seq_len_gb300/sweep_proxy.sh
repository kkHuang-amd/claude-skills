#!/usr/bin/env bash
# Proxy-MoE fixed_seq_len sweep on GB300 (InferenceX-style), one server per (shape, CONC) point.
# Same shapes / CONC / client as ../fixed_seq_len/sweep_proxy.sh (MI355X); only the platform differs.
#   SHAPES  : "1024/1024 8192/1024 70000/300"
#   CONCS   : 2..1024 for 1k/1k and 8k/1k; 2..32 for 70000/300 (override CONCS / CONCS_70K)
#   RECIPES : "tp4ep1" (12 layers on GPUs 0-3; one GB300 tray has 4 GPUs, 24L does not fit)
#   client  : sglang.bench_serving --backend sglang, random-ids + --tokenize-prompt,
#             range-ratio 1.0, np = CONC*8 (CONC*4 for ISL >= 32768), warmup = CONC*2,
#             request-rate inf, ignore_eos (default)
# Resumable: a point with DONE is skipped. Teardown kills our server by PID and waits for VRAM to drain.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-/shared_nfs/kk/results/proxy_12l/fixed_seq_len_gb300}
PORT=${PORT:-30000}
SHAPES=${SHAPES:-"1024/1024 8192/1024 70000/300"}
CONCS=${CONCS:-"2 4 8 16 32 64 128 256 512 1024"}
CONCS_70K=${CONCS_70K:-"2 4 8 16 32"}
RECIPES=${RECIPES:-"tp4ep1"}
NP_MULT=${NP_MULT:-}          # override np = CONC * NP_MULT (smoke tests)
mkdir -p "$OUT"

recipe_env() {  # recipe -> "TP EP MODEL GPUS"
  local m12=${MODEL_12L:-/sgl-workspace/proxy_moe_work/proxy_12l}
  case "$1" in
    tp4ep1) echo "4 1 $m12 0,1,2,3" ;; tp4ep4) echo "4 4 $m12 0,1,2,3" ;;
    *) echo "unknown recipe $1" >&2; return 1 ;;
  esac
}

teardown() {
  local pid="$1"
  [ -n "$pid" ] && kill "$pid" 2> /dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2> /dev/null || break; sleep 2; done
  kill -9 "$pid" 2> /dev/null
  # setproctitle rewrites argv (sglang::scheduler_TPn) - kill leftovers of OUR sweep only
  local left; left=$(ps -eo pid,args | grep -E '[s]glang::' | awk '{print $1}')
  [ -n "$left" ] && { echo "  leftover sglang pids: $left -> KILL"; kill -9 $left 2> /dev/null; sleep 5; }
  for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2> /dev/null | awk '$1>2048{c++}END{print c+0}')
    [ "${busy:-0}" -eq 0 ] && break; sleep 5
  done
}

for RECIPE in $RECIPES; do
  read -r TP EP MODEL GPUS <<< "$(recipe_env "$RECIPE")" || exit 1
  for SHAPE in $SHAPES; do
    ISL=${SHAPE%/*}; OSL=${SHAPE#*/}
    if [ "$ISL" -ge 32768 ]; then CL=$CONCS_70K; MULT=${NP_MULT:-4}; else CL=$CONCS; MULT=${NP_MULT:-8}; fi
    for CONC in $CL; do
      TAG="isl${ISL}-osl${OSL}-c${CONC}"
      RD="$OUT/$RECIPE/$TAG"; mkdir -p "$RD"
      NP=$((CONC * MULT)); WU=$((CONC * 2))
      echo "=== $(date +%H:%M:%S) $RECIPE $TAG (np=$NP warmup=$WU) ==="
      if [ -f "$RD/DONE" ]; then echo "  already done, skip"; continue; fi

      MODEL=$MODEL GPUS=$GPUS TP=$TP EP=$EP CONC=$CONC PORT=$PORT \
        SERVER_LOG="$RD/server.log" SERVER_PID_FILE="$RD/server.pid" \
        bash "$HERE/serve_proxy.sh" > "$RD/launch.log" 2>&1
      rc=$?; SPID=$(cat "$RD/server.pid" 2> /dev/null)
      if [ $rc -ne 0 ]; then
        echo "  SERVER FAILED (rc=$rc) - see $RD/server.log"; teardown "$SPID"; continue
      fi
      echo "  server up (pid $SPID), client"
      ( cd /tmp && python3 -m sglang.bench_serving --backend sglang \
          --host 127.0.0.1 --port "$PORT" --tokenizer "$MODEL" \
          --dataset-name random-ids --tokenize-prompt \
          --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
          --num-prompts "$NP" --max-concurrency "$CONC" --warmup-requests "$WU" \
          --request-rate inf --output-file "$RD/result.jsonl" --output-details ) > "$RD/client.log" 2>&1
      crc=$?
      if [ $crc -ne 0 ] || [ ! -s "$RD/result.jsonl" ]; then
        echo "  CLIENT FAILED (rc=$crc) - see $RD/client.log"
      else
        python3 - "$RD/result.jsonl" <<'PY'
import json, sys
d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
print(f"  -> completed {d['completed']}  TTT {d['total_throughput']:.0f} tok/s  "
      f"TTFT med {d['median_ttft_ms']:.0f} ms  ITL med {d['median_itl_ms']:.2f} ms")
PY
        touch "$RD/DONE"
      fi
      teardown "$SPID"
    done
  done
done
echo "=== sweep complete: $OUT  (report: python3 $HERE/../fixed_seq_len/report.py $OUT) ==="
