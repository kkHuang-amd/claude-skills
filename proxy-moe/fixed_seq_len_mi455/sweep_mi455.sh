#!/usr/bin/env bash
# Proxy-MoE fixed_seq_len sweep on MI455 (gfx1250, h13-17, 12-layer proxy, TP8), one server per point.
# Same client/report as ../fixed_seq_len (InferenceX-style); MI455 best-config env per recipe.
#   SHAPES    : "1024/1024 8192/1024 70000/300"
#   CONCS     : 1k/1k and 8k/1k (default 2..64);  CONCS_70K: 70000/300 (default 2..32)
#   RECIPES   : "tp8ep1 tp8ep8"
#   FUSED_PREP: 1 = SGLANG_OPT_LAGUNA_FUSED_QK_NORM_ROPE_KV=1 (default set below)
# Resumable: a point with DONE is skipped. Teardown kills our server by PID and waits for VRAM to drain.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT=${OUT:-/shared_nfs/kk/results/proxy_12l/fixed_seq_len_mi455_h13-17}
PORT=${PORT:-30000}
SHAPES=${SHAPES:-"1024/1024 8192/1024 70000/300"}
CONCS=${CONCS:-"2 4 8 16 32 64"}
CONCS_70K=${CONCS_70K:-"2 4 8 16 32"}
RECIPES=${RECIPES:-"tp8ep1 tp8ep8"}
FUSED_PREP=${FUSED_PREP:-1}
EP1_CSV=${EP1_CSV:-/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv}
EP8_CSV=${EP8_CSV:-/workspace/proxy_work/ep8_grouped_fmoe_gfx1250_v1.csv}
mkdir -p "$OUT"

# Shared MI455 best config: RCCL 2.32.3 (node-local swap) with CUMEM and no DDA, FlyDSL fp32 router,
# aiter custom AR for messages <= 1 MiB.
export NCCL_CUMEM_ENABLE=1 RCCL_DDA_ENABLE=0 SGLANG_OPT_LAGUNA_ROUTER_FLYDSL=1 AITER_CUSTOM_AR_MAX_SIZE=1048576
export SGLANG_OPT_LAGUNA_FUSED_QK_NORM_ROPE_KV=$FUSED_PREP

recipe_env() {  # recipe -> "TP EP" plus the tuned grouped MoE CSV for that EP size
  case "$1" in
    tp8ep1) export AITER_CONFIG_GROUPED_FMOE=$EP1_CSV; echo "8 1" ;;
    tp8ep8) export AITER_CONFIG_GROUPED_FMOE=$EP8_CSV; echo "8 8" ;;
    *) echo "unknown recipe $1" >&2; return 1 ;;
  esac
}

vram_busy() {  # GPUs (even = socket readout) using more than 5% VRAM
  amd-smi metric --mem-usage 2> /dev/null | awk '/TOTAL_VRAM:/{t=$2} /^ *USED_VRAM:/{if (t && $2/t > 0.05) c++} END{print c+0}'
}

teardown() {
  local pid="$1"
  [ -n "$pid" ] && kill "$pid" 2> /dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2> /dev/null || break; sleep 2; done
  kill -9 "$pid" 2> /dev/null
  local left; left=$(ps -eo pid,args | grep -E '[s]glang::' | awk '{print $1}')
  [ -n "$left" ] && { echo "  leftover sglang pids: $left -> KILL"; kill -9 $left 2> /dev/null; sleep 5; }
  for _ in $(seq 1 60); do [ "$(vram_busy)" -eq 0 ] && break; sleep 5; done
}

for RECIPE in $RECIPES; do
  # Not a subshell: recipe_env exports the MoE CSV for this recipe.
  recipe_env "$RECIPE" > /tmp/.recipe_tpep || exit 1
  read -r TP EP < /tmp/.recipe_tpep
  for SHAPE in $SHAPES; do
    ISL=${SHAPE%/*}; OSL=${SHAPE#*/}
    if [ "$ISL" -ge 32768 ]; then CL=$CONCS_70K; MULT=4; else CL=$CONCS; MULT=8; fi
    for CONC in $CL; do
      TAG="isl${ISL}-osl${OSL}-c${CONC}"
      RD="$OUT/$RECIPE/$TAG"; mkdir -p "$RD"
      NP=$((CONC * MULT)); WU=$((CONC * 2))
      echo "=== $(date +%H:%M:%S) $RECIPE $TAG (np=$NP warmup=$WU) ==="
      if [ -f "$RD/DONE" ]; then echo "  already done, skip"; continue; fi
      TP=$TP EP=$EP CONC=$CONC PORT=$PORT SERVER_LOG="$RD/server.log" SERVER_PID_FILE="$RD/server.pid" \
        bash "$HERE/serve_mi455.sh" > "$RD/launch.log" 2>&1
      rc=$?; SPID=$(cat "$RD/server.pid" 2> /dev/null)
      if [ $rc -ne 0 ]; then echo "  SERVER FAILED (rc=$rc) - see $RD/server.log"; teardown "$SPID"; continue; fi
      echo "  server up (pid $SPID), client"
      ( cd /tmp && python3 -m sglang.bench_serving --backend sglang \
          --host 127.0.0.1 --port "$PORT" --tokenizer /workspace/proxy_work/proxy_12l \
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
print(f"  -> completed {d['completed']}  TTT/GPU {d['total_throughput'] / 8:.1f} tok/s  "
      f"TTFT med {d['median_ttft_ms']:.0f} ms  ITL med {d['median_itl_ms']:.2f} p90 {d['p90_itl_ms']:.2f} ms")
PY
        touch "$RD/DONE"
      fi
      teardown "$SPID"
    done
  done
done
echo "=== sweep complete: $OUT  (report: python3 $HERE/../fixed_seq_len/report.py $OUT --md) ==="
