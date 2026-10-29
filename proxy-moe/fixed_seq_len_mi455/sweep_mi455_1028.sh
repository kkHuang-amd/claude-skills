#!/usr/bin/env bash
# Overnight fixed_seq_len sweep on MI455 (h13-17, 2026-10-28), 12-layer proxy, TP8, one server per point.
#   MODE=new : current best config ("all-on": allon_tp8.env incl. vectorized_5d KV, aiter attention, page 64, DDA on,
#              EP1 = v3 CSV + prefill tile rule, EP8 = prefill tile rule only, local aiter defaults incl. gather EP compact)
#   MODE=old : the 2026-10-23 sweep config (triton attention, DDA off, v3 / ep8 v1 CSVs, router flydsl + fused qk only),
#              with today's new local aiter defaults switched off (tile rule, gather EP compact) to approximate 10-23
# Same points / client as sweep_mi455.sh. Unattended hardening: client timeout, up to MAXTRY attempts per point,
# failures appended to $OUT/FAILED.txt and the sweep moves on; resumable via DONE markers.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE=${MODE:?MODE=new|old}
OUT=${OUT:?OUT required}
PORT=${PORT:-30000}
SHAPES=${SHAPES:-"1024/1024 8192/1024 70000/300"}
CONCS=${CONCS:-"2 4 8 16 32 64"}
CONCS_70K=${CONCS_70K:-"2 4 8 16 32"}
RECIPES=${RECIPES:-"tp8ep1 tp8ep8"}
MAXTRY=${MAXTRY:-3}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-3600}
EP1_CSV=${EP1_CSV:-/workspace/proxy_work/ep1_grouped_fmoe_gfx1250_v3.csv}
EP8_CSV_OLD=/workspace/proxy_work/ep8_grouped_fmoe_gfx1250_v1.csv
mkdir -p "$OUT"

# Start from a clean env for the knobs both modes touch.
unset $(env | grep -oE '^(SGLANG_OPT_[A-Z0-9_]*|SGLANG_AITER_[A-Z0-9_]*|AITER_[A-Z0-9_]*|RCCL_[A-Z0-9_]*|NCCL_CUMEM_ENABLE)=' | tr -d '=') 2> /dev/null
if [ "$MODE" = new ]; then
  set -a; . /workspace/proxy_work/allon_tp8.env; set +a
  export ATTN=aiter EXTRA_ARGS="--page-size 64"
elif [ "$MODE" = old ]; then
  export NCCL_CUMEM_ENABLE=1 RCCL_DDA_ENABLE=0 SGLANG_OPT_LAGUNA_ROUTER_FLYDSL=1 AITER_CUSTOM_AR_MAX_SIZE=1048576
  export SGLANG_OPT_LAGUNA_FUSED_QK_NORM_ROPE_KV=1
  export AITER_TDM_PREFILL_TILE_RULE=0 AITER_GR_EP_COMPACT=0
  export ATTN=triton EXTRA_ARGS=""
else
  echo "MODE must be new or old" >&2; exit 1
fi

recipe_env() {
  case "$1" in
    tp8ep1) export AITER_CONFIG_GROUPED_FMOE=$EP1_CSV; echo "8 1" ;;
    tp8ep8) if [ "$MODE" = old ]; then export AITER_CONFIG_GROUPED_FMOE=$EP8_CSV_OLD; else unset AITER_CONFIG_GROUPED_FMOE; fi
            echo "8 8" ;;
    *) echo "unknown recipe $1" >&2; return 1 ;;
  esac
}

vram_busy() {
  amd-smi metric --mem-usage 2> /dev/null | awk '/TOTAL_VRAM:/{t=$2} /^ *USED_VRAM:/{if (t && $2/t > 0.05) c++} END{print c+0}'
}

teardown() {
  local pid="$1"
  [ -n "$pid" ] && kill "$pid" 2> /dev/null
  for _ in $(seq 1 60); do kill -0 "$pid" 2> /dev/null || break; sleep 2; done
  [ -n "$pid" ] && kill -9 "$pid" 2> /dev/null
  local left; left=$(ps -eo pid,args | grep -E '[s]glang::|[s]glang.launch_server' | awk '{print $1}')
  [ -n "$left" ] && { echo "  leftover sglang pids: $(echo $left) -> KILL"; kill -9 $left 2> /dev/null; sleep 5; }
  for _ in $(seq 1 60); do [ "$(vram_busy)" -eq 0 ] && break; sleep 5; done
  [ "$(vram_busy)" -eq 0 ] || echo "  WARNING: VRAM still busy after teardown"
}

echo "=== $(date '+%F %T') MODE=$MODE OUT=$OUT"
env | grep -E '^(NCCL_|RCCL_|SGLANG_|AITER_)|^ATTN=|^EXTRA_ARGS=' | sort | sed 's/^/  env /'
for RECIPE in $RECIPES; do
  recipe_env "$RECIPE" > /tmp/.recipe_tpep_$MODE || exit 1
  read -r TP EP < /tmp/.recipe_tpep_$MODE
  for SHAPE in $SHAPES; do
    ISL=${SHAPE%/*}; OSL=${SHAPE#*/}
    if [ "$ISL" -ge 32768 ]; then CL=$CONCS_70K; MULT=4; else CL=$CONCS; MULT=8; fi
    for CONC in $CL; do
      TAG="isl${ISL}-osl${OSL}-c${CONC}"
      bash /workspace/claude-skills/proxy-moe/scripts/check_clk_lock.sh > /dev/null 2>&1 || { echo "CLOCK NOT LOCKED before $RECIPE $TAG, abort"; exit 3; }
      RD="$OUT/$RECIPE/$TAG"; mkdir -p "$RD"
      NP=$((CONC * MULT)); WU=$((CONC * 2))
      echo "=== $(date +%H:%M:%S) $RECIPE $TAG (np=$NP warmup=$WU) ==="
      if [ -f "$RD/DONE" ]; then echo "  already done, skip"; continue; fi
      for TRY in $(seq 1 "$MAXTRY"); do
        [ "$TRY" -gt 1 ] && { echo "  retry $TRY/$MAXTRY"; for f in server.log launch.log client.log result.jsonl; do [ -f "$RD/$f" ] && mv "$RD/$f" "$RD/$f.try$((TRY - 1))"; done; }
        TP=$TP EP=$EP CONC=$CONC PORT=$PORT SERVER_LOG="$RD/server.log" SERVER_PID_FILE="$RD/server.pid" \
          bash "$HERE/serve_mi455.sh" > "$RD/launch.log" 2>&1
        rc=$?; SPID=$(cat "$RD/server.pid" 2> /dev/null)
        if [ $rc -ne 0 ]; then
          echo "  SERVER FAILED (rc=$rc) try $TRY: $(grep -m1 -E 'Error|error:|Traceback|fault' "$RD/server.log" | cut -c1-160)"
          teardown "$SPID"; continue
        fi
        echo "  server up (pid $SPID), client"
        ( cd /tmp && timeout "$CLIENT_TIMEOUT" python3 -m sglang.bench_serving --backend sglang \
            --host 127.0.0.1 --port "$PORT" --tokenizer /workspace/proxy_work/proxy_12l \
            --dataset-name random-ids --tokenize-prompt \
            --random-input-len "$ISL" --random-output-len "$OSL" --random-range-ratio 1.0 \
            --num-prompts "$NP" --max-concurrency "$CONC" --warmup-requests "$WU" \
            --request-rate inf --output-file "$RD/result.jsonl" --output-details ) > "$RD/client.log" 2>&1
        crc=$?
        if [ $crc -eq 0 ] && [ -s "$RD/result.jsonl" ] && python3 - "$RD/result.jsonl" "$NP" <<'PY'
import json, sys
d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
ok = d["completed"] == int(sys.argv[2])
print(f"  -> completed {d['completed']}/{sys.argv[2]}  TTT/GPU {d['total_throughput'] / 8:.1f} tok/s  "
      f"TTFT med {d['median_ttft_ms']:.0f} ms  ITL med {d['median_itl_ms']:.2f} p90 {d['p90_itl_ms']:.2f} ms")
sys.exit(0 if ok else 3)
PY
        then
          touch "$RD/DONE"; teardown "$SPID"; break
        fi
        echo "  CLIENT FAILED (rc=$crc) try $TRY: $(grep -m1 -E 'Error|error|Traceback' "$RD/client.log" | cut -c1-160)"
        teardown "$SPID"
      done
      [ -f "$RD/DONE" ] || echo "$(date '+%F %T') $RECIPE $TAG failed after $MAXTRY tries" >> "$OUT/FAILED.txt"
    done
  done
done
echo "=== $(date '+%F %T') sweep MODE=$MODE complete: $(find "$OUT" -name DONE | wc -l) points done, failed: $(cat "$OUT/FAILED.txt" 2> /dev/null | wc -l)"
