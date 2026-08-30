#!/usr/bin/env bash
# 2026-08-29 03:5x -- delayer pair moved to CHUNK_PER_RANK 16384.
# Order is the user's: the c128 pair FIRST (known-good territory, the only pair left
# standing from the overnight run), then the three c256 arms that hung at 1x headroom.
# c256 dirs carry a -2x suffix so the hung originals stay on disk as evidence.
#   setsid nohup bash matrix_2x.sh </dev/null >/workspace/results/overnight/driver_2x_b.log 2>&1 &
set -u
SKILL_DIR=/workspace/claude-skills/agentx
OUT=/workspace/results/overnight
STATUS=$OUT/STATUS.txt
mkdir -p $OUT
RESFILE_TPL='dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic-b200align_c%s.json'

# 200 min, not 150: AGENTIC_WARMUP_GRACE_PERIOD=3600 for CONC>=32 means a HEALTHY
# arm can need ~145 min, and the old cap would have logged that as FAILED rc=124.
PER_RUN_TIMEOUT=$((200*60))
# Watchdog: give up on an arm whose scheduler has gone silent this long. All three
# c256 hangs showed hours of silence; a healthy arm's longest gap is the ~14 min
# finalise after the benchmark phase, so 25 min catches hangs with margin.
STALL_MIN=25

# name                 CONC  CHUNK_PER_RANK  GUARD
# Delayer pair moved from chunk 8192 to 16384: 8192 could not certify a run at
# c128 (aiperf coverage 94.1 % < 95 %), and c256 has higher TTFT still, so the
# 8192 delayer arm would likely have come back invalid for a reason unrelated to
# the guard. Guard-OFF runs first so the anchor exists even if only one lands.
# c128 pair and c256-chunk8192-2x are omitted: already attempted by matrix_2x.sh.
# name                    CONC  CHUNK_PER_RANK  GUARD
ARMS=(
  "c256-chunk16384-2x      256  16384  0"
  "c256-chunk16384-gd-2x   256  16384  1"
)

log(){ echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }

killpat(){ local me=$$ p
  for p in $(ps -eo pid,args | grep -E "[s]glang|[a]iperf" | grep -v "bash -c" | awk -v s=$me '$1!=s{print $1}'); do
    kill -9 $p 2>/dev/null
  done; }

kill_and_reclaim(){
  local i v
  for i in 1 2 3 4; do killpat; sleep 5; done
  for i in $(seq 1 120); do
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -E "VRAM%" | awk '{s+=$NF} END{print s+0}')
    [ "${v:-999}" -lt 10 ] && { log "  reclaim done (~$((i*10))s)"; return 0; }
    sleep 10
  done
  log "  WARNING: VRAM still held after 20 min (sum=${v:-?})"
}

log "=== 2x-headroom matrix: ${#ARMS[@]} arms (c128 pair first, then c256 at 512) ==="
idx=0
for arm in "${ARMS[@]}"; do
  idx=$((idx+1))
  read -r NAME CONC CHUNK GUARD <<<"$arm"
  D=$OUT/$NAME
  RES="$D/$(printf "$RESFILE_TPL" "$CONC")"
  if [ -f "$RES" ]; then log "[$idx/${#ARMS[@]}] $NAME  SKIP (result exists)"; continue; fi
  mkdir -p $D
  log "[$idx/${#ARMS[@]}] $NAME  START  conc=$CONC chunk/rank=$CHUNK guard=$GUARD max-running=$((2*CONC))"
  kill_and_reclaim

  timeout $PER_RUN_TIMEOUT env \
      EP_SIZE=1 CONC=$CONC DURATION=3600 CHUNK_PER_RANK=$CHUNK \
      SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=$GUARD RESULT_DIR=$D \
      bash $SKILL_DIR/agentx_b200align.sh > $D/launcher.out 2>&1 &
  lp=$!
  last=""; same=0; seen=0; stalled=0
  while kill -0 $lp 2>/dev/null; do
    sleep 60
    t=$(tail -c 400000 "$D/server.log" 2>/dev/null | grep -E "Prefill batch|Decode batch" | tail -1 | grep -oE "^\[[0-9-]+ [0-9:]+" || true)
    [ -n "$t" ] && seen=1
    if [ "$seen" = 1 ]; then
      if [ "$t" = "$last" ]; then same=$((same+1)); else same=0; fi
      last="$t"
      if [ "$same" -ge "$STALL_MIN" ]; then
        log "  WATCHDOG: scheduler silent ~${STALL_MIN} min (last batch $t) -- abandoning arm early"
        stalled=1; kill -9 $lp 2>/dev/null; killpat; break
      fi
    fi
  done
  wait $lp 2>/dev/null; rc=$?

  if [ -f "$RES" ]; then
    tps=$(python3 -c "import json;print('%.1f'%json.load(open('$RES'))['request_metrics']['throughput']['per_gpu']['total_tput_tps'])" 2>/dev/null)
    log "[$idx/${#ARMS[@]}] $NAME  DONE  rc=$rc  tok/s/GPU=$tps"
  elif [ "$stalled" = 1 ]; then
    log "[$idx/${#ARMS[@]}] $NAME  HUNG (watchdog, scheduler silent) -- continuing"
  else
    log "[$idx/${#ARMS[@]}] $NAME  FAILED rc=$rc (no result json) -- continuing"
  fi
done
kill_and_reclaim
log "=== 2x-headroom matrix finished; GPUs cleared ==="
