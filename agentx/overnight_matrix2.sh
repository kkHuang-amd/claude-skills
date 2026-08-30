#!/usr/bin/env bash
# Continuation driver, written 2026-08-28 20:40 after BOTH c256 arms hung the
# server in warmup (see SKILL.md "Matrix results as they land"). Runs only the
# c128 pair -- the user's own arms 5 and 6 -- and drops arm 4 (c256-chunk16384),
# which is the same configuration that has now failed twice.
#   setsid nohup bash overnight_matrix2.sh </dev/null >/workspace/results/overnight/driver2.log 2>&1 &
set -u
SKILL_DIR=/workspace/claude-skills/agentx
OUT=/workspace/results/overnight
STATUS=$OUT/STATUS.txt
mkdir -p $OUT
RESFILE_TPL='dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic-b200align_c%s.json'

# 200 min, NOT the original 150. The launcher sets AGENTIC_WARMUP_GRACE_PERIOD=3600
# for CONC>=32, so a HEALTHY arm may use 3600 s warmup + 3600 s benchmark + ~25 min
# startup = ~145 min, which the old 150 min cap would have killed as a "failure".
PER_RUN_TIMEOUT=$((200*60))

# name              CONC  CHUNK_PER_RANK  GUARD
ARMS=(
  "c128-chunk8192    128   8192  0"
  "c128-chunk16384   128  16384  0"
)

log(){ echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }

# As the original, plus [a]iperf: the aiperf controller/managers do NOT match
# "sglang" and survived the original routine, which would leave a second client
# alive against the next arm's server.
kill_and_reclaim(){
  local me=$$
  for i in 1 2 3 4; do
    for p in $(ps -eo pid,args | grep -E "[s]glang|[a]iperf" | grep -v "bash -c" | awk -v s=$me '$1!=s{print $1}'); do
      kill -9 $p 2>/dev/null
    done
    sleep 5
  done
  for i in $(seq 1 120); do
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -E "VRAM%" | awk '{s+=$NF} END{print s+0}')
    [ "${v:-999}" -lt 10 ] && { log "  reclaim done (~$((i*10))s)"; return 0; }
    sleep 10
  done
  log "  WARNING: VRAM still held after 20 min (sum=${v:-?})"
}

log "=== continuation driver: c128 pair only (c256 arms dropped, server hangs) ==="
idx=0
for arm in "${ARMS[@]}"; do
  idx=$((idx+1))
  read -r NAME CONC CHUNK GUARD <<<"$arm"
  D=$OUT/$NAME
  RES="$D/$(printf "$RESFILE_TPL" "$CONC")"
  if [ -f "$RES" ]; then log "[$idx/${#ARMS[@]}] $NAME  SKIP (result exists)"; continue; fi
  mkdir -p $D
  log "[$idx/${#ARMS[@]}] $NAME  START  conc=$CONC chunk/rank=$CHUNK guard=$GUARD"
  kill_and_reclaim
  timeout $PER_RUN_TIMEOUT env \
      EP_SIZE=1 CONC=$CONC DURATION=3600 CHUNK_PER_RANK=$CHUNK \
      SGLANG_PREFILL_DELAYER_MIXED_SLOT_GUARD=$GUARD RESULT_DIR=$D \
      bash $SKILL_DIR/agentx_b200align.sh > $D/launcher.out 2>&1
  rc=$?
  if [ -f "$RES" ]; then
    tps=$(python3 -c "import json;print('%.1f'%json.load(open('$RES'))['request_metrics']['throughput']['per_gpu']['total_tput_tps'])" 2>/dev/null)
    log "[$idx/${#ARMS[@]}] $NAME  DONE  rc=$rc  tok/s/GPU=$tps"
  else
    log "[$idx/${#ARMS[@]}] $NAME  FAILED rc=$rc (no result json) -- continuing"
  fi
done
kill_and_reclaim
log "=== continuation driver finished; GPUs cleared ==="
