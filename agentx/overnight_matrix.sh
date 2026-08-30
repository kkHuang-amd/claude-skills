#!/usr/bin/env bash
# Overnight 3600 s matrix. Sequential, resumable, unattended.
# Start with:  setsid nohup bash /workspace/claude-skills/agentx/overnight_matrix.sh </dev/null >/workspace/results/overnight/driver.log 2>&1 &
# Arms are ordered so that every COMPLETED arm answers a question on its own.
set -u
SKILL_DIR=/workspace/claude-skills/agentx
OUT=/workspace/results/overnight
STATUS=$OUT/STATUS.txt
mkdir -p $OUT
RESFILE_TPL='dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic-b200align_c%s.json'
PER_RUN_TIMEOUT=$((150*60))     # hard cap; a healthy 3600 s arm needs ~110 min

# name              CONC  CHUNK_PER_RANK  GUARD
ARMS=(
  "c64-chunk16384      64  16384  0"
  "c256-chunk8192     256   8192  0"
  "c256-chunk8192-gd  256   8192  1"
  "c256-chunk16384    256  16384  0"
  "c128-chunk8192     128   8192  0"
  "c128-chunk16384    128  16384  0"
)

log(){ echo "[$(date '+%F %T')] $*" | tee -a $STATUS; }

kill_and_reclaim(){
  local me=$$
  for i in 1 2 3 4; do
    for p in $(ps -eo pid,args | grep -E "[s]glang" | grep -v "bash -c" | awk -v s=$me '$1!=s{print $1}'); do
      kill -9 $p 2>/dev/null
    done
    sleep 5
  done
  # KFD reclaim is bursty and can hold ~90% for minutes with no process (trap 4)
  for i in $(seq 1 120); do
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -E "VRAM%" | awk '{s+=$NF} END{print s+0}')
    [ "${v:-999}" -lt 10 ] && { log "  reclaim done (~$((i*10))s)"; return 0; }
    sleep 10
  done
  log "  WARNING: VRAM still held after 20 min (sum=${v:-?})"
}

log "=== overnight matrix start: ${#ARMS[@]} arms, 3600 s each ==="
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
log "=== overnight matrix finished ==="
