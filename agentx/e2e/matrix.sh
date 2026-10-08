#!/usr/bin/env bash
# Smoke-gated sequence of AgentX points on one node.
#   1. SMOKE=1 on the first point; abort everything if it fails.
#   2. each point at DURATION (default 3600 s), in order; a failed point is
#      logged and the next one still runs.
# Completed points (.ok marker) are skipped, so re-launching resumes.
#
# usage: TAG=<label> [POINTS="c1 c4 c16 c256"] [POINT_SET=ci36401947630 | POINTS_DIR=<dir>]
#        [SKIP_SMOKE=1] bash matrix.sh
#   results: /shared_nfs/kk/results/DeepSeek-V4-Pro-0813/e2e-<point>-<TAG>/, log: .../e2e-matrix-<TAG>.status
# Inherits PYTHONPATH/SGL_DIR from the caller (which sglang is under test).
#
# The scripts and the point set are snapshotted to $BASE/e2e-matrix-<TAG>.code/
# and the matrix re-execs from there, so the shared e2e/ dir (on NFS, used by
# several nodes) can be edited while runs are live: bash reads scripts lazily.
set -uo pipefail
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TAG:?set TAG}"
BASE=${RESULTS_BASE:-/shared_nfs/kk/results/DeepSeek-V4-Pro-0813}
if [ -z "${AGENTX_SNAPSHOT:-}" ]; then
  SNAP=$BASE/e2e-matrix-$TAG.code
  SRC_POINTS=${POINTS_DIR:-$SKILL_DIR/points/${POINT_SET:-ci36401947630}}
  [ -d "$SRC_POINTS" ] || { echo "no point set $SRC_POINTS"; exit 2; }
  rm -rf "$SNAP"; mkdir -p "$SNAP"
  cp "$SKILL_DIR"/{matrix.sh,run_point.sh,env.sh,stack_fingerprint.sh} "$SNAP/"
  cp -r "$SRC_POINTS" "$SNAP/points"
  echo "$SRC_POINTS" > "$SNAP/points.src"
  exec env AGENTX_SNAPSHOT="$SNAP" POINTS_DIR="$SNAP/points" bash "$SNAP/matrix.sh" "$@"
fi
POINTS=(${POINTS:-c1 c4 c16 c256})
STATUS=$BASE/e2e-matrix-$TAG.status
RUNNER="$AGENTX_SNAPSHOT/run_point.sh"
cd /workspace   # never under /sgl-workspace: cwd sglang/ shadows the import
log() { echo "$(date '+%F %T') $*" | tee -a "$STATUS"; }

reclaim() {  # kill leftovers, then wait for every GPU <= 10 % VRAM
  ps -eo pid,args | awk '$2 ~ /^sglang::/ || ($2 ~ /python3?$/ && $0 ~ /sglang\.launch_server|sglang_router\.launch_router/) {print $1}' \
    | xargs -r kill -9 2>/dev/null
  for _ in $(seq 1 90); do
    max=$(rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): [0-9]+' | grep -oE '[0-9]+$' | sort -n | tail -1)
    [ "${max:-100}" -le 10 ] && return 0
    sleep 10
  done
  log "WARN VRAM still ${max}% after 15 min"
}

result_json() { ls "$1"/dsv4_fp4_sglang_*_agentic.json 2>/dev/null | head -1; }
# A run that exits non-zero still writes its partial JSON (aiperf aborts after
# writing available results), so "json exists" is not "point done". A point
# counts as done only with a .ok marker, written when rc=0.
point_done() { [ -f "$1/.ok" ]; }

log "matrix $TAG start: points=${POINTS[*]} set=$(cat "$AGENTX_SNAPSHOT/points.src") PYTHONPATH=${PYTHONPATH:-<none>}"
first=${POINTS[0]}
D=$BASE/e2e-$first-$TAG-smoke
if [ "${SKIP_SMOKE:-0}" = 1 ]; then
  log "SMOKE skipped (SKIP_SMOKE=1)"
elif [ -z "$(result_json "$D")" ]; then
  reclaim
  log "SMOKE $first start"
  POINT=$first RESULT_DIR=$D SMOKE=1 bash "$RUNNER" > "$D.driver.log" 2>&1; rc=$?
  reclaim
  if [ $rc -ne 0 ] || [ -z "$(result_json "$D")" ]; then
    log "SMOKE $first FAILED rc=$rc -- matrix aborted"; exit 1
  fi
fi
[ "${SKIP_SMOKE:-0}" = 1 ] || log "SMOKE $first ok"

for pt in "${POINTS[@]}"; do
  D=$BASE/e2e-$pt-$TAG
  if point_done "$D"; then log "SKIP $pt (done)"; continue; fi
  if [ -d "$D" ]; then mv "$D" "$D.void-$(date +%H%M%S)"; log "moved stale $D aside"; fi
  reclaim
  log "START $pt"
  POINT=$pt RESULT_DIR=$D DURATION=${DURATION:-3600} timeout ${PER_POINT_TIMEOUT:-4h} \
    bash "$RUNNER" > "$D.driver.log" 2>&1; rc=$?
  reclaim
  j=$(result_json "$D")
  if [ $rc -eq 0 ] && [ -n "$j" ]; then
    touch "$D/.ok"
    m=$(python3 -c "import json;r=json.load(open('$j'))['request_metrics'];print('tok/s/GPU=%.1f intvty_p90=%.2f' % (r['throughput']['per_gpu']['total_tput_tps'], r['latency']['intvty']['p90']))" 2>/dev/null)
    log "DONE $pt rc=$rc $m"
  else
    log "FAILED $pt rc=$rc (partial json: ${j:-none}) -- numbers NOT valid"
  fi
done
log "finished"
