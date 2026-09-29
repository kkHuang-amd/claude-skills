#!/usr/bin/env bash
# Wait for a launched server, then GSM8K + throughput sweep. One summary line per step -> $SUMMARY.
#   TAG=<name> PORT=30000 SERVER_LOG=/shared_nfs/kk/dsv41/server.log SERVER_PID=<pid>
#   SUMMARY=/shared_nfs/kk/dsv41/summary.txt  SKIP_GSM8K=0 SKIP_PERF=0
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SUMMARY=${SUMMARY:-/shared_nfs/kk/dsv41/summary.txt}
say(){ echo "[$(date +%T)] $TAG: $*" >> "$SUMMARY"; }
until grep -qE 'ready to roll' "$SERVER_LOG"; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null || grep -qE 'Scheduler hit an exception|Initialization failed' "$SERVER_LOG"; then
    say "SERVER FAILED: $(grep -oE '\w+Error: .{0,120}' "$SERVER_LOG" | tail -1)"; exit 1; fi
  sleep 10; done
say "server ready"
[ "${SKIP_GSM8K:-0}" = 1 ] || say "gsm8k $(TAG=$TAG PORT=$PORT bash "$D/run_gsm8k.sh" | tail -1)"
[ "${SKIP_PERF:-0}" = 1 ] || TAG=$TAG PORT=$PORT bash "$D/run_throughput.sh" | while read -r l; do say "perf $l"; done
say "DONE"
