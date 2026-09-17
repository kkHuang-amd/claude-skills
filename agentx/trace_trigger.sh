#!/usr/bin/env bash
# Trigger a torch-profiler capture on a RUNNING agentx server.
#
#   ./trace_trigger.sh <launch-log> [port] [num_steps]
#
# Timing is the whole point. Do NOT use a fixed sleep: warmup length moves with
# page-cache warmth, and the KV working set at capture time is what a trace is
# read against. This waits for the launcher to report warmup complete, then
# settles, then fires.
#
# With DP attention the router owns $PORT and the SERVER owns $PORT+1;
# /start_profile exists only on the server. Health is /health_generate,
# not /health.
set -uo pipefail
LOG="${1:?usage: trace_trigger.sh <launch-log> [port] [num_steps]}"
PORT="${2:-8888}"
STEPS="${3:-40}"
SETTLE="${SETTLE:-120}"
SRV="http://127.0.0.1:$((PORT + 1))"

echo "[trigger] waiting for warmup to complete in $LOG"
for _ in $(seq 1 360); do
    rg -q 'Phase warmup .*complete' "$LOG" 2>/dev/null && break
    sleep 10
done
rg -q 'Phase warmup .*complete' "$LOG" 2>/dev/null || {
    echo "[trigger] FAIL: warmup never completed"; exit 1; }

echo "[trigger] warmup done, settling ${SETTLE}s so the KV working set is steady"
sleep "$SETTLE"

code=$(curl -s -o /dev/null -w '%{http_code}' "$SRV/health_generate" || echo 000)
echo "[trigger] health_generate=$code"

echo "[trigger] starting profile, num_steps=$STEPS"
curl -fsS -X POST "$SRV/start_profile" -H "Content-Type: application/json" \
  -d "{\"activities\":[\"CPU\",\"GPU\"],\"num_steps\":$STEPS,\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false}" \
  | head -c 300
echo
echo "[trigger] posted at $(date -u +%H:%M:%S)"
