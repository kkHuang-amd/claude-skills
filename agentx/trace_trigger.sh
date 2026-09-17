#!/usr/bin/env bash
# Trigger a torch-profiler capture on a RUNNING agentx server, at steady state.
#
#   ./trace_trigger.sh <launch-log> <server-log> [port] [num_steps]
#
# Timing is the whole point, and a fixed sleep gets it wrong. The first attempt
# settled 120 s after warmup and captured bs=5-7 against a reference trace at
# bs=9-20 -- unusable, because matched bs IS the method. Warmup length also
# moves with page-cache warmth, so no constant is right twice.
#
# So this waits on the KV WORKING SET, which is what a trace is read against:
# tok/req = `#full token` / `#running-req` from the server's own Decode lines.
# Steady state for c128 is ~165-170k; the default gate is 150k.
#
# num_steps defaults to 12, not 40. At 40 a rank died mid-capture and the rest
# cascaded through NCCL heartbeat, so only 4 of 8 ranks ever flushed.
#
# With DP attention the router owns $PORT and the SERVER owns $PORT+1;
# /start_profile exists only on the server. Health is /health_generate.
set -uo pipefail
LOG="${1:?usage: trace_trigger.sh <launch-log> <server-log> [port] [num_steps]}"
SRVLOG="${2:?need the server.log path}"
PORT="${3:-8888}"
STEPS="${4:-12}"
TOKREQ_MIN="${TOKREQ_MIN:-140000}"
BS_MIN="${BS_MIN:-10}"       # reference trace is bs 9-20; below this nothing is comparable
MAX_WAIT_MIN="${MAX_WAIT_MIN:-45}"
SRV="http://127.0.0.1:$((PORT + 1))"

# Prints "<median bs> <median tok/req>" over the last 60 Decode lines.
#
# tok/req ALONE IS THE WRONG GATE, and it cost a whole capture. tok/req is
# `#full token` / `#running-req`, so it is largest exactly when the batch is
# smallest: during the post-warmup drain a handful of very long requests read
# 293k tok/req at bs=1-5. The gate passed instantly and the trace was useless.
# Steady state needs BOTH a full batch and a large working set.
state() {
    rg -o '#running-req: [0-9]+, #full token: [0-9]+' "$SRVLOG" 2>/dev/null \
      | tail -60 | python3 -c '
import re, sys, statistics
bs, tr = [], []
for ln in sys.stdin:
    m = re.search(r"#running-req: (\d+), #full token: (\d+)", ln)
    if m and int(m.group(1)) > 0:
        bs.append(int(m.group(1)))
        tr.append(int(m.group(2)) / int(m.group(1)))
print(int(statistics.median(bs)) if bs else 0,
      int(statistics.median(tr)) if tr else 0)'
}

echo "[trigger] waiting for warmup to complete in $LOG"
for _ in $(seq 1 360); do
    rg -q 'Phase warmup .*complete' "$LOG" 2>/dev/null && break
    sleep 10
done
rg -q 'Phase warmup .*complete' "$LOG" 2>/dev/null || {
    echo "[trigger] FAIL: warmup never completed"; exit 1; }
echo "[trigger] warmup done at $(date -u +%H:%M:%S)"

echo "[trigger] waiting for bs >= $BS_MIN AND tok/req >= $TOKREQ_MIN"
for i in $(seq 1 $((MAX_WAIT_MIN * 2))); do
    read -r b t <<<"$(state)"
    [ $((i % 4)) -eq 1 ] && echo "[trigger] bs=$b tok/req=$t at $(date -u +%H:%M:%S)"
    if [ "${b:-0}" -ge "$BS_MIN" ] && [ "${t:-0}" -ge "$TOKREQ_MIN" ]; then
        break
    fi
    sleep 30
done
read -r b t <<<"$(state)"
echo "[trigger] proceeding at bs=$b tok/req=$t"
if [ "${b:-0}" -lt "$BS_MIN" ]; then
    echo "[trigger] WARNING: bs never reached $BS_MIN -- the capture will not be"
    echo "[trigger] comparable to the reference trace (bs 9-20). Firing anyway."
fi

code=$(curl -s -o /dev/null -w '%{http_code}' "$SRV/health_generate" || echo 000)
echo "[trigger] health_generate=$code"

echo "[trigger] starting profile, num_steps=$STEPS"
curl -fsS -X POST "$SRV/start_profile" -H "Content-Type: application/json" \
  -d "{\"activities\":[\"CPU\",\"GPU\"],\"num_steps\":$STEPS,\"profile_by_stage\":true,\"record_shapes\":true,\"with_stack\":false}" \
  | head -c 300
echo
echo "[trigger] posted at $(date -u +%H:%M:%S)"
