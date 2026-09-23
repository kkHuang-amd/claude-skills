#!/usr/bin/env bash
# Waits for the c192 arm to finish, applies the cache gate, and only then
# launches c224.
#
# The gate is the operator's own criterion: the GPU-tier prefix-cache hit rate
# must stay at or above 90 %. That is NOT vacuous here -- there is no CPU tier
# to demote to (KV_OFFLOADING=none, no --enable-hierarchical-cache), so a
# saturating KV pool evicts outright and the hit rate falls. c192's live figure
# was already 90.6 % against c160's 94.1 %, with 76 requests queued, so this can
# genuinely stop the sweep.
#
# Own session/pgid via setsid, so a group-wide signal on the caller cannot take
# it down mid-wait.
set -u
HERE=/workspace/claude-skills/agentx
PREV=/workspace/results/fp4-dptbo-c192
MARK=/workspace/results/CHAIN_C224.txt
: >"$MARK"

for _ in $(seq 1 480); do        # up to 4 h at 30 s
    if grep -q 'ARM_EXIT=' "$PREV/run.log" 2>/dev/null; then
        echo "$(date '+%F %T')  c192 finished $(grep -o 'ARM_EXIT=[0-9]*' "$PREV/run.log" | tail -1)" >>"$MARK"
        # Let the c192 waiter run its own export wait, cleanup and gate first.
        sleep 180
        if python3 "$HERE/cache_tier_gate.py" fp4-dptbo-c192 >>"$MARK" 2>&1; then
            echo "$(date '+%F %T')  gate PASSED -- launching c224" >>"$MARK"
            exec setsid bash "$HERE/wait_and_launch.sh" 224
        fi
        echo "$(date '+%F %T')  gate FAILED -- c224 NOT launched, sweep paused" >>"$MARK"
        exit 1
    fi
    sleep 30
done
echo "$(date '+%F %T')  timed out waiting for c192 to finish" >>"$MARK"
