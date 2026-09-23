#!/usr/bin/env bash
# Generic: wait for a shared node to go idle, run one FP4 arm, clean up after it.
#   wait_and_launch.sh <conc>
# Expects /workspace/claude-skills/agentx/fp4_dptbo_c<conc>.sh to exist.
#
# Requires THREE consecutive idle checks 30 s apart, not one. A single zero can
# land in the gap between a launcher tearing down its server and its aiperf
# export finishing, and starting there would collide with another session's run.
#
# Bracketed grep patterns only -- a bare `pkill -f aiperf` matches the caller's
# own command line and has already killed a cleanup shell on this node.
set -u
CONC=${1:?usage: wait_and_launch.sh <conc>}
HERE=/workspace/claude-skills/agentx
SCRIPT=$HERE/fp4_dptbo_c${CONC}.sh
DIR=/workspace/results/fp4-dptbo-c${CONC}
MARK=$DIR/WAITER.txt

test -x "$SCRIPT" || { echo "FATAL: no $SCRIPT"; exit 2; }
mkdir -p "$DIR"
: >"$MARK"

busy_count() {
    ps -eo args \
        | grep -Ec "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf (system_controller|profile)"
}

idle=0
for _ in $(seq 1 960); do        # up to 8 h at 30 s
    n=$(busy_count)
    if [ "$n" -eq 0 ]; then
        idle=$((idle + 1))
    else
        [ "$idle" -ne 0 ] && echo "$(date '+%F %T')  busy again ($n), idle streak reset" >>"$MARK"
        idle=0
    fi
    if [ "$idle" -ge 3 ]; then
        echo "$(date '+%F %T')  node idle for 3 consecutive checks -- launching c$CONC" >>"$MARK"
        cd /workspace/InferenceX || exit 2
        bash "$SCRIPT" >"$DIR/run.log" 2>&1
        echo "$(date '+%F %T')  arm finished $(grep -o 'ARM_EXIT=[0-9]*' "$DIR/run.log" | tail -1)" >>"$MARK"
        sleep 90        # let aiperf finish its export / certification
        for p in $(ps -eo pid,args \
            | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" \
            | awk '{print $1}'); do
            [ "$p" = "$$" ] && continue
            kill -9 "$p" 2>/dev/null
        done
        sleep 10
        echo "$(date '+%F %T')  cleanup done; procs now: $(busy_count)" >>"$MARK"
        md5sum -c "$DIR/TREE_CHECKSUMS_AT_START.txt" >>"$MARK" 2>&1
        # Cache-tier / ceiling gate. Recorded, NOT acted on: whether the sweep
        # continues is a call for the operator, since the CPU-tier check is
        # vacuous without hicache and the ceiling threshold is a judgement.
        python3 "$HERE/cache_tier_gate.py" "fp4-dptbo-c$CONC" >>"$MARK" 2>&1
        echo "$(date '+%F %T')  gate exit=$?" >>"$MARK"
        exit 0
    fi
    sleep 30
done
echo "$(date '+%F %T')  timed out waiting for an idle node" >>"$MARK"
