#!/usr/bin/env bash
# Waits for another session's arm to finish, then runs the c96 FP4 arm and
# cleans up after it.
#
# Requires THREE consecutive idle checks 30 s apart, not one. A single zero can
# be caught in the gap between a launcher tearing down its server and its
# aiperf export finishing, and starting there would collide with the other
# session's run.
#
# Bracketed grep patterns only -- a bare `pkill -f aiperf` matches the caller's
# own command line and has already killed a cleanup shell on this node.
set -u
HERE=/workspace/claude-skills/agentx
DIR=/workspace/results/fp4-dptbo-c96
MARK=$DIR/WAITER.txt
mkdir -p "$DIR"
: >"$MARK"

busy_count() {
    ps -eo args \
        | grep -Ec "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf (system_controller|profile)"
}

idle=0
for i in $(seq 1 960); do        # up to 8 h at 30 s
    n=$(busy_count)
    if [ "$n" -eq 0 ]; then
        idle=$((idle + 1))
    else
        [ "$idle" -ne 0 ] && echo "$(date '+%F %T')  busy again ($n), idle streak reset" >>"$MARK"
        idle=0
    fi
    if [ "$idle" -ge 3 ]; then
        echo "$(date '+%F %T')  node idle for 3 consecutive checks -- launching c96" >>"$MARK"
        cd /workspace/InferenceX || exit 2
        bash "$HERE/fp4_dptbo_c96.sh" >"$DIR/run.log" 2>&1
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
        exit 0
    fi
    sleep 30
done
echo "$(date '+%F %T')  timed out waiting for an idle node" >>"$MARK"
