#!/usr/bin/env bash
# One-shot watchdog for an unattended arm. Waits for the launcher to print
# ARM_EXIT, then kills the server it leaves behind (the launcher never does, and
# the leftover holds ~275 GB/GPU plus the ports the next arm needs).
#
# Started with `setsid` so it gets its own session and process group: the arm
# itself shares pgid with the agent's tool shell, and a group-wide signal there
# would take this down with it.
#
# Bracketed grep patterns only -- a bare `pkill -f aiperf` matches the caller's
# own command line and has already killed a cleanup shell on this node once.
set -u
LOG="${1:-/workspace/results/dptbo-c128/run.log}"
MARK="$(dirname "$LOG")/CLEANUP_DONE.txt"

for _ in $(seq 1 720); do   # up to 6 h at 30 s
    if grep -q 'ARM_EXIT=' "$LOG" 2>/dev/null; then
        sleep 90             # let aiperf finish its export / certification
        {
            date '+%F %T'
            grep -o 'ARM_EXIT=[0-9]*' "$LOG" | tail -1
        } >"$MARK"
        for p in $(ps -eo pid,args \
            | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" \
            | awk '{print $1}'); do
            [ "$p" = "$$" ] && continue
            kill -9 "$p" 2>/dev/null
        done
        sleep 10
        echo "leftovers killed; sglang/aiperf procs now: $(ps -eo args \
            | grep -Ec '[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf')" >>"$MARK"
        exit 0
    fi
    sleep 30
done
echo "timed out waiting for ARM_EXIT in $LOG" >"$MARK"
