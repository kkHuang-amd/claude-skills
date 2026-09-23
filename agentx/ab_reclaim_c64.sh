#!/usr/bin/env bash
# Runs the c64 HSA_NO_SCRATCH_RECLAIM A/B back to back, unattended.
#
# reclaim=1 goes first: it is the arm that might repeat the c128 OOR abort, and
# a failure is cheaper to learn about at the start than after three hours.
#
# Each arm script carries its own three-shape kill preamble, so a leftover from
# the previous arm is cleared before the next one starts; this driver adds the
# same cleanup after the LAST arm, which nothing else would do.
#
# Bracketed grep patterns only -- a bare `pkill -f aiperf` matches the caller's
# own command line and has already killed a cleanup shell on this node once.
set -u
HERE=/workspace/claude-skills/agentx
MARK=/workspace/results/AB_RECLAIM_C64_DONE.txt
: >"$MARK"

kill_leftovers() {
    for p in $(ps -eo pid,args \
        | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" \
        | awk '{print $1}'); do
        [ "$p" = "$$" ] && continue
        kill -9 "$p" 2>/dev/null
    done
    sleep 12
}

for arm in fp4-dptbo-c64-reclaim1 fp4-dptbo-c64-reclaim0; do
    script="$HERE/${arm//-/_}.sh"
    test -x "$script" || { echo "FATAL: no $script" >>"$MARK"; exit 2; }
    dir="/workspace/results/$arm"
    mkdir -p "$dir"
    echo "$(date '+%F %T')  START $arm" >>"$MARK"
    cd /workspace/InferenceX || exit 2
    bash "$script" >"$dir/run.log" 2>&1
    echo "$(date '+%F %T')  END   $arm  $(grep -o 'ARM_EXIT=[0-9]*' "$dir/run.log" | tail -1)" >>"$MARK"
    sleep 60          # let aiperf finish its export / certification
    kill_leftovers
done

echo "$(date '+%F %T')  ALL DONE; sglang/aiperf procs now: $(ps -eo args \
    | grep -Ec '[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf')" >>"$MARK"
