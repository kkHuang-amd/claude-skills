#!/usr/bin/env bash
# Fast AgentX debug loop: keep ONE server resident, fire short client probes.
#
# A full launcher run is ~90 min, most of it weight load (~25 min) and the
# 10-request-per-lane warmup (~40 min at conc 64). For debugging you rarely
# need either. This replays the argv+env that a previous run already persisted
# (sglang_command.txt / the SGLANG_* dump at the top of server.log), so the
# server is bit-identical to that run without re-deriving anything.
#
#   agentx_debug.sh serve  <ref_result_dir> [port]
#   agentx_debug.sh probe  <ref_result_dir> <out_dir> [duration] [warmup_per_lane]
#   agentx_debug.sh status
#   agentx_debug.sh stop
#
# Typical loop (server stays up across probes):
#   agentx_debug.sh serve /workspace/results/armB-tp8-c64
#   agentx_debug.sh probe /workspace/results/armB-tp8-c64 /tmp/p1 300 1
#   agentx_debug.sh probe /workspace/results/armB-tp8-c64 /tmp/p2 300 1
#   agentx_debug.sh stop
#
# Probe numbers are for TRENDS AND BUGS ONLY. duration<900 makes AIPerf stamp
# submission_valid=false, and a 1-per-lane warmup does not reach the steady
# state the published arms measure. Never quote a probe against the leaderboard.
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SKILL_DIR/agentx_env.sh"

PIDFILE=/tmp/agentx_debug_server.pid

_ports_busy() { ss -lntp 2>/dev/null | grep -cE ':(8888|8889)\b' || true; }

_assert_clean() {
    local v n pb
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{if($2>m)m=$2}END{print m+0}')
    n=$(ps -eo cmd | grep -cE '[s]glang.launch_server|[s]glang::router' || true)
    pb=$(_ports_busy)
    # A stale sglang::router keeps 8888 bound and answers /health, so a launch
    # "succeeds" and then every request fails against a dead backend. VRAM and
    # launch_server checks alone do not catch it.
    if [ "$n" != "0" ] || [ "${v:-0}" -gt 5 ] || [ "$pb" != "0" ]; then
        echo "REFUSING: vram%max=$v sglang_procs=$n ports_8888_8889=$pb" >&2
        echo "Run '$0 stop' first." >&2
        exit 1
    fi
}

cmd_serve() {
    local ref="${1:?usage: serve <ref_result_dir> [port]}"
    local port="${2:-8888}"
    [ -f "$ref/sglang_command.txt" ] || { echo "no sglang_command.txt in $ref" >&2; exit 1; }
    _assert_clean

    local log=/tmp/agentx_debug_server.log
    {
        # Replay the SGLANG_* env the reference run recorded, then its argv.
        sed -n '2,/^====*$/p' "$ref/server.log" | grep -E '^SGLANG_[A-Z0-9_]+=' \
            | sed 's/^/export /' > /tmp/agentx_debug_env.sh
        echo "restored $(wc -l < /tmp/agentx_debug_env.sh) SGLANG_* vars from $ref/server.log"
    } >&2
    set -a; . /tmp/agentx_debug_env.sh; set +a

    ( eval "exec $(cat "$ref/sglang_command.txt")" ) > "$log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "server pid=$(cat $PIDFILE) log=$log port=$port"
    echo "waiting for readiness (weight load is still ~20-25 min the first time)..."
    for i in $(seq 1 400); do
        grep -q 'ready to roll' "$log" 2>/dev/null && { echo "READY after $((i*15))s"; return 0; }
        kill -0 "$(cat $PIDFILE)" 2>/dev/null || { echo "server died; tail:"; tail -5 "$log" | cut -c1-160; return 1; }
        sleep 15
    done
    echo "TIMEOUT waiting for readiness"; return 1
}

cmd_probe() {
    local ref="${1:?usage: probe <ref_result_dir> <out_dir> [duration] [warmup_per_lane]}"
    local out="${2:?}" dur="${3:-300}" warm="${4:-1}"
    [ -f "$ref/benchmark_command.txt" ] || { echo "no benchmark_command.txt in $ref" >&2; exit 1; }
    mkdir -p "$out"
    # Rewrite only duration / warmup / artifact dir; every other flag stays as
    # the reference run had it so the probe exercises the same code paths.
    local c
    c=$(cat "$ref/benchmark_command.txt")
    c=$(echo "$c" | sed -E "s#--benchmark-duration [0-9]+#--benchmark-duration $dur#; \
                            s#--warmup-requests-per-lane [0-9]+#--warmup-requests-per-lane $warm#; \
                            s#--output-artifact-dir [^ ]+#--output-artifact-dir $out/aiperf_artifacts#")
    grep -q -- '--unsafe-override' <<<"$c" || c="$c --unsafe-override"
    echo "$c" > "$out/benchmark_command.txt"
    echo "probe: duration=${dur}s warmup=${warm}/lane -> $out"
    eval "$c" 2>&1 | tee "$out/benchmark.log" | \
        grep -E 'Phase (warmup|profiling)|Requests:|error|ERROR' | cut -c1-170 || true
}

cmd_runs() {
    # Discover every run under RESULTS_ROOT and classify it, so a FRESH session
    # (which has none of the previous session's PIDs or background watchers)
    # can tell what finished, what died, and what is still going.
    local root="${RESULTS_ROOT:-/workspace/results}"
    printf '%-34s %-10s %-9s %s\n' DIR STATE AGE DETAIL
    for d in "$root"/*/; do
        [ -d "$d" ] || continue
        local name age agg state detail slog
        name=$(basename "$d")
        case "$name" in .*) continue ;; esac
        slog="$d/server.log"
        agg=$(ls "$d"/*.json 2>/dev/null | grep -vE 'power|identity' | head -1)
        age=$(( ($(date +%s) - $(stat -c %Y "$d" 2>/dev/null || echo 0)) / 60 ))
        if [ -n "$agg" ]; then
            state=DONE
            detail=$(python3 -c "
import json,sys
try:
    d=json.load(open('$agg')); rm=d['request_metrics']
    print('tput/chip=%.1f intvty_p90=%.2f reqs=%d' % (rm['throughput']['per_gpu']['total_tput_tps'], rm['latency']['intvty']['p90'], d['num_requests_successful']))
except Exception as e: print('unreadable aggregate')" 2>/dev/null)
        elif grep -q 'ERROR: agentic trace replay\|Process died before\|Terminal warmup failure' "$d"/../*"$name"*.log 2>/dev/null; then
            state=FAILED; detail="see the wrapper log next to $root"
        elif [ "$age" -lt 120 ] && [ -f "$slog" ] && [ $(( ($(date +%s) - $(stat -c %Y "$slog")) / 60 )) -lt 10 ]; then
            state=RUNNING
            detail=$(grep -o 'ready to roll' "$slog" >/dev/null 2>&1 && echo "server up, in replay" || echo "still loading weights")
        else
            state=STALE; detail="no aggregate, server.log idle - died or was killed"
        fi
        printf '%-34s %-10s %5dm    %s\n' "$name" "$state" "$age" "$detail"
    done
}

cmd_status() {
    local v n pb
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{printf "%s ",$2}')
    n=$(ps -eo cmd | grep -cE '[s]glang.launch_server|[s]glang::router' || true)
    pb=$(ss -lntp 2>/dev/null | grep -E ':(8888|8889)\b' | wc -l)
    echo "vram: $v"
    echo "sglang/router procs: $n   ports 8888/8889 bound: $pb"
    [ -f "$PIDFILE" ] && echo "tracked pid: $(cat $PIDFILE) alive=$(kill -0 "$(cat $PIDFILE)" 2>/dev/null && echo yes || echo no)"
}

cmd_stop() {
    ps -eo pid,cmd | grep -E '[s]glang.launch_server|[s]glang::router' | awk '{print $1}' \
        | while read -r p; do kill -TERM "$p" 2>/dev/null && echo "TERM $p"; done
    rm -f "$PIDFILE"
    for i in $(seq 1 40); do
        local v n pb
        v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{if($2>m)m=$2}END{print m+0}')
        n=$(ps -eo cmd | grep -cE '[s]glang.launch_server|[s]glang::router' || true)
        pb=$(_ports_busy)
        if [ "$n" = "0" ] && [ "${v:-0}" -le 5 ] && [ "$pb" = "0" ]; then
            echo "CLEAN after $((i*5))s"; return 0
        fi
        sleep 5
    done
    echo "still draining (vram=$v procs=$n ports=$pb)"; return 1
}

case "${1:-}" in
    serve)  shift; cmd_serve "$@" ;;
    probe)  shift; cmd_probe "$@" ;;
    runs)   cmd_runs ;;
    status) cmd_status ;;
    stop)   cmd_stop ;;
    *) sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
