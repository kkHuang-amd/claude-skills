#!/usr/bin/env bash
# Run DP2 + TP MoE + bounded replay AgentX points back to back on GPUs 0,1 (fix tree), one server at a time.
#   RUNS="32:32 64:4 64:16 64:32 128:4 128:16" setsid nohup bash dp2tp_queue.sh > /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/dp2tp_queue.nohup 2>&1 < /dev/null &
# Each point -> /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/dp2tp_replay_c<CONC>_pdi<P>/ ; progress + results -> dp2tp_queue.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
Q=$OUT/dp2tp_queue.txt
SRC=${SRC:-/sgl-workspace/sglang-dsv41-dp-fault/python}
PER_RUN_TIMEOUT=${PER_RUN_TIMEOUT:-6300}
say() { echo "[$(date -u +%F' '%T) UTC] $*" >> "$Q"; }

# Our own leftovers only: port holders and sglang:: workers in this container. Never kill by rocm-smi PIDs
# (those are host PIDs and include other containers' processes).
cleanup() {
    local pids
    pids=$( (ss -ltnp 2>/dev/null | grep -E ':(9123|8888|8889) ' | grep -oE 'pid=[0-9]+' | cut -d= -f2
             ps -eo pid=,comm= | awk '$2 ~ /^sglang::/ {print $1}'
             pgrep -f 'bin/aiperf|aiperf profile|sglang_router|sglang.launch_server' | grep -vx "$$") | sort -u | tr '\n' ' ')
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    sleep 10
}

summarize() {
    local tag=$1 f
    f=$(ls "$OUT/$tag"/*agentic_c*.json 2>/dev/null | head -1)
    local crash
    crash=$(grep -cE 'bonus out of|HSA_STATUS|Fatal Python|Scheduler hit an exception' "$OUT/$tag/server.log" 2>/dev/null)
    local req tput
    req=$(grep -oE 'Requests: [0-9]+ successful / [0-9]+ total \([0-9]+ warmup, [0-9]+ error dropped\)' "$OUT/$tag.nohup" | tail -1)
    tput=$(grep -oE 'Throughput per GPU: [0-9]+' "$OUT/$tag.nohup" | tail -1)
    local lat=""
    [ -n "$f" ] && lat=$(python3 -c "
import json;d=json.load(open('$f'))['request_metrics']['latency']
print('P90 intvty %.1f, p50 intvty %.1f, TTFT p50 %.2f s' % (d['intvty']['p90'], d['intvty']['p50'], d['ttft']['p50']))" 2>/dev/null)
    say "RESULT $tag | ${tput:-no throughput} | ${lat:-no json} | ${req:-no request line} | crash_lines=${crash:-?}"
}

say "QUEUE START runs='${RUNS}' src=$SRC"
for item in ${RUNS}; do
    c=${item%%:*}
    p=${item#*:}
    tag=dp2tp_replay_c${c}_pdi${p}
    cleanup
    [ -e "$OUT/$tag" ] && mv "$OUT/$tag" "$OUT/${tag}_old_$(date +%H%M%S)"
    say "START $tag"
    cd "$OUT" || exit 1
    SRC=$SRC REPLAY=1 PREFILL_DECODE_INTERVAL=$p DP_ATTENTION=true DP_MOE=tp TP=2 CONC=$c GPUS=0,1 PORT=8888 TAG=$tag \
        timeout "$PER_RUN_TIMEOUT" bash "$D/agentx_colleague_run.sh" > "$OUT/$tag.nohup" 2>&1 < /dev/null
    rc=$?
    say "END $tag rc=$rc"
    summarize "$tag"
done
cleanup
say "QUEUE DONE"
