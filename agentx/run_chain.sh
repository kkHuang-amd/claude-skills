#!/usr/bin/env bash
# Run several arms back to back, unattended.
#
#   setsid nohup ./run_chain.sh <arm.sh> [arm2.sh ...] > /shared_nfs/kk/chain.log 2>&1 &
#
# Written to survive the operator disconnecting. Three things it has to get
# right, each of which cost a manual session today:
#
# 1. THE LAUNCHER NEVER EXITS. When the benchmark finishes, the agentx launcher
#    leaves the server running forever. So completion is detected from the
#    RESULT LOG, not from process exit.
# 2. `pgrep '^sglang::'` DOES NOT MATCH THE PARENT. `python3 -m
#    sglang.launch_server` is reparented to init and keeps respawning tokenizer
#    workers, holding the port. It must be killed FIRST, and the pattern must
#    not match this script's own command line (hence the `launch_serve[r]`
#    bracket, which cost a self-kill to learn).
# 3. VRAM DRAINS SLOWLY. It sits on a multi-GB plateau for 20-35 min after the
#    processes are gone. Starting an arm on the plateau OOMs at cuda-graph
#    capture. Gate on the 0.28 GB baseline, confirmed twice a minute apart.
set -uo pipefail

KK=/shared_nfs/kk
SUMMARY="$KK/chain_summary.md"
DONE_RE='Validated aiperf request error rate|All results received, initiating shutdown'
ARM_TIMEOUT_MIN="${ARM_TIMEOUT_MIN:-160}"
VRAM_TIMEOUT_MIN="${VRAM_TIMEOUT_MIN:-60}"
VRAM_BASELINE=400000000        # 0.4 GB; the idle baseline measures 0.298 GB

log() { echo "[chain $(date -u +%H:%M:%S)] $*"; }

vram_used() {
    rocm-smi --showmeminfo vram 2>/dev/null \
      | rg -o 'Used Memory \(B\): [0-9]+' | rg -o '[0-9]+$' | sort -rn | head -1
}

cleanup() {
    log "cleanup: killing launch_server parents first"
    ps -eo pid,args | rg 'launch_serve[r]' | awk '{print $1}' > /tmp/chain_p.txt
    xargs -r kill -9 < /tmp/chain_p.txt 2>/dev/null
    sleep 5
    pgrep '^sglang::' > /tmp/chain_s.txt
    log "cleanup: killing $(wc -l < /tmp/chain_s.txt) sglang:: workers"
    xargs -r kill -9 < /tmp/chain_s.txt 2>/dev/null
    sleep 15
    log "cleanup: sglang=$(pgrep -c '^sglang::') launch_server=$(ps -eo args | rg -c 'launch_serve[r]')"
}

wait_for_vram() {
    log "waiting for the VRAM cliff (<= $VRAM_BASELINE B), max ${VRAM_TIMEOUT_MIN}min"
    for i in $(seq 1 $((VRAM_TIMEOUT_MIN * 2))); do
        u=$(vram_used); p=$(pgrep -c '^sglang::')
        [ $((i % 6)) -eq 1 ] && log "  vram=${u:-?} procs=$p"
        if [ "${u:-999999999999}" -lt "$VRAM_BASELINE" ] && [ "$p" = "0" ]; then
            sleep 60                       # second reading, a minute apart
            u2=$(vram_used)
            if [ "${u2:-999999999999}" -lt "$VRAM_BASELINE" ]; then
                log "  cliff confirmed twice: $u then $u2"
                return 0
            fi
        fi
        sleep 30
    done
    log "  TIMEOUT waiting for VRAM"
    return 1
}

run_arm() {
    local script="$1" tag
    tag=$(basename "$script" .sh)
    local llog="$KK/chain_${tag}.log"
    log "=== ARM $tag -> $llog"

    if ! wait_for_vram; then
        log "ARM $tag SKIPPED: node never came back to baseline"
        echo "- **$tag: SKIPPED** (VRAM never returned to baseline)" >> "$SUMMARY"
        return 1
    fi

    bash "/workspace/claude-skills/agentx/$script" > "$llog" 2>&1 &
    local pid=$!
    log "ARM $tag launched, wrapper pid=$pid"

    local ok=1
    for _ in $(seq 1 $((ARM_TIMEOUT_MIN * 2))); do
        if rg -q "$DONE_RE" "$llog" 2>/dev/null; then ok=0; break; fi
        sleep 30
    done

    if [ $ok -ne 0 ]; then
        log "ARM $tag TIMEOUT after ${ARM_TIMEOUT_MIN}min"
        echo "- **$tag: TIMEOUT** after ${ARM_TIMEOUT_MIN} min, see \`$llog\`" >> "$SUMMARY"
    else
        log "ARM $tag benchmark complete"
    fi

    kill "$pid" 2>/dev/null
    cleanup

    # Record the result immediately: if anything later goes wrong, the numbers
    # for the arms that DID finish are already on disk.
    local rdir
    rdir=$(rg -o 'RESULT_DIR:-[^}]*' "/workspace/claude-skills/agentx/$script" | head -1 | sed 's/RESULT_DIR:-//')
    {
        echo ""
        echo "## $tag ($(date -u +'%Y-%m-%d %H:%M') UTC)"
        echo ""
        echo "results: \`$rdir\`  launch log: \`$llog\`"
        echo ""
        echo '```'
        if [ -f "$rdir/server.log" ]; then
            (cd /workspace/claude-skills/agentx && \
             python3 analysis/decode_stats.py "$rdir/server.log" 2>&1 | head -40)
        else
            echo "no server.log at $rdir"
        fi
        echo '```'
        rg -o 'Validated aiperf request error rate.*' "$llog" | tail -1
    } >> "$SUMMARY"
    log "ARM $tag recorded in $SUMMARY"
    return $ok
}

{
    echo "# Unattended chain started $(date -u +'%Y-%m-%d %H:%M') UTC"
    echo ""
    echo "Arms: $*"
} >> "$SUMMARY"

log "chain starting with arms: $*"
for arm in "$@"; do
    run_arm "$arm"
done
cleanup
log "CHAIN COMPLETE. Summary: $SUMMARY"
echo "" >> "$SUMMARY"
echo "**Chain complete $(date -u +'%Y-%m-%d %H:%M') UTC.**" >> "$SUMMARY"
