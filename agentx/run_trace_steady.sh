#!/usr/bin/env bash
# Re-capture the split-K=4 trace at STEADY STATE, unattended, then answer the
# one question the arm result cannot: did shrinking the MLA straggler also help
# MegaMoE's FIRST kernel (`megamoe_prepare_compact`), or is that kernel a pure
# cross-rank barrier whose duration just absorbs whatever slack MLA leaves?
#
#   STEPS=24 setsid nohup ./run_trace_steady.sh > /shared_nfs/kk/trace_run_s24.log 2>&1 &
#
# STEPS IS A TWO-SIDED CONSTRAINT and both sides have now been hit. 40 kills a
# rank mid-capture (only 4 of 8 flushed on 2026-09-17; on B200 the same value
# times out `deep_gemm barrier.cuh:45`). 12 flushed all 8 ranks but the window
# fit inside ONE 1,076 ms chunked-prefill step, so the trace held `EXTENDx1`
# and zero `TARGET_VERIFY` steps -- `prepare_wait.py` had nothing to read, and
# decode is the whole question. A full TARGET_VERIFY step is 74-120 ms, so 24
# spans decode comfortably while staying well under the value that killed a
# rank. The file is named `-DECODE` either way: that name is NOT evidence, so
# this script now checks the step annotations before it trusts the capture.
#
# Survives operator disconnect. Structure and the three traps (launcher never
# exits, `launch_serve[r]` must be killed first, VRAM plateaus 20-35 min) are
# lifted from run_chain.sh -- see its header before editing either.
#
# Why this run exists: the 2026-09-17 capture failed twice over. Only 4 of 8
# ranks flushed (num_steps=40 -> a rank died mid-capture and the rest cascaded
# through NCCL heartbeat), and the window landed at bs=5-7 while the reference
# trace is bs=9-20 -- matched bs IS the method, so it was unusable. Both fixes
# now live in trace_trigger.sh: it gates on bs AND tok/req (not a fixed sleep,
# and not tok/req alone, which peaks during the post-warmup drain at bs=1-5),
# and num_steps defaults to 12.
#
# It does NOT wait for the benchmark to finish. Once all 8 ranks have flushed
# the trace there is nothing left to learn from the remaining ~40 min of
# DURATION, so it kills the arm and goes straight to analysis. The result dir
# is therefore NOT a valid throughput arm -- do not put it in summary_table.py.
set -uo pipefail

KK=/shared_nfs/kk
SKILL_DIR=/workspace/claude-skills/agentx
STEPS="${STEPS:-24}"
TRACE_DIR="$KK/pr35619/trace_hcasplit4_steady_s${STEPS}"
REF_TRACE="$KK/pr35619/trace_c128_pdi24_steady"
RESULT_DIR="/workspace/results/megamoe-eplb-c128-hcasplit4-trace-s${STEPS}"
ARMLOG="$KK/trace_arm_s${STEPS}.log"
TRIGLOG="$KK/trace_trigger_s${STEPS}.log"
OUT="$KK/trace_summary_s${STEPS}.md"
PORT=8888
RANKS=8
FLUSH_TIMEOUT_MIN="${FLUSH_TIMEOUT_MIN:-40}"
VRAM_TIMEOUT_MIN="${VRAM_TIMEOUT_MIN:-60}"
VRAM_BASELINE=400000000

log() { echo "[trace $(date -u +%H:%M:%S)] $*"; }

vram_used() {
    rocm-smi --showmeminfo vram 2>/dev/null \
      | rg -o 'Used Memory \(B\): [0-9]+' | rg -o '[0-9]+$' | sort -rn | head -1
}

cleanup() {
    log "cleanup: killing launch_server parents first"
    ps -eo pid,args | rg 'launch_serve[r]' | awk '{print $1}' > /tmp/trace_p.txt
    xargs -r kill -9 < /tmp/trace_p.txt 2>/dev/null
    sleep 5
    pgrep '^sglang::' > /tmp/trace_s.txt
    log "cleanup: killing $(wc -l < /tmp/trace_s.txt) sglang:: workers"
    xargs -r kill -9 < /tmp/trace_s.txt 2>/dev/null
    sleep 15
    log "cleanup: sglang=$(pgrep -c '^sglang::') launch_server=$(ps -eo args | rg -c 'launch_serve[r]')"
}

wait_for_vram() {
    log "waiting for the VRAM cliff (<= $VRAM_BASELINE B), max ${VRAM_TIMEOUT_MIN}min"
    for i in $(seq 1 $((VRAM_TIMEOUT_MIN * 2))); do
        u=$(vram_used); p=$(pgrep -c '^sglang::')
        [ $((i % 6)) -eq 1 ] && log "  vram=${u:-?} procs=$p"
        if [ "${u:-999999999999}" -lt "$VRAM_BASELINE" ] && [ "$p" = "0" ]; then
            sleep 60
            u2=$(vram_used)
            if [ "${u2:-999999999999}" -lt "$VRAM_BASELINE" ]; then
                log "  cliff confirmed twice: $u then $u2"; return 0
            fi
        fi
        sleep 30
    done
    log "  TIMEOUT waiting for VRAM"; return 1
}

gz_count() { ls "$TRACE_DIR"/*.trace.json.gz 2>/dev/null | wc -l; }

mkdir -p "$TRACE_DIR" "$RESULT_DIR"
log "trace dir $TRACE_DIR (starting count $(gz_count)), ref $REF_TRACE"

wait_for_vram || { log "ABORT: node never returned to baseline"; exit 1; }

export SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"
export RESULT_DIR="$RESULT_DIR"
export SGLANG_MLA_HCA_KV_SPLITS=4
bash "$SKILL_DIR/agentx_c128_hcasplit_trace.sh" > "$ARMLOG" 2>&1 &
ARM_PID=$!
log "arm launched, wrapper pid=$ARM_PID, launch log $ARMLOG"

# trace_trigger.sh owns the warmup wait, the steady-state gate and the POST.
BS_MIN=10 TOKREQ_MIN=140000 MAX_WAIT_MIN=45 \
  bash "$SKILL_DIR/trace_trigger.sh" "$ARMLOG" "$RESULT_DIR/server.log" "$PORT" "$STEPS" \
  > "$TRIGLOG" 2>&1
log "trigger exited rc=$? -- see $TRIGLOG"
rg -o 'proceeding at bs=.*|WARNING:.*|FAIL:.*' "$TRIGLOG" | tail -3

log "waiting for all $RANKS ranks to flush, max ${FLUSH_TIMEOUT_MIN}min"
for i in $(seq 1 $((FLUSH_TIMEOUT_MIN * 2))); do
    n=$(gz_count)
    [ $((i % 6)) -eq 1 ] && log "  flushed=$n/$RANKS"
    [ "$n" -ge "$RANKS" ] && break
    sleep 30
done
N=$(gz_count)
log "flushed $N/$RANKS ranks"

kill "$ARM_PID" 2>/dev/null
cleanup

# The step ANNOTATION, not the file name, decides whether this capture can
# answer anything about decode. `-DECODE` in the name means nothing.
TP7=$(ls "$TRACE_DIR"/*TP-7-*.gz 2>/dev/null | head -1)
VERIFY="?"
if [ -n "$TP7" ]; then
    VERIFY=$(cd "$SKILL_DIR" && python3 analysis/trace_summary.py "$TP7" 2>/dev/null \
      | rg -o 'TARGET_VERIFYx[0-9]+' | rg -o '[0-9]+$' | head -1)
    VERIFY="${VERIFY:-0}"
fi
log "TARGET_VERIFY steps in TP-7: $VERIFY (0 => capture landed on prefill again, question unanswered)"

{
    echo "# Steady-state split-K=4 trace, num_steps=$STEPS — $(date -u +'%Y-%m-%d %H:%M') UTC"
    echo ""
    echo "**TARGET_VERIFY steps captured (TP-7): $VERIFY.** 0 means the window"
    echo "landed on a chunked-prefill EXTEND step again and nothing below"
    echo "addresses decode — that is what happened at num_steps=12."
    echo ""
    echo "trace: \`$TRACE_DIR\` ($N/$RANKS ranks)   reference: \`$REF_TRACE\`"
    echo "arm log: \`$ARMLOG\`   trigger log: \`$TRIGLOG\`"
    echo ""
    echo "Operating point actually captured (matched bs is the method — if this"
    echo "is not inside the reference's bs 9-20, the comparison below is void):"
    echo ""
    echo '```'
    rg -o 'proceeding at bs=.*|WARNING: bs never.*' "$TRIGLOG" | tail -2
    echo '```'
    if [ "$N" -lt "$RANKS" ]; then
        echo ""
        echo "**PARTIAL: only $N of $RANKS ranks flushed.** Rank-comparative"
        echo "numbers below are over the ranks that wrote and are not a"
        echo "cross-rank picture."
    fi
    for pair in "split4:$TRACE_DIR" "reference:$REF_TRACE"; do
        tag=${pair%%:*}; dir=${pair#*:}
        echo ""
        echo "## $tag — bucket summary (TP-7)"
        echo ""
        echo '```'
        f=$(ls "$dir"/*TP-7-*.gz 2>/dev/null | head -1)
        if [ -n "$f" ]; then
            (cd "$SKILL_DIR" && python3 analysis/trace_summary.py "$f" 2>&1 | head -40)
        else
            echo "no TP-7 trace in $dir"
        fi
        echo '```'
        echo ""
        echo "## $tag — is \`megamoe_prepare_compact\` a barrier or real work?"
        echo ""
        echo '```'
        (cd "$SKILL_DIR" && python3 analysis/prepare_wait.py "$dir" 2>&1 | head -60)
        echo '```'
    done
    echo ""
    echo "**Read this against the question:** MLA got ~7.4 ms/step cheaper. If"
    echo "\`megamoe_prepare_compact\` is a barrier (anti-correlated, r<-0.9 both"
    echo "sides) its per-call time will ABSORB part of that win rather than"
    echo "shrink with it, and the MoE dispatch is then the next straggler. If it"
    echo "shrinks in proportion, the attention fix helped it directly."
    echo ""
    echo "**Trace run complete $(date -u +'%Y-%m-%d %H:%M') UTC.**"
} >> "$OUT"

log "ANALYSIS WRITTEN to $OUT"
log "TRACE RUN COMPLETE"
