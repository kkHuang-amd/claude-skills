#!/usr/bin/env bash
# --prefill-decode-interval 10 -> 20 at c128. Controlled partner for
# dptbo_c128.sh: identical in every other resolved flag, FP4 indexer OFF, TBO ON,
# mem-frac 0.90, chunk 16,384/rank.
#
# WHY. The decode-stall dose-response (DATA_AND_ANALYSIS_20260902.md §4) shows
# per-step cost is flat at 1-2 prefills per 40-iteration log window and jumps
# 62-120 % at 3+, and the interval's own ceiling at 10 is 3-4 prefills per
# window. So the quota is exactly as permissive as the knee. Raising it to 20
# halves how many prefills can be admitted per unit time.
#
# HOW TO SCORE IT -- as a PAIR, not on ITL alone (§10). ATOM's TTFT is *worse*
# than ours at every concurrency we both have (10.9 s vs our 8.50 s at c128,
# 10.5 s vs 4.62 s at c64) while its ITL is better (61.5 vs 78.4 ms). The two
# engines look like different points on one prefill/decode trade-off, so we have
# ~2.4 s of TTFT headroom at c128 to spend on ITL.
#
#   PASS  : ITL p90 <= 61.5 ms AND TTFT avg <= 10.9 s  -> the gap was a tuning
#           choice, and the report should say so.
#   USEFUL: ITL improves and TTFT stays under 10.9 s   -> sweep on to 40.
#   FAIL  : ITL flat, or TTFT blows past 10.9 s        -> we are not on ATOM's
#           trade-off curve; then profile a decode step, which is the first time
#           that becomes the right move rather than a guess.
#
# Do NOT run this at c192. There the failure is prefill work per request
# (20,918 new tokens vs 12,304 at c160, §9), not prefill aggressiveness, so
# slowing admission would make TTFT and prefix eviction worse.
set -u

HERE=/workspace/claude-skills/agentx
RESULT_DIR=/workspace/results/interval20-c128
mkdir -p "$RESULT_DIR"

# Shared node. This script REFUSES rather than kills: a blind kill preamble
# destroyed another session's c96 arm at 05:12:25 on 2026-09-02. Three
# consecutive idle checks, because a single zero can land in the gap between a
# launcher tearing down its server and its aiperf export finishing.
# Bracketed patterns only -- a bare `pkill -f aiperf` matches this script's own
# command line.
busy_count() {
    ps -eo args | grep -Ec "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf (system_controller|profile)"
}
idle=0
for _ in $(seq 1 60); do        # up to 30 min at 30 s
    n=$(busy_count)
    if [ "$n" -eq 0 ]; then idle=$((idle + 1)); else
        [ "$idle" -ne 0 ] && echo "$(date '+%F %T') busy again ($n), idle streak reset"
        idle=0
    fi
    [ "$idle" -ge 3 ] && break
    sleep 30
done
if [ "$idle" -lt 3 ] ; then
    echo "FATAL: node still busy after 30 min ($(busy_count) procs) -- refusing to start"
    exit 3
fi
echo "$(date '+%F %T') node idle x3 -- launching"

cd /workspace/InferenceX
source "$HERE/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=128 DURATION=3600 PORT=8888
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0

export ENABLE_TBO=1          # measured: TBO off is worse on ITL, TTFT and tok/s
export CHUNK_PER_RANK=16384
export HSA_NO_SCRATCH_RECLAIM=0
export MEM_FRACTION_STATIC=0.90

# THE VARIABLE UNDER TEST. The launcher hardcodes --prefill-decode-interval 10
# inside PARALLEL_ARGS (b200align_mtp.sh:235, expanded at :353); EXTRA_ARGS is
# expanded later at :372, so this later occurrence is the one argparse keeps.
# VERIFY in $RESULT_DIR/sglang_command.txt and in server.log's server_args before
# trusting the result -- if both 10 and 20 appear, the last one must be 20.
export EXTRA_SERVER_ARGS="--prefill-decode-interval 20"

# FP4 indexer stays OFF, matching the dptbo-c128 baseline. (Every FP4 number on
# the board is stale anyway since 33979a814b changed the FP4 scoring path.)

export RESULT_DIR
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c128"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }

# The tree is now committed, so record the SHA as well as the md5s. HEAD should
# be 33979a814b (bounded prefill logits buffer) on top of 83310485e1 (#37353).
(cd /sgl-workspace/sglang && git log -2 --format='%h %s' && git status --porcelain | wc -l) \
    >"$RESULT_DIR/TREE_SHA_AT_START.txt" 2>&1
md5sum \
    /sgl-workspace/sglang/python/sglang/srt/arg_groups/serving_hook.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
    /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
    /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
    >"$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" 2>&1

bash "$HERE/vram_sampler.sh" "$RESULT_DIR/vram.csv" 15 &
SAMPLER_PID=$!
echo "vram sampler PID: $SAMPLER_PID"
trap 'kill -9 "$SAMPLER_PID" 2>/dev/null' EXIT

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"

kill -9 "$SAMPLER_PID" 2>/dev/null
trap - EXIT
sleep 90        # let aiperf finish its export / certification

# The launcher never kills its own server; it holds ~275 GB/GPU and the next arm
# then sits in the 15-min drain gate and fails. Skip our own PID.
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" | awk '{print $1}'); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
done
sleep 10

echo "=== interval really 20? (last occurrence wins) ==="
tr ' ' '\n' <"$RESULT_DIR/sglang_command.txt" | grep -A1 -x -- '--prefill-decode-interval' | grep -v '^--' | tr '\n' ' '; echo
echo "=== TBO on, FP4 off? ==="
echo "tbo=$(grep -c -- '--enable-two-batch-overlap' "$RESULT_DIR/sglang_command.txt") fp4=$(grep -c -- '--enable-deepseek-v4-fp4-indexer' "$RESULT_DIR/sglang_command.txt")"
echo "=== tree unchanged? ==="
md5sum -c --quiet "$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
echo "=== vram ==="
awk -F, 'NR>1 { if (m == "" || $5 < m) m = $5 } END { print "min free_gb seen: " m }' "$RESULT_DIR/vram.csv"
echo "=== late Triton device loads (OOR trigger; expect these, note the free mem) ==="
rg -c 'device-loaded after serving started' "$RESULT_DIR/server.log" || echo 0
echo "=== report ==="
python3 "$HERE/arm_report.py" interval20-c128 dptbo-c128
