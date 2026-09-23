#!/usr/bin/env bash
# HiCache CPU-tier SMOKE at c192, DURATION=300. NOT a certifiable arm -- the
# duration gate (3500-3750 s) will fail by construction. Its only job is to
# answer, in ~30 min instead of ~92, four yes/no questions:
#
#   1. Does the server start at all with --enable-hierarchical-cache? (The
#      launcher's own validation is the first hurdle: KV_OFFLOADING must be
#      `dram`, NOT `hicache` -- benchmark_lib.sh:44-67 only accepts none|dram
#      and requires KV_OFFLOAD_BACKEND=hicache plus a positive
#      TOTAL_CPU_DRAM_GB. ITL_GAP_FINDINGS.md's `KV_OFFLOADING=hicache` is
#      wrong and would exit 1 before the model ever loads.)
#   2. Does the host pinned pool actually allocate at ratio 1.5, and how big is
#      it? (`Allocating N GB host memory for V4 paged pool` --
#      memory_pool_host.py:258, one line per pool per rank.)
#   3. Does it serve without the rust `DeepseekV4C4IndexerScale` pool-name
#      problem? Pre-checked and expected to be a non-issue: there are ZERO .rs
#      references to that name in this tree, PoolName already carries
#      DEEPSEEK_V4_C4_INDEXER_SCALE (hicache_storage.py:71), and CACHE_ARGS sets
#      no --hicache-storage-backend, so no rust storage tier is in play at all.
#      FP4 is off here anyway, which is what the original inference relied on.
#   4. Does the miss rate move in the right direction at all?
#
# THE REAL PASS CRITERION IS THE FULL ARM'S, NOT THIS ONE'S (§9): the
# token-weighted miss rate Sum(new)/Sum(new+cached) over `Prefill batch` lines must
# fall from 11.70 % (fp4-dptbo-c192) below 8.6 % (c160's level), and TTFT must
# follow. 300 s of measurement is far too short for a trace replay whose ISL
# mean is ~100k -- treat any miss-rate number here as direction only, never as
# a result.
#
# CONFOUND TO STATE WHEN REPORTING: the c192 baseline `fp4-dptbo-c192` ran with
# the FP4 indexer ON, and this runs with it OFF, so hicache is not the only
# variable. Defensible for the *cache* metric (the FP4 indexer changes attention
# scoring, not what the radix tree stores or hits) but it is NOT defensible for
# tok/s. If the full arm's headline needs to be a clean pair, the baseline has
# to be re-run FP4-off -- and every FP4 number is stale since 33979a814b anyway.
set -u

HERE=/workspace/claude-skills/agentx
RESULT_DIR=/workspace/results/hicache-smoke-c192
mkdir -p "$RESULT_DIR"

# Shared node. REFUSES rather than kills: a blind kill preamble destroyed
# another session's c96 arm at 05:12:25 on 2026-09-02. Bracketed patterns only,
# so this script's own command line is not matched.
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
export CONC=192 DURATION=300 PORT=8888
export IS_AGENTIC=1

# THE VARIABLE UNDER TEST.
export KV_OFFLOADING="dram"
export KV_OFFLOAD_BACKEND="hicache"
# Undocumented THIRD requirement: process_agentic_result.py:89 needs metadata
# whose .name equals KV_OFFLOAD_BACKEND, or the arm exits 1 AFTER a fully
# successful benchmark and writes no result JSON.
export KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
export TOTAL_CPU_DRAM_GB=1024      # declared budget only; this launcher just
                                   # echoes it. Actual host pinned bytes are
                                   # HICACHE_RATIO x device KV pool x TP, which
                                   # at ratio 1.5 lands around 500 GB against
                                   # 2,960 GB available -- check question 2.
export HICACHE_RATIO=1.5           # launcher defaults, pinned explicitly so the
export HICACHE_WRITE_POLICY=write_through   # smoke and the full arm cannot
export HICACHE_IO_BACKEND=direct            # silently differ.
export HICACHE_MEM_LAYOUT=page_first_direct

export ENABLE_TBO=1          # settled: TBO off is worse on ITL, TTFT and tok/s
export CHUNK_PER_RANK=16384
export HSA_NO_SCRATCH_RECLAIM=0
export MEM_FRACTION_STATIC=0.90   # do NOT lower: ATOM runs 0.9 and every arm on
                                  # the board is 0.90

# No EXTRA_SERVER_ARGS: the launcher's own --prefill-decode-interval 10 stands,
# matching the c192 baseline. The interval sweep is a separate, unrelated track
# (task 1) and must not be entangled with this one. FP4 indexer off.

export RESULT_DIR
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c192"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }

(cd /sgl-workspace/sglang && git log -2 --format='%h %s' && git status --porcelain | wc -l) \
    >"$RESULT_DIR/TREE_SHA_AT_START.txt" 2>&1
md5sum \
    /sgl-workspace/sglang/python/sglang/srt/arg_groups/serving_hook.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
    /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
    /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
    >"$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" 2>&1

# Host DRAM before, so question 2 can be answered from the delta as well as the
# log line. hicache pins memory, which `free` shows as used, not buff/cache.
free -g >"$RESULT_DIR/host_dram_before.txt"

bash "$HERE/vram_sampler.sh" "$RESULT_DIR/vram.csv" 15 &
SAMPLER_PID=$!
echo "vram sampler PID: $SAMPLER_PID"
trap 'kill -9 "$SAMPLER_PID" 2>/dev/null' EXIT

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"

free -g >"$RESULT_DIR/host_dram_after.txt"
kill -9 "$SAMPLER_PID" 2>/dev/null
trap - EXIT
sleep 60

# The launcher never kills its own server; it holds ~275 GB/GPU and the next arm
# would then sit in the idle gate and fail. Skip our own PID.
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" | awk '{print $1}'); do
    [ "$p" = "$$" ] && continue
    kill -9 "$p" 2>/dev/null
done
sleep 10

echo "=== Q1: did hicache reach the server command? ==="
for f in --enable-hierarchical-cache --hicache-ratio --hicache-write-policy \
         --hicache-io-backend --hicache-mem-layout; do
    printf '%s=%s ' "$f" "$(grep -c -- "$f" "$RESULT_DIR/sglang_command.txt" 2>/dev/null || echo 0)"
done; echo
echo "server_args hierarchical:"
rg -o "'enable_hierarchical_cache': [A-Za-z]+|'hicache_ratio': [0-9.]+|'hicache_mem_layout': '[a-z_]+'" \
    "$RESULT_DIR/server.log" 2>/dev/null | sort -u | head -5

echo "=== Q2: host pool allocated, and how much? ==="
rg -o "Allocating [0-9.]+ GB host memory for V4 paged pool '[a-z0-9_]+'" "$RESULT_DIR/server.log" 2>/dev/null \
    | sort | uniq -c | head -12
echo -n "sum over all ranks: "
rg -o 'Allocating ([0-9.]+) GB host memory' -r '$1' "$RESULT_DIR/server.log" 2>/dev/null \
    | awk '{s += $1} END {printf "%.1f GB\n", s}'
echo "host DRAM used, before -> after (GB):"
paste <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_before.txt") \
      <(awk '/^Mem:/{print $3}' "$RESULT_DIR/host_dram_after.txt")

echo "=== Q3: did it serve, or die? ==="
echo "FP4 off? fp4=$(grep -c -- '--enable-deepseek-v4-fp4-indexer' "$RESULT_DIR/sglang_command.txt" 2>/dev/null || echo 0) tbo=$(grep -c -- '--enable-two-batch-overlap' "$RESULT_DIR/sglang_command.txt" 2>/dev/null || echo 0)"
rg -c 'Traceback|HSA_STATUS_ERROR|OutOfMemory|Aborting with error' "$RESULT_DIR/server.log" 2>/dev/null || echo "no fatal patterns: 0"
rg -o 'ValueError: Unsupported layout.*|KeyError.*[Pp]ool.*|Unsupported pool.*' "$RESULT_DIR/server.log" 2>/dev/null | head -3 | cut -c1-200
echo "decode batches seen: $(rg -c 'Decode batch' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)"

echo "=== Q4: miss rate direction (DIRECTION ONLY -- 300 s is not a result) ==="
echo "baseline fp4-dptbo-c192 = 11.70 %, target < 8.6 % (c160), on a FULL arm"
rg -o '#new-token: [0-9]+, #cached-token: [0-9]+' "$RESULT_DIR/server.log" 2>/dev/null \
    | awk -F'[:,]' '{n += $2; c += $4} END {if (n + c > 0) printf "miss = %.2f %% over %d prefill batches\n", 100 * n / (n + c), NR}'
echo "=== tier split: is any hit now served off-GPU? (0 by construction until now) ==="
# Takes an ARM NAME, not a path, and reads the arm's json + run.log. Its tier
# check has been vacuous on every arm so far precisely because hicache was off;
# a non-zero cpu/external share here is the whole point of this run.
python3 "$HERE/cache_tier_gate.py" hicache-smoke-c192 2>&1 | tail -12 | cut -c1-200

echo "=== vram (hicache must not have cost device pool) ==="
awk -F, 'NR>1 { if (m == "" || $5 < m) m = $5 } END { print "min free_gb seen: " m }' "$RESULT_DIR/vram.csv"
echo "late Triton device loads: $(rg -c 'device-loaded after serving started' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)"
echo "=== tree unchanged? ==="
md5sum -c --quiet "$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
echo
echo "SMOKE DONE. Duration gate WILL fail (300 s vs 3500-3750) -- that is by"
echo "design. Go/no-go for the full arm is Q1-Q3, not Q4."
