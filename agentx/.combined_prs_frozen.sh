#!/usr/bin/env bash
# A/B the four decode-path PRs (#37423 / #37658 / #34624 / #37580) on ONE tree.
#
# Usage:  CONC_TARGET=128 PR_GATES=off bash .combined_prs_frozen.sh
#         CONC_TARGET=128 PR_GATES=on  bash .combined_prs_frozen.sh
#
# Config is byte-for-byte hicache_fp4_int20_fuse_conc.sh -- DP attention, TBO,
# FP4 indexer, HiCache ratio 3.0, --prefill-decode-interval 20, shared-experts
# fusion ON, mem-fraction-static 0.90, CHUNK_PER_RANK 16384, DURATION 3600 --
# with exactly two additions: it serves from the combined worktree, and it sets
# the four PRs' env gates.
#
# WHY BOTH ARMS AND NOT ONE.
#
#   The combined worktree is origin/main 8770c1db1f + the four PR merges. The
#   benchmark tree /sgl-workspace/sglang is efaeb6f664, whose base 2641e427be is
#   62 commits behind that main. So a single gates-ON arm scored against
#   hicache-fp4-int20-c128-fuse-mf090 moves the base AND the features at once,
#   and the base bump is not small (a previous main bump measured +7.5-8 %).
#   Replicate spread here is 5.67 % and the PRs' published effects are +2.7 % and
#   +2.95 %, i.e. already inside it. Only OFF-vs-ON on the SAME tree can say
#   anything about the features.
#
#   The gates-OFF arm doubles as the base-bump measurement: OFF vs
#   hicache-fp4-int20-c128-fuse-mf090 is the 62 commits, same config otherwise.
#
# WHAT THIS A/B CANNOT SEPARATE.
#
#   #37580 has NO env var -- it is an unconditional `#ifdef USE_ROCM` early
#   return in TopKKernel::plan. It is therefore in BOTH arms and the OFF-vs-ON
#   delta covers three PRs, not four. #37580's own claim is 1.2-1.5 us per
#   top-k v2 metadata build; it rides in the base-bump comparison instead.
#
# ONE GATE ONLY BECAME LIVE TODAY.
#
#   SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE was INERT on every FP4 arm ever run:
#   compressor_v2 required plan.is_decode, and with MTP the steady-state modes
#   are TARGET_VERIFY / DRAFT_EXTEND, so no decode plan exists. The extend
#   variant added 2026-09-04 (FP4_COMPRESS_FUSION.md) is what makes it bind.
#   The ON arm must print `[compress-fusion] ACTIVE, fp4 extend epilogue` 8x,
#   once per TP worker. If it does not, the arm scored the baseline -- see the
#   verification block below, which fails loudly rather than quietly.
set -u

HERE=/workspace/claude-skills/agentx
WORKTREE=/shared_nfs/kk/tmp/combined
CONC_TARGET="${CONC_TARGET:?set CONC_TARGET, e.g. 128}"
PR_GATES="${PR_GATES:?set PR_GATES=off or PR_GATES=on}"
case "$PR_GATES" in off|on) ;; *) echo "FATAL: PR_GATES must be off|on"; exit 2 ;; esac

ARM_BASE=combined-prs-c${CONC_TARGET}-${PR_GATES}
LOG_DIR=/shared_nfs/kk/logs/$ARM_BASE
mkdir -p "$LOG_DIR"

test -d "$WORKTREE/python/sglang" || { echo "FATAL: no worktree at $WORKTREE"; exit 2; }

# ---------------------------------------------------------------- node gates
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
if [ "$idle" -lt 3 ]; then
    echo "FATAL: node still busy after 30 min ($(busy_count) procs) -- refusing to start"
    exit 3
fi
echo "$(date '+%F %T') node idle x3 -- launching"

# SECOND GATE: VRAM, not just processes. A job in another container's PID
# namespace is invisible to busy_count() but its allocation is not, and starting
# on top of it silently shrinks the KV pool and breaks the memory matching.
held_gb() {
    rocm-smi --showmeminfo vram 2>/dev/null \
        | grep -oE 'Total Used Memory \(B\): [0-9]+' \
        | awk '{s += $NF} END {printf "%.0f", s / 1e9}'
}
vram_ok=0
for _ in $(seq 1 60); do
    h=$(held_gb)
    if [ "${h:-999}" -lt 10 ]; then vram_ok=$((vram_ok + 1)); else
        [ "$vram_ok" -ne 0 ] && echo "$(date '+%F %T') VRAM back up (${h} GB), streak reset"
        vram_ok=0
    fi
    [ "$vram_ok" -ge 3 ] && break
    sleep 30
done
if [ "$vram_ok" -lt 3 ]; then
    echo "FATAL: $(held_gb) GB still held across the 8 GPUs after 30 min with no"
    echo "visible process. Refusing: the KV pool would be smaller than every other"
    echo "arm's and the result would not be comparable. Do NOT kill blindly --"
    echo "check 'ps -eo pid,lstart,args' and ask before touching another session's job."
    exit 4
fi
echo "$(date '+%F %T') VRAM clear ($(held_gb) GB) -- launching"

# ------------------------------------------------------------------- settings
cd /workspace/InferenceX
source "$HERE/agentx_env.sh"

# Serve from the combined worktree. The launcher runs `python3 -m
# sglang.launch_server`, so shadowing the package is enough and
# /sgl-workspace/sglang stays untouched. Proven by the 2026-09-04 GSM8K arm,
# which reached worktree-only code through exactly this path.
export PYTHONPATH="$WORKTREE/python"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC="$CONC_TARGET" DURATION=3600 PORT=8888
export IS_AGENTIC=1

export FUSE_SHARED_EXPERTS=1

export KV_OFFLOADING="dram"
export KV_OFFLOAD_BACKEND="hicache"
# Undocumented third requirement: process_agentic_result.py:89 needs metadata
# whose .name equals KV_OFFLOAD_BACKEND, or the arm exits 1 AFTER a fully
# successful benchmark and writes no result JSON.
export KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
export TOTAL_CPU_DRAM_GB=2048
export HICACHE_RATIO=3.0
export HICACHE_WRITE_POLICY=write_through
export HICACHE_IO_BACKEND=direct
export HICACHE_MEM_LAYOUT=page_first_direct

export ENABLE_TBO=1
export CHUNK_PER_RANK=16384
export HSA_NO_SCRATCH_RECLAIM=0

export SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=1000
export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer --prefill-decode-interval 20"

# ------------------------------------------------------- THE VARIABLE UNDER TEST
# All four are read at IMPORT time (module-level constants or import-time
# binding), so they must be set before launch_server starts. Flipping one later
# does nothing.
#
# SGLANG_OPT_FP8_WO_A_GEMM defaults TRUE upstream and, on this branch, binds by
# itself on HIP because model_hook now clears it only when
# _rocm_fp8_wo_a_supported() is False -- and that returns True here. So the OFF
# arm has to set it to 0 explicitly; leaving it unset is an ON arm.
if [ "$PR_GATES" = "on" ]; then
    export SGLANG_OPT_FP8_WO_A_GEMM=1           # 37423
    export SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=1  # 37658, default off
    export SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE=1 # 34624, inert before today
    export SGLANG_OPT_NATIVE_BPRESHUFFLE_SCALE=1 # 34624
else
    export SGLANG_OPT_FP8_WO_A_GEMM=0
    export SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=0
    export SGLANG_OPT_FUSE_COMPRESS_NORM_ROPE=0
    export SGLANG_OPT_NATIVE_BPRESHUFFLE_SCALE=0
fi

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }

# --------------------------------------------------------------- attempt loop
oor_hit() {   # $1 = result dir
    rg -c 'HSA_STATUS_ERROR_OUT_OF_RESOURCES|out of memory|OutOfMemoryError|CUDA error: out of memory' \
        "$1/server.log" 2>/dev/null | head -1
}

WINNER=""
for MF in 0.90 0.87 0.85; do
    ARM="$ARM_BASE-mf${MF/./}"
    RESULT_DIR=/workspace/results/$ARM
    mkdir -p "$RESULT_DIR"
    export MEM_FRACTION_STATIC="$MF"
    export RESULT_DIR
    export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
    export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c$CONC_TARGET"

    echo "############################################################"
    echo "$(date '+%F %T') ATTEMPT mem-fraction-static=$MF -> $ARM  (PR_GATES=$PR_GATES)"
    [ "$MF" != "0.90" ] && echo "  !! BELOW 0.90: this attempt is NOT comparable to ATOM or to the board !!"
    echo "############################################################"

    # WHICH TREE ACTUALLY SERVED. The stock script records
    # /sgl-workspace/sglang's SHA, which is the wrong tree here and would read as
    # if the arm ran efaeb6f664. Record the worktree, and resolve the package the
    # way the launcher's interpreter will, so "it served from the worktree" is
    # evidence rather than an assumption.
    (cd "$WORKTREE" && git log -3 --format='%h %s' && git status --porcelain | wc -l) \
        >"$RESULT_DIR/TREE_SHA_AT_START.txt" 2>&1
    {
        echo "PYTHONPATH=$PYTHONPATH"
        python3 -c "import sglang, sys; print('sglang.__file__', sglang.__file__)" 2>&1
        echo "--- benchmark tree (NOT served, for the record) ---"
        (cd /sgl-workspace/sglang && git log -1 --format='%h %s')
    } >"$RESULT_DIR/SERVED_FROM.txt" 2>&1
    env | grep -E '^SGLANG_OPT_' | sort >"$RESULT_DIR/PR_GATES.txt"
    md5sum \
        "$WORKTREE"/python/sglang/srt/layers/attention/dsv4/indexer.py \
        "$WORKTREE"/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
        "$WORKTREE"/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py \
        "$WORKTREE"/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py \
        "$WORKTREE"/python/sglang/srt/layers/attention/dsv4/compressor_v2.py \
        /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
        >"$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" 2>&1
    free -g >"$RESULT_DIR/host_dram_before.txt"

    bash "$HERE/vram_sampler.sh" "$RESULT_DIR/vram.csv" 15 &
    SAMPLER_PID=$!
    trap 'kill -9 "$SAMPLER_PID" 2>/dev/null' EXIT

    bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
    RC=$?
    echo "ARM_EXIT=$RC  (attempt mf=$MF)"

    free -g >"$RESULT_DIR/host_dram_after.txt"
    kill -9 "$SAMPLER_PID" 2>/dev/null
    trap - EXIT
    sleep 90        # let aiperf finish its export / certification

    # The launcher never kills its own server; it holds ~275 GB/GPU and the next
    # attempt would then sit in the idle gate and fail. Skip our own PID.
    for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf" | awk '{print $1}'); do
        [ "$p" = "$$" ] && continue
        kill -9 "$p" 2>/dev/null
    done
    sleep 20

    # ---------------------------------------------- did the gates actually bind?
    # A gate that cannot be confirmed cannot be scored. This is the whole reason
    # the marker exists: three GSM8K arms on 2026-08-29 were scored before anyone
    # checked, and all three had the fusion INACTIVE.
    echo "--- PR gates as the server saw them ---"
    cat "$RESULT_DIR/PR_GATES.txt" | sed 's/^/  /'
    n_active=$(rg -c 'compress-fusion\] ACTIVE' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)
    n_skip=$(rg -c 'compress-fusion\] gate set but SKIPPED' "$RESULT_DIR/server.log" 2>/dev/null || echo 0)
    echo "  [compress-fusion] ACTIVE lines: $n_active   SKIPPED lines: $n_skip"
    rg -o 'compress-fusion\] ACTIVE, [a-z0-9 ]+ epilogue' "$RESULT_DIR/server.log" 2>/dev/null \
        | sort | uniq -c | sed 's/^/  /'
    if [ "$PR_GATES" = "on" ] && [ "${n_active:-0}" -eq 0 ]; then
        echo "  !! GATES ON BUT FUSION NEVER BOUND -- this arm scored the baseline."
        echo "     Do NOT report it as a PR result. Check the SKIPPED lines above."
    fi
    if [ "$PR_GATES" = "off" ] && [ "${n_active:-0}" -ne 0 ]; then
        echo "  !! GATES OFF BUT FUSION BOUND -- the OFF arm is not a baseline."
    fi
    echo "--- served from (must be the worktree, not /sgl-workspace) ---"
    sed 's/^/  /' "$RESULT_DIR/SERVED_FROM.txt"

    echo "--- shared-experts fusion really on? (expect True, n_share_experts_fusion 8) ---"
    rg -o "'enforce_shared_experts_fusion': [A-Za-z]+|'n_share_experts_fusion': [0-9]+" \
        "$RESULT_DIR/server.log" 2>/dev/null | sort -u | head -3
    grep -c -- '--disable-shared-experts-fusion' "$RESULT_DIR/sglang_command.txt" 2>/dev/null \
        | sed 's/^/  disable flag count (must be 0): /'
    echo "--- interval override took? (must be 20; launcher hardcodes 10 first) ---"
    rg -o "'prefill_decode_interval': [0-9]+" "$RESULT_DIR/server.log" 2>/dev/null | sort -u | head -2
    echo "--- KV pool (c128 fusion got 7,412,480; rows compare only if this matches within ~2 %) ---"
    rg -o 'max_total_num_tokens=[0-9]+' "$RESULT_DIR/server.log" 2>/dev/null | tail -1
    echo "--- OOR / fatal patterns ---"
    rg -o 'HSA_STATUS_ERROR_OUT_OF_RESOURCES|OutOfMemoryError|CUDA error: out of memory|Traceback' \
        "$RESULT_DIR/server.log" 2>/dev/null | sort | uniq -c | head -6

    # Same contract arm_report.py:24 uses. NOT a bare *.json glob: the power
    # artifacts (agentic_power_window.json) match that and would fake a success.
    n_res=$(find "$RESULT_DIR" -maxdepth 1 -name 'dsv4_fp4_sglang_*_c*.json' -size +0 | wc -l)
    if [ "$RC" -eq 0 ] && [ "$n_res" -gt 0 ]; then
        WINNER="$ARM"
        echo "$(date '+%F %T') attempt mf=$MF SUCCEEDED ($n_res result json)"
        break
    fi
    echo "$(date '+%F %T') attempt mf=$MF produced no scorable result (rc=$RC, json=$n_res)"

    n_oor=$(oor_hit "$RESULT_DIR"); n_oor=${n_oor:-0}
    if [ "${n_oor:-0}" -eq 0 ]; then
        echo "$(date '+%F %T') attempt mf=$MF failed with NO OOR signature."
        echo "Lowering mem-frac would not address it -- stopping so the real cause"
        echo "is not masked. Check $RESULT_DIR/server.log."
        break
    fi
    echo "$(date '+%F %T') attempt mf=$MF hit OOR ($n_oor matches) -- stepping mem-frac down"
done

# ----------------------------------------------------------------- reporting
echo "############################################################"
if [ -z "$WINNER" ]; then
    echo "NO ATTEMPT SUCCEEDED. Nothing to score."
    exit 5
fi
echo "WINNER: $WINNER"
case "$WINNER" in
    *mf090) echo "mem-frac 0.90 -- ATOM-matched, this IS board-comparable." ;;
    *)      echo "!! mem-frac BELOW 0.90 -- NOT comparable to ATOM or to any 0.90 arm." ;;
esac
echo "############################################################"

R=/workspace/results/$WINNER

if [ "$PR_GATES" = "on" ]; then
    OFF_ARM="combined-prs-c${CONC_TARGET}-off-mf090"
    if [ -d "/workspace/results/$OFF_ARM" ]; then
        echo "=== THE COMPARISON THAT COUNTS: gates ON vs OFF, same tree, same conc ==="
        echo "    Covers #37423 + #37658 + #34624. NOT #37580 (no env var, in both)."
        echo "    Replicate spread is 5.67 % -- a smaller delta is null, say 'null'."
        python3 "$HERE/arm_report.py" "$WINNER" "$OFF_ARM"
    else
        echo "=== no gates-OFF arm yet ($OFF_ARM); run PR_GATES=off first ==="
    fi
else
    # Baseline MUST follow CONC_TARGET. Hardcoding the c128 arm here printed a
    # cross-concurrency comparison on the first c192 run, which is invalid --
    # trace mix and ISL move with concurrency (conc-and-trace-mix.md 19.6).
    # arm_report.py flags it rather than faking a number, but do not rely on that.
    BOARD_ARM="hicache-fp4-int20-c${CONC_TARGET}-fuse-mf090"
    if [ -d "/workspace/results/$BOARD_ARM" ]; then
        echo "=== BASE BUMP: gates OFF vs the board's c${CONC_TARGET}, 62 commits of main ==="
        echo "    Same config, same conc, different tree. This is NOT a PR result."
        python3 "$HERE/arm_report.py" "$WINNER" "$BOARD_ARM"
    else
        echo "=== no board arm $BOARD_ARM to compare against ==="
    fi
fi

echo "=== late Triton loads ==="
LATE=$(rg -c 'device-loaded after serving started' "$R/server.log" 2>/dev/null || echo 0)
echo "  late Triton device loads: $LATE   (threshold was 1000 GB, so this IS measured)"

echo "=== free VRAM ==="
awk -F, 'NR>1 {print $5}' "$R/vram.csv" | sort -n \
    | awk '{a[NR]=$1} END {if(NR)printf "  free_gb  n=%d min=%.2f p10=%.2f med=%.2f max=%.2f\n", NR, a[1], a[int(NR*0.1)+1], a[int(NR/2)], a[NR]}'

echo "=== host DRAM used, before -> after (GB) ==="
paste <(awk '/^Mem:/{print $3}' "$R/host_dram_before.txt") \
      <(awk '/^Mem:/{print $3}' "$R/host_dram_after.txt") 2>/dev/null | sed 's/^/  /'
