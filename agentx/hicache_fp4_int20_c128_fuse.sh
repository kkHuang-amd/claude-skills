#!/usr/bin/env bash
# SHARED-EXPERTS FUSION ON, everything else identical to
# hicache-fp4-int20-c128-postswap. c128, DP attention, TBO, FP4 indexer,
# HiCache ratio 3.0, --prefill-decode-interval 20.
#
# WHY THIS IS A CLEAN ARM NOW. The launcher holds fusion OFF in the DP branch
# on purpose (:176-192) because "fusion has never been combined with DP here",
# and it left fusion-with-DP on the backlog as its own single-variable
# experiment. That experiment is this one. It is only clean because the
# post-swap bridge arm exists: its partner was measured on THIS image, on THIS
# tree, so fusion is the single variable. Run against a pre-swap arm it would
# move the image too.
#
# Flipped via FUSE_SHARED_EXPERTS=1, the launcher's own knob (:188), NOT via
# EXTRA_SERVER_ARGS. EXTRA_ARGS expands at :372, after SHARED_EXPERTS_ARGS at
# :359, so an EXTRA_SERVER_ARGS override would put BOTH
# --enforce-shared-experts-fusion and --disable-shared-experts-fusion on the
# command line and rely on argparse keeping the last. The knob keeps one flag.
#
# BASELINE: hicache-fp4-int20-c128-postswap
#   30,352 tok/s/GPU, ITL p90 57.75 ms, TTFT 12.16 s, cache 0.949,
#   KV pool 7,187,200, GPU pool 75 %, CPU-tier hit 0.004.
#
# READ THIS BEFORE SCORING IT: fusion is NOT memory-neutral. It replicates the
# shared experts (n_share_experts_fusion tracks tp_size), so static weight
# memory grows. --mem-fraction-static is a TOTAL budget, so at a fixed 0.90 the
# KV pool absorbs the difference and comes out SMALLER. The arm is therefore
# single-variable in the flag sense but NOT memory-matched, and a KV-pool
# change moves cache hit and TTFT on its own. Report max_total_num_tokens next
# to every number, and attribute nothing until the pool delta is known.
#
# THE MEM-FRACTION LADDER, AND WHAT IT COSTS. Requested behaviour: if the arm
# hits OOR, step --mem-fraction-static down and retry. Implemented as
# 0.90 -> 0.87 -> 0.85, each attempt in its own result dir so nothing is
# overwritten.
#
#   !! A SUCCESSFUL RETRY IS NOT BOARD-COMPARABLE. !!
#   ATOM runs --gpu-memory-utilization 0.9 and every arm on the board is 0.90.
#   That matching is what the whole ATOM comparison rests on, which is why
#   "do not lower mem-frac" is a settled rule in ITL_GAP_FINDINGS. An arm that
#   only survives at 0.87 answers "does fusion run", not "is fusion faster than
#   the baseline". If attempt 1 fails and a later one passes, that is itself the
#   finding: fusion does not fit at ATOM-matched memory.
set -u

HERE=/workspace/claude-skills/agentx
ARM_BASE=hicache-fp4-int20-c128-fuse
LOG_DIR=/shared_nfs/kk/logs/$ARM_BASE
mkdir -p "$LOG_DIR"

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
# Refuses rather than kills: we cannot tell whose allocation it is.
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

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=128 DURATION=3600 PORT=8888
export IS_AGENTIC=1

# THE VARIABLE UNDER TEST.
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

# REPORT EVERY LATE TRITON LOAD, not only the ones under 1 GiB of headroom.
# triton_load_watch._on_kernel_load returns early unless free VRAM is below
# SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB (default 1). The 2026-09-03 fusion arm
# ran with ~15 GB free, so its "0 late loads" proved NOTHING -- any load would
# have been invisible. Raising the threshold makes the count a real gate that is
# independent of how much headroom the arm happens to have. Warn-only, not
# SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY, so a stray load cannot kill the arm.
export SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=1000
export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer --prefill-decode-interval 20"

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
    export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c128"

    echo "############################################################"
    echo "$(date '+%F %T') ATTEMPT mem-fraction-static=$MF -> $ARM"
    [ "$MF" != "0.90" ] && echo "  !! BELOW 0.90: this attempt is NOT comparable to ATOM or to the board !!"
    echo "############################################################"

    (cd /sgl-workspace/sglang && git log -3 --format='%h %s' && git status --porcelain | wc -l) \
        >"$RESULT_DIR/TREE_SHA_AT_START.txt" 2>&1
    md5sum \
        /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
        /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
        /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py \
        /sgl-workspace/sglang/python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py \
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

    echo "--- fusion really on? (expect enforce_shared_experts_fusion True, n_share_experts_fusion 8) ---"
    rg -o "'enforce_shared_experts_fusion': [A-Za-z]+|'n_share_experts_fusion': [0-9]+" \
        "$RESULT_DIR/server.log" 2>/dev/null | sort -u | head -3
    grep -c -- '--enforce-shared-experts-fusion' "$RESULT_DIR/sglang_command.txt" 2>/dev/null \
        | sed 's/^/  enforce flag count: /'
    grep -c -- '--disable-shared-experts-fusion' "$RESULT_DIR/sglang_command.txt" 2>/dev/null \
        | sed 's/^/  disable flag count (must be 0): /'
    echo "--- KV pool (baseline 7,187,200; fusion grows weights so expect SMALLER) ---"
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
    *)      echo "!! mem-frac BELOW 0.90 -- NOT comparable to ATOM or to any 0.90 arm."
            echo "   The finding is that fusion does not fit at ATOM-matched memory."
            echo "   Do not put this row in DATA_AND_ANALYSIS sec 1 next to the others." ;;
esac
echo "############################################################"

echo "=== SINGLE VARIABLE: fusion on vs off, same image, same tree ==="
python3 "$HERE/arm_report.py" "$WINNER" hicache-fp4-int20-c128-postswap

echo "=== P0 gate (still unproven on a real arm as of sec 18) ==="
R=/workspace/results/$WINNER
rg -o 'Preloaded unified_kv prefill index kernels for compress ratios [^ ]*' \
    "$R/server.log" 2>/dev/null | sort -u | head -3
LATE=$(rg -c 'device-loaded after serving started' "$R/server.log" 2>/dev/null || echo 0)
echo "  late Triton device loads: $LATE   (PASS is 0; postswap arm had 74 before the Wc fix)"
echo -n "  gate meaningful? warn threshold must be >> free VRAM: "
rg -o 'SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB[^ ]*' "$R/sglang_command.txt" 2>/dev/null \
    || echo "env-only (expect 1000; if 1 GiB the count is NOT a gate)"
[ "${LATE:-1}" -eq 0 ] && echo "  -> P0 GATE PASS" || {
    echo "  -> P0 GATE FAIL. Which kernels:"
    rg -o "Triton kernel '[^']+' device-loaded" "$R/server.log" 2>/dev/null | sort | uniq -c | head -5
}

echo "=== free VRAM ==="
awk -F, 'NR>1 {print $5}' "$R/vram.csv" | sort -n \
    | awk '{a[NR]=$1} END {if(NR)printf "  free_gb  n=%d min=%.2f p10=%.2f med=%.2f max=%.2f\n", NR, a[1], a[int(NR*0.1)+1], a[int(NR/2)], a[NR]}'

echo "=== hicache tier (baseline: 38.5 %% full, CPU-tier hit 0.004 = no-op at c128) ==="
python3 - "$R/aiperf_artifacts/server_metrics_export.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1]))['metrics']
except Exception as e:
    print("  no server metrics:", e); raise SystemExit
tot, use = d.get('sglang:hicache_host_total_tokens'), d.get('sglang:hicache_host_used_tokens')
if tot and use:
    T = sum((s.get('stats') or {}).get('max') or 0 for s in tot['series'])
    U = sum((s.get('stats') or {}).get('avg') or 0 for s in use['series'])
    if T:
        print(f"  tier {100 * U / T:.1f} % full on average")
PY

echo "=== host DRAM used, before -> after (GB) ==="
paste <(awk '/^Mem:/{print $3}' "$R/host_dram_before.txt") \
      <(awk '/^Mem:/{print $3}' "$R/host_dram_after.txt")
echo "=== tree unchanged? ==="
md5sum -c --quiet "$R/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
