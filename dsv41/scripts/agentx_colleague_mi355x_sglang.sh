#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4.1-Flash AgentX on MI355X (gfx950) with SGLang native DSpark.
# The model-side recipe follows dsv41flash_fp4_b300_sglang_mtp.sh, the sibling
# on the same 288 GiB card: same golden AL, memory split, SWA tails and bounded
# prefill chunk. The platform handling (HIP visibility, AITER, breakable prefill
# graphs) follows the validated dsv4_fp4_mi355x_sglang_mtp.sh, and the topology
# bounds follow the MI355X vLLM sibling, which measured TP2 and TP4 on this SKU.
# https://lmsysorg.mintlify.app/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1
# LOCAL COPY of /sgl-workspace/dsv41flash_fp4_mi355x_sglang_mtp.sh (colleague recipe, 2026-09-26). Only changes:
#   benchmark_lib path, hf download skipped for an existing local MODEL_PATH, EXTRA_ARGS knob. Run via agentx_colleague_run.sh.
source "${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX}/benchmarks/benchmark_lib.sh"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY SPEC_DECODING DP_ATTENTION PORT
require_agentic_kv_offload_none
export GPU_COUNT="$TP"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

# ROCR/HIP visibility under slurm cgroups.
if [[ -n "${ROCR_VISIBLE_DEVICES:-}" ]]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

# Complete/resume partial downloads instead of trusting nonempty directories.
if [[ -f "${MODEL_PATH:-}/config.json" ]]; then
    echo "Using local MODEL_PATH=$MODEL_PATH (skip hf download)"
elif [[ -n "${MODEL_PATH:-}" && "$MODEL_PATH" != "$MODEL" ]]; then
    hf download "$MODEL" --local-dir "$MODEL_PATH"
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi

rocm-smi || true
amd-smi || true

# Pin the full-context corpus for this 1M-context recipe, as the MI300X/MI325X/
# MI355X vLLM DSv4.1-Flash arms do.
export WEKA_LOADER_OVERRIDE=semianalysis_cc_traces_weka_062126
resolve_trace_source
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Agentic warmup dispatches hundreds of large prompts at once and SGLang's
# tokenizer can leave bytes unacknowledged past AIPerf's default 30 s
# TCP_USER_TIMEOUT, so Linux aborts live localhost connections.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# Outlast AIPerf's pooled connections so an inter-turn idle gap cannot race
# Uvicorn's five-second keep-alive closure.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# AgentX measures the thinking-on regime, which is also the committed golden-AL
# curve. SGLang ships thinking off by default for this model.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV41_REASONING_EFFORT=high

# ---- ROCm / AITER platform settings -----------------------------------------
# Hardware-level only. The DSv4-Pro MI355X recipe additionally carries
# SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton with SGLANG_DSV4_UNIFIED_KV_FP8
# and the SGLANG_OPT_* MLA/cache flags; those are gated on the DSv4-Pro unified
# pool and are deliberately not carried over to DSv4.1-Flash, which reaches its
# KV through the SWA + Engram path instead.
export SGLANG_USE_AITER=1
export SGLANG_USE_ROCM700A=0
# Triton's HIP async-copy lowering, off in the qualified gfx950 DSv4.1-Flash
# bring-up. Re-measure before turning it back on.
export TRITON_HIP_USE_ASYNC_COPY=0
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export AITER_BF16_FP8_MOE_BOUND=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HSA_NO_SCRATCH_RECLAIM=0
export GPU_MAX_HW_QUEUES=2

# ---- Topology ---------------------------------------------------------------
# The DP-attention + sglang-router topology is validated for DSv4-Pro on this
# SKU, not for DSv4.1-Flash: the DSpark worker requires attn_tp=1 under DP and
# the Engram host tables are sized per rank. Reject it rather than ignore it.
case "$DP_ATTENTION" in
    false) ;;
    true) echo "Error: DP_ATTENTION=true is not supported by this recipe" >&2; exit 1 ;;
    *) echo "Error: DP_ATTENTION must be true or false, got '$DP_ATTENTION'" >&2; exit 1 ;;
esac

case "$TP" in
    2|4) ;;
    *) echo "Unsupported DSpark TP=$TP; expected 2 or 4" >&2; exit 1 ;;
esac
# Row-sharded host Engram tables at every TP, as the B200 SGLang recipe does.
# GPU-resident tables (~47.2 GiB per rank at TP4) left TP4 c64 with almost no
# prefix-cache hits and a prefill queue that never drained (tp4m_c64).
# ENGRAM_HOST_TABLE=0 keeps them resident.
if [[ "${ENGRAM_HOST_TABLE:-1}" == 0 ]]; then
    export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
    unset SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT
else
    export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
    export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
fi

# AgentX concurrency counts live session trees, not individual requests.
# Allow subagent fan-out to exceed CONC without clipping request bursts, but
# never let the pool exceed the decode graph batch: a DSpark verify step for a
# batch above the captured 64 runs eagerly and allocates its attention
# workspace on the fly, which OOMed the H200 eval at 128 running requests.
# Batches within the graph tier reuse the capture-time workspace instead.
CUDA_GRAPH_MAX_BS=64
MAX_RUNNING_REQUESTS=$((2 * CONC))
if (( MAX_RUNNING_REQUESTS > CUDA_GRAPH_MAX_BS )); then
    MAX_RUNNING_REQUESTS=$CUDA_GRAPH_MAX_BS
fi

# The memory split follows B300, an SGLang sibling on an identically sized
# 288 GiB card whose committed search space runs TP2 across the full 1-128
# curve. AgentX reuses long prefixes across turns even
# at low session concurrency, and TP2's ~147.76 GiB of target plus draft weights
# leaves far less KV room than TP4's share does, so give TP2 more of the budget
# as concurrency climbs rather than retaining more SWA tails. This node measured
# 42.69 GiB free after graph capture at 0.80, within a GiB of the 43.59 GiB B300
# recorded at the same fraction before deciding C64 needed 0.85.
MEM_FRACTION_STATIC_DEFAULT=0.70
if (( TP == 2 && CONC >= 32 )); then
    MEM_FRACTION_STATIC_DEFAULT=0.85
elif (( TP == 2 && CONC >= 16 )); then
    MEM_FRACTION_STATIC_DEFAULT=0.80
fi
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-$MEM_FRACTION_STATIC_DEFAULT}"

# Chunked requests leave reusable SWA tails in the radix tree. Size retained
# tails by session concurrency, rather than the capped running-request count.
# The tails and full KV share a fixed pool; the cap preserves full-prefix space.
# 64*CONC on both TP bands, as B300 does.
SWA_PREFIX_TAILS=$((64 * CONC))
if (( SWA_PREFIX_TAILS < 128 )); then
    SWA_PREFIX_TAILS=128
elif (( SWA_PREFIX_TAILS > 4096 )); then
    SWA_PREFIX_TAILS=4096
fi

# 4096 on both TP bands, as B300 does: the sparse-attention indexer and the
# DSpark prefill buffers scale with the chunk times the 1M context, and the
# upstream 16384 exhausts HBM on the first 66k-99k-token AgentX prompts. It also
# bounds the decode stall a queued prefill imposes on in-flight requests, which
# is what the ITL tail measures.
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-4096}"
# Forces N decode steps between one prefill batch and the next, so a queue of
# long AgentX prefixes cannot starve in-flight decode. 16 is the latency-oriented
# cadence every other DSv4.1-Flash recipe runs (B200/H200/GB300 hardcode it;
# B300 drops to 4 to buy prefill duty at TP2 c32+). TP2 c64 drops to 4 here: at
# 16 its prefill queue averaged 41 requests and median TTFT reached 35 s.
# Env-overridable so a sweep can retune it without editing the recipe.
PREFILL_DECODE_INTERVAL_DEFAULT=16
if (( TP == 2 && CONC >= 64 )); then
    PREFILL_DECODE_INTERVAL_DEFAULT=4
fi
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-$PREFILL_DECODE_INTERVAL_DEFAULT}"

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# The MI355X runner assigns a per-runner port; do not reselect it here.
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
echo "Using SGLang endpoint ${AIPERF_SERVER_URL}"

# DSpark is the checkpoint's own bundled draft: no EAGLE/MTP path and no
# --speculative-num-steps knob; the block size is the only tunable. Throughput
# uses the committed golden acceptance curve; evals verify real draft tokens.
# Leave SGLang's default draft computation and precision.
case "$SPEC_DECODING" in
    mtp|draft_model) ;;
    *) echo "Unsupported SPEC_DECODING=$SPEC_DECODING; expected mtp or draft_model" >&2; exit 1 ;;
esac
unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
DSPARK_BLOCK_SIZE=5
# golden_al_distribution/dsv41flash_dspark.yaml, thinking_on, five draft tokens.
DSV41_GOLDEN_AL=3.51
if [[ "$EVAL_ONLY" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark block size: $DSPARK_BLOCK_SIZE, golden AL=$DSV41_GOLDEN_AL"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    # Backends resolve automatically on gfx950 (dsv4 attention, AITER MoE); the
    # cookbook warns that overriding them costs decode speed, and the DSv4-Pro
    # MI355X recipe's explicit --attention-backend/--page-size/--kv-cache-dtype
    # belong to its unified-KV pool, not to this model.
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
    --swa-prefix-tails "$SWA_PREFIX_TAILS"
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    # tc_piecewise is unavailable on HIP, and naming the backend explicitly is
    # also what skips the auto-disable cascade that would otherwise drop prefill
    # graphs entirely. Both qualified gfx950 SGLang launches pin it this way.
    --cuda-graph-backend-prefill breakable
    --reasoning-parser auto
    --tool-call-parser auto
    # Draft-token forward passes under long-context agentic load block the
    # scheduler long enough to trip the 1800 s default watchdog mid-warmup.
    --watchdog-timeout 3600
    --enable-metrics
)
# LOCAL: EXTRA_ARGS="..." appended verbatim (e.g. --fp8-gemm-backend aiter)
[ -n "${EXTRA_ARGS:-}" ] && SGLANG_CMD+=(${EXTRA_ARGS})
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

# A leaked server keeps its HBM and blocks the next job on this node; always
# tear the tree down.
SERVER_PID=""
cleanup_server() {
    local rc=$?
    trap - EXIT INT TERM
    stop_background_process_tree "$SERVER_PID" "SGLang server" 60
    exit "$rc"
}
trap cleanup_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# LOCAL: SERVER_ONLY=1 keeps the server up (no aiperf) for profiling; stop it by killing this script's PID.
if [[ "${SERVER_ONLY:-0}" == 1 ]]; then
    echo "SERVER_ONLY: ready on port $PORT, server PID $SERVER_PID"
    wait "$SERVER_PID"
    exit $?
fi

if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
