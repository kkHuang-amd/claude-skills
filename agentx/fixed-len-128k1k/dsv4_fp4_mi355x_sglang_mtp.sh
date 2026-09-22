#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP4 + SGLang DSpark MTP on MI355X, FIXED sequence length.
#
# Purpose: an ISL/OSL-controlled companion to
#   benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
# whose partner on the NVIDIA side is dsv4_fp4_b200_sglang_mtp.sh in THIS
# directory. Every request is the same length, so DP ranks cannot go unbalanced
# and the remaining MI355X-vs-B200 delta is kernel-wise.
#
# Server configuration is copied from the AGENTIC MI355X launcher (DSpark gamma,
# pinned acceptance length, dsv4 attention backend, FP4 indexer, page size, SWA
# ratio, the DP env block). Deltas, all forced by the fixed-length workload:
#
#   * No sglang-router. Fixed length makes every request identical, so the DP
#     controller's own admission is already balanced and cache-aware routing has
#     nothing to route on. Clients hit the backend port directly.
#     (non-PD load_balance_method resolves to round_robin --
#     srt/arg_groups/serving_hook.py:200)
#   * --disable-radix-cache, no HiCache. Random fixed-length prompts share no
#     prefix; a prefix cache would only add nondeterministic hits.
#   * max-running-requests = CONC, not 2*CONC. The agentic 2x headroom exists
#     for subagent fan-out inside a session tree. Here in-flight == CONC.
#   * infx bench_serving (random dataset, --ignore-eos) instead of AIPerf trace
#     replay. --dsv4 does the DeepSeek-V4 framing CLIENT-side and the client
#     posts to /v1/completions, so --chat-template is deliberately not passed
#     (it would be a no-op here and the agentic launcher does not pass it
#     either -- see claude-skills/agentx/references/b200-alignment.md 24.1).
#
# ---------------------------------------------------------------------------
# KV LAYOUT -- read before comparing decode numbers against B200
# ---------------------------------------------------------------------------
# --kv-cache-dtype is a NO-OP for this model. srt/arg_groups/overrides.py:947
# (_deepseek_v4_kv_cache_dtype) rewrites "auto" to "fp8_e4m3" for
# DeepseekV4ForCausalLM, so passing it explicitly and omitting it land on the
# same value. It is kept below only for parity with the agentic launcher.
#
# What actually picks the layout is two HIP-only gates in
# kernels/ops/attention/dsv4/unified_kv_kernels/env_gate.py:
#   is_unified_kv_triton() = is_hip() and SGLANG_HACK_FLASHMLA_BACKEND == unified_kv_triton
#   is_unified_kv_fp8()    = the above AND SGLANG_DSV4_UNIFIED_KV_FP8=1 AND gfx95
# and model_executor/pool_configurator.py:998 branches on them:
#   unified      -> kv_bytes = dsv4_unified_row_bytes()  (kv_cache_dtype ignored)
#   not unified  -> kv_bytes = qk_nope + qk_rope*2 + 8   (the B200/CUDA path)
#
# For DSv4 (head_dim 512 => qk_nope 448, qk_rope 64):
#   MI355X, UNIFIED_KV_FP8=0 (agentx default)   (448+64)*2 = 1024 B/token
#   MI355X, UNIFIED_KV_FP8=1                                 640 B/token
#   B200 (never unified -- is_hip() is False)   448+128+8  =  584 B/token
#
# So the default arm moves 1.75x the KV bytes per token that B200 does. At
# ISL 128k decode is KV-bandwidth bound, so leaving this at 0 folds a
# quantisation difference into what you will read as a kernel gap. Run the
# UNIFIED_KV_FP8=1 arm as well before attributing anything to kernels.
# ---------------------------------------------------------------------------

# ---- Locate the InferenceX checkout -----------------------------------------
# This script is portable: it may live inside the repo
# (benchmarks/single_node/fixed_seq_len/) or outside it (the claude-skills
# bundle). Set INFERENCEX_ROOT when it is outside and the default is wrong.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -z "${INFERENCEX_ROOT:-}" ]; then
    if [ -f "$SCRIPT_DIR/../../benchmark_lib.sh" ]; then
        INFERENCEX_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
    else
        INFERENCEX_ROOT="/workspace/InferenceX"
    fi
fi
export INFERENCEX_ROOT
if [ ! -f "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh" ]; then
    echo "Error: no InferenceX checkout at INFERENCEX_ROOT='$INFERENCEX_ROOT'" >&2
    echo "       (expected \$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh)" >&2
    exit 1
fi
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh"

check_env_vars \
    MODEL \
    TP \
    DP_ATTENTION \
    EP_SIZE \
    CONC \
    ISL \
    OSL \
    RANDOM_RANGE_RATIO \
    RESULT_FILENAME

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

MODEL_PATH="${MODEL_PATH:-$MODEL}"
if [[ "$MODEL_PATH" != /* ]]; then hf download "$MODEL"; fi

PORT=${PORT:-8888}
RESULT_DIR=${RESULT_DIR:-/workspace}
SERVER_LOG="${SERVER_LOG:-$RESULT_DIR/server.log}"
mkdir -p "$RESULT_DIR"

# A server killed minutes earlier can still be draining HBM; booting into a
# half-drained node fails RCCL init with HIP 'unhandled cuda error'.
wait_for_amd_gpu_clean 10

# ---- Workload ---------------------------------------------------------------
# ISL 131072 / OSL 1024 at RANDOM_RANGE_RATIO 1.0 is the point this script
# exists for; everything is env-overridable so the same file serves a sweep.
NUM_PROMPTS=${NUM_PROMPTS:-$((CONC * 3))}
# ISL + OSL + slack for the DSv4 framing the client adds. Extra context length
# does not cost KV -- the pool is sized from mem-fraction-static, not from this.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( ((ISL + OSL + 4096 + 255) / 256) * 256 ))}

# ---- Env: copied verbatim from the agentic MI355X launcher ------------------
export PYTHONNOUSERSITE=1
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
export TORCH_BLAS_PREFER_HIPBLASLT=1
export HSA_NO_SCRATCH_RECLAIM=0
# aiter batched GEMM for the absorbed MLA projections; off by default in environ.py.
export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1

# Two-pool fp8 unified KV. Default 0 reproduces the agentic arm; 1 brings the
# per-token KV footprint (640 B) close to B200's 584 B. See the header block.
# Incompatible with HiCache offload, which this script does not use.
if [ "${UNIFIED_KV_FP8:-0}" = "1" ]; then
    export SGLANG_DSV4_UNIFIED_KV_FP8=1
else
    unset SGLANG_DSV4_UNIFIED_KV_FP8
fi

# The agentic launcher's unified radix tree / out-of-window SWA release exist to
# protect prefix-cache hit rate on multi-turn traces. Radix cache is off here,
# so they are deliberately not exported.

# ---- Parallelism ------------------------------------------------------------
case "$TP" in
    4|8) ;;
    *) echo "Error: unsupported TP '$TP' (expected: 4 or 8)" >&2; exit 1 ;;
esac

MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.86}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.10}"
export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES:-2}"
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
SHARED_EXPERTS_ARGS=(--enforce-shared-experts-fusion)

if [ "$DP_ATTENTION" = "true" ]; then
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES="${GPU_MAX_HW_QUEUES_DP:-5}"
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC_DP:-0.92}"

    # The flag is engine-wide and DP divides it by dp_size, so pass 8192*TP to
    # keep CHUNK_PER_RANK per rank.
    CHUNKED_PREFILL_SIZE=$(( ${CHUNK_PER_RANK:-8192} * TP ))

    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-dp-lm-head
        --enable-dp-attention-local-control-broadcast
        --tokenizer-worker-num "$TP"
        --stream-interval 20
        --prefill-decode-interval "${PREFILL_DECODE_INTERVAL:-24}"
        --load-balance-method "${LOAD_BALANCE_METHOD:-total_requests}"
    )
    # The agentic launcher also passes --enable-prefill-delayer. The B200
    # launcher does NOT, in either branch, so enabling it here would be a
    # one-sided scheduling change on top of the kernel comparison. Default off;
    # ENABLE_PREFILL_DELAYER=1 restores the agentic behaviour.
    if [ "${ENABLE_PREFILL_DELAYER:-0}" = "1" ]; then
        PARALLEL_ARGS+=(
            --enable-prefill-delayer
            --prefill-delayer-token-usage-low-watermark "${DP_PREFILL_DELAYER_LOW_WATERMARK:-0.7}"
        )
    fi
else
    CHUNKED_PREFILL_SIZE=$(( ${CHUNK_PER_RANK:-8192} * TP ))
    PARALLEL_ARGS+=(--prefill-decode-interval "${PREFILL_DECODE_INTERVAL:-20}")
fi

if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
    SHARED_EXPERTS_ARGS=(--disable-shared-experts-fusion)
fi

# No session fan-out in a fixed-length run: in-flight requests are exactly CONC.
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$CONC}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$CONC}
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# ---- Speculative decoding ---------------------------------------------------
# The DSpark draft head is bundled in the target checkpoint (dspark_* keys in
# config.json), so there is no separate draft path. gamma=6 is AL-optimal on the
# golden curve (golden_al_distribution/dsv4-pro-0813-dspark.yaml, thinking_on).
DSV4_DSPARK_GAMMA="${DSV4_DSPARK_GAMMA:-6}"
SPEC_ARGS=(
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSV4_DSPARK_GAMMA"
    --speculative-num-steps 1
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens $((DSV4_DSPARK_GAMMA + 1))
)

# Pinning acceptance length is the whole point of this comparison: it removes
# per-platform AL drift so MI355X and B200 verify the same number of tokens per
# step. Same value and method as both agentic launchers.
DSV4_GOLDEN_AL="${DSV4_GOLDEN_AL:-3.77}"
if [ "${EVAL_ONLY}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV4_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark gamma=$DSV4_DSPARK_GAMMA (verify window $((DSV4_DSPARK_GAMMA + 1))), pinned AL=$DSV4_GOLDEN_AL"
echo "Workload: ISL=$ISL OSL=$OSL ratio=$RANDOM_RANGE_RATIO conc=$CONC prompts=$NUM_PROMPTS ctx=$MAX_MODEL_LEN"
echo "KV layout: unified_kv_triton=on unified_fp8=${UNIFIED_KV_FP8:-0} chunked_prefill=$CHUNKED_PREFILL_SIZE"

start_gpu_monitor

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend dsv4
    --enable-deepseek-v4-fp4-indexer
    --page-size 256
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
    # No-op for DSv4 (overrides.py:947 forces fp8_e4m3 from "auto"); kept for
    # parity with the agentic launcher. See the KV LAYOUT block above.
    --kv-cache-dtype fp8_e4m3
    "${SHARED_EXPERTS_ARGS[@]}"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --context-length "$MAX_MODEL_LEN"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --disable-radix-cache
    "${SPEC_ARGS[@]}"
    --watchdog-timeout 3600
    --enable-metrics
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for MI355X (fixed ISL=$ISL OSL=$OSL)..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID  (kill with: kill -- -\$(ps -o pgid= $SERVER_PID | tr -d ' '))"

wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

# --dsv4 routes prompts through encoding_dsv4.py, emitting the
# <bos><User>...<Assistant><think> framing DeepSeek-V4-Pro expects. Speculative
# acceptance regresses on raw random tokens, so MTP benchmarks must use
# chat-formatted inputs even when the acceptance length is pinned.
run_benchmark_serving \
    --model "$MODEL" \
    --port "$PORT" \
    --backend vllm \
    --input-len "$ISL" \
    --output-len "$OSL" \
    --random-range-ratio "$RANDOM_RANGE_RATIO" \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONC" \
    --result-filename "$RESULT_FILENAME" \
    --result-dir "$RESULT_DIR" \
    --bench-serving-dir "$INFERENCEX_ROOT" \
    --server-pid "$SERVER_PID" \
    --dsv4

stop_gpu_monitor
