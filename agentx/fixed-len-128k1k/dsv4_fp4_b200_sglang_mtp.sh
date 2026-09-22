#!/usr/bin/env bash
set -eo pipefail

# DeepSeek-V4-Pro FP4 + SGLang DSpark MTP on B200, FIXED sequence length.
#
# NVIDIA partner of dsv4_fp4_mi355x_sglang_mtp.sh in this directory. Read that
# file's header first -- it carries the shared rationale and the KV-layout
# analysis. This file only records what is B200-specific.
#
# Server configuration is copied from
#   benchmarks/single_node/agentic/dsv4_fp4_b200_sglang_mtp.sh
# with the same fixed-length deltas as the MI355X side: no sglang-router,
# --disable-radix-cache, no HiCache, max-running-requests = CONC, and
# infx bench_serving instead of AIPerf trace replay.
#
# Two further deltas from the B200 agentic launcher, both for symmetry with the
# MI355X arm:
#   * chunked-prefill-size is CHUNK_PER_RANK*TP with CHUNK_PER_RANK=8192.
#     The agentic B200 DP branch uses 6144/rank. Prefill chunk size sets the
#     prefill GEMM/attention shape, so it must match the other platform or the
#     prefill comparison is confounded. Override CHUNK_PER_RANK to get 6144 back.
#   * --chat-template is not passed. infx bench_serving posts to
#     /v1/completions and --dsv4 applies the DeepSeek-V4 framing client-side, so
#     a server-side template would not be consulted; the MI355X arm does not
#     pass one either.
#
# KV layout: is_unified_kv_triton() is gated on is_hip(), so B200 can never take
# the unified pool. pool_configurator.py:998 non-unified branch gives
# qk_nope + qk_rope*2 + 8 = 448 + 128 + 8 = 584 B/token. --kv-cache-dtype is not
# passed because overrides.py:947 already forces fp8_e4m3 for this model.

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
source "$INFERENCEX_ROOT/benchmarks/benchmark_lib.sh" --validation-only
# The B200 DeepSeek-V4 image installs SGLang editable under /workspace, so the
# runner mounts InferenceX elsewhere and exports this. Default it for a plain
# manual run outside that runner.
export INFMAX_CONTAINER_WORKSPACE="${INFMAX_CONTAINER_WORKSPACE:-$INFERENCEX_ROOT}"

# The B200 DeepSeek-V4 image installs SGLang editable under /workspace, so its
# launcher mounts InferenceX elsewhere. Resolve results against the real mount.
if [[ "${RESULT_DIR:-}" == /workspace/* && "$INFMAX_CONTAINER_WORKSPACE" != /workspace ]]; then
    export RESULT_DIR="$INFMAX_CONTAINER_WORKSPACE/${RESULT_DIR#/workspace/}"
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

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "JOB $SLURM_JOB_ID running on ${SLURMD_NODENAME:-unknown}"
fi

MODEL_PATH="${MODEL_PATH:-$MODEL}"
if [[ "$MODEL_PATH" != /* ]]; then hf download "$MODEL"; fi
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true

PORT=${PORT:-8888}
RESULT_DIR=${RESULT_DIR:-$INFMAX_CONTAINER_WORKSPACE}
SERVER_LOG="${SERVER_LOG:-$RESULT_DIR/server.log}"
mkdir -p "$RESULT_DIR"

NUM_PROMPTS=${NUM_PROMPTS:-$((CONC * 3))}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( ((ISL + OSL + 4096 + 255) / 256) * 256 ))}

# ---- Env: copied from the agentic B200 launcher -----------------------------
export PYTHONNOUSERSITE=1
export TORCH_CUDA_ARCH_LIST=10.0
export SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1
export SGLANG_OPT_USE_JIT_NORM=1
export SGLANG_OPT_USE_JIT_INDEXER_METADATA=1
export SGLANG_OPT_USE_TOPK_V2=1
export SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=1

TRITON_PTXAS_PATH=$(find \
    /usr/local/cuda* \
    /usr/local/lib/python*/dist-packages/nvidia \
    /usr/local/lib/python*/site-packages/nvidia \
    -type f -name ptxas -perm -u+x -print -quit 2>/dev/null || true)
if [ -n "$TRITON_PTXAS_PATH" ]; then
    export TRITON_PTXAS_PATH
    echo "Using ptxas for Triton: $TRITON_PTXAS_PATH"
fi

# ---- Parallelism ------------------------------------------------------------
PARALLEL_ARGS=(--tp "$TP")
METRICS_ARGS=(--enable-metrics)
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO:-0.1}"
CHUNKED_PREFILL_SIZE=$(( ${CHUNK_PER_RANK:-8192} * TP ))

if [ "$DP_ATTENTION" = "true" ]; then
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320
    # Leave HBM headroom for the FP4 indexer's context-dependent workspace.
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC_DP:-0.88}"
    SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO_DP:-0.02}"
    METRICS_ARGS+=(--load-snapshot-publish-interval 1)
    PARALLEL_ARGS+=(
        --dp "$TP"
        --tokenizer-worker-num "$TP"
        --prefill-decode-interval "${PREFILL_DECODE_INTERVAL:-24}"
        --load-balance-method "${LOAD_BALANCE_METHOD:-total_requests}"
        --enable-dp-attention
        --enable-dp-lm-head
        --enable-dp-attention-local-control-broadcast
        --incremental-streaming-output
        --stream-interval 20
        --dist-init-addr "127.0.0.1:$((PORT + 2000))"
        --ep-size "$EP_SIZE"
        --moe-a2a-backend megamoe
        --enable-w4a4-mxfp4-megamoe
        --enable-deepseek-v4-fp4-indexer
        --disable-shared-experts-fusion
        --disable-flashinfer-autotune
    )
else
    PARALLEL_ARGS+=(
        --moe-runner-backend flashinfer_mxfp4
        --enable-deepseek-v4-fp4-indexer
        --disable-flashinfer-autotune
    )
fi

MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-$CONC}
CUDA_GRAPH_MAX_BS=${CUDA_GRAPH_MAX_BS:-$CONC}
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# ---- Speculative decoding: identical to the MI355X arm ----------------------
DSV4_DSPARK_GAMMA="${DSV4_DSPARK_GAMMA:-6}"
SPEC_ARGS=(
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSV4_DSPARK_GAMMA"
    --speculative-num-steps 1
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens $((DSV4_DSPARK_GAMMA + 1))
)
DSV4_GOLDEN_AL="${DSV4_GOLDEN_AL:-3.77}"
if [ "${EVAL_ONLY}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV4_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi
echo "DSpark gamma=$DSV4_DSPARK_GAMMA (verify window $((DSV4_DSPARK_GAMMA + 1))), pinned AL=$DSV4_GOLDEN_AL"
echo "Workload: ISL=$ISL OSL=$OSL ratio=$RANDOM_RANGE_RATIO conc=$CONC prompts=$NUM_PROMPTS ctx=$MAX_MODEL_LEN"
echo "KV layout: non-unified (CUDA), chunked_prefill=$CHUNKED_PREFILL_SIZE"

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --context-length "$MAX_MODEL_LEN"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --disable-radix-cache
    --watchdog-timeout 1800
    "${SPEC_ARGS[@]}"
    # The B200 checkpoint lives on Lustre: prefetch sequentially across local
    # ranks so post-load repacking reads from page cache.
    --weight-loader-prefetch-checkpoints
    --model-loader-extra-config '{"enable_multithread_load": true}'
    "${METRICS_ARGS[@]}"
)

write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for B200 (fixed ISL=$ISL OSL=$OSL)..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID  (kill with: kill -- -\$(ps -o pgid= $SERVER_PID | tr -d ' '))"

wait_for_ready \
    --endpoint "http://localhost:$PORT/health" \
    --log "$SERVER_LOG" \
    --pid "$SERVER_PID"

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
