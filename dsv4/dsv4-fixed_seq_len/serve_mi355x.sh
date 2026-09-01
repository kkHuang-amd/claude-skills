#!/usr/bin/env bash
# Server launch ONLY, extracted verbatim from
#   InferenceX benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_sglang.sh
#   (InferenceX 8fcfc62830, 2026-08-26) lines 27-91.
# Config entry: dsv4-fp4-mi355x-sglang (configs/amd-master.yaml)
#   image lmsysorg/sglang-rocm:v0.5.14-rocm720-mi35x-20260706
#   model deepseek-ai/DeepSeek-V4-Pro, runner mi355x
#
# Does NOT source benchmark_lib.sh (a copy outside fixed_seq_len/ cannot find
# ../../benchmark_lib.sh). PORT default and wait_for_server_ready are inlined.
#
# !! CONC IS A SERVER-SIDE VARIABLE HERE. Upstream sets both
#    --cuda-graph-max-bs and --max-running-requests to $CONC, so every
#    concurrency point needs its OWN server. See README.
set -uo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
TP="${TP:-8}"
DP_ATTENTION="${DP_ATTENTION:-true}"
EP_SIZE="${EP_SIZE:-1}"
ISL="${ISL:-8192}"
CONC="${CONC:?CONC is required -- it sets cuda-graph-max-bs and max-running-requests}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"   # benchmark_lib.sh:946 default
PORT="${PORT:-8888}"
SERVER_LOG="${SERVER_LOG:-$PWD/server.log}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-}"

# Upstream resolves this relative to its own $0; a copy must point at the real
# InferenceX tree or sglang starts with the wrong (default) chat template.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/workspace/InferenceX/benchmarks/single_node/chat_templates/deepseek_v4_thinking.jinja}"
[ -r "$CHAT_TEMPLATE" ] || { echo "FAIL: chat template not readable: $CHAT_TEMPLATE" >&2; exit 1; }

# --- common env (original lines 27-31) -----------------------------------
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=max
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0

# --- parallelism (original lines 43-64) ----------------------------------
PARALLEL_ARGS=(
    --tensor-parallel-size "$TP"
)
CHUNKED_PREFILL_SIZE=$ISL
if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
    export GPU_MAX_HW_QUEUES=5

    CHUNKED_PREFILL_SIZE=$((ISL * TP))
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --enable-prefill-delayer
        --enable-two-batch-overlap
    )
fi
if [ "${EP_SIZE:-1}" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE_OVERRIDE:-$CHUNKED_PREFILL_SIZE}"

EVAL_CONTEXT_ARGS=()
[ -n "$EVAL_MAX_MODEL_LEN" ] && EVAL_CONTEXT_ARGS=(--context-length "$EVAL_MAX_MODEL_LEN")

{
    echo "=== dsv4_fp4_mi355x_sglang server launch ==="
    echo "TP=$TP DP_ATTENTION=$DP_ATTENTION EP_SIZE=$EP_SIZE ISL=$ISL CONC=$CONC PORT=$PORT"
    echo "chunked_prefill_size=$CHUNKED_PREFILL_SIZE max_model_len=$MAX_MODEL_LEN"
    echo "cuda_graph_max_bs=$CONC max_running_requests=$CONC"
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_|^AITER_|^GPU_MAX_HW_QUEUES' | sort
    echo "==========================================="
} | tee "$SERVER_LOG"

set -x
sglang serve \
    --model-path "$MODEL" \
    --host=0.0.0.0 \
    --port "$PORT" \
    "${PARALLEL_ARGS[@]}" \
    --trust-remote-code \
    --disable-radix-cache \
    --attention-backend dsv4 \
    --cuda-graph-max-bs "${CONC}" \
    --max-running-requests "${CONC}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC:-0.90}" \
    --swa-full-tokens-ratio "${SWA_FULL_TOKENS_RATIO:-0.15}" \
    --page-size 256 \
    --kv-cache-dtype fp8_e4m3 \
    --context-length "$MAX_MODEL_LEN" \
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE" \
    --disable-shared-experts-fusion \
    --tool-call-parser deepseekv4 \
    --reasoning-parser deepseek-v4 \
    --chat-template "$CHAT_TEMPLATE" \
    --watchdog-timeout 1800 "${EVAL_CONTEXT_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x
echo "$SERVER_PID" > "${SERVER_PID_FILE:-$PWD/server.pid}"

# --- inlined wait_for_server_ready ---------------------------------------
deadline=$(( $(date +%s) + ${READY_TIMEOUT:-3600} ))
while :; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server pid $SERVER_PID died; traceback is in $SERVER_LOG" >&2
        tail -30 "$SERVER_LOG" >&2; exit 1
    fi
    if curl -sf -o /dev/null "http://0.0.0.0:${PORT}/health"; then
        if ss -lntp 2>/dev/null | grep -q ":${PORT}\b.*pid=${SERVER_PID}\b"; then
            echo "READY: port $PORT owned by pid $SERVER_PID"; break
        fi
        echo "WARN: :$PORT answers /health but is NOT pid $SERVER_PID (stale router?)" >&2
    fi
    [ "$(date +%s)" -gt "$deadline" ] && { echo "FAIL: not ready in ${READY_TIMEOUT:-3600}s" >&2; exit 1; }
    sleep 5
done
echo "server pid $SERVER_PID on port $PORT; kill BY PID when done (pkill -f 'sglang::' does not work)"
