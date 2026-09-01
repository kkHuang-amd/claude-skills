#!/usr/bin/env bash
# Server launch ONLY, extracted verbatim from
#   InferenceX benchmarks/single_node/fixed_seq_len/dsv4_fp4_b200.sh
#   (InferenceX 8fcfc62830, 2026-08-26) lines 23-126.
# Config entry: dsv4-fp4-b200-sglang (configs/nvidia-master.yaml)
#   image lmsysorg/sglang:nightly-dev-cu13-20260628-da802ddc
#   model deepseek-ai/DeepSeek-V4-Pro, runner b200-dsv4
#
# Deliberately does NOT source benchmark_lib.sh -- a copied launcher cannot find
# ../../benchmark_lib.sh and dies with exit 1 instantly. Everything it needed
# (PORT default, wait_for_server_ready) is inlined below.
#
# Dropped from the original (client-side, not server launch): hf download,
# nvidia-smi, start/stop_gpu_monitor, run_benchmark_serving, run_eval.
# EVAL_ONLY/--context-length is kept as an optional passthrough.
set -uo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
TP="${TP:-8}"
DP_ATTENTION="${DP_ATTENTION:-true}"
ISL="${ISL:-8192}"                 # 1024 or 8192; see SWA note below
CONC="${CONC:-0}"                  # logged only -- see README "b200: one server per sweep"
PORT="${PORT:-8888}"
SERVER_LOG="${SERVER_LOG:-$PWD/server.log}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-}"

# --- common env (original line 24) ---------------------------------------
export SGLANG_OPT_SWA_SPLIT_LEAF_ON_INSERT=1

# --- SWA ratio (original lines 45-50) ------------------------------------
# 1k inputs need more SWA cache headroom than 8k inputs do.
if [[ "$ISL" == "1024" ]]; then
    SWA_FULL_TOKENS_RATIO=0.5
else
    SWA_FULL_TOKENS_RATIO=0.1
fi

# --- recipe select (original lines 55-101) -------------------------------
if [ "${DP_ATTENTION}" = "true" ]; then
    export SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8
    export SGLANG_OPT_SWA_EVICT_DROP_PAGE_MARGIN=1
    export SGLANG_OPT_USE_FAST_MASK_EP=1
    export SGLANG_OPT_FIX_MEGA_MOE_MEMORY=1
    export SGLANG_OPT_FIX_NEXTN_MEGA_MOE=1
    export NVSHMEM_DISABLE_IB=1
    export SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1
    export SGLANG_OPT_USE_ONLINE_COMPRESS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=2048
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1
    export SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1
    export SGLANG_EXPERIMENTAL_ENABLE_PIECEWISE_CUDA_GRAPH_MOE_A2A=1
    export NCCL_MNNVL_ENABLE=1
    export NCCL_CUMEM_ENABLE=1
    export MC_FORCE_MNNVL=1
    export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=True

    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.835}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-4352}"
    # NOTE: upstream overwrites the ISL-derived ratio here, so in DP mode
    # 1k/1k and 8k/1k launch with the SAME 0.12. Preserved as-is.
    SWA_FULL_TOKENS_RATIO="${SWA_FULL_TOKENS_RATIO_OVERRIDE:-0.12}"
    CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-544}"

    PARALLEL_ARGS=(
        --dp-size "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
        --enable-mixed-chunk
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}"
        --max-prefill-tokens "${MAX_PREFILL_TOKENS:-16384}"
        --tokenizer-worker-num 8
        --stream-interval 30
        --enable-prefill-delayer
    )
else
    MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
    MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-512}"
    CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-512}"
    PARALLEL_ARGS=(
        --moe-runner-backend flashinfer_mxfp4
        --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-8192}"
        --disable-flashinfer-autotune
        --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
        --tokenizer-worker-num 8
        --stream-interval 30
        --enable-prefill-delayer
    )
fi

EVAL_CONTEXT_ARGS=()
[ -n "$EVAL_MAX_MODEL_LEN" ] && EVAL_CONTEXT_ARGS=(--context-length "$EVAL_MAX_MODEL_LEN")

{
    echo "=== dsv4_fp4_b200 server launch ==="
    echo "TP=$TP DP_ATTENTION=$DP_ATTENTION ISL=$ISL CONC=$CONC PORT=$PORT"
    echo "mem_fraction_static=$MEM_FRACTION_STATIC max_running_requests=$MAX_RUNNING_REQUESTS"
    echo "swa_full_tokens_ratio=$SWA_FULL_TOKENS_RATIO cuda_graph_max_bs=$CUDA_GRAPH_MAX_BS"
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

set -x
PYTHONNOUSERSITE=1 sglang serve \
    --model-path "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --tp "$TP" \
    --disable-radix-cache \
    --max-running-requests "$MAX_RUNNING_REQUESTS" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --swa-full-tokens-ratio "$SWA_FULL_TOKENS_RATIO" \
    "${PARALLEL_ARGS[@]}" "${EVAL_CONTEXT_ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
set +x
echo "$SERVER_PID" > "${SERVER_PID_FILE:-$PWD/server.pid}"

# --- inlined wait_for_server_ready ---------------------------------------
# /health returning 200 is NOT proof it is your server: a stale router from an
# earlier arm can own the port. Confirm the listener is this PID.
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
