#!/usr/bin/env bash
# AgentX (InferenceX) DSV4.1-Flash SGLang DSpark on MI355X, ported from InferenceX origin/main
# benchmarks/single_node/agentic/dsv41flash_fp4_b200_sglang_mtp.sh (d3ddb20bb, 2026-09-22).
# Kept from B200: engram per_rank host table, max-running min(2*CONC,64), SWA prefix tails, TP2 mem 0.92 /
#   chunk 2048, golden AL 3.51 match-expected simulation, watchdog 3600, warmup grace 3600 at CONC >= 32.
# Changed: EP_SIZE default 1; B200's TP2 "CONC <= 8" limit dropped (288 GB HBM); --prefill-decode-interval is
#   4 at CONC 64 and 16 otherwise (PDI=<n> overrides); no nvidia-smi / hf download / expandable_segments;
#   local branch via PYTHONPATH + validated ROCm env (RUNBOOK.md); cookbook ROCm prefill graph
#   (--cuda-graph-backend-prefill breakable, max bs = chunk); OPUS prefill on (OPUS=0 to disable).
#   TP=2 EP_SIZE=1 CONC=4 GPUS=4,5 PORT=8888 DURATION=3600 TAG=<name> PDI= OPUS=1 QR=NONE MEM= CHUNK=
# Result dir: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/<TAG>/ ; driven by agentx_series.sh SCRIPT=agentx_dsv41_b200port_mi355x.sh
set -eo pipefail
source /workspace/claude-skills/agentx/agentx_env.sh
export AIPERF_PYTHON_VERSION=${AIPERF_PYTHON_VERSION:-3.11}
IX=$INFMAX_CONTAINER_WORKSPACE
source "$IX/benchmarks/benchmark_lib.sh"

export TP=${TP:-2} EP_SIZE=${EP_SIZE:-1} CONC=${CONC:-4} DURATION=${DURATION:-3600}
export MODEL="deepseek-ai/DeepSeek-V4.1-Flash" MODEL_PREFIX="dsv41flash"
export MODEL_PATH=${MODEL_PATH:-$( [ -d /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash ] && echo /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash || echo /shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash)}
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0 EVAL_ONLY=${EVAL_ONLY:-false} SPEC_DECODING=mtp
TAG=${TAG:-b200port_tp${TP}_c${CONC}_$(date +%m%d_%H%M)}
export RESULT_DIR=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/$TAG AGENTIC_OUTPUT_DIR=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/$TAG
export RESULT_FILENAME="dsv41flash_fp4_sglang_tp${TP}-ep${EP_SIZE}_spec-dspark_agentic_c${CONC}"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION
check_env_vars EVAL_ONLY SPEC_DECODING
export GPU_COUNT="$TP"

resolve_trace_source
# install_agentic_deps starts with rm -rf of the shared venv: install once, never from parallel runs
[ -x "$AIPERF_VENV/bin/aiperf" ] && export AIPERF_DEPS_READY=1
install_agentic_deps
mkdir -p "$RESULT_DIR"
SERVER_LOG="$RESULT_DIR/server.log"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV41_REASONING_EFFORT=high

# ---- ROCm / local branch (RUNBOOK.md) ----
export HIP_VISIBLE_DEVICES=${GPUS:-4,5}
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_USE_AITER=1 SGLANG_MOE_PADDING=1 AITER_FLYDSL_FORCE_REDUCE=1 ROCM_QUICK_REDUCE_QUANTIZATION=${QR:-NONE}
export AITER_BF16_FP8_MOE_BOUND=0 TRITON_HIP_USE_ASYNC_COPY=0 SGLANG_USE_ROCM700A=0
if [ "${OPUS:-1}" = 1 ]; then export SGLANG_OPT_DSV41_OPUS_PREFILL=1; else unset SGLANG_OPT_DSV41_OPUS_PREFILL; fi

case "$TP" in
    2|4) export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 ;;
    *) echo "Unsupported DSpark TP=$TP; expected 2 or 4" >&2; exit 1 ;;
esac
export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=${ENGRAM_LAYOUT:-per_rank}   # ENGRAM_LAYOUT=shared: per_rank hit RCCL OUT_OF_RESOURCES on MI355X TP2

CUDA_GRAPH_MAX_BS=64
MAX_RUNNING_REQUESTS=$((2 * CONC))
if (( MAX_RUNNING_REQUESTS > CUDA_GRAPH_MAX_BS )); then
    MAX_RUNNING_REQUESTS=$CUDA_GRAPH_MAX_BS
fi
SWA_PREFIX_TAILS=$((64 * CONC))
MEM_FRACTION_STATIC=0.80
CHUNKED_PREFILL_SIZE=4096
if (( TP == 2 )); then
    SWA_PREFIX_TAILS=$((128 * CONC))
    MEM_FRACTION_STATIC=0.92
    CHUNKED_PREFILL_SIZE=2048
fi
if (( SWA_PREFIX_TAILS < 128 )); then
    SWA_PREFIX_TAILS=128
elif (( SWA_PREFIX_TAILS > 4096 )); then
    SWA_PREFIX_TAILS=4096
fi
MEM_FRACTION_STATIC=${MEM:-$MEM_FRACTION_STATIC}
CHUNKED_PREFILL_SIZE=${CHUNK:-$CHUNKED_PREFILL_SIZE}
if (( CONC == 64 )); then PREFILL_DECODE_INTERVAL=4; else PREFILL_DECODE_INTERVAL=16; fi
PREFILL_DECODE_INTERVAL=${PDI:-$PREFILL_DECODE_INTERVAL}

if (( CONC >= 32 )); then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

export PORT=${PORT:-8888}
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}"
export AIPERF_SERVER_METRICS_URLS="${AIPERF_SERVER_URL}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"

unset SGLANG_SIMULATE_ACC_LEN SGLANG_SIMULATE_ACC_METHOD SGLANG_SIMULATE_ACC_TOKEN_MODE
DSPARK_BLOCK_SIZE=5
DSV41_GOLDEN_AL=3.51
if [[ "$EVAL_ONLY" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN="$DSV41_GOLDEN_AL"
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL"
    --host 0.0.0.0 --port "$PORT"
    --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --prefill-decode-interval "$PREFILL_DECODE_INTERVAL"
    --swa-prefix-tails "$SWA_PREFIX_TAILS"
    --speculative-algorithm DSPARK
    --speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill "$CHUNKED_PREFILL_SIZE"
    --reasoning-parser auto
    --tool-call-parser auto
    --watchdog-timeout 3600
    --enable-metrics
)
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{
    echo "=== env at launch ==="
    env | grep -E '^(SGLANG_|AITER_|ROCM_|HIP_VISIBLE)' | sort
    echo "==="
} > "$SERVER_LOG"
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [[ "${EVAL_ONLY}" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
