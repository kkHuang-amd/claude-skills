#!/usr/bin/env bash
# AgentX trace replay (InferenceX inferencex-agentx-mvp) for DeepSeek-V4.1-Flash, SGLang DSpark on MI355X.
# Adapted from InferenceX benchmarks/single_node/agentic/dsv41flash_fp4_b200_sglang_mtp.sh (DSpark block 5,
# simulated AL 3.51, thinking on, reasoning effort high, engram host table, chunked prefill 4096,
# max-running = min(2*CONC, 64), radix cache ON) + this skill's validated ROCm settings (launch_server.sh).
#   TP=2 EP_SIZE=$TP CONC=4 GPUS=0,1 PORT=8888 DURATION=1200 TAG=<name>
#   OPUS=1 (SGLANG_OPT_DSV41_OPUS_PREFILL; needs branch opus-prefill)  MIXED=0 (--enable-mixed-chunk)  QR=NONE
#   CHUNK=4096 (--chunked-prefill-size)  PGRAPH=4096 (--cuda-graph-max-bs-prefill)
#   MEM=0.85 (weights ~160 GB/GPU at TP2 with the 203 GB engram table on host)
#   PDI=<n> (--prefill-decode-interval; unset = off)  PDI_AUTO=1 -> 4 at CONC 64, 16 otherwise (B200 recipe uses 16)
#   EVAL_ONLY=true -> real acceptance + run_eval instead of replay (accuracy check)
# Result dir: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/<TAG>/ ; one row -> results/agentx.md via agentx_summary.py
set -eo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /workspace/claude-skills/agentx/agentx_env.sh
export AIPERF_PYTHON_VERSION=${AIPERF_PYTHON_VERSION:-3.11}   # needed by resolve_trace_source too; normally set by the InferenceX launcher
IX=$INFMAX_CONTAINER_WORKSPACE
source "$IX/benchmarks/benchmark_lib.sh"

export TP=${TP:-2} EP_SIZE=${EP_SIZE:-${TP:-2}} CONC=${CONC:-4} DURATION=${DURATION:-1200}
export MODEL="deepseek-ai/DeepSeek-V4.1-Flash" MODEL_PREFIX="dsv41flash"
export MODEL_PATH=${MODEL_PATH:-$( [ -d /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash ] && echo /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash || echo /shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash)}
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0 EVAL_ONLY=${EVAL_ONLY:-false}
TAG=${TAG:-tp${TP}_c${CONC}_$(date +%m%d_%H%M)}
export RESULT_DIR=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/$TAG AGENTIC_OUTPUT_DIR=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/$TAG
export RESULT_FILENAME="dsv41flash_fp4_sglang_tp${TP}-ep${EP_SIZE}_spec-dspark_agentic_c${CONC}"
check_env_vars MODEL TP EP_SIZE CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EVAL_ONLY
export GPU_COUNT="$TP"
mkdir -p "$RESULT_DIR"; SERVER_LOG="$RESULT_DIR/server.log"

resolve_trace_source
# install_agentic_deps starts with rm -rf of the shared venv: install once, never from parallel runs
[ -x "$AIPERF_VENV/bin/aiperf" ] && export AIPERF_DEPS_READY=1
install_agentic_deps

# ---- ROCm / this branch (see RUNBOOK.md) ----
export HIP_VISIBLE_DEVICES=${GPUS:-0,1}
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_USE_AITER=1 SGLANG_MOE_PADDING=1 AITER_FLYDSL_FORCE_REDUCE=1 ROCM_QUICK_REDUCE_QUANTIZATION=${QR:-NONE}
export AITER_BF16_FP8_MOE_BOUND=0 TRITON_HIP_USE_ASYNC_COPY=0 SGLANG_USE_ROCM700A=0
[ "${OPUS:-1}" = 1 ] && export SGLANG_OPT_DSV41_OPUS_PREFILL=1
# ---- from the b200 script ----
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 AIPERF_HTTP_TCP_USER_TIMEOUT=900000 SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING=1 SGLANG_DSV41_REASONING_EFFORT=high SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
CUDA_GRAPH_MAX_BS=64
MAX_RUNNING_REQUESTS=$((2 * CONC)); (( MAX_RUNNING_REQUESTS > CUDA_GRAPH_MAX_BS )) && MAX_RUNNING_REQUESTS=$CUDA_GRAPH_MAX_BS
(( CONC >= 32 )) && export AGENTIC_WARMUP_GRACE_PERIOD=3600
export PORT=${PORT:-8888}
select_available_server_port
export AIPERF_SERVER_URL="http://localhost:${PORT}" AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="sglang:"
if [[ "$EVAL_ONLY" != true ]]; then
    export SGLANG_SIMULATE_ACC_LEN=3.51 SGLANG_SIMULATE_ACC_METHOD=match-expected SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

SGLANG_CMD=(python3 -m sglang.launch_server
    --model-path "$MODEL_PATH" --served-model-name "$MODEL" --host 0.0.0.0 --port "$PORT" --trust-remote-code
    --tp "$TP" --ep-size "$EP_SIZE" --mem-fraction-static "${MEM:-0.85}" --chunked-prefill-size "${CHUNK:-4096}"
    --speculative-algorithm DSPARK --speculative-dspark-block-size 5
    --max-running-requests "$MAX_RUNNING_REQUESTS" --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill "${PGRAPH:-4096}"
    --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 3600 --enable-metrics)
[ "${MIXED:-0}" = 1 ] && SGLANG_CMD+=(--enable-mixed-chunk)
[ "${PDI_AUTO:-0}" = 1 ] && [ -z "${PDI:-}" ] && { (( CONC == 64 )) && PDI=4 || PDI=16; }
[ -n "${PDI:-}" ] && SGLANG_CMD+=(--prefill-decode-interval "$PDI")
write_command "$RESULT_DIR/sglang_command.txt" "${SGLANG_CMD[@]}"
{ echo "=== env at launch ==="; env | grep -E '^(SGLANG_|AITER_|ROCM_|HIP_VISIBLE)' | sort; echo "==="; } > "$SERVER_LOG"
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
wait_for_server_ready --port "$PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"
if [[ "$EVAL_ONLY" == true ]]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics ${AIPERF_SERVER_METRICS_URLS}"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
