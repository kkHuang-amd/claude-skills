#!/usr/bin/env bash
# Launcher shim for agentx_colleague_mi355x_sglang.sh: provides the env the InferenceX runner normally sets, runs the
# local sglang branch via PYTHONPATH (OPUS off, like the colleague's numbers), and pins GPUs.
#   TAG CONC OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 PORT=8888 DURATION=3600 PREFILL_DECODE_INTERVAL=<override, else recipe default>
# Result dir: /shared_nfs/kk/dsv41/agentx/<TAG>/ ; driven by agentx_series.sh SCRIPT=agentx_colleague_run.sh
set -eo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /workspace/claude-skills/agentx/agentx_env.sh
export AIPERF_PYTHON_VERSION=${AIPERF_PYTHON_VERSION:-3.11}
[ -x "$AIPERF_VENV/bin/aiperf" ] && export AIPERF_DEPS_READY=1
export TP=${TP:-2} EP_SIZE=${EP_SIZE:-1} CONC=${CONC:-4} DURATION=${DURATION:-3600} PORT=${PORT:-8888}
# benchmark_lib.sh resets AIPERF_DEPS_READY and rm -rf's the venv on every run; per-port venvs keep parallel lanes
# from deleting each other's live aiperf install.
export AIPERF_VENV=$AIPERF_RUNTIME_DIR/venv_p$PORT
export MODEL="deepseek-ai/DeepSeek-V4.1-Flash" MODEL_PREFIX="dsv41flash"
export MODEL_PATH=${MODEL_PATH:-$( [ -d /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash ] && echo /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash || echo /shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash)}
export IS_AGENTIC=1 KV_OFFLOADING=none TOTAL_CPU_DRAM_GB=0 EVAL_ONLY=${EVAL_ONLY:-false} SPEC_DECODING=mtp DP_ATTENTION=false
TAG=${TAG:-colleague_tp${TP}_c${CONC}_$(date +%m%d_%H%M)}
export RESULT_DIR=/shared_nfs/kk/dsv41/agentx/$TAG AGENTIC_OUTPUT_DIR=/shared_nfs/kk/dsv41/agentx/$TAG
export RESULT_FILENAME="dsv41flash_fp4_sglang_tp${TP}-ep${EP_SIZE}_spec-dspark_agentic_c${CONC}"
export HIP_VISIBLE_DEVICES=${GPUS:-4,5}
unset ROCR_VISIBLE_DEVICES SGLANG_OPT_DSV41_OPUS_PREFILL
[ "${OPUS:-0}" = 1 ] && export SGLANG_OPT_DSV41_OPUS_PREFILL=1   # OPUS=1: our OPUS sparse prefill (colleague numbers: off)
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
exec bash "$D/agentx_colleague_mi355x_sglang.sh"
