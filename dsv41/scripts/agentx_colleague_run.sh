#!/usr/bin/env bash
# Launcher shim for agentx_colleague_mi355x_sglang.sh: provides the env the InferenceX runner normally sets, runs the
# local sglang branch via PYTHONPATH (OPUS off, like the colleague's numbers), and pins GPUs.
#   TAG CONC OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 PORT=8888 DURATION=3600 PREFILL_DECODE_INTERVAL=<override, else recipe default>
#   Defaults = best config: SRC=/sgl-workspace/sglang/python, REPLAY=1, SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1,
#   EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion", per-CONC PDI / chunk / mem in the recipe.
#   DP_ATTENTION=true -> DEP<TP>: EP_SIZE=TP, MegaMoE a2a, sglang-router on PORT, server on PORT+1 (use ports 2 apart).
# Result dir: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/<TAG>/ ; driven by agentx_series.sh SCRIPT=agentx_colleague_run.sh
set -eo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /workspace/claude-skills/agentx/agentx_env.sh
export AIPERF_PYTHON_VERSION=${AIPERF_PYTHON_VERSION:-3.11}
[ -x "$AIPERF_VENV/bin/aiperf" ] && export AIPERF_DEPS_READY=1
export TP=${TP:-2} EP_SIZE=${EP_SIZE:-1} CONC=${CONC:-4} DURATION=${DURATION:-3600} PORT=${PORT:-8888}
# benchmark_lib.sh resets AIPERF_DEPS_READY, rm -rf's the venv on every run AND re-derives AIPERF_VENV as
# $AIPERF_RUNTIME_DIR/venv (an AIPERF_VENV export alone is ignored), so parallel lanes need a per-port runtime dir.
export AIPERF_RUNTIME_DIR=/workspace/agentx-runtime/p$PORT AIPERF_VENV=/workspace/agentx-runtime/p$PORT/venv
if [ ! -d "$AIPERF_RUNTIME_DIR/uv" ] && [ -d /workspace/agentx-runtime/uv ]; then   # seed uv + cache (hardlinks)
  mkdir -p "$AIPERF_RUNTIME_DIR" && cp -al /workspace/agentx-runtime/uv /workspace/agentx-runtime/uv-cache "$AIPERF_RUNTIME_DIR/" 2>/dev/null || true
fi
export MODEL="deepseek-ai/DeepSeek-V4.1-Flash" MODEL_PREFIX="dsv41flash"
export MODEL_PATH=${MODEL_PATH:-$( [ -d /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash ] && echo /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash || echo /shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash)}
export DP_ATTENTION=${DP_ATTENTION:-false}
export DP_MOE=${DP_MOE:-megamoe}
[ "$DP_ATTENTION" = true ] && [ "$DP_MOE" = megamoe ] && export EP_SIZE=$TP
export IS_AGENTIC=1 KV_OFFLOADING=none TOTAL_CPU_DRAM_GB=0 EVAL_ONLY=${EVAL_ONLY:-false} SPEC_DECODING=mtp
TAG=${TAG:-colleague_tp${TP}_c${CONC}_$(date +%m%d_%H%M)}
export RESULT_DIR=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/$TAG AGENTIC_OUTPUT_DIR=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/$TAG
export RESULT_FILENAME="dsv41flash_fp4_sglang_tp${TP}-ep${EP_SIZE}$([ "$DP_ATTENTION" = true ] && echo -dpa)$([ "$DP_ATTENTION" = true ] && [ "$DP_MOE" = tp ] && echo -tpmoe)_spec-dspark_agentic_c${CONC}"
export HIP_VISIBLE_DEVICES=${GPUS:-4,5}
unset ROCR_VISIBLE_DEVICES SGLANG_OPT_DSV41_OPUS_PREFILL
[ "${OPUS:-0}" = 1 ] && export SGLANG_OPT_DSV41_OPUS_PREFILL=1   # OPUS=1: our OPUS sparse prefill (colleague numbers: off)
# Best-config defaults (results/agentx.md, REL_REGRESS_1005.md); every one is env-overridable, EXTRA_ARGS="" drops both flags.
export SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=${SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL:-1}
# DP_ATTENTION=true: MegaMoE needs the shared expert unfused, so the recipe adds --disable-shared-experts-fusion instead.
if [ "$DP_ATTENTION" = true ] && [ "$DP_MOE" = megamoe ]; then
    export EXTRA_ARGS=${EXTRA_ARGS---fp8-gemm-backend aiter}
else
    export EXTRA_ARGS=${EXTRA_ARGS---fp8-gemm-backend aiter --enforce-shared-experts-fusion}
fi
[ -d /sgl-workspace/mori ] && [[ ":${PYTHONPATH:-}:" != *:/sgl-workspace/mori:* ]] && export PYTHONPATH=/sgl-workspace/mori${PYTHONPATH:+:$PYTHONPATH}
export PYTHONPATH=${SRC:-/sgl-workspace/sglang/python}${PYTHONPATH:+:$PYTHONPATH}
exec bash "$D/agentx_colleague_mi355x_sglang.sh"
