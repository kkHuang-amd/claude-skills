#!/usr/bin/env bash
# DSV4.1-Flash on 4x MI355X, TP4+EP4, from the cookbook MI350X recipe
# (docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx).
# Runs the PR branch source via PYTHONPATH; no reinstall needed.
#   DSPARK=1  -> low-latency cell (DSpark block 5); default is high-throughput cell.
#   SIM_AL=3.51 -> SGLANG_SIMULATE_ACC_LEN (DSpark only; golden AL for DSV4.1-Flash thinking_on, 5 draft tokens,
#                 InferenceX golden_al_distribution/dsv41flash_dspark.yaml). Throughput only -- never for GSM8K.
#   QR=NONE (default, recipe; deterministic) or INT8/INT6/INT4 -> ROCM_QUICK_REDUCE_QUANTIZATION (quantized all-reduce)
#   GPUS=0,1,2,3  PORT=30000  LOG=/shared_nfs/kk/dsv41/server.log
set -euo pipefail
SRC=${SRC:-/sgl-workspace/sglang-dsv41/python}
MODEL=${MODEL:-/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash}
export HIP_VISIBLE_DEVICES=${GPUS:-0,1,2,3}
export PYTHONPATH=$SRC${PYTHONPATH:+:$PYTHONPATH}
export SGLANG_USE_AITER=1 SGLANG_MOE_PADDING=1 AITER_FLYDSL_FORCE_REDUCE=1 ROCM_QUICK_REDUCE_QUANTIZATION=${QR:-NONE}
# Not in the V4.1 cookbook but set by the PR body/commits and the DSV4 cookbook: without it M<256 MoE
# takes aiter's bf16-activation route -> CK stage1 "Unsupported kernel config for moe heuristic dispatch"
# (fix is unmerged ROCm/aiter#5802). Crash seen at prefill graph capture num_tokens=240.
export AITER_BF16_FP8_MOE_BOUND=${AITER_BF16_FP8_MOE_BOUND:-0} TRITON_HIP_USE_ASYNC_COPY=0 SGLANG_USE_ROCM700A=0
ARGS=(--trust-remote-code --model-path "$MODEL" --tp 4 --ep-size 4 --disable-radix-cache
      --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 4096
      --reasoning-parser auto --tool-call-parser auto --host 0.0.0.0 --port "${PORT:-30000}")
if [ "${DSPARK:-0}" = 1 ]; then
  ARGS+=(--mem-fraction-static 0.8 --speculative-algorithm DSPARK --speculative-dspark-block-size 5
         --cuda-graph-max-bs-decode 64)
fi
[ -n "${SIM_AL:-}" ] && export SGLANG_SIMULATE_ACC_LEN=$SIM_AL SGLANG_RAGGED_VERIFY_MODE=static
exec python3 -m sglang.launch_server "${ARGS[@]}"
