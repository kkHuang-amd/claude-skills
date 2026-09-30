#!/usr/bin/env bash
# 5ec406bb76 AgentX queue: c1 chunk4096, c2 chunk4096, c2 chunk16384 (PDI16, official client params).
set -uo pipefail
S=/workspace/claude-skills/dsv41/scripts
export PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori
export SRC=/sgl-workspace/sglang-rolao-opt/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 SCRIPT=agentx_colleague_run.sh
BASE="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
EXTRA_ARGS="$BASE" RUNS="1:16 2:16" PREFIX=f5ec406 bash $S/agentx_series.sh
EXTRA_ARGS="$BASE --chunked-prefill-size 16384" RUNS="2:16" PREFIX=f5ec406_chunk16k bash $S/agentx_series.sh
