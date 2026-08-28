#!/usr/bin/env bash
# Smoke run of the InferenceX AgentX (agentic trace replay) pipeline.
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/models/DeepSeek-V4-Pro"
export TP="${TP:-8}"
export EP_SIZE="${EP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-false}"
export CONC="${CONC:-2}"
export IS_AGENTIC=1
export KV_OFFLOADING="none"
export TOTAL_CPU_DRAM_GB=0
export DURATION="${DURATION:-300}"
export PORT="${PORT:-8888}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/smoke-tp${TP}-c${CONC}}"
export RESULT_FILENAME="${RESULT_FILENAME:-dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
export AIPERF_WARMUP_REQUESTS_PER_LANE="${AIPERF_WARMUP_REQUESTS_PER_LANE:-1}"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
