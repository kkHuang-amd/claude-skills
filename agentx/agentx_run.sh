#!/usr/bin/env bash
# Published-arm run of the InferenceX AgentX dsv4-fp4-mi355x-sglang-agentic-mtp recipe.
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/models/DeepSeek-V4-Pro"
export TP="${TP:?}" CONC="${CONC:?}"
export EP_SIZE=1
export DP_ATTENTION="false"
export SPEC_DECODING="mtp"
export IS_AGENTIC=1
export KV_OFFLOADING="${KV_OFFLOADING:?}"
# process_agentic_result needs both when KV_OFFLOADING != none; the CI matrix
# normally supplies the metadata blob.
if [ "$KV_OFFLOADING" != "none" ]; then
  export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
  export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"hicache\"}}"
fi
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:?}"
export DURATION="${DURATION:-3600}"
export PORT="${PORT:-8888}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/armA-tp${TP}-c${CONC}}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
