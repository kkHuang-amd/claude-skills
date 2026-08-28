#!/usr/bin/env bash
# Aiter MegaMoEv2 (sglang#35619, unmerged) feasibility/A-B run on MI355X.
# Mirrors the B200 megamoe arm shape: tp8 + ep8 + dp-attn + hicache.
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/models/DeepSeek-V4-Pro"
export TP="${TP:-8}" CONC="${CONC:-32}"
export EP_SIZE="${EP_SIZE:-8}"
export DP_ATTENTION="${DP_ATTENTION:-true}"
export SPEC_DECODING="mtp"
export IS_AGENTIC=1
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
# process_agentic_result requires BOTH the name and the metadata blob whenever
# KV_OFFLOADING != none; the CI matrix normally supplies this one.
export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"hicache\"}}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2399}"
export DURATION="${DURATION:-3600}"
export PORT="${PORT:-8888}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-tp${TP}-c${CONC}}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic-megamoe_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_megamoe_mtp.sh
