#!/usr/bin/env bash
# AgentX MegaMoE + EPLB at c64 on the PR#35619 tree with mega_moe_eplb.diff applied.
# Derived from /workspace/claude-skills/agentx/agentx_megamoe.sh with three changes:
#   - MODEL_PATH moved to /shared_nfs/deepseek-ai (the old /shared_nfs/models path is gone)
#   - ENABLE_EPLB=1 and SGLANG_AITER_MEGA_RANK_SYNC=1 (the IndexError that forced 0 is fixed)
#   - launcher swapped for the copy that emits --cuda-graph-max-bs-decode
set -eo pipefail
SKILL_DIR=/workspace/claude-skills/agentx
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP="${TP:-8}" CONC="${CONC:-64}"
export EP_SIZE="${EP_SIZE:-8}"
export DP_ATTENTION="${DP_ATTENTION:-true}"
export SPEC_DECODING="mtp"
export IS_AGENTIC=1
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"hicache\"}}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2399}"
export DURATION="${DURATION:-900}"
export PORT="${PORT:-8888}"

export ENABLE_EPLB="${ENABLE_EPLB:-1}"
export SGLANG_AITER_MEGA_RANK_SYNC="${SGLANG_AITER_MEGA_RANK_SYNC:-1}"
export SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY="${SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY:-1}"
export SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD="${SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD:-1}"

export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c${CONC}-d${DURATION}}"
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$RESULT_DIR/eplb"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic-megamoe-eplb_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR" "$SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_megamoe_mtp_cgbsfix.sh
