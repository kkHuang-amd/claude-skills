#!/usr/bin/env bash
# AgentX dp8+tp8 at concurrency 128 on dsv4_fp4_mi355x_sglang_mtp.sh (the plain
# DP arm, ENABLE_MEGAMOE stays 0). Target to match: 34,432.5 tok/s/GPU,
# P90 interactivity 18.8.
#
# MODEL is the local checkpoint path, not the HF repo id: benchmark_lib passes
# $MODEL straight to aiperf as --model/--tokenizer, and
# deepseek-ai/DeepSeek-V4-Pro-0813 is not in /shared_nfs/hf_cache, so the repo
# id would 401 the way it did in the runbook replay.
set -eo pipefail
SKILL_DIR=/workspace/claude-skills/agentx
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

export MODEL="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813"
export MODEL_PREFIX="dsv4"
export TP="${TP:-8}" CONC="${CONC:-128}"
# EP off: the launcher only adds --ep-size when EP_SIZE>1, and keeps
# --enforce-shared-experts-fusion at EP_SIZE=1. Matches the TP/DP-only table.
export EP_SIZE="${EP_SIZE:-1}"
export DP_ATTENTION="${DP_ATTENTION:-true}"
export SPEC_DECODING="mtp"
export IS_AGENTIC=1
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"hicache\"}}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2399}"
export DURATION="${DURATION:-3600}"
export PORT="${PORT:-8888}"

export RESULT_DIR="${RESULT_DIR:-/workspace/results/dp8-c${CONC}-20260914}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
