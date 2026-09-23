#!/usr/bin/env bash
# Smoke arm for the rebuilt node: DURATION=300, CONC=32, TP8, no DP/TBO.
# Proves repo -> venv -> traces -> weights -> server -> aiperf -> result json.
set -u

cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="false"
export CONC=32 DURATION=300 PORT=8888
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0
export AIPERF_WARMUP_REQUESTS_PER_LANE=1
export RESULT_DIR="/workspace/results/smoke-tp8-c32"
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_agentic_c32"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
echo "SMOKE_EXIT=$?"
