#!/usr/bin/env bash
# DPA+TBO c64, 3600 s. Controlled partner for mori-mxfp8-ep8-c64: identical
# except for the mode (no EP, no MoRI a2a, TBO on).
set -u

cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=64 DURATION=3600 PORT=8888
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0

# The mode, and the only intended difference from mori_mxfp8_c64.sh:
# no MOE_A2A_BACKEND (so no --moe-a2a-backend / --deepep-mode /
# --moe-dense-tp-size 1 / --enable-dp-lm-head), EP_SIZE=1, TBO on.
export ENABLE_TBO=1

export CHUNK_PER_RANK=16384

# Held at the mori arm's 0.85 rather than the launcher's 0.90 default, so both
# arms get the same KV pool. Without mori there is no symmetric heap, so this
# arm simply leaves the extra VRAM idle. Note this makes it NOT comparable to
# the previous node's DPA+TBO rows, which ran at 0.90.
export MEM_FRACTION_STATIC=0.85

export RESULT_DIR=/workspace/results/dptbo-c64
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c64"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"
