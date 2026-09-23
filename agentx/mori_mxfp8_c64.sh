#!/usr/bin/env bash
# MoRI-EP + MXFP8 dispatch, c64, 3600 s.
# Requires the three worktree patches recorded in AGENTX_20260901.md
# (aiter#4954, aiter dsv4_ep_tune, mori#600) — without aiter#4954 sglang
# silently falls back to bf16 dispatch and this arm measures nothing.
set -u

cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=8 DP_ATTENTION="true"
export CONC=64 DURATION=3600 PORT=8888
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0

export MOE_A2A_BACKEND=mori
export DEEPEP_MODE=normal
export SGLANG_MORI_DISPATCH_DTYPE=mxfp8
export ENABLE_TBO=0

# 16384/rank => --chunked-prefill-size 131072, and the launcher derives
# SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384 from it. That doubles the
# mori dispatch/combine buffers, which live in the symmetric heap, so the heap
# is raised from the launcher's 16G default. The heap sits outside
# --mem-fraction-static, so this trades against free VRAM, not the KV pool.
export CHUNK_PER_RANK=16384
export MORI_SHMEM_HEAP_SIZE=24G

# mori's symmetric heap (16G) is allocated OUTSIDE this budget, and plain EP8
# at c64 already OOR'd at 0.90 on the previous node with 388 MB to spare.
export MEM_FRACTION_STATIC=0.85

export RESULT_DIR=/workspace/results/mori-mxfp8-ep8-c64
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep8-dpatrue_disagg-false_spec-mtp_agentic_c64"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"
