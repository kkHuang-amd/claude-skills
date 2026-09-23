#!/usr/bin/env bash
# FP4 C4 indexer (sglang#37353) on DPA+TBO, c128, 3600 s. Controlled partner
# for dptbo_c128.sh -- identical except for EXTRA_SERVER_ARGS below.
set -u

# The launcher never kills its own server, so a previous arm may still be
# running. Three process shapes have to go; missing any one kills this arm at
# startup with "[Errno 98] Address already in use" or "port_base at 9123 is not
# available in 30 seconds". A clean rocm-smi is NOT evidence the node is free:
# the tokenizer workers hold ports without holding VRAM.
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
done
sleep 12

cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=128 DURATION=3600 PORT=8888
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0

export ENABLE_TBO=1
export CHUNK_PER_RANK=16384

# The container default is 1, which retains the scratch a large-grid kernel
# allocated instead of returning it. The first fp4-dptbo-c128 attempt died at
# 44/60 min with HSA_STATUS_ERROR_OUT_OF_RESOURCES / "Available Free mem : 0 MB"
# right after two 16,384-token prefill chunks, on a demonstrably healthy
# workload (3,623 decode batches, accept len ~2.4, no --chat-template). That is
# exactly the condition the b200align launcher's comment reserved for re-adding
# this, so the result now counts as evidence rather than a workaround. Set on
# both arms of the pair so the environment stays matched.
export HSA_NO_SCRATCH_RECLAIM=0

# The launcher's DP+TBO default (sglang#29362), NOT the 0.85 the c64 arms used
# to match mori's out-of-budget symmetric heap. There is no mori here, so the
# tuned value applies. This makes the c128 pair internally consistent but NOT
# comparable to the c64 rows in the table.
export MEM_FRACTION_STATIC=0.90

# hicache stays off: PR #37353's rust-side pool-name variant was deliberately
# not applied (it needs a rust rebuild), so the unified-radix path must not run.

# THE VARIABLE UNDER TEST. Needs sglang#37353 applied to the worktree and the
# EXTRA_SERVER_ARGS hook in the b200align launcher. The flag is gated on
# is_gfx95_supported() (True here); on an unpatched tree the server would
# refuse to start rather than silently fall back, which is what we want.
export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer"

export RESULT_DIR=/workspace/results/fp4-dptbo-c128
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c128"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"
