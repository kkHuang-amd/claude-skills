#!/usr/bin/env bash
# Baseline: DPA+TBO, c128, 3600 s. Controlled partner for fp4_dptbo_c128.sh --
# identical except that this one does NOT set EXTRA_SERVER_ARGS, so the FP4 C4
# indexer stays off and the existing fp8 indexer path runs.
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

# Must match fp4_dptbo_c128.sh -- see the longer note there. The container
# default (=1) retains large-grid scratch and cost the first FP4 attempt its run
# at 44/60 min with HSA_STATUS_ERROR_OUT_OF_RESOURCES.
export HSA_NO_SCRATCH_RECLAIM=0

# The launcher's DP+TBO default (sglang#29362), NOT the 0.85 the c64 arms used
# to match mori's out-of-budget symmetric heap. There is no mori here, so the
# tuned value applies. This makes the c128 pair internally consistent but NOT
# comparable to the c64 rows in the table.
export MEM_FRACTION_STATIC=0.90

# hicache stays off: PR #37353's rust-side pool-name variant was deliberately
# not applied (it needs a rust rebuild), so the unified-radix path must not run.

export RESULT_DIR=/workspace/results/dptbo-c128
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c128"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"
