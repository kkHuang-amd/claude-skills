#!/usr/bin/env bash
# MoRI-EP + MXFP8 dispatch + SGLANG_MORI_RECV_BOUND=true, c64, 3600 s.
# Controlled partner for mori-mxfp8-ep8-c64: identical except the recv bound.
set -u

# The launcher never kills its own server, so the PREVIOUS arm is still running
# when this one starts. Three different process shapes have to go, and missing
# any one of them kills this arm at startup:
#   python3 -m sglang.launch_server   holds port_base 9123 and 8889
#   sglang::tokenizer_worker:N        holds 8889, uses NO VRAM
#   sglang::router                    holds 8888
# A clean rocm-smi is NOT evidence the node is free -- the workers hold ports
# without holding memory. Symptoms are "[Errno 98] Address already in use" or
# "port_base at 9123 is not available in 30 seconds".
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
done
sleep 12

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

# The one variable under test (sglang#36130, merge aa0a0aa3c3, already in HEAD).
# Decode only -- prefill always stays unbounded -- so on this prefill-heavy
# trace the leverage is small. Verify it actually engaged by grepping server.log
# for "mori recv bound active"; absence means it degraded to unbounded.
export SGLANG_MORI_RECV_BOUND=true

export CHUNK_PER_RANK=16384
export MORI_SHMEM_HEAP_SIZE=24G
export MEM_FRACTION_STATIC=0.85

export RESULT_DIR=/workspace/results/mori-mxfp8-ep8-c64-recvbound
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep8-dpatrue_disagg-false_spec-mtp_agentic_c64"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"
