#!/usr/bin/env bash
# Direction A A/B: --load-balance-method total_tokens, everything else matched
# to /workspace/results/megamoe-eplb-c128-b200aligned (the reference arm).
#
# Two questions, one arm:
#
#  1. Does `total_tokens` help end to end? It fights `cache_aware` prefix reuse
#     (the router is a separate process, --policy cache_aware, launcher :368),
#     so expect a TTFT regression and report ITL, TTFT and cache hit together.
#
#  2. It is also a FALSIFICATION TEST. The microbenchmark showed the MLA decode
#     kernel's cost follows the LONGEST sequence in the batch, not the total
#     (+71 % for 2x the max at constant total, -4 % for half the total at
#     constant max). `total_tokens` equalises the total. Prediction: MLA
#     us/call barely moves. If it moves a lot, the straggler finding is wrong.
#
# SGLANG_MLA_KVLEN_STATS=1 adds `kvlen mean/max/min` and `kvlen straggler` to
# each decode log line, which is what sizes a straggler-aware kernel rewrite.
# Overhead is emitted in ONE layer per forward, measured at noise (344.1 vs
# 346.2 us standalone), so it does not disturb the timings above.
#
# pdi is NOT set here on purpose: at CONC=128 the launcher defaults it to 24
# (dsv4_fp4_mi355x_sglang_mtp.sh:203), which is what the reference arm ran.
# VERIFY it in the produced sglang_command.txt rather than trusting this note.
set -eo pipefail
SKILL_DIR=/workspace/claude-skills/agentx
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

export MODEL="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813"
export MODEL_PATH="$MODEL"
export MODEL_PREFIX="dsv4"
export TP="${TP:-8}" CONC="${CONC:-128}"
export EP_SIZE="${EP_SIZE:-8}"
export DP_ATTENTION="${DP_ATTENTION:-true}"
export ENABLE_MEGAMOE=1
export ENABLE_EPLB="${ENABLE_EPLB:-1}"
export SPEC_DECODING="mtp"
export IS_AGENTIC=1
export KV_OFFLOADING="${KV_OFFLOADING:-dram}"
export KV_OFFLOAD_BACKEND="${KV_OFFLOAD_BACKEND:-hicache}"
export KV_OFFLOAD_BACKEND_METADATA="${KV_OFFLOAD_BACKEND_METADATA:-{\"name\":\"hicache\"}}"
export TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-2399}"
export DURATION="${DURATION:-3600}"
export PORT="${PORT:-8888}"

# The launcher's MegaMoE+DP branch defaults mem-fraction-static to 0.65
# (dsv4_fp4_mi355x_sglang_mtp.sh:216), but the reference arm ran 0.85. Left
# alone this silently changes the KV pool, and with it cache hit and batch
# composition -- a confound that would have invalidated the whole arm. Verify
# with analysis/cmd_diff.py against the reference ~60 s after launch.
export MEM_FRACTION_STATIC_DP_MEGAMOE="${MEM_FRACTION_STATIC_DP_MEGAMOE:-0.85}"

# The two variables under test.
export LOAD_BALANCE_METHOD="${LOAD_BALANCE_METHOD:-total_tokens}"
export SGLANG_MLA_KVLEN_STATS="${SGLANG_MLA_KVLEN_STATS:-1}"

export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-b200aligned-totaltokens}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
