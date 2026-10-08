#!/usr/bin/env bash
# TP4 c1/c8/c16 AgentX rerun for TP4_GAP_1006.md: agentx_lane.sh + the full env the InferenceX runner would set.
#   bash tp4_gap_lane.sh            # launch the lane (GPUs 0-3, port 8888)
#   CHECK_ONLY=1 bash tp4_gap_lane.sh   # only verify every check_env_vars name in benchmark_lib.sh is set
# benchmark_lib.sh (InferenceX b5d0e56a2) aborts aiperf after server start when any of these is unset; aiperf values
# match the B200 vLLM run 37070984585 (threshold 0.10, idle gap cap 300, warmup 10/lane, grace 1800). Power capture off.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source /workspace/claude-skills/agentx/agentx_env.sh >/dev/null 2>&1
source "${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX}/benchmarks/runtime_settings.sh"
export AIPERF_EXPERIMENTAL_FAST=0 ENABLE_AGENTX_POWER=0 REQUIRE_POWER=0 IS_MULTINODE=false PP_SIZE=1 PCP_SIZE=1
export PYTHONPATH=/sgl-workspace/mori SRC=/sgl-workspace/sglang/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=${SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL:-1} OPUS=0
export EXTRA_ARGS="${EXTRA_ARGS:---fp8-gemm-backend aiter --enforce-shared-experts-fusion}"   # never empty (lane would pass "")
export TP=${TP:-4} EP_SIZE=1 GPUS=${GPUS:-0,1,2,3} PORT=${PORT:-8888}
export POINTS=${POINTS:-"m255_tp4_c1_rep:1:16:16384:0.70 m255_tp4_c8_rep:8:16:16384:0.70 m255_tp4_c16_rep:16:16:16384:0.80"}

L=${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX}/benchmarks/benchmark_lib.sh
# Names set per run by agentx_colleague_run.sh, or only checked on multinode / swebench / kimi paths.
skip=" CONC DURATION MODEL MODEL_PREFIX MODEL_PATH RESULT_FILENAME RESULT_DIR AGENTIC_OUTPUT_DIR IS_AGENTIC SPEC_DECODING
 KV_OFFLOADING TOTAL_CPU_DRAM_GB DP_ATTENTION EVAL_ONLY INFMAX_CONTAINER_WORKSPACE SWEBENCH_GEN_MODE SWEBENCH_USE_MODAL "
skip+=" VENDOR_VERIFIER_PYTHON VAR1 VAR2 "
missing=""
for v in $(awk '/check_env_vars/{f=1} f{print} f&&!/\\$/{f=0}' "$L" | grep -oE '\b[A-Z][A-Z0-9_]{2,}\b' | sort -u); do
    [[ "$skip" == *" $v "* ]] && continue
    [ -z "${!v+x}" ] && missing+=" $v"
done
if [ -n "$missing" ]; then echo "UNSET:$missing"; exit 1; fi
echo "env check OK"
[ "${CHECK_ONLY:-0}" = 1 ] && exit 0
exec bash "$D/agentx_lane.sh"
