#!/usr/bin/env bash
# FP4 C4 indexer (sglang#37353) on DPA+TBO, c160, 3600 s.
# Byte-for-byte the settings of fp4_dptbo_c96.sh except CONC=160 (and the
# result paths), so it extends the FP4 curve to a further operating point.
# max-running-requests and cuda-graph-max-bs are derived from CONC by the
# launcher, so they follow automatically (320 / 160 here).
set -u

# NOTE: this arm does NOT carry the usual three-process kill preamble.
# Another session shares this node, and a blind kill of every sglang process
# would destroy its running arm. Instead we refuse to start if the node is
# busy, and the caller waits. A clean rocm-smi is NOT evidence the node is
# free -- the tokenizer workers hold ports without holding VRAM -- so this
# checks processes, not memory.
busy=$(ps -eo args \
    | grep -Ec "[s]glang::|[s]glang\.launch_server|[s]glang_router|[a]iperf (system_controller|profile)")
if [ "$busy" -ne 0 ]; then
    echo "FATAL: node busy ($busy sglang/aiperf processes) -- refusing to start."
    echo "       Another session's arm is probably still running. Wait for 0."
    exit 3
fi

cd /workspace/InferenceX
source /workspace/claude-skills/agentx/agentx_env.sh

export MODEL="deepseek-ai/DeepSeek-V4-Pro"
export MODEL_PREFIX="dsv4"
export MODEL_PATH="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro"
export TP=8 EP_SIZE=1 DP_ATTENTION="true"
export CONC=160 DURATION=3600 PORT=8888
export IS_AGENTIC=1 KV_OFFLOADING="none" TOTAL_CPU_DRAM_GB=0

export ENABLE_TBO=1
export CHUNK_PER_RANK=16384

# Kept at the certified fp4-dptbo-c128 arm's value so this row is comparable to
# it. Note the 2x2 at c64/c128 showed this flag does NOT decide whether the
# ~45-minute HSA_STATUS_ERROR_OUT_OF_RESOURCES abort happens -- each setting has
# one survival and one abort -- so this arm may still die that way.
export HSA_NO_SCRATCH_RECLAIM=0

export MEM_FRACTION_STATIC=0.90

# hicache stays off: PR #37353's rust-side pool-name variant was deliberately
# not applied (it needs a rust rebuild), so the unified-radix path must not run.

export EXTRA_SERVER_ARGS="--enable-deepseek-v4-fp4-indexer"

export RESULT_DIR=/workspace/results/fp4-dptbo-c160
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c160"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

# Integrity snapshot. The FP4 adapter's aiter imports are lazy, inside
# functions, so this arm reads the working tree for its whole 1.5 h. A branch
# switch mid-run already destroyed one arm; verify these at the end.
md5sum /sgl-workspace/sglang/python/sglang/srt/arg_groups/serving_hook.py \
       /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
       /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
       /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
    >"$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt"

# VRAM sampler. gpu_metrics.csv carries clocks, power and activity but NO
# memory column, which is why no arm so far can say whether the ~45-minute OOR
# abort comes from a monotonic decay or a single spike. 30 s cadence is far
# below the telemetry already running, so it cannot perturb the measurement.
(
    while :; do
        printf '%s ' "$(date '+%F %T')"
        rocm-smi --showmeminfo vram 2>/dev/null \
            | rg -o 'Used Memory \(B\): [0-9]+' | awk '{printf "%.1f ", $NF/1073741824}'
        printf '\n'
        sleep 30
    done
) >"$RESULT_DIR/vram_samples.txt" 2>&1 &
SAMPLER_PID=$!
trap 'kill -9 $SAMPLER_PID 2>/dev/null' EXIT

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"
