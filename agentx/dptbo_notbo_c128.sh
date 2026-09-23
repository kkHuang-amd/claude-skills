#!/usr/bin/env bash
# EXPERIMENT 1 of the ITL-gap investigation: DPA with TBO **OFF**, c128, 3600 s.
#
# Controlled partner for dptbo_c128.sh -- identical except ENABLE_TBO=0 and
# RESULT_DIR, so the pair isolates two-batch overlap and nothing else. Report as
#   python3 arm_report.py dptbo-notbo-c128 dptbo-c128
#
# WHY THIS ARM. SGLang is +27..34 % worse than ATOM on ITL p90 at matched
# concurrency (78.4 vs 61.5 ms at c128) on a workload verified identical flag by
# flag, with cache hit flat and fp8 KV on both sides. The 2026-09-02 probe
# (results/tbo-debug-probe) showed why TBO is the prime suspect: SGLang splits
# DSV4's MTP **target-verify** batches into two ubatches, at capture and at
# replay (two_batch_overlap.py:360 captures ONLY tbo=true graphs;
# compute_split_indices_for_cuda_graph_replay splits decode/target-verify
# unconditionally), whereas ATOM's `--enable-tbo` is prefill-only by construction
# (argparse const=prefill -> enable_tbo_decode=False, and decode-TBO is
# MTP-incompatible there). ITL is set on the decode path, which is exactly where
# the two engines differ. SGLang has no prefill-only TBO mode -- server_args.py
# offers only enable_two_batch_overlap and tbo_token_distribution_threshold -- so
# on/off is the only experiment available.
#
# READING IT. ITL p90 toward ~61.5 ms => decode-side TBO is the cause, and the
# finding is upstream-shaped (DSV4 wants a prefill-only TBO mode), not a tuning
# fix. ITL still ~78 ms => the scheduler is exonerated and the next step is
# profiling a decode step, NOT another arm. Throughput will also move; the c128
# throughput gap (-9.2 %) is only just above this node's 5.67 % replicate
# spread, so treat tok/s as secondary and ITL as the signal.
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

# THE VARIABLE UNDER TEST. b200align_mtp.sh:157 drops --enable-two-batch-overlap
# when this is not "1"; every other resolved flag is unchanged. Verify in
# $RESULT_DIR/sglang_command.txt before trusting the result.
export ENABLE_TBO=0

export CHUNK_PER_RANK=16384

# Matched to dptbo_c128.sh, not re-tuned. The container default (=1) retains
# large-grid scratch and cost the first FP4 attempt its run at 44/60 min with
# HSA_STATUS_ERROR_OUT_OF_RESOURCES -- though the c64 A/B later showed a crossed
# pattern, so this is matching the partner, not a fix.
export HSA_NO_SCRATCH_RECLAIM=0

# The launcher's DP+TBO default (sglang#29362) and what dptbo-c128 ran at. Kept
# even though TBO is off here, because changing it would break the pair.
export MEM_FRACTION_STATIC=0.90

# hicache stays off: PR #37353's rust-side pool-name variant was deliberately
# not applied (it needs a rust rebuild), so the unified-radix path must not run.

export RESULT_DIR=/workspace/results/dptbo-notbo-c128
export RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c128"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"

test -f "$MODEL_PATH/config.json" || { echo "FATAL: no config.json at $MODEL_PATH"; exit 2; }
mkdir -p "$RESULT_DIR"

# Tree snapshot. The FP4 integration is uncommitted working-tree state shared
# with other people's local edits; a mid-run change already destroyed one arm,
# because the adapter's aiter imports are lazy and read the tree throughout.
# Verify these four md5s again when the arm finishes.
md5sum \
    /sgl-workspace/sglang/python/sglang/srt/arg_groups/serving_hook.py \
    /sgl-workspace/sglang/python/sglang/srt/layers/attention/dsv4/indexer.py \
    /sgl-workspace/sglang/python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
    /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4_prefill.py \
    >"$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" 2>&1

# Every arm from now on carries this: gpu_metrics.csv has no memory column, so
# none of the six arms on file can say whether the ~45-min OOR abort is a
# monotonic decay or a spike. Validated on the tbo-debug probe (peak 255.3 GB
# used, ~33 GB free per GPU at mem-frac 0.90).
bash /workspace/claude-skills/agentx/vram_sampler.sh "$RESULT_DIR/vram.csv" 15 &
SAMPLER_PID=$!
echo "vram sampler PID: $SAMPLER_PID"
trap 'kill -9 "$SAMPLER_PID" 2>/dev/null' EXIT

bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_b200align_mtp.sh
echo "ARM_EXIT=$?"

kill -9 "$SAMPLER_PID" 2>/dev/null
trap - EXIT

# The launcher does NOT kill the server: it keeps ~275 GB/GPU and the next arm
# then sits in the 15-min drain gate and fails. Kill by PID here.
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
done

echo "=== TBO off? (expect NO --enable-two-batch-overlap) ==="
grep -c -- '--enable-two-batch-overlap' "$RESULT_DIR/sglang_command.txt" || true
echo "=== tree unchanged? ==="
md5sum -c --quiet "$RESULT_DIR/TREE_CHECKSUMS_AT_START.txt" && echo "TREE OK"
echo "=== vram ==="
awk -F, 'NR>1 { if (m == "" || $5 < m) m = $5 } END { print "min free_gb seen: " m }' "$RESULT_DIR/vram.csv"
