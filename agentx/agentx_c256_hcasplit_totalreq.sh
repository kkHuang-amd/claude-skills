#!/usr/bin/env bash
# c256 + LAYER-AWARE SPLIT-K + total_tokens balancing.
#
# Same code and same gates as the c128 split-K arm; only CONC and the
# balancer differ. The launcher derives max-running-requests (512) and
# prefill-decode-interval (20) from CONC -- verify both in the produced
# sglang_command.txt, and expect the KV pool to differ from c128
# (11,906,048 vs 12,077,312), which is normal at this concurrency.
#
# ORIGINAL HEADER FOLLOWS -- the prediction in it is the c128 one.
#
# LAYER-AWARE SPLIT-K ARM. The first arm in this line that ships a real
# optimization rather than a counterfactual.
#
# `_kv_splits_heuristic` is an OCCUPANCY rule: split only when the base grid
# underfills the device. At bs=14 the grid nearly saturates (196 CTAs against a
# 384 target) so it picks splits=1 -- but the cost is set by ONE straggler CTA
# walking ~5,000 KV entries beside CTAs walking 200, which occupancy cannot see.
#
# The discriminator is static and available at capture time: `compress_ratio`,
# read at the only call site that knows it
# (deepseek_v4_backend_hip_radix.py, `runtime.decode(... kv_splits=...)`).
# CSA (ratio 4) is clamped to index_topk+128 = 1152 and split-K LOSES on it
# (+5.1 %); HCA (ratio 128) is unclamped, reaches ~5,000, and split-K wins at
# every kv_len the run traverses. Only HCA is overridden.
#
# This retires the standing "uniform split-K is the wrong tool" verdict, whose
# premise was that capture-time scalars cannot separate ragged from uniform
# shapes. compress_ratio is exactly that separator.
#
# PREDICTION, WRITTEN BEFORE THE RUN. HCA is ~80 % of MLA time (per-call 1,884
# vs 504 us at bs=14, with the layers split ~30/31), rank 1's MLA is
# 330.7 us x 61 = 20.17 ms/step, so HCA is ~15.9 ms. The sweep says splits=4
# takes ~50 % off it at the steady-state kv_len => about -8 ms.
#   reference megamoe-eplb-c128-b200aligned, bs=14 p50 = 123.83 ms
#   expected  ~116 ms, i.e. ~44 % of the fake-kernel arm's -18.19 ms ceiling
# A drop under 2 ms means the synthetic microbench does not transfer to the
# real per-layer shapes, and the next step is to find out why -- NOT to tune
# the split count.
#
# Unlike the fake-kernel arm this one produces CORRECT output, so throughput,
# TTFT and cache hit are all readable. Still quote matched-bs step time as the
# primary number, for comparability with everything before it.
#
# Validated before launch by `analysis/layer_split_validate.py` (GPU0, ~8 s):
# stream gating, the override reaching the kernel, numerics against the fused
# path (relL2 2.46e-03), and CUDAGraph capture + replay on the split path --
# which is a DIFFERENT code path (partial buffers + reduce kernel) from the
# fused one production has been capturing until now.
#
# Set SGLANG_MLA_HCA_KV_SPLITS=0 to fall back to the heuristic without editing
# code, which is how to A/B this arm.

set -eo pipefail
SKILL_DIR=/workspace/claude-skills/agentx
cd /workspace/InferenceX
source "$SKILL_DIR/agentx_env.sh"

# ⚠ PIN THE TREE. The reference arm ran from /sgl-workspace/sglang-MegaMoE
# (36 occurrences in its server.log), but a bare `import sglang` now resolves to
# /sgl-workspace/sglang, which is a DIFFERENT, actively-edited checkout (27
# dirty files as of 2026-09-17, including dp_attn.py and forward_batch_info.py,
# with a live vim session in it). Two launches against that tree OOMed
# identically at cuda-graph capture -- 236.93 GiB allocated, 588 MiB free --
# where the reference had ~20 GiB spare at the same point. Even had it started,
# the arm would have been confounded by another person's work in progress.
# cmd_diff.py cannot catch this: it compares CLI flags, not code.
#
# All three values below are RECOVERED FROM THE REFERENCE ARM'S OWN LAUNCH LOG
# (/shared_nfs/kk/pr35619/b200aligned_c128.log), not guessed:
#   MEM_FRACTION_STATIC=0.85
#   MORI_SHMEM_HEAP_SIZE=17179869184        (= 16 GiB, NOT the launcher's 40G)
#   PYTHONPATH=/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori:
export PYTHONPATH="/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori${PYTHONPATH:+:$PYTHONPATH}"

export MODEL="/shared_nfs/deepseek-ai/DeepSeek-V4-Pro-0813"
export MODEL_PATH="$MODEL"
export MODEL_PREFIX="dsv4"
export TP="${TP:-8}" CONC="${CONC:-256}"
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
export DURATION="${DURATION:-3600}"   # matches the reference arm
export PORT="${PORT:-8888}"

# The launcher's MegaMoE+DP branch defaults mem-fraction-static to 0.65
# (dsv4_fp4_mi355x_sglang_mtp.sh:216), but EVERY MegaMoE number we have
# published was taken at 0.85 -- so the default is off-matrix, not just
# different from this arm's reference. Left alone it silently changes the KV
# pool, and with it cache hit and batch composition. Verify with
# analysis/cmd_diff.py against the reference ~60 s after launch.
export MEM_FRACTION_STATIC_DP_MEGAMOE="${MEM_FRACTION_STATIC_DP_MEGAMOE:-0.85}"

# The launcher gained `MORI_SHMEM_HEAP_SIZE:-40G` after the reference arm ran,
# and that heap is charged OUTSIDE mem-fraction-static, so 40G does not fit:
#   236.93 (PyTorch static at 0.85) + 40 (heap) + 20.99 (target-verify capture)
#   = 297.9 GiB > 287.98 GiB, before the ~5 GiB driver context.
# Four launches OOMed on exactly that, each on a different GPU, <1 GiB free.
# The reference ran 16 GiB (see the launch log quoted above), which is also the
# value the launcher's own comment records as measured-sufficient.
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-16G}"

# The single variable under test. 4, not 8: it wins outright at low kv_len
# (-41.8 % vs -35.8 %), gives ~93 % of 8 at steady state, and halves the
# partial buffers (acc_partial 103 MB vs 205 MB at bs=14, inside the graph
# pool at mem-fraction 0.85). 0 = keep the occupancy heuristic.
export SGLANG_MLA_HCA_KV_SPLITS="${SGLANG_MLA_HCA_KV_SPLITS:-4}"

# OFF -- that was the counterfactual arm, and it produces garbage output.
export SGLANG_MLA_FAKE_KVLEN="${SGLANG_MLA_FAKE_KVLEN:-0}"

# OFF: ~14 % per call, and this is a timing arm.
export SGLANG_MLA_KVLEN_STATS="${SGLANG_MLA_KVLEN_STATS:-0}"

# total_tokens, as asked. NOTE this makes the arm differ from the c256
# reference (megamoe-eplb-c256-b200aligned, total_requests) in TWO ways,
# balancer and split-K, so the two cannot be attributed apart from this
# run alone. At c128 total_tokens was measured null on step time
# (0 to +2.4 % at matched bs), so the delta here should be mostly split-K.
export LOAD_BALANCE_METHOD="${LOAD_BALANCE_METHOD:-total_requests}"

export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c256-hcasplit4-totalreq}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
