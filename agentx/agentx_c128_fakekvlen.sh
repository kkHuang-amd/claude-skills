#!/usr/bin/env bash
# FAKE-KERNEL ARM. Not an optimization -- a counterfactual intervention, and the
# decisive test of the whole MLA line.
#
# Everything established so far about MLA is component-level: the kernel is at
# 1.2-3.1 % of both rooflines, its cost follows the batch's LONGEST sequence,
# and a trace inversion prices "MLA free" at -19.54 ms of the 74.02 ms pure
# decode step. What has NEVER been tested is the chain "MLA faster => step wall
# drops". That chain has been wrong twice: the cross-rank balancing route died
# when `total_tokens` cut the skew 2.08x->1.69x and the step did not move, and
# the `floor` = 5.19 ms it rests on was measured with MLA UNCHANGED. Whether the
# floor grows or a new serialisation appears once MLA shrinks, no model can say.
#
# SGLANG_MLA_FAKE_KVLEN=128 clamps every token's kv_len inside
# `_sparse_attn_v4_paged_decode_triton` (paged_decode.py, `_fake_clamp_indptr`),
# which removes ~95 % of the kernel's work (measured eager at the production
# distribution: 1891.7 -> 90.9 us/call). It is the upper bound of any MLA work,
# straggler-aware or otherwise.
#
# PREDICTION, WRITTEN BEFORE THE RUN. Reference `megamoe-eplb-c128-b200aligned`
# has bs=14 p50 = 123.83 ms (decode_stats.py, n=247). MLA free is -19.5 ms on
# the pure-decode scale, so bs=14 should land at ~104 ms. A drop of only 3-5 ms
# falsifies the line, and neither the microbenchmark nor any kernel work should
# then proceed.
#
# READ ONLY STEP-LEVEL NUMBERS: log-implied step ms at matched bs, and ITL p90.
# The output is GARBAGE, so OSL, queueing, TTFT, throughput and cache hit are
# all meaningless here. `accept len` is not a confound -- AgentX pins it at
# SGLANG_SIMULATE_ACC_LEN=3.77 (launcher :302-311) -- but verify it in the log.
#
# Capture-safety was verified BEFORE this script was allowed to run:
# `analysis/fake_kvlen_validate.py` (GPU0, ~7 s) checks the indptr arithmetic on
# a hand-computed case, that the clamp removes the work, and that a local
# torch.cuda.CUDAGraph capture + replay around the real kernel succeeds. Launch
# #4 of this project died with hipErrorStreamCaptureUnsupported because a probe
# wrote a host scalar into a device tensor; that check is not optional.
#
# DURATION 3600 and `total_requests` both match the reference arm exactly, so
# the matched-bs comparison is against like conditions. pdi is NOT set: at
# CONC=128 the launcher defaults it to 24, which is what the reference ran --
# verify in the produced sglang_command.txt rather than trusting this note.

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

# The single variable under test. 128 = the SWA window, i.e. the smallest span
# the kernel can be given without changing its shape. 0 disables the clamp.
export SGLANG_MLA_FAKE_KVLEN="${SGLANG_MLA_FAKE_KVLEN:-128}"

# OFF. The probe is capture-safe now, but it adds ~14 % per call and this is a
# TIMING arm; the distribution it measures is already recorded.
export SGLANG_MLA_KVLEN_STATS="${SGLANG_MLA_KVLEN_STATS:-0}"

# Matches the reference arm. `total_tokens` is closed (see CONTINUE_HERE).
export LOAD_BALANCE_METHOD="${LOAD_BALANCE_METHOD:-total_requests}"

export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-fakekvlen}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
