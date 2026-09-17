#!/usr/bin/env bash
# kv_len DISTRIBUTION PROBE. Not a performance arm -- it exists to measure one
# number: (max - mean) / max of the per-token kv_len, logged per decode line as
# `kvlen straggler`.
#
# Why that number decides the next month of work: the MLA decode kernel's cost
# follows the batch's LONGEST sequence, not its total (microbenchmark: +71 % for
# 2x the max at constant total, -4 % for half the total at constant max). So a
# straggler-aware kernel can only win what the spread inside a batch gives it.
# If max ~= mean the whole line closes here, for the cost of one short arm.
#
# It also discriminates two very different worlds. The 2.9x spread we know about
# (per-rank implied kv_len 244-708) is BETWEEN ranks; nobody has measured the
# spread WITHIN a batch. If the dispersion is within-batch a straggler fix
# works; if ranks are internally uniform and merely sit at different levels, it
# does nothing and the only route left is making MLA faster outright.
#
# `total_requests` on purpose: it matches the trace the counterfactual is built
# on, and the balancer does not affect this distribution anyway.
#
# DURATION 1800: the distribution must be read at a representative KV working
# set, never on a timer. tok/req reaches ~140k by minute 15 and 147-172k by
# 25-40 (steady state is 165-170k), so read the trend against tok/req and
# discard the early samples rather than trusting the run length.
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
export DURATION="${DURATION:-1800}"
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

# The two variables under test.
export LOAD_BALANCE_METHOD="${LOAD_BALANCE_METHOD:-total_requests}"
# OFF. The probe as written is NOT cuda-graph-capture-safe: writing the
# host-side `d.numel()` into the device buffer is a pageable H2D copy, which
# aborts capture with hipErrorStreamCaptureUnsupported. Lazily allocating the
# buffer inside capture is the second problem. Fix both (drop the host scalar,
# allocate the buffer from the backend's init, outside capture) before turning
# this on, or sample the distribution from a --disable-cuda-graph run instead,
# where the distribution is identical and Python runs every step.
export SGLANG_MLA_KVLEN_STATS="${SGLANG_MLA_KVLEN_STATS:-1}"

export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-kvlenprobe}"
export RESULT_FILENAME="dsv4_fp4_sglang_tp${TP}-pp1-dcp1-pcp1-ep${EP_SIZE}-dpa${DP_ATTENTION}_disagg-false_spec-mtp_agentic_c${CONC}"
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
mkdir -p "$RESULT_DIR"
exec bash benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh
