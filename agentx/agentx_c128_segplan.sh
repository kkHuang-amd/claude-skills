#!/usr/bin/env bash
# PHASE 3 ARM — length-aware split-KV ("segment plan") on the dsv4 MLA decode
# kernel, c128. Port of sgl-project/sglang#39172 (which fixed this for the
# gfx950 ASM kernel only) into dsv4/unified_kv_kernels/paged_decode.py.
#
# WHAT CHANGES. The shipped split path gives every token the same number of
# segments, so per-CTA work stays proportional to that token's kv_len and the
# layer still ends when the longest token ends. The plan picks one segment
# LENGTH for the batch, so every CTA does at most T tiles regardless of how
# skewed the batch is.
#
# MEASURED STANDALONE on the PRODUCTION (bf16, no kv_scales) path, ragged
# real_lens distribution, PYTHONPATH pinned to sglang-MegaMoE:
#
#   bs=12  split-K=4 237.4 us -> plan 153.4 us (-35 %), balance floor 143.0
#   bs=24  split-K=4 382.6 us -> plan 299.0 us (-22 %)
#   bs=48  split-K=4 658.0 us -> plan 559.0 us (-15 %)
#   numerics relL2 1.7-2.0e-03 at all three (split-K shipped at 2.46e-03)
#
# PRE-REGISTERED: -1 to -3 ms/step at matched bs against
# megamoe-eplb-c128-hcasplit4. `_paged_decode_split_kernel` is 7.97 ms/step on
# rank 7 at bs=12, and the microbench's distribution (p50 1,090 / max 5,000) is
# probably harsher than production's, so the ceiling is ~-2.8 ms and the floor
# of usefulness is low. FALSIFICATION: better than -0.5 ms => the imbalance the
# microbench prices is not what production's shapes look like, and the next step
# is a kv_len histogram from the arm's own server.log, NOT more tuning.
#
# Settings from the post-bounds-fix sweep: seg_max 16 (32 and 64 are worse --
# more segments cost more reduce than they buy in balance) and wg_mult 4
# (156.5 us vs 176.4 at x2, bs=12, bf16 path). This is also the pair the GSM8K
# A/B was run with, so accuracy and timing refer to the same configuration.
#
# ACCURACY GATE PASSED FIRST (2026-09-18, gsm8k 1319, max-new-tokens 8192,
# EVAL_ONLY=true so MTP acceptance is real, not the golden 3.77 pin):
#   plan off 0.937  /  plan on 0.939   Invalid 0.000 both
# +0.002 is ~3 questions, i.e. inside noise, which is what a different fp32
# reduce order across new segment boundaries should look like.
#
# NOTE this applies to every stream whose kv_splits > 1, not just HCA, unlike
# the split-K arm which overrode HCA alone. If the arm regresses, check the CSA
# and SWA streams' splits before blaming the plan.
set -eo pipefail

export SGLANG_MLA_SEG_PLAN="${SGLANG_MLA_SEG_PLAN:-1}"
export SGLANG_MLA_SEG_MAX="${SGLANG_MLA_SEG_MAX:-16}"
export SGLANG_MLA_SEG_WG_MULT="${SGLANG_MLA_SEG_WG_MULT:-4}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-segplan}"
exec bash /workspace/claude-skills/agentx/agentx_c128_hcasplit.sh
