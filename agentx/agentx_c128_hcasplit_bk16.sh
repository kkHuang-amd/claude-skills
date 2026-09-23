#!/usr/bin/env bash
# PHASE 3 ARM — fp8 MLA decode with BLOCK_K=16 instead of 32, c128.
#
# One line in paged_decode.py forced BLOCK_K=32 on every quantised-KV decode
# call ("fp8 dequant inflates per-tile ALU work ~4x"), justified at bs=512 /
# ctx=4096. At the agentic operating point that is exactly backwards: measured
# standalone, one config per fresh process, BLOCK_K=16 is 2.0x faster
# (396.4 vs 789.3 us at bs=12 ragged splits=4) and still wins at bs=512 with
# short uniform kv_len (1821 vs 2026 us). relL2 between the two = 1.76e-03.
#
# THE ONLY DIFFERENCE vs agentx_c128_hcasplit.sh is SGLANG_MLA_FP8_BLOCK_K=16.
# splits stay at 4, tree/mem-frac/heap/balancer/DURATION all inherited, so the
# matched-bs delta is attributable to the K tile.
#
# PRE-REGISTERED: -3 to -5 ms/step at matched bs against
# megamoe-eplb-c128-hcasplit4 (`_paged_decode_split_kernel` is 7.97 ms/step on
# rank 7 at bs=12, so a 2x is ~-4 ms). FALSIFICATION: better than -1 ms => the
# microbench does not transfer to the graph-captured production path, and the
# next question is what differs there -- NOT to tune block_k further.
#
# Sanity checks once it is up:
#   rg -o 'accept len: [0-9.]+' $RESULT_DIR/server.log | tail -1   # ~3.77
#   python3 analysis/cmd_diff.py <this arm> <hcasplit4>            # flags identical
set -eo pipefail

export SGLANG_MLA_FP8_BLOCK_K="${SGLANG_MLA_FP8_BLOCK_K:-16}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-hcasplit4-bk16}"
exec bash /workspace/claude-skills/agentx/agentx_c128_hcasplit.sh
