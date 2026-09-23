#!/usr/bin/env bash
# PHASE 3a ARM — SGLANG_MLA_HCA_KV_SPLITS=8 instead of 4, c128.
#
# The microbench chose 4 and was right about the magnitude (-7.42/-7.50 ms
# measured against a pre-registered -8.00), but 8 was never tested in situ. The
# synthetic sweep said 8 wins only at low kv_len (-41.8 % vs -35.8 % for 4 at
# small kv_len) and that 4 gives ~93 % of 8 at steady state, while degrading
# more gracefully across the batch range.
#
# PRE-REGISTERED: 0 to -1.5 ms/step at matched bs against
# megamoe-eplb-c128-hcasplit4 (NOT against b200aligned -- the question is 8 vs
# 4, not split-K vs none). FALSIFICATION: worse than +1 ms => close the knob and
# keep 4.
#
# WATCH: acc_partial doubles, 205 MB vs 103 MB at bs=14, and it lives in the
# cuda-graph pool at mem-frac 0.85. If capture OOMs, that is the answer to "why
# not 8" and is worth recording as such rather than retrying at a lower
# mem-fraction, which would confound the comparison.
#
# Everything else is inherited from agentx_c128_hcasplit.sh unchanged.
set -eo pipefail

export SGLANG_MLA_HCA_KV_SPLITS="${SGLANG_MLA_HCA_KV_SPLITS:-8}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-hcasplit8}"
exec bash /workspace/claude-skills/agentx/agentx_c128_hcasplit.sh
