#!/usr/bin/env bash
# PHASE 1 ARM — split-K=4 + two-batch overlap, c128.
#
# The question: the decode step is a strictly serial chain (`ovl = ksum/wall =
# 1.00` on every rank, 3,065 launches/step) and 226 of 256 CUs sit unclaimed for
# the 10-16 ms/step that `megamoe_prepare_compact` spins on its epoch gate. TBO
# is the existing mechanism for putting another ubatch's work into that window.
#
# THE ONLY DIFFERENCE vs agentx_c128_hcasplit.sh is the flag. Everything else --
# tree pin, mem-frac 0.85, mori heap 16G, splits=4, total_requests, DURATION --
# is inherited, so the matched-bs delta is attributable to TBO.
#
# ⚠ `ENABLE_TBO=1` DOES NOT WORK ON THIS LAUNCHER. That variable is read only by
# dsv4_fp4_mi355x_sglang_b200align_mtp.sh:157 (the fp4_dptbo_* arms). The
# MegaMoE arms run dsv4_fp4_mi355x_sglang_mtp.sh, which had no TBO wiring and no
# pass-through, so ENABLE_TBO=1 would have been silently swallowed and produced
# a clean null. The pass-through (EXTRA_SERVER_ARGS) was added to that launcher
# on 2026-09-18 for exactly this arm.
#
# Feasibility, checked before spending the run: check_two_batch_overlap
# (arg_groups/validation_hook.py:481) only rejects TBO when moe_a2a_backend is
# "none" AND DP attention is off. MegaMoE is an a2a backend with DPA on, so the
# flag is accepted. Whether the MegaMoE path actually splits ubatches is the
# open question -- agentx_mori.sh:19 records "TBO x a2a is untested here".
#
# PRE-REGISTERED: -5 to -10 ms/step at matched bs against
# megamoe-eplb-c128-b200aligned. FALSIFICATION: better than -2 ms => TBO cannot
# exploit this window. And read the MECHANISM from `ovl`, not the step time: the
# claim is "kernels now overlap", so a trace must show ksum/wall > 1.0. A step
# win with ovl still 1.00 means something else moved.
#
# FIRST CHECK AFTER LAUNCH (~60 s), before trusting anything:
#   rg -o 'two.batch[a-z-]*' $RESULT_DIR/sglang_command.txt
# If that prints nothing the flag never reached the server and the arm is void.
set -eo pipefail

export EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:---enable-two-batch-overlap}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/megamoe-eplb-c128-hcasplit4-tbo}"
exec bash /workspace/claude-skills/agentx/agentx_c128_hcasplit.sh
