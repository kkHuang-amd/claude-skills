#!/usr/bin/env bash
# MoRI EP a2a arm on top of the b200align AgentX shape (2026-08-31).
#
# Context: sglang#36130 ("bound the MoRI receive buffer during decode") is
# ALREADY in HEAD cdbfe90b4a -- it only adds the opt-in env SGLANG_MORI_RECV_BOUND
# (default off). This wrapper is the expensive half: turning the mori a2a path on
# at all. Relative to c64-chunk16384-newmain it moves several things at once
# (EP1->EP8, mori a2a, moe_dense_tp_size, dp_lm_head, TBO off, mem-fraction), so
# it is NOT a single-variable comparison against that baseline. Its comparable
# partner is another mori arm, or the megamoe arms (also ep8 + mori-based a2a).
#
# Usage:
#   DURATION=300 CONC=64 RESULT_DIR=/workspace/results/mori-smoke bash agentx_mori.sh
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

export MOE_A2A_BACKEND=mori
export EP_SIZE="${EP_SIZE:-8}"            # mori is an EP a2a backend; ep1 dispatches to itself
export ENABLE_TBO="${ENABLE_TBO:-0}"      # TBO x a2a is untested here
export CHUNK_PER_RANK="${CHUNK_PER_RANK:-16384}"   # matches every arm on file
export MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.85}"  # 0.90 + out-of-budget heap = expected OOR
export SGLANG_MORI_RECV_BOUND="${SGLANG_MORI_RECV_BOUND:-1}"  # the PR's knob
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-16G}"
export CONC="${CONC:-64}"
export DURATION="${DURATION:-3600}"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/mori-c${CONC}-bound${SGLANG_MORI_RECV_BOUND}}"
exec bash "$SKILL_DIR/agentx_b200align.sh"
