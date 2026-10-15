#!/usr/bin/env bash
# RFC refactor A (step 5's TPOT cost: the clock / STOP read once every CHECK_EVERY = 32 polls), snap_rfc_a_1014:
# 1. fault injection (rank 1, attn, from the first pair, 10 s deadline): every rank stops
# 2. A/B s4 (no guard) / c (tip 2d998c2496) / a: GSM8K 1319 + NIAH + TPOT x3 at 8k / 128k, c1 / c4
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
NEW=/sgl-workspace/snap_rfc_a_1014
say() { echo "[$(date +%F' '%T)] $*"; }
say "1. fault injection"
VARIANTS="fault:$NEW:MONO_ARG=--enable-monokernel,SGLANG_TEST_MONO_FAULT_INJECT=1:attn:0,EXTRA_ADD=--watchdog-timeout=20" \
  PHASES=gsm NQ=20 TAGP=rfcafault bash "$D/fw_ab.sh"
T=mi355-4_rfcafault_fault_gsm
say "memory faults: $(grep -c 'Memory access fault' $OUT/$T/server.log) (nil: $(grep -c 'on address (nil)' $OUT/$T/server.log)); leftover: $(ps -eo comm | grep -c '^sglang::')"
say "2. A/B s4 / c / a"
VARIANTS="s4:/sgl-workspace/snap_rfc_s4_1014:MONO_ARG=--enable-monokernel c:/sgl-workspace/snap_rfc_c_1014:MONO_ARG=--enable-monokernel a:$NEW:MONO_ARG=--enable-monokernel" \
  PHASES="gsm ab" NQ=1319 REPS=3 TAGP=rfca bash "$D/fw_ab.sh"
say "RFC A DONE"
