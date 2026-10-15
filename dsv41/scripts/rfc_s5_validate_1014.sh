#!/usr/bin/env bash
# RFC #43949 refactor step 5 (fail-stop: bounded waits in K1 / K2, stop of every rank, fault injection):
# 1. compile-only ISA dump (no build errors)
# 2. fault injection: rank 1 stalls its "attn" waits from the first pair, --watchdog-timeout=20 (10 s deadline):
#    every rank must stop (GPU fault), no hang
# 3. A/B snap_rfc_s4_1014 vs snap_rfc_s5_1014: GSM8K 1319 + NIAH + TPOT x3 at 8k / 128k, c1 / c4
#   setsid bash rfc_s5_validate_1014.sh > /shared_nfs/kk/rfc_s5_validate_1014.log 2>&1 < /dev/null &
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=/shared_nfs/kk/results/DeepSeek-V4.1-Flash
OUT=$R/agentx
OLD=/sgl-workspace/snap_rfc_s4_1014 NEW=/sgl-workspace/snap_rfc_s5_1014
say() { echo "[$(date +%F' '%T)] $*"; }

say "1. ISA"
(cd /tmp && HIP_VISIBLE_DEVICES=4 PYTHONPATH=$NEW/python python3 "$D/mono_isa_dump.py" $R/isa_rfc_s5 > /tmp/isa_rfc_s5.log 2>&1)
say "isa builds=$(grep -c . $R/isa_rfc_s5/index.txt) errors=$(grep -c ERROR /tmp/isa_rfc_s5.log)"

say "2. fault injection"
t0=$(date +%s)
VARIANTS="fault:$NEW:MONO_ARG=--enable-monokernel,SGLANG_TEST_MONO_FAULT_INJECT=1:attn:0,EXTRA_ADD=--watchdog-timeout=20" \
  PHASES=gsm NQ=20 TAGP=rfcs5fault bash "$D/fw_ab.sh"
T=mi355-4_rfcs5fault_fault_gsm
say "fault run took $(( $(date +%s) - t0 )) s"
grep -E 'attached in' "$OUT/$T/server.log" 2>/dev/null | cut -c1-160 | head -2
grep -E -i 'memory access fault|HSA_STATUS|page not present|Memory access|core dumped|Aborted|exited with|died' \
  "$OUT/$T/server.log" "$OUT/$T.nohup" 2>/dev/null | cut -c1-220 | head -12
say "leftover GPU 4-7 processes: $(ps -eo pid,comm | awk '$2 ~ /^sglang::/' | wc -l)"

say "3. A/B s4 vs s5"
VARIANTS="s4:$OLD:MONO_ARG=--enable-monokernel s5:$NEW:MONO_ARG=--enable-monokernel" PHASES="gsm ab" NQ=1319 REPS=3 \
  TAGP=rfcs5 bash "$D/fw_ab.sh"
say "RFC S5 DONE"
