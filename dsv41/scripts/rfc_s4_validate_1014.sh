#!/usr/bin/env bash
# RFC #43949 refactor step 4 (--enable-monokernel, RFC refusal list, requires enforced, no lazy width builds):
# 1. compile-only ISA dump vs isa_task26  2. smoke (GSM8K 200 + NIAH 3 x 4)  3. refusal: + --enable-two-batch-overlap
#   setsid bash rfc_s4_validate_1014.sh > /shared_nfs/kk/rfc_s4_validate_1014.log 2>&1 < /dev/null &
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=/shared_nfs/kk/results/DeepSeek-V4.1-Flash
OUT=$R/agentx
NEW=/sgl-workspace/snap_rfc_s4_1014
say() { echo "[$(date +%F' '%T)] $*"; }

if [ "${ONLY_RECHECK:-0}" = 1 ]; then
  say "3b. refusal (routed-expert capture)"
  VARIANTS="rex:$NEW:MONO_ARG=--enable-monokernel,EXTRA_ADD=--enable-return-routed-experts" PHASES=gsm NQ=20 \
    TAGP=rfcs4refuse bash "$D/fw_ab.sh"
  grep -h -E -A4 'enable-monokernel refused' "$OUT/mi355-4_rfcs4refuse_rex_gsm/server.log" \
    "$OUT/mi355-4_rfcs4refuse_rex_gsm.nohup" 2>/dev/null | head -8 | cut -c1-220
  say "4. GSM8K 1319 + NIAH 60"
  VARIANTS="rfc4:$NEW:MONO_ARG=--enable-monokernel" PHASES=gsm NQ=1319 TAGP=rfcs4full bash "$D/fw_ab.sh"
  say "RFC S4 RECHECK DONE"
  exit 0
fi
say "1. ISA"
(cd /tmp && HIP_VISIBLE_DEVICES=4 PYTHONPATH=$NEW/python python3 "$D/mono_isa_dump.py" $R/isa_rfc_s4 > /tmp/isa_rfc_s4.log 2>&1)
say "isa errors=$(grep -c ERROR /tmp/isa_rfc_s4.log) diff_files=$(diff -r -q $R/isa_task26 $R/isa_rfc_s4 | wc -l)"

say "2. smoke $NEW"
VARIANTS="rfc4:$NEW:MONO_ARG=--enable-monokernel" PHASES=gsm NQ=200 NIAH_N=4 TAGP=rfcs4smoke bash "$D/fw_ab.sh"
T=mi355-4_rfcs4smoke_rfc4_gsm
grep -E 'attached in|accepted the configuration|refused' "$OUT/$T/server.log" 2>/dev/null | cut -c1-260 | head -4

say "3. refusal (two-batch overlap)"
VARIANTS="tbo:$NEW:MONO_ARG=--enable-monokernel,EXTRA_ADD=--enable-two-batch-overlap" PHASES=gsm NQ=20 TAGP=rfcs4refuse \
  bash "$D/fw_ab.sh"
grep -E -A3 'enable-monokernel refused' "$OUT/mi355-4_rfcs4refuse_tbo_gsm/server.log" "$OUT/mi355-4_rfcs4refuse_tbo_gsm.nohup" \
  2>/dev/null | head -8 | cut -c1-220
say "RFC S4 DONE"
