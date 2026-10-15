#!/usr/bin/env bash
# c1 +2% after C: prototype of the one-line HIP hook (snap_rfc_ch_1014 = snap_rfc_c_1014 + radix filter
# k2i_hook.skips_schedule) vs c. Waits for the c1 profile chain (PGID $1) to end. GSM8K 1319 + NIAH + TPOT x3 8k/128k c1/c4.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
say() { echo "[$(date +%F' '%T)] $*"; }
while [ -n "${1:-}" ] && kill -0 -- -"$1" 2>/dev/null; do sleep 20; done
say "start c vs ch"
VARIANTS="c:/sgl-workspace/snap_rfc_c_1014:MONO_ARG=--enable-monokernel ch:/sgl-workspace/snap_rfc_ch_1014:MONO_ARG=--enable-monokernel" \
  PHASES="gsm ab" NQ=1319 REPS=3 TAGP=rfch bash "$D/fw_ab.sh"
T=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/mi355-4_rfch_ch
say "PROTO skip lines: gsm $(grep -c 'PROTO skips_schedule' ${T}_gsm/server.log) ab $(grep -c 'PROTO skips_schedule' ${T}_ab/server.log)"
say "RFC HOOK AB DONE"
