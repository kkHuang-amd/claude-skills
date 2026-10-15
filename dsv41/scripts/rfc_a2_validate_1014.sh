#!/usr/bin/env bash
# The formal HIP hook (mono.skip_metadata, snap_rfc_h_1014) and A2 on top of it (guard.py: STOP read once every
# PEER_EVERY = 32 polls, clock every poll, one loop; snap_rfc_a2_1014): GSM8K 1319 + NIAH + TPOT x3 8k / 128k c1 / c4,
# then s4 (no guard) TPOT as the reference. Tags mi355-4_rfca2_{h,a2,s4}_*.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
say() { echo "[$(date +%F' '%T)] $*"; }
say "1. h / a2: gsm + ab"
VARIANTS="h:/sgl-workspace/snap_rfc_h_1014:MONO_ARG=--enable-monokernel a2:/sgl-workspace/snap_rfc_a2_1014:MONO_ARG=--enable-monokernel" \
  PHASES="gsm ab" NQ=1319 REPS=3 TAGP=rfca2 bash "$D/fw_ab.sh"
say "2. s4: ab"
VARIANTS="s4:/sgl-workspace/snap_rfc_s4_1014:MONO_ARG=--enable-monokernel" PHASES=ab REPS=3 TAGP=rfca2 bash "$D/fw_ab.sh"
say "RFC A2 DONE"
