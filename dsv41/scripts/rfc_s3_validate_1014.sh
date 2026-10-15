#!/usr/bin/env bash
# RFC #43949 refactor step 3 (hooks; attach in precompile_kernels_after_loading, every capture width built up front):
# 1. smoke of snap_rfc_s3_1014 (GSM8K 200 + NIAH 3 x 4, startup log checked); stops the chain on failure
# 2. A/B snap_fw_f6_1013 (base) vs snap_rfc_s3_1014: GSM8K 1319 + NIAH + TPOT x3 at 8k / 128k, c1 / c4
#   setsid bash rfc_s3_validate_1014.sh > /shared_nfs/kk/rfc_s3_validate_1014.log 2>&1 < /dev/null &
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
NEW=/sgl-workspace/snap_rfc_s3_1014 BASE=/sgl-workspace/snap_fw_f6_1013
say() { echo "[$(date +%F' '%T)] $*"; }

say "1. smoke $NEW"
VARIANTS="rfc:$NEW:MONO_ARG=--layer-backend=mono" PHASES=gsm NQ=200 NIAH_N=4 TAGP=rfcs3smoke bash "$D/fw_ab.sh"
T=mi355-4_rfcs3smoke_rfc_gsm
grep -E 'attached in|mono kernels on|refused|MonoRefused' "$OUT/$T/server.log" 2>/dev/null | cut -c1-260 | head -6
acc=$(grep -oE 'Accuracy: [0-9.]+' "/shared_nfs/kk/results/DeepSeek-V4.1-Flash/gsm8k_$T.log" 2>/dev/null | tail -1 | awk '{print $2}')
say "smoke GSM8K accuracy=${acc:-none}"
if [ -z "$acc" ] || awk "BEGIN{exit !($acc < 0.85)}"; then say "SMOKE FAILED: chain stops"; exit 1; fi
if ! grep -q 'attached in' "$OUT/$T/server.log"; then say "SMOKE FAILED: no attach line"; exit 1; fi

say "2. A/B base vs rfc"
VARIANTS="base:$BASE:MONO_ARG=--layer-backend=mono rfc:$NEW:MONO_ARG=--layer-backend=mono" PHASES="gsm ab" NQ=1319 REPS=3 \
  TAGP=rfcs3 bash "$D/fw_ab.sh"
say "RFC S3 DONE"
