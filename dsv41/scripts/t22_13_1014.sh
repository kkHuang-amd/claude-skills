#!/usr/bin/env bash
# Tasks 22 + 13 (MONO_KERNEL_mi355-4.md) on snap_t22_1014, after WAIT_PGID (the S3 validation chain) and free GPUs 4-7.
#   22: check mode, eager, 600 checked steps (GSM8K 16 + NIAH 1 x 64k) -> [mono-check] lines with relL2 / cos per part
#   13: GSM8K 1319 x3 mono vs module, interleaved (NIAH minimal)
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ts() { date '+%F %T'; }
if [ -n "${WAIT_PGID:-}" ]; then
  while pgrep -g "$WAIT_PGID" > /dev/null; do sleep 60; done
  echo "[$(ts)] chain $WAIT_PGID done"
fi
for _ in $(seq 1 30); do
  used=$(rocm-smi --showmemuse 2>/dev/null | grep -E 'GPU\[[4-7]\].*Allocated' | awk '{s+=$NF} END{print s+0}')
  [ "$used" -le 4 ] && break; sleep 60
done
SNAP=/sgl-workspace/snap_t22_1014
echo "[$(ts)] 22. check mode"
TAGP=t22chk NQ=16 NIAH_N=1 NIAH_LENGTHS=64000 PHASES=gsm \
  VARIANTS="chk:$SNAP:MONO_ARG=--layer-backend=mono,SGLANG_MONO_CHECK=1,SGLANG_MONO_CHECK_STEPS=600,EXTRA_ADD=--disable-cuda-graph" \
  bash "$D/fw_ab.sh"
echo "[$(ts)] 13. GSM8K x3 mono vs module"
for r in 1 2 3; do
  TAGP=t13r$r NQ=1319 NIAH_N=1 NIAH_LENGTHS=32000 PHASES=gsm \
    VARIANTS="mono:$SNAP:MONO_ARG=--layer-backend=mono module:$SNAP:MONO_ARG=--layer-backend=module" \
    bash "$D/fw_ab.sh"
done
echo "T22_13 DONE $(ts)"
