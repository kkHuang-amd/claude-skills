#!/usr/bin/env bash
# Task 27a (MONO_KERNEL_mi355-4.md): A/B snap_t22_1014 (base) vs snap_t27a_1014 (ATOM #2545 non-token-parallel ports),
# both IQ=0: GSM8K 1319 + NIAH, TPOT x3 at ISL 8k / 128k, c1 / c4. Waits for WAIT_PGID and free GPUs 4-7.
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
M=MONO_ARG=--layer-backend=mono
TAGP=t27a NQ=1319 REPS=3 PHASES="gsm ab" \
  VARIANTS="base:/sgl-workspace/snap_t22_1014:$M p27:/sgl-workspace/snap_t27a_1014:$M" bash "$D/fw_ab.sh"
echo "T27A DONE $(ts)"
