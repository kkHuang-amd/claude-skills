#!/usr/bin/env bash
# Task 17 S3 (MONO_KERNEL_mi355-4.md): validate K1's index query on snap_s3_1014 once the task 25 chain is done.
#   1. eager check: IQ=1 + K2A_CHECK, no CUDA graphs, GSM8K 200 -> [K2A_CHECK] "K1 index q" byte counts + Jaccard
#   2. A/B IQ=0 vs IQ=1: GSM8K 1319 + NIAH, then TPOT x3 at ISL 8k / 128k (fw_ab.sh defaults)
# WAIT_PGID: a chain to wait for first (GPUs 4-7 must also be free). NIAH 2 x 64k in the check run.
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
echo "[$(ts)] GPUs 4-7 VRAM% sum=$used"
SNAP=/sgl-workspace/snap_s3_1014
M=MONO_ARG=--layer-backend=mono
echo "[$(ts)] 1. eager check"
TAGP=s3chk NQ=200 NIAH_N=2 NIAH_LENGTHS=64000 PHASES=gsm \
  VARIANTS="iqchk:$SNAP:$M,SGLANG_MONO_DSV41_K2A_IQ=1,SGLANG_MONO_DSV41_K2A_CHECK=1,EXTRA_ADD=--disable-cuda-graph" \
  bash "$D/fw_ab.sh"
echo "[$(ts)] 2. A/B IQ=0 vs IQ=1"
TAGP=s3ab NQ=1319 REPS=3 PHASES="gsm ab" \
  VARIANTS="iq0:$SNAP:$M,SGLANG_MONO_DSV41_K2A_IQ=0 iq1:$SNAP:$M,SGLANG_MONO_DSV41_K2A_IQ=1" \
  bash "$D/fw_ab.sh"
echo "S3 VALIDATE DONE $(ts)"
