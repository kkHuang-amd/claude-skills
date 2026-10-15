#!/usr/bin/env bash
# Task 27b (TOKEN_PAR_PORT_PLAN_mi355-4.md): bitwise check of the token-parallel K2 against the regular K2, one frozen
# snapshot per step. SGLANG_MONO_DSV41_TOK_PAR_CHECK=1, no CUDA graphs: every eager K2 at M >= 12, M % TP == 0 runs
# token parallel first on the same inputs; the runner logs "mono tok_par check" (MISMATCH lines + per-200 summaries).
# Each check: TP4 (GPUs 4-7) then TP2 (GPUs 4,5), GSM8K 200 (parallel 8: M = 6 .. 48 target rows, 5 .. 40 draft rows)
# + NIAH 1 x 16k. Steps in order, so a mismatch is pinned to the first step that shows it.
#   CHECKS="s2:/sgl-workspace/snap_t27b_s2_1014 s3:/sgl-workspace/snap_t27b_s3_1014" (default)
# WAIT_PGID: a chain to wait for first (GPUs 4-7 must also be free).
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
CHECKS=${CHECKS:-"s2:/sgl-workspace/snap_t27b_s2_1014 s3:/sgl-workspace/snap_t27b_s3_1014"}
LOGS=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
for chk in $CHECKS; do
  step=${chk%%:*} snap=${chk#*:}
  V="tokchk:$snap:MONO_ARG=--layer-backend=mono,SGLANG_MONO_DSV41_TOK_PAR_CHECK=1,EXTRA_ADD=--disable-cuda-graph"
  for tp in 4 2; do
    gpus=$([ $tp = 4 ] && echo 4,5,6,7 || echo 4,5)
    tag=t27b${step}tp$tp
    echo "[$(ts)] $step TP$tp check ($snap)"
    FW_TP=$tp FW_GPUS=$gpus TAGP=$tag NQ=200 NIAH_N=1 NIAH_LENGTHS=16000 PHASES=gsm VARIANTS="$V" bash "$D/fw_ab.sh"
    L=$LOGS/mi355-4_${tag}_tokchk_gsm/server.log
    echo "[$step TP$tp] mismatch lines: $(grep -c 'tok_par check.*MISMATCH' "$L" 2>/dev/null)"
    grep -E 'tok_par check rank [0-9]+: ' "$L" 2>/dev/null | tail -"$tp" | sed "s/^/[$step TP$tp] /" | cut -c1-200
    grep -E 'tok_par check.*MISMATCH' "$L" 2>/dev/null | head -3 | sed "s/^/[$step TP$tp] /" | cut -c1-260
  done
done
echo "T27B CHECK DONE $(ts)"
