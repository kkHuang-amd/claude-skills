#!/usr/bin/env bash
# Run rccl_comminit_stress.py variants; each variant gets a hard timeout (a hang = exit 124).
# Usage: rccl_comminit_stress.sh <run-tag>   env: NP (4), ITERS (150), TMO (600), VARIANTS, REPEAT (1)
set -u
OUT=/shared_nfs/kk/results/proxy_12l/$1; mkdir -p "$OUT"; cd /tmp
export NCCL_CUMEM_ENABLE=1 HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 PYTHONUNBUFFERED=1
PY=/workspace/claude-skills/proxy-moe/scripts/rccl_comminit_stress.py
for v0 in ${VARIANTS:-plain busy jitter}; do for r in $(seq 1 ${REPEAT:-1}); do
  v=${v0}_r$r
  case $v0 in plain) ARGS="";; busy) ARGS="--busy-ms 20";; jitter) ARGS="--jitter-ms 50";; esac
  timeout -k 20 ${TMO:-600} torchrun --nproc-per-node ${NP:-4} --master-port $((29500 + RANDOM % 400)) $PY \
    --iters ${ITERS:-150} --out "$OUT/$v" $ARGS > "$OUT/$v.log" 2>&1
  rc=$?
  last=$(for f in "$OUT/$v"/rank*.txt; do echo "$(basename $f .txt):$(tail -1 $f)"; done | tr '\n' ' ')
  echo "$v: exit=$rc $(grep -oE 'PASS.*|WRONG RESULT.*' "$OUT/$v.log" | head -2 | tr '\n' ' ') | $last" | cut -c1-300
  pkill -9 -f '[r]ccl_comminit_stress[.]py' 2>/dev/null; sleep 5
done; done
