#!/usr/bin/env bash
# V41 whole-layer validation (MONO_KERNEL_mi355-4.md P2b phase 1): default main KV layout (V41 on gfx950) with the
# compressed cache forced to fp8 (V41_FP4 reads not ported yet), queued behind WAIT_PID on the same GPUs.
#   WAIT_PID=<pid> GPUS=4,5,6,7 bash mono_v41_chain.sh
# 1. GSM8K 1319 parallel 8 (tag mi355-4_monov41_gsm8k_p8); 2. TPOT sim-AL whole-layer mono (tag mi355-4_tpotv41whole).
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
GPUS=${GPUS:-4,5,6,7}
cleanup() {
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3" {print $1}'); do
    tr "\0" "\n" 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p" 2>/dev/null
  done
  sleep 10
}
if [ -n "${WAIT_PID:-}" ]; then
  echo "[$(date +%T)] waiting for PID $WAIT_PID"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "[$(date +%T)] PID $WAIT_PID exited"; sleep 30; cleanup
fi
unset SGLANG_DSV4_KV_LAYOUT
export SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp8 SGLANG_MONO_DSV41_WHOLE=1
echo "[$(date +%T)] 1 V41 whole-layer GSM8K"
TAG=mi355-4_monov41_gsm8k_p8 MONO=1 NQ=1319 PARALLEL=8 timeout 3600 bash "$D/mono_smoke.sh"; echo "rc=$?"; cleanup
echo "[$(date +%T)] 2 V41 whole-layer TPOT"
CONFIGS="mono" TAGP=mi355-4_tpotv41whole OSL=1024 timeout 3600 bash "$D/mono_tpot_ab.sh"; echo "rc=$?"; cleanup
echo "[$(date +%T)] chain done"
