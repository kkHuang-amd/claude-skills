#!/usr/bin/env bash
# All 40 layers whole on PRODUCTION KV layouts (main V41 + compressed V41_FP4, no layout env overrides;
# MONO_KERNEL_mi355-4.md task 3), queued behind WAIT_PID on the same GPUs.
#   WAIT_PID=<pid> GPUS=4,5,6,7 bash mono_prod_chain.sh
# 1. GSM8K 1319 parallel 8 (tag mi355-4_monoprod_gsm8k_p8); 2. TPOT sim-AL (tag mi355-4_tpotprod).
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
unset SGLANG_DSV4_KV_LAYOUT SGLANG_DSV4_COMPRESSED_KV_LAYOUT
export SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1
echo "[$(date +%T)] 1 production-layout all-40 GSM8K"
TAG=mi355-4_monoprod_gsm8k_p8 MONO=1 NQ=1319 PARALLEL=8 timeout 3600 bash "$D/mono_smoke.sh"; echo "rc=$?"
grep -o "layers=[0-9]* ({[^}]*})" /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/mi355-4_monoprod_gsm8k_p8/server.log | head -1
cleanup
echo "[$(date +%T)] 2 production-layout all-40 TPOT"
CONFIGS="mono" TAGP=mi355-4_tpotprod OSL=1024 timeout 3600 bash "$D/mono_tpot_ab.sh"; echo "rc=$?"; cleanup
echo "[$(date +%T)] chain done"
