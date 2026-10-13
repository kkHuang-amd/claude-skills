#!/usr/bin/env bash
# Mono kernel smoke (MONO_KERNEL_mi355-4.md): TP4 server with --enable-mono-kernel (shared-experts fusion off,
# EVAL_ONLY=true -> no simulated acceptance), check the mono runtime came up, GSM8K, stop the server.
#   TAG=mono_smoke GPUS=4,5,6,7 PORT=8890 NQ=1319 MONO=1 bash mono_smoke.sh     (MONO=0: same config, mono off = A/B)
# Uses the CI-recipe AgentX env (InferenceX-ci37423). Out: agentx/<TAG>/server.log, gsm8k_<TAG>.log; markers only.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INFMAX_CONTAINER_WORKSPACE=${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX-ci37423/inferencex-e2e}
export AGENTX_RUNTIME_ROOT=${AGENTX_RUNTIME_ROOT:-/workspace/agentx-runtime-ci37423}
TAG=${TAG:-mono_smoke_$(date +%m%d_%H%M)} GPUS=${GPUS:-4,5,6,7} PORT=${PORT:-8890} MONO=${MONO:-1}
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx; mkdir -p "$OUT"
EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion"; [ "$MONO" = 1 ] && EA="$EA --enable-mono-kernel"
TAG=$TAG TP=4 GPUS=$GPUS PORT=$PORT CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 MEM_FRACTION_STATIC=0.70 \
  SERVER_ONLY=1 EVAL_ONLY=true EXTRA_ARGS="$EA" setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$TAG.nohup" 2>&1 < /dev/null &
LP=$!
echo "server lane PGID $LP (kill -- -$LP); log $OUT/$TAG/server.log"
for _ in $(seq 1 180); do
  grep -q 'SERVER_ONLY: ready' "$OUT/$TAG.nohup" && break
  kill -0 $LP 2>/dev/null || break
  sleep 10
done
SL=$OUT/$TAG/server.log
if grep -q 'SERVER_ONLY: ready' "$OUT/$TAG.nohup"; then
  echo "ready; mono lines:"; grep -E 'mono kernels on|enable-mono-kernel|mono .*refused|MonoRefused' "$SL" | head -4 | cut -c1-220
  TAG=$TAG PORT=$PORT NQ=${NQ:-1319} SRC=/sgl-workspace/sglang/python bash "$D/run_gsm8k.sh"
  echo "captures:"; grep -oE "mono graph capture: [0-9]+ rows -> [a-z]+" "$SL" | sort -t: -k2 -n | uniq | tr "\n" ";" | cut -c1-600; echo
  echo "errors:"; grep -E "Traceback|MonoKernelError|build refused|first reached inside" "$SL" | head -5 | cut -c1-220
else
  echo "server not ready"; grep -E 'Traceback|Error|refused' "$OUT/$TAG.nohup" "$SL" 2>/dev/null | tail -8 | cut -c1-240
fi
kill -- -$LP 2>/dev/null; sleep 5
for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do
  tr "\0" "\n" 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
echo "done"
