#!/usr/bin/env bash
# Decode TPOT A/B for the mono kernels (MONO_KERNEL_mi355-4.md): one TP4 server per config (CONC=8 recipe args,
# EVAL_ONLY=false (default) = simulated golden acceptance so TPOT tracks step time, as in CI; true = real draft),
# decode_load_profile.py at CONCS on each, no profiler.
#   CONFIGS="prod nomono mono" CONCS=1,2,4,8 ISL=4096 OSL=512 GPUS=4,5,6,7 PORT=8890 TAGP=mi355-4_tpot bash mono_tpot_ab.sh
#   prod = recipe (--enforce-shared-experts-fusion); nomono = fusion off; mono = fusion off + --enable-mono-kernel
# Out: agentx/<TAGP>_<cfg>/server.log + load.out; prints the [clean cN] lines per config.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INFMAX_CONTAINER_WORKSPACE=${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX-ci37423/inferencex-e2e}
export AGENTX_RUNTIME_ROOT=${AGENTX_RUNTIME_ROOT:-/workspace/agentx-runtime-ci37423}
CONFIGS=${CONFIGS:-prod nomono mono} CONCS=${CONCS:-1,2,4,8} ISL=${ISL:-4096} OSL=${OSL:-512}
GPUS=${GPUS:-4,5,6,7} PORT=${PORT:-8890} TAGP=${TAGP:-mi355-4_tpot_$(date +%m%d_%H%M)}
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
for cfg in $CONFIGS; do
  case $cfg in
    prod) EA="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" ;;
    nomono) EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion" ;;
    mono) EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel" ;;
    *) echo "bad config $cfg"; continue ;;
  esac
  T=${TAGP}_$cfg
  TAG=$T TP=4 GPUS=$GPUS PORT=$PORT CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 MEM_FRACTION_STATIC=0.70 \
    SERVER_ONLY=1 EVAL_ONLY=${EVAL_ONLY:-false} EXTRA_ARGS="$EA" setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$T.nohup" 2>&1 < /dev/null &
  LP=$!
  for _ in $(seq 1 180); do
    grep -q 'SERVER_ONLY: ready' "$OUT/$T.nohup" && break
    kill -0 $LP 2>/dev/null || break
    sleep 10
  done
  if grep -q 'SERVER_ONLY: ready' "$OUT/$T.nohup"; then
    sleep 15
    python3 -I "$D/decode_load_profile.py" --port "$PORT" --conc "$CONCS" --isl "$ISL" --osl "$OSL" > "$OUT/$T/load.out" 2>&1
    echo "== $cfg (load exit=$?)"; grep '^\[clean' "$OUT/$T/load.out" | cut -c1-200
  else
    echo "== $cfg: server not ready"; grep -E 'Traceback|Error|refused' "$OUT/$T.nohup" | tail -4 | cut -c1-200
  fi
  kill -- -$LP 2>/dev/null; sleep 5
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do
    tr "\0" "\n" 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
  sleep 10
done
echo done
