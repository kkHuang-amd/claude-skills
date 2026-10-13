#!/usr/bin/env bash
# Task 8 debug: eager TP4 server (--disable-cuda-graph, real acceptance) with the draft stages on mono and
# SGLANG_MONO_DSV41_DRAFT_CHECK=1 (per-stage relative error vs the SGLang path on TP0), a short GSM8K, then stop.
#   SNAP=/sgl-workspace/snap_task8_dbg NQ=20 bash task8_draft_check.sh
# Out: agentx/mi355-4_task8_check/server.log; prints the first check lines + accept len.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423
SNAP=${SNAP:-/sgl-workspace/snap_task8_dbg} GPUS=4,5,6,7 PORT=8890 T=${TAG:-mi355-4_task8_check}
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
export QR_QUANT=INT8 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 \
       SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1 \
       SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1 SGLANG_MONO_DSV41_DRAFT=1 \
       SGLANG_MONO_DSV41_DRAFT_CHECK=1
SRC=$SNAP/python TAG=$T TP=4 GPUS=$GPUS PORT=$PORT CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 \
  MEM_FRACTION_STATIC=0.70 SERVER_ONLY=1 EVAL_ONLY=true \
  EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel --disable-cuda-graph" \
  setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$T.nohup" 2>&1 < /dev/null &
LP=$!
for _ in $(seq 1 180); do grep -q 'SERVER_ONLY: ready' "$OUT/$T.nohup" && break; kill -0 $LP 2>/dev/null || break; sleep 10; done
if grep -q 'SERVER_ONLY: ready' "$OUT/$T.nohup"; then
  TAG=$T PORT=$PORT NQ=${NQ:-20} PARALLEL=1 SRC=$SNAP/python timeout 1800 bash "$D/run_gsm8k.sh" | cut -c1-200
else
  echo "server not ready"; grep -E 'Traceback|Error' "$OUT/$T.nohup" | tail -4 | cut -c1-200
fi
SL=$OUT/$T/server.log
echo "check lines: $(grep -c 'mono draft check' "$SL")"
grep 'mono draft check' "$SL" | head -3 | sed 's/.*mono draft check/check/' | cut -c1-400
grep 'mono draft check' "$SL" | tail -2 | sed 's/.*mono draft check/check/' | cut -c1-400
grep -o 'accept len: [0-9.]*' "$SL" | awk '{s+=$3;n++} END{if(n) printf "accept len n=%d mean=%.3f\n",n,s/n}'
grep -E 'Traceback' -A12 "$SL" | head -16 | cut -c1-200
kill -- -$LP 2>/dev/null; sleep 5
for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3"{print $1}'); do
  tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
echo "CHECK DONE"
