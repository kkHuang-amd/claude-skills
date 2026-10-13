#!/usr/bin/env bash
# Task 9c (MONO_KERNEL_mi355-4.md): host-side sampling trace of steady c1 decode, GPUs 4-7, all-40 mono, sim AL.
#   SNAP=/sgl-workspace/snap_task9c_1012 TAG=t9c_trace bash task9c_hosttrace.sh
# py-spy (nonblocking, --idle --threads, raw collapsed stacks) on the TP0 scheduler for DUR s while one ISL 131072
# request decodes OSL 16384; no torch profiler (it drops verify-graph kernels under the nosync patch).
# Out: agentx/mi355-4_<TAG>/{pyspy_tp0.txt,load.out,server.log}; markers -> stdout
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INFMAX_CONTAINER_WORKSPACE=${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX-ci37423/inferencex-e2e}
export AGENTX_RUNTIME_ROOT=${AGENTX_RUNTIME_ROOT:-/workspace/agentx-runtime-ci37423}
SNAP=${SNAP:-/sgl-workspace/snap_task9c_1012} TAG=mi355-4_${TAG:-t9c_trace} DUR=${DUR:-10} RATE=${RATE:-1000}
GPUS=4,5,6,7 PORT=8890 OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
export QR_QUANT=INT8 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 \
       SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1 \
       SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1 SGLANG_MONO_DSV41_DRAFT=1
EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel"

if [ "${MODE:-pyspy}" = rocwrap ]; then   # whole server under rocprofv3, kernel trace only inside the -P window
  mkdir -p "$OUT/$TAG/rocprof"
  export SERVER_CMD_PREFIX="rocprofv3 --kernel-trace -f csv -P ${ROC_P:-400:1:1} -d $OUT/$TAG/rocprof --"
fi
SRC=$SNAP/python TAG=$TAG TP=4 GPUS=$GPUS PORT=$PORT CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 \
  MEM_FRACTION_STATIC=0.70 SERVER_ONLY=1 EVAL_ONLY=false EXTRA_ARGS="$EA" \
  setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$TAG.nohup" 2>&1 < /dev/null &
LP=$!
stop() {
  kill -- -"$LP" 2>/dev/null; sleep 5
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3"{print $1}'); do
    tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
}
trap stop EXIT
ready=0
for _ in $(seq 1 180); do
  grep -q 'SERVER_ONLY: ready' "$OUT/$TAG.nohup" && { ready=1; sleep 15; break; }
  kill -0 $LP 2>/dev/null || break; sleep 10
done
[ $ready = 1 ] || { echo "[$TAG] server not ready"; exit 1; }
TP0=$(ps -eo pid,args | awk '/sglang::scheduler_TP0/ && !/awk/{print $1; exit}')
echo "[$TAG] ready, TP0 scheduler pid=$TP0"

timeout 900 python3 -I "$D/decode_load_profile.py" --port $PORT --conc 1 --isl 131072 --osl ${OSL:-16384} --seed 11 \
  > "$OUT/$TAG/load.out" 2>&1 &
LD=$!
for _ in $(seq 1 120); do grep -q '^\[prime 0\]' "$OUT/$TAG/load.out" && break; sleep 2; done
sleep 4   # clean c1 decode is running (~25 s at OSL 16384)
if [ "${MODE:-pyspy}" = rocwrap ]; then
  echo "[$TAG] rocwrap: trace window ROC_P=${ROC_P:-400:1:1} s after each process start; waiting for the load"
elif [ "${MODE:-pyspy}" = rocprof ]; then   # HSA-level kernel trace, includes graph-replayed kernels
  timeout 300 rocprofv3 --attach "$TP0" --attach-duration-msec "${ATTACH_MS:-3000}" --kernel-trace -f csv \
    -d "$OUT/$TAG/rocprof" -o tp0 > "$OUT/$TAG/rocprof.log" 2>&1
  echo "[$TAG] rocprofv3 rc=$? files $(find "$OUT/$TAG/rocprof" -name '*kernel_trace*' 2>/dev/null | wc -l)"
else
timeout $((DUR + 60)) py-spy record --pid "$TP0" --nonblocking --idle --threads --rate "$RATE" --duration "$DUR" \
  --format raw -o "$OUT/$TAG/pyspy_tp0.txt" > "$OUT/$TAG/pyspy.log" 2>&1
echo "[$TAG] py-spy rc=$? samples file $(wc -l < "$OUT/$TAG/pyspy_tp0.txt" 2>/dev/null) lines"
fi
wait $LD
grep '^\[clean' "$OUT/$TAG/load.out" | cut -c1-200
grep -c Traceback "$OUT/$TAG/server.log" | sed "s/^/[$TAG] tracebacks: /"
echo "TRACE DONE $(date +%F' '%T)"
