#!/usr/bin/env bash
# Task 9b (MONO_KERNEL_mi355-4.md): host-bubble patch A/B without profiler, GPUs 4-7, all-40 mono, production layouts.
#   VARIANTS="orig:/sgl-workspace/snap_task9_1012 nosync:/sgl-workspace/snap_task9_nosync" REPS=3 SOAK=nosync GSM=nosync \
#     bash task9b_ab.sh
# Per variant: one sim-AL server (CONC 8 recipe args), REPS x decode_load_profile c1,4 at ISL 131072 OSL 1024; for the
# SOAK variant also c1,2,4,8 ISL 32768 OSL 4096 (hang check). Then GSM8K 1319 (parallel 8, real acceptance) on GSM.
# Every load has a timeout; on timeout: py-spy dumps of the schedulers, "HANG" marker, server killed.
# Out: agentx/mi355-4_task9b_<variant>{,_gsm}/ ; markers -> stdout (log it to profile_tp4/task9b_ab_mi355-4.log)
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INFMAX_CONTAINER_WORKSPACE=${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX-ci37423/inferencex-e2e}
export AGENTX_RUNTIME_ROOT=${AGENTX_RUNTIME_ROOT:-/workspace/agentx-runtime-ci37423}
VARIANTS=${VARIANTS:-"orig:/sgl-workspace/snap_task9_1012 nosync:/sgl-workspace/snap_task9_nosync"}
REPS=${REPS:-3} SOAK=${SOAK:-nosync} GSM=${GSM:-nosync} GPUS=4,5,6,7 PORT=8890
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
export QR_QUANT=INT8 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 \
       SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1 \
       SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1
EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel"

stop() {  # $1 = launcher PGID
  kill -- -"$1" 2>/dev/null; sleep 5
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3"{print $1}'); do
    tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
  sleep 10
}
start() {  # $1 tag, $2 src, $3 EVAL_ONLY -> sets LP; returns 0 when ready
  SRC=$2/python TAG=$1 TP=4 GPUS=$GPUS PORT=$PORT CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 \
    MEM_FRACTION_STATIC=0.70 SERVER_ONLY=1 EVAL_ONLY=$3 EXTRA_ARGS="$EA" \
    setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$1.nohup" 2>&1 < /dev/null &
  LP=$!
  for _ in $(seq 1 180); do
    grep -q 'SERVER_ONLY: ready' "$OUT/$1.nohup" && { sleep 15; return 0; }
    kill -0 $LP 2>/dev/null || break; sleep 10
  done
  echo "[$1] server not ready"; grep -E 'Traceback|Error' "$OUT/$1.nohup" "$OUT/$1/server.log" 2>/dev/null | tail -4 | cut -c1-200
  return 1
}
load() {  # $1 tag, $2 label, $3 timeout s, rest = decode_load_profile args; returns 1 on hang
  local t=$1 l=$2 to=$3; shift 3
  timeout "$to" python3 -I "$D/decode_load_profile.py" --port $PORT "$@" > "$OUT/$t/load_$l.out" 2>&1
  local rc=$?
  if [ $rc = 124 ]; then
    echo "[$t $l] HANG (timeout ${to}s)"
    for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::schedul/{print $1}'); do
      timeout 30 py-spy dump --pid "$p" > "$OUT/$t/pyspy_${l}_$p.txt" 2>&1; done
    return 1
  fi
  grep '^\[clean' "$OUT/$t/load_$l.out" | sed "s/^/[$t $l] /" | cut -c1-200
  return 0
}

for v in $VARIANTS; do
  name=${v%%:*} snap=${v#*:} T=mi355-4_task9b_$name
  echo "[$(date +%F' '%T)] start $T ($snap)"
  if start "$T" "$snap" false; then
    ok=1
    for r in $(seq 1 "$REPS"); do load "$T" "r$r" 900 --conc ${AB_CONCS:-1,4} --isl 131072 --osl 1024 --seed "$r" ${LOAD_EXTRA:-} || { ok=0; break; }; done
    if [ $ok = 1 ] && [ "$name" = "$SOAK" ]; then load "$T" soak 2400 --conc 1,2,4,8 --isl 32768 --osl 4096 --seed 7 || ok=0; fi
    grep -cE 'Traceback' "$OUT/$T/server.log" | sed "s/^/[$T] tracebacks: /"
  fi
  stop "$LP"
  if [ "$name" = "$GSM" ]; then
    G=${T}_gsm; echo "[$(date +%F' '%T)] start $G"
    if start "$G" "$snap" true; then
      TAG=$G PORT=$PORT NQ=1319 PARALLEL=8 SRC=$snap/python timeout 3600 bash "$D/run_gsm8k.sh" | cut -c1-200
      grep -E 'Accuracy|Invalid' "/shared_nfs/kk/results/DeepSeek-V4.1-Flash/gsm8k_$G.log" | tail -3 | sed "s/^/[$G] /" | cut -c1-200
    fi
    stop "$LP"
  fi
done
echo "AB DONE $(date +%F' '%T)"
