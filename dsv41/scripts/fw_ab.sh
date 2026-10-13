#!/usr/bin/env bash
# Task 20 framework branch (MONO_COMPARE_mi355-4.md): A/B of frozen snapshots, GPUs 4-7, TP4. Derived from task17_ab.sh;
# the mono switch and its env are per variant (no global mono env), so a variant with none runs the defaults.
#   VARIANTS="base:/sgl-workspace/snap_v41mono_rebased_1013:MONO_ARG=--enable-mono-kernel,SGLANG_MONO_DSV41_WHOLE=1,...
#             fw:/sgl-workspace/snap_fw_f1_1013:MONO_ARG=--layer-backend=mono" PHASES="gsm ab" bash fw_ab.sh
# A variant is name:snapshot:ENV=V,ENV=V (EXTRA_ADD=... appends server args, e.g. EXTRA_ADD=--disable-cuda-graph).
# gsm: EVAL_ONLY server -> GSM8K NQ (1319, parallel 8) + NIAH (i4_niah_eval.py, NIAH_LENGTHS, NIAH_N).
# ab:  sim-AL server -> REPS x decode_load_profile AB_CONCS at each ISL in ISLS (OSL 1024).
# Out: agentx/mi355-4_<TAGP>_<variant>_{gsm,ab}/ ; markers -> stdout.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export INFMAX_CONTAINER_WORKSPACE=${INFMAX_CONTAINER_WORKSPACE:-/workspace/InferenceX-ci37423/inferencex-e2e}
export AGENTX_RUNTIME_ROOT=${AGENTX_RUNTIME_ROOT:-/workspace/agentx-runtime-ci37423}
REPS=${REPS:-3} PHASES=${PHASES:-"gsm ab"} ISLS=${ISLS:-"8192 131072"} GPUS=4,5,6,7 PORT=8890
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
export QR_QUANT=INT8 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 \
       SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1
EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion"

stop() {  # $1 = launcher PGID
  kill -- -"$1" 2>/dev/null; sleep 5
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3"{print $1}'); do
    tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
  sleep 10
}
start() {  # $1 tag, $2 src, $3 EVAL_ONLY, $4 extra server args -> sets LP; returns 0 when ready
  SRC=$2/python TAG=$1 TP=4 GPUS=$GPUS PORT=$PORT CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 \
    MEM_FRACTION_STATIC=0.70 SERVER_ONLY=1 EVAL_ONLY=$3 EXTRA_ARGS="$EA ${MONO_ARG:?variant must set MONO_ARG} $4" \
    setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$1.nohup" 2>&1 < /dev/null &
  LP=$!
  for _ in $(seq 1 180); do
    grep -q 'SERVER_ONLY: ready' "$OUT/$1.nohup" && { sleep 15; return 0; }
    kill -0 $LP 2>/dev/null || break; sleep 10
  done
  echo "[$1] server not ready"; grep -E 'Traceback|Error|refused' "$OUT/$1.nohup" "$OUT/$1/server.log" 2>/dev/null | tail -4 | cut -c1-200
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
summary() {  # $1 tag
  grep -cE 'Traceback' "$OUT/$1/server.log" | sed "s/^/[$1] tracebacks: /"
  grep -E '\[K2A_CHECK\]' "$OUT/$1/server.log" 2>/dev/null | tail -2 | sed "s/^/[$1] /" | cut -c1-220
  grep -E 'mono kernels on|accepted the configuration' "$OUT/$1/server.log" 2>/dev/null | head -2 | sed "s/^/[$1] /" | cut -c1-220
}

for v in $VARIANTS; do
  IFS=: read -r name snap envs <<< "$v"
  extra=""
  ( # the variant's env stays in this subshell
    for kv in ${envs//,/ }; do
      case $kv in EXTRA_ADD=*) extra="$extra ${kv#EXTRA_ADD=}";; *) export "$kv";; esac
    done
    for ph in $PHASES; do
      T=mi355-4_${TAGP:-fw}_${name}_$ph
      echo "[$(date +%F' '%T)] start $T ($snap; $envs)"
      if [ "$ph" = gsm ]; then
        if start "$T" "$snap" true "$extra"; then
          TAG=$T PORT=$PORT NQ=${NQ:-1319} PARALLEL=8 SRC=$snap/python timeout 3600 bash "$D/run_gsm8k.sh" | cut -c1-200
          grep -E 'Accuracy|Invalid' "/shared_nfs/kk/results/DeepSeek-V4.1-Flash/gsm8k_$T.log" | tail -3 | sed "s/^/[$T] /" | cut -c1-200
          timeout 3600 python3 -I "$D/i4_niah_eval.py" --port $PORT --lengths "${NIAH_LENGTHS:-32000,64000,120000}" \
            --n "${NIAH_N:-20}" --conc 4 --out "$OUT/$T/niah.jsonl" > "$OUT/$T/niah.out" 2>&1
          tail -n 4 "$OUT/$T/niah.out" | sed "s/^/[$T niah] /" | cut -c1-200
          summary "$T"
        fi
      else
        if start "$T" "$snap" false "$extra"; then
          ok=1
          for isl in $ISLS; do
            for r in $(seq 1 "$REPS"); do
              load "$T" "isl${isl}_r$r" 1500 --conc "${AB_CONCS:-1,4}" --isl "$isl" --osl 1024 --seed "$r" ${LOAD_EXTRA:-} \
                || { ok=0; break 2; }
            done
          done
          summary "$T"
        fi
      fi
      stop "$LP"
    done
  )
done
echo "AB DONE $(date +%F' '%T)"
