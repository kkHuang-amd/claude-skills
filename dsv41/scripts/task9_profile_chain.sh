#!/usr/bin/env bash
# Task 9 (MONO_KERNEL_mi355-4.md): c1 decode step profiles on GPUs 4-7 from a frozen sglang snapshot, production KV
# layouts, sim AL 3.51, sweep env (QR INT8 + opt1008 flags). Points (TAG suffix: cfg_isl):
#   mono_d128k: all 40 whole mono, ISL 131072 | base_d128k: no mono, prod flags | mono_d8k: all 40 whole, ISL 8192
#   SNAP=/sgl-workspace/snap_task9_1012 POINTS="mono:131072 base:131072 mono:8192" bash task9_profile_chain.sh
# Out: profile_tp4/mi355-4_task9_<cfg>_d<k>k_c1/ ; markers in profile_tp4/task9_chain_mi355-4.log
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SNAP=${SNAP:-/sgl-workspace/snap_task9_1012}
POINTS=${POINTS:-"mono:131072 base:131072 mono:8192"}
if [ ! -d "$SNAP/python" ]; then mkdir -p "$SNAP" && cp -a /sgl-workspace/sglang/python "$SNAP/"; fi
export SRC=$SNAP/python GPUS=4,5,6,7 PORT=8890 CONC=1 OSL=${OSL:-1024} STEPS=${STEPS:-40}
export QR_QUANT=INT8 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 \
       SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1
for p in $POINTS; do
  cfg=${p%%:*} isl=${p##*:}
  case $cfg in
    mono) EA="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel"
          ME="SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1" ;;
    base) EA="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"; ME="" ;;
    *) echo "bad cfg $cfg"; continue ;;
  esac
  T=mi355-4_task9_${cfg}_d$((isl / 1024))k_c1${TSUF:-}
  echo "[$(date +%F' '%T)] start $T SRC=$SRC"
  env $ME TAG=$T ISL=$isl EXTRA_ARGS="$EA" bash "$D/tp4_decode_profile.sh"
  echo "[$(date +%F' '%T)] end $T rc=$?"
  sleep 15
done
echo "CHAIN DONE"
