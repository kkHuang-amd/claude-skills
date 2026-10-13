#!/usr/bin/env bash
# Task 9 follow-up: same as task9_profile_chain.sh mono:131072 but from /sgl-workspace/snap_task9_hipwait, where
# overlap_utils.py uses publish_ready.wait() on HIP instead of the #26672 synchronize() workaround (host bubble test).
# Waits for the chain log's CHAIN DONE first.  Out: profile_tp4/mi355-4_task9_mono_d128k_c1_hipwait/
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CL=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_tp4/task9_chain_mi355-4.log
until grep -q 'CHAIN DONE' "$CL"; do sleep 20; done
sleep 20
export SRC=/sgl-workspace/snap_task9_hipwait/python GPUS=4,5,6,7 PORT=8890 CONC=1 OSL=1024 STEPS=40 ISL=131072
export QR_QUANT=INT8 ROCM_QUICK_REDUCE_QUANTIZATION=INT8 SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 \
       SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1
export SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1
echo "[$(date +%F' '%T)] start hipwait"
TAG=mi355-4_task9_mono_d128k_c1_hipwait EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel" \
  bash "$D/tp4_decode_profile.sh"
echo "[$(date +%F' '%T)] HIPWAIT DONE rc=$?"
