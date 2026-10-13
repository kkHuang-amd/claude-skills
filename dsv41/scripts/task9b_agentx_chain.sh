#!/usr/bin/env bash
# Task 9b follow-up (MONO_KERNEL_mi355-4.md): from the frozen snapshot with the nosync patch (/sgl-workspace/snap_task9b_1012)
#   1. GPU-only c1 128k profile (remaining host gap), guarded by a timeout (c1 /start_profile hangs seen with CPU activity)
#   2. AgentX TP4 c1 (sweep_ci37423_opt.sh, production layouts, all-40 mono) -> agentx/mi355-4_nosync_tp4_c1/
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SNAP=${SNAP:-/sgl-workspace/snap_task9b_1012} GPUS=4,5,6,7
kill_lane() {
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3"{print $1}'); do
    tr '\0' '\n' 2>/dev/null < /proc/$p/environ | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
  sleep 15
}
echo "[$(date +%F' '%T)] profile start"
SNAP=$SNAP PROF_ACT=GPU TSUF=_nosync_gpuonly POINTS="mono:131072" timeout 1800 bash "$D/task9_profile_chain.sh"
rc=$?; echo "[$(date +%F' '%T)] profile rc=$rc"; [ $rc = 124 ] && { echo "PROFILE HANG"; kill_lane; }
echo "[$(date +%F' '%T)] agentx c1 start"
export SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1
INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423 \
  SRC=$SNAP/python LANE=tp4 CONCS="1" TAGP=mi355-4_nosync \
  SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel" bash "$D/sweep_ci37423_opt.sh"
echo "[$(date +%F' '%T)] agentx rc=$?"
kill_lane
echo "CHAIN9B DONE"
