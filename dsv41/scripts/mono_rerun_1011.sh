#!/usr/bin/env bash
# 2026-10-11 reruns after the back.py fix (MONO_KERNEL_mi355-4.md): all-40 whole (V4), V41 whole, then the whole-layer
# AgentX c4/c8 from a FROZEN sglang snapshot (sweeps reload code at every server start).
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
M=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono
SNAP=/sgl-workspace/sglang_snap_1011
rm -rf "$SNAP" && mkdir -p "$SNAP" && cp -a /sgl-workspace/sglang/python "$SNAP/" && echo "snapshot $SNAP"
WAIT_PID= bash "$D/mono_all40_chain.sh"
WAIT_PID= bash "$D/mono_v41_chain.sh"
echo "[$(date +%T)] AgentX whole c4/c8 from snapshot"
export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423
SRC=$SNAP/python SGLANG_DSV4_KV_LAYOUT=v4 SGLANG_MONO_DSV41_WHOLE=1 LANE=tp4 CONCS="4 8" TAGP=mi355-4_monowhole TSUF=_r2 \
  SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel" bash "$D/sweep_ci37423_opt.sh"
echo "[$(date +%T)] rerun done"
