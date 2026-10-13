#!/usr/bin/env bash
# Task 20 framework branch final validation (MONO_COMPARE_mi355-4.md), GPUs 4-7:
# 1) fw6 vs base same session: GSM8K 1319, NIAH 32k/64k/120k x20, TPOT 8k/128k c1/c4 x3
# 2) AgentX TP4 c1/c4 on fw6 (compare with mi355-4_task18reb = snap_v41mono_rebased_1013, K2A=1)
set -uo pipefail
S=/workspace/claude-skills/dsv41/scripts
FW=${FW:-/sgl-workspace/snap_fw_f6_1013}
BASE_ENV="MONO_ARG=--enable-mono-kernel,SGLANG_MONO_DSV41_WHOLE=1,SGLANG_MONO_DSV41_INDEXER=1,SGLANG_MONO_DSV41_EDGE=1,SGLANG_MONO_DSV41_DRAFT=1,SGLANG_MONO_DSV41_K2A=1"
TAGP=fwfinal NQ=1319 REPS=3 ISLS="8192 131072" PHASES="gsm ab" \
  VARIANTS="fw6:$FW:MONO_ARG=--layer-backend=mono base:/sgl-workspace/snap_v41mono_rebased_1013:$BASE_ENV" bash $S/fw_ab.sh
echo "[$(date +%F' '%T)] AgentX fw6 TP4 c1 c4"
INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423 \
  SRC=$FW/python LANE=tp4 CONCS="1 4" TAGP=mi355-4_fwfinal \
  SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --layer-backend mono" bash $S/sweep_ci37423_opt.sh
python3 -I $S/agentx_table.py mi355-4_fwfinal mi355-4_task18reb 2>&1 | tail -12
echo "FINAL CHAIN DONE $(date +%F' '%T)"
