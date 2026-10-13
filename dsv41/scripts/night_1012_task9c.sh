#!/usr/bin/env bash
# 2026-10-12 night queue (MONO_KERNEL_mi355-4.md, user request): all from snap_task9c_1012, GPUs 4-7 only, one lane at a time.
#   1. TP2 mono GSM8K smoke (GPUs 4,5; gates step 3: accuracy >= 0.895)
#   2. TP4 c4 c16, then c8 rerun (TSUF=_r2)
#   3. TP2 c1 c4 c8 c16 (GPUs 4,5)
# Markers -> stdout (log to agentx/night_1012_task9c_mi355-4.log); per-point results agentx/task9c_tp{2,4}_c*/
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=/shared_nfs/kk/results/DeepSeek-V4.1-Flash
export SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1 SGLANG_MONO_DSV41_DRAFT=1 \
       SGLANG_DSPARK_SIMULATE_FOLD=1 \
       INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423 \
       SRC=/sgl-workspace/snap_task9c_1012/python TAGP=task9c \
       SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel"
ts() { date '+%F %T'; }

echo "[$(ts)] 1. TP2 smoke (GSM8K)"
MODE=smoke LANE=tp2 GPUS=4,5 PORT=8888 bash "$D/sweep_ci37423_opt.sh" 2>&1 | grep -E 'SMOKE|Accuracy|Invalid|rc=' | cut -c1-200
ACC=$(grep -oE 'Accuracy: [0-9.]+' "$R/gsm8k_task9c_tp2_smoke.log" 2>/dev/null | tail -1 | awk '{print $2}')
echo "[$(ts)] TP2 GSM8K accuracy=${ACC:-none}"

echo "[$(ts)] 2. TP4 c4 c16"
LANE=tp4 CONCS="4 16" bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
echo "[$(ts)] 2b. TP4 c8 rerun"
LANE=tp4 CONCS="8" TSUF=_r2 bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1

if [ -n "$ACC" ] && python3 -c "import sys; sys.exit(0 if float('$ACC') >= 0.895 else 1)"; then
  echo "[$(ts)] 3. TP2 c1 c4 c8 c16"
  LANE=tp2 GPUS=4,5 PORT=8888 CONCS="1 4 8 16" bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
else
  echo "[$(ts)] 3. SKIPPED: TP2 GSM8K ${ACC:-missing} < 0.895"
fi
for f in "$R"/agentx/lane_8890.txt "$R"/agentx/lane_8888.txt; do grep -E 'END task9c' "$f" 2>/dev/null | tail -8 | cut -c1-120; done
echo "NIGHT DONE $(ts)"
