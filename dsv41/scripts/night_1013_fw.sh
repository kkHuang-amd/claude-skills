#!/usr/bin/env bash
# Night 10-13 (MONO_KERNEL_mi355-4.md "AgentX night 10-13"): the 10-12 night sweep on perf/v41-mono 134919476c
# (snap_fw_f6_1013, --layer-backend mono, no mono env). First TP4 c4_r2 for fw6 and for the base (task18reb, same
# session). TP4 c1/c4 already ran as mi355-4_fwfinal; adds TP4 c8, c16, and TP2 c1/c4/c8/c16 (GPUs 4,5) gated on a TP2 GSM8K smoke >= 0.895. GPUs 4-7 only.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=/shared_nfs/kk/results/DeepSeek-V4.1-Flash
export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423 \
       SRC=/sgl-workspace/snap_fw_f6_1013/python TAGP=mi355-4_fwfinal \
       SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --layer-backend mono"
ts() { date '+%F %T'; }
echo "[$(ts)] 0. TP4 c4 rerun: fw6, then base (same session)"
LANE=tp4 CONCS="4" TSUF=_r2 bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1 SGLANG_MONO_DSV41_DRAFT=1 SGLANG_MONO_DSV41_K2A=1 \
  SRC=/sgl-workspace/snap_v41mono_rebased_1013/python TAGP=mi355-4_task18reb \
  SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-mono-kernel" \
  LANE=tp4 CONCS="4" TSUF=_r2 bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
python3 -I "$D/agentx_table.py" mi355-4_fwfinal mi355-4_task18reb 2>&1 | grep -E '^\| 4 \| 4' | cut -c1-200
echo "[$(ts)] 1. TP2 smoke (GSM8K)"
MODE=smoke LANE=tp2 GPUS=4,5 PORT=8888 bash "$D/sweep_ci37423_opt.sh" 2>&1 | grep -E 'SMOKE|Accuracy|Invalid|rc=' | cut -c1-200
ACC=$(grep -oE 'Accuracy: [0-9.]+' "$R/gsm8k_${TAGP}_tp2_smoke.log" 2>/dev/null | tail -1 | awk '{print $2}')
echo "[$(ts)] TP2 GSM8K accuracy=${ACC:-none}"
echo "[$(ts)] 2. TP4 c8 c16"
LANE=tp4 CONCS="8 16" bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
if [ -n "$ACC" ] && python3 -c "import sys; sys.exit(0 if float('$ACC') >= 0.895 else 1)"; then
  echo "[$(ts)] 3. TP2 c1 c4 c8 c16"
  LANE=tp2 GPUS=4,5 PORT=8888 CONCS="1 4 8 16" bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
else
  echo "[$(ts)] 3. SKIPPED: TP2 GSM8K ${ACC:-missing} < 0.895"
fi
for f in "$R"/agentx/lane_8890.txt "$R"/agentx/lane_8888.txt; do grep -E "END ${TAGP}" "$f" 2>/dev/null | tail -9 | cut -c1-140; done
python3 -I "$D/agentx_table.py" mi355-4_fwfinal task9c 2>&1 | head -14
echo "NIGHT DONE $(ts)"
