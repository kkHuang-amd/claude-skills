#!/usr/bin/env bash
# OPT_SWEEP_1008.md: wait for the TP2 sweep (chain_opt1008.sh pid $1) to exit, then run the TP4 sweep alone.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); S=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/chain_opt1008.txt
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date +%F' '%T)] tp2 done; tp4 sweep start (alone)" >> "$S"
QR_QUANT=${QR_QUANT:-INT8} TAGP=${TAGP:-opt1008} LANE=tp4 bash "$D/sweep_ci37423_opt.sh" > /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/opt1008_lane_tp4.out 2>&1
echo "[$(date +%F' '%T)] tp4 sweep done" >> "$S"
