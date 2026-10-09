#!/usr/bin/env bash
# OPT_SWEEP_1008.md: wait for pid $1 (the TP4 sweep waiter) to exit, rerun the listed TP4 points alone (CONCS, TSUF),
# then append their vs-CI rows to the doc (the cmp watcher exits after tp4_c16).
#   setsid nohup bash after_tp4_rerun.sh <pid> &      env: CONCS="1" TSUF=_r2 QR_QUANT=INT8 TAGP=opt1008
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); R=/shared_nfs/kk/results/DeepSeek-V4.1-Flash
S=$R/agentx/chain_opt1008.txt; CONCS=${CONCS:-1}; TSUF=${TSUF:-_r2}; TAGP=${TAGP:-opt1008}; H=$(hostname -s); H=${H#crsuse2-}
while kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "[$(date +%F' '%T)] tp4 rerun start CONCS=[$CONCS] TSUF=$TSUF (alone)" >> "$S"
CONCS="$CONCS" TSUF=$TSUF QR_QUANT=${QR_QUANT:-INT8} TAGP=$TAGP LANE=tp4 bash "$D/sweep_ci37423_opt.sh" > $R/agentx/${TAGP}_tp4_rerun${TSUF}.out 2>&1
for c in $CONCS; do
  t=${TAGP}_tp4_c$c$TSUF; f=$(ls $R/agentx/$t/*_agentic_c$c.json 2>/dev/null | head -1)
  rc=$(command grep -aoE "END $t rc=[0-9]+" $R/agentx/lane_8890.txt | tail -1 | command grep -oE 'rc=[0-9]+')
  if [ -n "$f" ]; then python3 "$D/agentx_agg_table.py" --ref $R/ref_ci_37423021942/agg_bmk.json --row "$H $(date +%m-%d) $t $rc" "$f" >> "$D/../OPT_SWEEP_1008.md"
  else echo "| $H $(date +%m-%d) $t $rc | NO RESULT | | | | | | |" >> "$D/../OPT_SWEEP_1008.md"; fi
done
echo "[$(date +%F' '%T)] tp4 rerun done" >> "$S"
