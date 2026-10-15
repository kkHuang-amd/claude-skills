#!/usr/bin/env bash
# AgentX TP4 then TP2 (GPUs 4-7 / 4,5), c1 / c4 / c8, --enable-monokernel, no mono env:
#   s4 = snap_rfc_s4_1014 (no in-kernel guard, old srt schedule skip)
#   h  = snap_rfc_h_1014  (020f4c2584: guard + mono.skip_metadata hook)
# Interleaved per point (s4, h) so a partial night still compares fairly. Tags mi355-4_n1014{s4,h}_tp<TP>_c<C>.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
A=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423 \
       SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --enable-monokernel"
ts() { date '+%F %T'; }
for lane in tp4:4,5,6,7:8890 tp2:4,5:8888; do
  IFS=: read -r L G P <<< "$lane"
  for c in 1 4 8; do
    for v in s4 h; do
      echo "[$(ts)] start $v $L c$c"
      SRC=/sgl-workspace/snap_rfc_${v}_1014/python TAGP=mi355-4_n1014$v LANE=$L GPUS=$G PORT=$P CONCS="$c" \
        bash "$D/sweep_ci37423_opt.sh" > /dev/null 2>&1
      grep -E "END mi355-4_n1014${v}_${L}_c$c " "$A/lane_$P.txt" | tail -1 | cut -c1-90
    done
  done
done
python3 -I "$D/agentx_table.py" mi355-4_n1014s4 mi355-4_n1014h 2>&1 | head -14 | cut -c1-220
echo "NIGHT DONE $(ts)"
