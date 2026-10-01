#!/usr/bin/env bash
# One GPU lane of a parallel AgentX sweep: runs points back to back on GPUS/PORT, waiting for this lane's port to
# free between points (agentx_series.sh waits on node-wide VRAM, so two series lanes would block each other).
#   GPUS=4,5 PORT=8888 POINTS="tag:conc:pdi:chunk:mem ..." bash agentx_lane.sh   (other env, e.g. SRC/EXTRA_ARGS, inherited)
# Progress -> /shared_nfs/kk/dsv41/agentx/lane_<PORT>.txt ; each point -> <tag>/ and <tag>.nohup
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); OUT=/shared_nfs/kk/dsv41/agentx; S=$OUT/lane_${PORT}.txt
say(){ echo "[$(date +%F' '%T)] $*" >> "$S"; }
port_busy(){ ss -ltn | grep -q ":${PORT} "; }
for pt in $POINTS; do
  IFS=: read -r tag conc pdi chunk mem <<< "$pt"
  for i in $(seq 1 120); do port_busy || break; sleep 5; done
  sleep 60
  say "START $tag conc=$conc pdi=$pdi chunk=$chunk mem=$mem gpus=$GPUS"
  TAG=$tag CONC=$conc PREFILL_DECODE_INTERVAL=$pdi CHUNKED_PREFILL_SIZE=$chunk MEM_FRACTION_STATIC=$mem \
    bash "$D/agentx_colleague_run.sh" > "$OUT/$tag.nohup" 2>&1
  rc=$?
  f=$(ls "$OUT/$tag"/*_agentic_c${conc}.json 2>/dev/null | head -1)
  say "END $tag rc=$rc result=${f:+yes}${f:-MISSING}"
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"
  done
done
say "LANE DONE"
