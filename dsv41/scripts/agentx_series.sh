#!/usr/bin/env bash
# Run AgentX points one after another on the same GPUs (clean comparison), waiting for VRAM to drain between runs.
#   CONCS="4 16 64" TP=2 EP_SIZE=1 GPUS=0,1 DURATION=3600 PREFIX=tp2 bash agentx_series.sh
#   SCRIPT=agentx_dsv41_mi355x_sglang.sh (default) | agentx_dsv41_b200port_mi355x.sh ; extra env (OPUS, PDI, ...) is inherited
# Each point -> /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/<PREFIX>_c<CONC>/ ; progress lines -> /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/series.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx; S=$OUT/series.txt
say(){ echo "[$(date +%F' '%T)] $*" >> "$S"; }
#   RUNS="64:16 64:4 ..." -> per-run conc:prefill-decode-interval (exports PREFILL_DECODE_INTERVAL and PDI; tag gets _pdiN)
for item in ${RUNS:-${CONCS:-4 16 64}}; do
  c=${item%%:*}; p=""; [ "$item" != "$c" ] && p=${item#*:}
  for i in $(seq 1 120); do u=$(rocm-smi --showmeminfo vram 2>/dev/null | grep Used | awk '{s+=$NF} END{print int(s/1e9)}'); [ "$u" -lt 10 ] && break; sleep 5; done
  tag=${PREFIX:-tp${TP:-2}ep${EP_SIZE:-1}}_c$c${p:+_pdi$p}; say "START $tag (vram ${u}G)"
  ( [ -n "$p" ] && export PREFILL_DECODE_INTERVAL=$p PDI=$p
    TAG=$tag CONC=$c TP=${TP:-2} EP_SIZE=${EP_SIZE:-1} GPUS=${GPUS:-0,1} DURATION=${DURATION:-3600} bash "$D/${SCRIPT:-agentx_dsv41_mi355x_sglang.sh}" > "$OUT/$tag.driver.log" 2>&1 )
  say "END $tag rc=$? $(ls "$OUT/$tag"/*.json 2>/dev/null | head -1)"
done
say "SERIES DONE"
