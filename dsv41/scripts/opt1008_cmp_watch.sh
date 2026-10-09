#!/usr/bin/env bash
# OPT_SWEEP_1008.md: on every "END <tag> rc=.. result=.." in the AgentX lane files, append a vs-CI row to the doc.
# Polls every 30 s by line number (a tail -F Monitor missed lane lines on NFS). Exits after tp4_c16.
#   setsid nohup bash opt1008_cmp_watch.sh &
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); DOC=$D/../OPT_SWEEP_1008.md
A=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx; REF=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_ci_37423021942/agg_bmk.json
H=$(hostname -s); H=${H#crsuse2-}
declare -A seen; for L in "$A/lane_8888.txt" "$A/lane_8890.txt"; do seen[$L]=$(wc -l < "$L" 2>/dev/null || echo 0); done
while sleep 30; do for L in "$A/lane_8888.txt" "$A/lane_8890.txt"; do
  n=$(wc -l < "$L" 2>/dev/null || echo 0); [ "$n" -gt "${seen[$L]}" ] || continue
  new=$(sed -n "$(( ${seen[$L]} + 1 )),${n}p" "$L"); seen[$L]=$n
  while read -r line; do
  [[ "$line" == *" END "* ]] || continue
  tag=$(sed -E 's/.* END ([^ ]+) .*/\1/' <<< "$line"); rc=$(grep -oE 'rc=[0-9]+' <<< "$line")
  f=$(grep -oE 'result=yes\S+' <<< "$line" | sed 's/^result=yes//')
  qr=$(grep -aoE '^ROCM_QUICK_REDUCE_QUANTIZATION=\S+' "$A/$tag.nohup" 2>/dev/null | head -1 | cut -d= -f2)
  label="$H $(date +%m-%d) $tag ${rc}${qr:+ QR=$qr}"
  if [ -n "$f" ] && [ -f "$f" ]; then
    python3 "$D/agentx_agg_table.py" --ref "$REF" --row "$label" "$f" >> "$DOC" 2>/dev/null || echo "| $label | compare failed | | | | | | |" >> "$DOC"
  else echo "| $label | NO RESULT | | | | | | |" >> "$DOC"; fi
  [[ "$tag" == *_tp4_c16 ]] && exit 0
  done <<< "$new"
done; done
