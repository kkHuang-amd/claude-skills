#!/usr/bin/env bash
# Per sweep dir: config count, failures, default baseline median, and the 3 fastest passing configs.
R=${R:-/shared_nfs/kk/results/proxy_12l/moe_csv_prefill_h13-17_1024}
for d in $R/sweep/*/; do f=$d/summary.txt; [ -f $f ] || continue
  echo "$(basename $d): n=$(wc -l < $f) fail=$(grep -vc 'pass=True' $f)  best: $(grep 'pass=True' $f | sort -n | head -3 | awk '{printf "%s %s | ", $1, $2}')"
done | sed 's/_rc=0//g' | cut -c1-260
