#!/usr/bin/env bash
# Per-GPU VRAM used/free over time, for the OOR-abort diagnosis (report §5).
#
# Why this exists: gpu_metrics.csv is 41 MB of clocks/power/activity with NO
# memory column, so none of the six arms on file can say whether free VRAM
# decays monotonically or dies to a single spike. That is the one fact that
# decides between lowering mem-frac, lowering CHUNK_PER_RANK, and reporting a
# leak upstream.
#
# usage: vram_sampler.sh <outfile> [interval_s]
#
# Interval is 15 s, not the 30 s the plan called for: the leading hypothesis is
# a *transient* indexer logits buffer (total_tokens x max_seq_len x 4 B, 13.7 GB
# at ISL p90), so the sampler has to have some chance of landing inside a spike.
# 8 rows every 15 s for 1.5 h is ~110 KB -- the cost is nothing either way.
set -u
out="${1:?usage: vram_sampler.sh <outfile> [interval_s]}"
iv="${2:-15}"

echo "epoch,iso,gpu,used_gb,free_gb,total_gb" >"$out"
while true; do
    rocm-smi --showmeminfo vram 2>/dev/null | awk \
        -v ts="$(date +%s)" -v iso="$(date -Is)" '
        function gpuid(s) { gsub(/[^0-9]/, "", s); return s + 0 }
        /VRAM Total Memory \(B\)/      { tot[gpuid($1)]  = $NF }
        /VRAM Total Used Memory \(B\)/ { used[gpuid($1)] = $NF }
        END {
            for (g = 0; g < 16; g++) {
                if (!(g in tot)) continue
                printf "%s,%s,%d,%.2f,%.2f,%.2f\n", ts, iso, g,
                    used[g] / 1073741824,
                    (tot[g] - used[g]) / 1073741824,
                    tot[g] / 1073741824
            }
        }' >>"$out"
    sleep "$iv"
done
