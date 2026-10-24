#!/usr/bin/env bash
# Default-tile baseline, server path (NOBIAS=1), 3 reps per key. One GPU per queue, queues in parallel.
R=${R:-/shared_nfs/kk/results/proxy_12l/moe_csv_prefill_h13-17_1024}/def
S=/workspace/claude-skills/proxy-moe/scripts/gfx1250_moe_tile_sweep.sh
unset AITER_CONFIG_GROUPED_FMOE
q() {  # gpu E TOPK INTER tokens...
  g=$1 e=$2 k=$3 i=$4; shift 4
  for t in "$@"; do for r in 1 2 3; do
    GPU=$g STAGE=0 E=$e TOPK=$k INTER=$i bash $S $R/$([ $e = 512 ] && echo ep1 || echo ep8)_r$r $t > /dev/null 2>&1
  done; done
}
q 0 512 16 512 8192 &
q 1 512 16 512 4096 1024 &
q 2 512 16 512 2048 &
q 3 64 2 4096 32 64 &
wait
