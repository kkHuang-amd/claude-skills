#!/usr/bin/env bash
# Stage1 / stage2 tile sweeps, server path (NOBIAS=1), single runs; the other stage pinned to defaults.
R=${R:-/shared_nfs/kk/results/proxy_12l/moe_csv_prefill_h13-17_1024}/sweep
S=/workspace/claude-skills/proxy-moe/scripts/gfx1250_moe_tile_sweep.sh
unset AITER_CONFIG_GROUPED_FMOE
export NS="128 256" KS="128 256" NBS="2 3 4" WS="2 4"
ep1() { GPU=$1 E=512 TOPK=16 INTER=512 MS="32 64 128" STAGE=$3 bash $S $R/ep1_t$2_s$3 $2 > /dev/null 2>&1; }
ep8() { GPU=$1 E=64 TOPK=2 INTER=4096 MS="16 32 64" STAGE=$3 bash $S $R/ep8_t$2_s$3 $2 > /dev/null 2>&1; }
(ep1 0 8192 1; ep1 0 8192 2) &
(ep1 1 4096 1; ep1 1 4096 2) &
(ep1 2 2048 1; ep1 2 2048 2; ep1 2 1024 1) &
(ep8 3 32 1; ep8 3 32 2; ep8 3 64 1; ep8 3 64 2; ep1 3 1024 2) &
wait
