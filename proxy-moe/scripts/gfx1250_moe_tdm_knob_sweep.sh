#!/usr/bin/env bash
# One-factor-at-a-time sweep of the grouped TDM MoE CSV-only knobs (tdm_b_th, next_stage_prefetch,
# persistent_workers, cluster_m/n) on top of a base tile row. Per-kernel gemm1/gemm2 us (--scenario kernel),
# server path (--no-bias). One single-row CSV per variant via AITER_CONFIG_GROUPED_FMOE.
# Usage: gfx1250_moe_tdm_knob_sweep.sh <out> <tokens> <E> <TOPK> <INTER> "<base tile cols>"
#   base tile cols = tile_m,tile_n,tile_k,num_buffers,n_warp,tile_m2,tile_n2,tile_k2,num_buffer_stage2,n_warp2
#   VARIANTS env overrides the list ("col=val" items, "base" = no extra column).
set -u
OUT=$1; T=$2; E=$3; K=$4; I=$5; BASE=$6; mkdir -p "$OUT"; cd /sgl-workspace/aiter
HDR="token,model_dim,inter_dim,expert,topk,tile_m,tile_n,tile_k,num_buffers,n_warp,tile_m2,tile_n2,tile_k2,num_buffer_stage2,n_warp2"
VARIANTS=${VARIANTS:-"base tdm_b_th=1 tdm_b_th=2 tdm_b_th=3 tdm_b_th=4 tdm_b_th=5 next_stage_prefetch=1 persistent_workers=96 persistent_workers=192 persistent_workers=384 cluster_n=2 cluster_m=2"}
for v in $VARIANTS; do
  csv="$OUT/$v.csv"
  if [ "$v" = base ]; then
    printf '%s\n%s,16384,%s,%s,%s,%s\n' "$HDR" "$T" "$I" "$E" "$K" "$BASE" > "$csv"
  else
    printf '%s,%s\n%s,16384,%s,%s,%s,%s,%s\n' "$HDR" "${v%%=*}" "$T" "$I" "$E" "$K" "$BASE" "${v#*=}" > "$csv"
  fi
  AITER_CONFIG_GROUPED_FMOE="$csv" timeout 900 python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --scenario kernel \
    --no-check-aot-cache --data-init uniform --scale-init auto --data-format a4w4 --experts $E --topk $K \
    --model-dim 16384 --inter-dim $I --tokens $T --act silu --no-bias > "$OUT/$v.log" 2>&1
  rc=$?
  miss=$(grep -c 'no grouped CSV config matched' "$OUT/$v.log")
  echo "$v rc=$rc miss=$miss $(grep -oE 'gemm[12]: us = [0-9.]+' "$OUT/$v.log" | tr '\n' ' ')" | tee -a "$OUT/summary.txt"
done
