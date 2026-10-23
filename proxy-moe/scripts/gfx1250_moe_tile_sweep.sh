#!/usr/bin/env bash
# Sweep gfx1250 grouped MoE TDM tiles on one GPU (test_flydsl_grouped_gemm.py, hipgraph, end-to-end us).
# Usage: gfx1250_moe_tile_sweep.sh <out-dir> [tokens] [a4w4|a8w4]
#   STAGE=1 (default): sweep stage1 (gate-up) tiles, stage2 pinned to M2/N2/K2/NB2.
#   STAGE=2: sweep stage2 (down) tiles, stage1 pinned to M1/N1/K1/NB1.
#   STAGE=0: run once with no tile env (kernel defaults).
# NOBIAS=1 (default) passes --no-bias = the SERVER path (bias None -> fused stage1 act+MX quant epilogue,
#   which requires (tile_n / n_warp) % 64 == 0). NOBIAS=0 runs the biased, unfused path.
# Shape env: E/TOPK/INTER. Defaults = one EP8 rank (64 local experts, top-2 = top-16/512, inter 4096).
#   EP1 rank: E=512 TOPK=16 INTER=512.
set -u
OUT=$1; T=${2:-64}; FMT=${3:-a4w4}; mkdir -p "$OUT"; cd /sgl-workspace/aiter
export HIP_VISIBLE_DEVICES=${GPU:-0}
E=${E:-64}; TOPK=${TOPK:-2}; INTER=${INTER:-4096}; STAGE=${STAGE:-1}
run() {  # $1=tag; tile env already exported by caller
  timeout 600 python3 op_tests/flydsl_tests/test_flydsl_grouped_gemm.py --no-check-aot-cache \
    --data-init uniform --scale-init auto --data-format $FMT --experts $E --topk $TOPK \
    --model-dim 16384 --inter-dim $INTER --tokens $T --act silu $( [ "${NOBIAS:-1}" = 1 ] && echo --no-bias ) > "$OUT/$1.log" 2>&1
  rc=$?
  us=$(grep -oE 'end-to-end us = [0-9.]+' "$OUT/$1.log" | head -1 | awk '{print $NF}')
  ok=$(grep -oE '\| (True|False) +\|' "$OUT/$1.log" | head -1 | tr -d '| ')
  echo "${us:-NA} $1 rc=$rc pass=${ok:-NA}" | tee -a "$OUT/summary.txt"
}
if [ "$STAGE" = 0 ]; then run t${T}_default; exit 0; fi
for m in ${MS:-16 32 64}; do for n in ${NS:-128 256}; do for k in ${KS:-128 256}; do for nb in ${NBS:-2 3 4}; do for w in ${WS:-4}; do
  if [ "$STAGE" = 1 ]; then
    export AITER_TDM_TILE_M=$m AITER_TDM_TILE_N=$n AITER_TDM_TILE_K=$k AITER_TDM_NUM_BUFFERS=$nb AITER_TDM_N_WARP=$w
    export AITER_TDM_TILE_M2=${M2:-64} AITER_TDM_TILE_N2=${N2:-256} AITER_TDM_TILE_K2=${K2:-128} AITER_TDM_NUM_BUFFERS2=${NB2:-3}
  else
    export AITER_TDM_TILE_M=${M1:-64} AITER_TDM_TILE_N=${N1:-256} AITER_TDM_TILE_K=${K1:-128} AITER_TDM_NUM_BUFFERS=${NB1:-3}
    export AITER_TDM_TILE_M2=$m AITER_TDM_TILE_N2=$n AITER_TDM_TILE_K2=$k AITER_TDM_NUM_BUFFERS2=$nb AITER_TDM_N_WARP2=$w
  fi
  run s${STAGE}_t${T}_m${m}_n${n}_k${k}_b${nb}_w${w}
done; done; done; done; done
