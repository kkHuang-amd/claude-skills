#!/usr/bin/env bash
# One arm of the clock-lock A/B (h13-17 1029): mb_skew_set.sh variants + stream/light, results to $OUT/<arm>.txt.
# Usage: mb_skew_ab.sh <arm>      (e.g. dpm1, lock800_1, dpm2)
cd "$(dirname "$0")"; ARM=$1; OUT=${OUT:-/shared_nfs/kk/results/proxy_12l/clklock_ab_mb_h13-17_1029}; mkdir -p "$OUT"
source <(grep -E '^(NCCL_|RCCL_)' /workspace/proxy_work/allon_tp8.env | sed 's/^/export /')
run() { echo "### $1"; CUDA_VISIBLE_DEVICES=$2 torchrun --nproc-per-node $3 --master-port 29611 mb_skew.py --gpus $2 "${@:4}" 2>&1 | grep -E 'load=|compute|corr|AR own|Error|Traceback' | cut -c1-320; }
{
  echo "arm=$ARM $(date +%T)"
  for f in /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input; do echo "$f $(( $(cat "$f") / 1000000 )) MHz"; done
  run "8r gemm"              0,1,2,3,4,5,6,7 8 --load gemm --size 2048 --iters 1500
  run "4r 1-per-socket gemm" 0,2,4,6         4 --load gemm --size 2048 --iters 1500
  run "2r cross socket gemm" 0,2             2 --load gemm --size 2048 --iters 1500
  run "8r stream"            0,1,2,3,4,5,6,7 8 --load stream --iters 1500
  run "8r light"             0,1,2,3,4,5,6,7 8 --load light --iters 1500
  echo "done $(date +%T)"
} > "$OUT/$ARM.txt" 2>&1
