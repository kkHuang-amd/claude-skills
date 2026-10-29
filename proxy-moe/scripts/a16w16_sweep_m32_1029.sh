#!/bin/bash
# M=32 bucket (70k/300 c=32 decode) for the proxy TP8 a16w16 shapes; run after a16w16_sweep_1029.sh. Resumable.
R=${R:-/shared_nfs/kk/results/proxy_12l/a16w16_sweep_h13-17_1029}
cd /sgl-workspace/aiter/aiter/ops/triton/utils/_triton/tuning
for job in "32:2304:16384" "32:16384:2048" "32:25256:16384"; do
  IFS=: read M N K <<< "$job"
  python3 sweep_configs.py gemm_a16w16 --N $N --K $K --M $M --gpu 0 1 2 3 4 5 6 7 --runs-dir $R > $R/drv_M${M}_N${N}.log 2>&1
  echo "M=$M N=$N K=$K exit=$? $(grep -oE 'baseline \(installed[^:]*: [0-9.]+ us' $R/drv_M${M}_N${N}.log | tail -1)"
done
