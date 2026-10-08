#!/usr/bin/env bash
# wait for the smoke driver to exit, then run the full 20-point sweep
while kill -0 1916333 2>/dev/null; do sleep 10; done
echo "=== smoke finished, starting full sweep $(date +%H:%M:%S) ==="
exec bash /shared_nfs/kk/results/DeepSeek-V4-Pro/dsv4-fixed_seq_len/sweep_mi355x.sh
