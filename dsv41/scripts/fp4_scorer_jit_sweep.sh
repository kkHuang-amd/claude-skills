#!/usr/bin/env bash
# P0.4 driver: fp4_scorer_jit_sweep.py for MODE A and B, each first with EMPTY FlyDSL/Triton caches (cold node), then
# again with the same caches (process restart, disk cache warm). GPU ${GPU:-0}, aiter ${AITER:-/sgl-workspace/aiter-6145}.
# Output: /shared_nfs/kk/dsv41/fp4_index_port/p04_jit_sweep.log (one line per case).
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
L=/shared_nfs/kk/dsv41/fp4_index_port/p04_jit_sweep.log; : > "$L"
for M in A B; do
    C=/tmp/p04_cache_$M; rm -rf "$C"; mkdir -p "$C/flydsl" "$C/triton"
    for pass in cold warm; do
        echo "== MODE $M pass $pass" >> "$L"
        (cd /tmp && MODE=$M PYTHONPATH=${AITER:-/sgl-workspace/aiter-6145} HIP_VISIBLE_DEVICES=${GPU:-0} \
            FLYDSL_RUNTIME_CACHE_DIR=$C/flydsl TRITON_CACHE_DIR=$C/triton \
            timeout 3000 python3 "$D/fp4_scorer_jit_sweep.py") >> "$L" 2>&1
        echo "exit=$?" >> "$L"
    done
done
