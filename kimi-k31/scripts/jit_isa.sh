#!/usr/bin/env bash
# Device ISA + resource summary for the newest sglang JIT build of <module>.
# Usage: jit_isa.sh <module dir suffix, e.g. sparse_mla_prefill_gfx950_2 (name + template args)> [out.s]
set -euo pipefail
M=$1
OUT=${2:-/tmp/${M}.s}
D=$(ls -td /root/.cache/sglang/jit/gfx950/sgl_kernel_jit_${M}/build-*/deps-* 2>/dev/null | head -1 || true)
test -n "$D" || { echo "no build for $M"; exit 1; }
NVCC=$(sed -n 's/^nvcc *= *//p' "$D/build.ninja")
FLAGS=$(sed -n 's/^cudaflags *= *//p' "$D/build.ninja")
cd "$D"
# shellcheck disable=SC2086
eval "$NVCC $FLAGS --cuda-device-only -S cuda.cu -o $OUT" 2>&1 | rg -v 'warning|^\s*$|\^|~' | head -5 || true
rg -o '\.(vgpr_count|agpr_count|sgpr_count|vgpr_spill_count|sgpr_spill_count|group_segment_fixed_size|private_segment_fixed_size):\s+\d+' "$OUT" | sort -u
rg -o '; Occupancy: \d+' "$OUT" | sort -u
echo "isa: $OUT"
