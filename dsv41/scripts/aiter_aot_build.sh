#!/usr/bin/env bash
# Full AOT build + editable install of /sgl-workspace/aiter for gfx950 (incl. FlyDSL run_aot), as in
# sglang docker/rocm.Dockerfile (BUILD_AITER_ALL=1). Prepare the source tree first (checkout + patches; see the
# calling doc's Environment section) and move old aiter/jit/*.so + aiter/jit/build out of the way.
# Knobs: AITER_DIR (default /sgl-workspace/aiter), GPU_ARCHS (default gfx950), LOG (default below).
# Output: full log in $LOG; prints only "exit=" and error markers.
set -u
AITER_DIR=${AITER_DIR:-/sgl-workspace/aiter}
GPU_ARCHS=${GPU_ARCHS:-gfx950}
LOG=${LOG:-/shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/aiter_aot_build_$(hostname)_$(date +%m%d_%H%M).log}
mkdir -p "$(dirname "$LOG")"
cd "$AITER_DIR" || exit 1
export AITER_USE_SYSTEM_TRITON=1
{
  echo "[$(date +%F' '%T)] aiter $(git rev-parse --short HEAD) GPU_ARCHS=$GPU_ARCHS"
  PREBUILD_KERNELS=1 GPU_ARCHS=$GPU_ARCHS python setup.py build_ext --inplace \
    && GPU_ARCHS=$GPU_ARCHS pip install --no-build-isolation --config-settings editable_mode=compat -e .
  rc=$?
  echo "[$(date +%F' '%T)] done rc=$rc"
  exit $rc
} > "$LOG" 2>&1
rc=$?
echo "exit=$rc log=$LOG"
grep -E "error:|Error|FAILED|Traceback" "$LOG" | tail -20 | cut -c1-200
exit $rc
