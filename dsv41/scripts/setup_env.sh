#!/usr/bin/env bash
# Idempotent: bring a fresh rocm/sglang gfx950 container to the dsv41 working state (see RUNBOOK.md).
#   SKIP_KERNEL=1 to skip the sgl-kernel rebuild.  Prints one line per step; exits non-zero on failure.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
SRC=${SRC:-/sgl-workspace/sglang-dsv41}; AITER=${AITER:-/sgl-workspace/aiter}
AITER_PIN=acf8fdf9307431ece8ee275971c41cb3d1a7020b; SGL_PIN=e2e824dc58
SP=$(python3 -c 'import site;print(site.getsitepackages()[0])'); LOGD=/shared_nfs/kk/dsv41; mkdir -p "$LOGD"
say(){ echo "[setup] $*"; }

# 1. sglang branch
if [ ! -d "$SRC/.git" ]; then git clone -b dsv41-amd-main https://github.com/kevin-mii/sglang.git "$SRC"; fi
git -C "$SRC" remote get-url upstream >/dev/null 2>&1 || git -C "$SRC" remote add upstream https://github.com/sgl-project/sglang.git
say "sglang HEAD=$(git -C "$SRC" rev-parse --short HEAD) (runbook pin $SGL_PIN)"

# 2. aiter pin + patches (git apply is skipped when already applied)
[ "$(git -C "$AITER" rev-parse HEAD)" = "$AITER_PIN" ] || { say "WARN aiter HEAD != $AITER_PIN"; }
for p in "$HERE"/patches/aiter_*.patch; do
  if git -C "$AITER" apply --reverse --check "$p" 2>/dev/null; then say "aiter patch already applied: $(basename "$p")"
  else git -C "$AITER" apply "$p" && say "aiter patch applied: $(basename "$p")"; fi
done

# 3. sgl-kernel from branch (needs sort_output in deepseek_v4_topk_transform_512)
has_sort(){ (cd /tmp && python3 -c "import torch,sgl_kernel,sys;sys.exit(0 if 'sort_output' in str(torch.ops.sgl_kernel.deepseek_v4_topk_transform_512.default._schema) else 1)") 2>/dev/null; }
if has_sort; then say "sgl-kernel already has sort_output"
elif [ "${SKIP_KERNEL:-0}" = 1 ]; then say "WARN sgl-kernel lacks sort_output (SKIP_KERNEL=1)"
else
  [ -d "$LOGD/sgl_kernel_backup_orig" ] || cp -a "$SP/sgl_kernel" "$LOGD/sgl_kernel_backup_orig"
  rm -rf /tmp/aot_build && cp -a "$SRC/python/sglang/kernels/aot" /tmp/aot_build
  (cd /tmp/aot_build && rm -f pyproject.toml && mv pyproject_rocm.toml pyproject.toml && \
   AMDGPU_TARGET=gfx950 MAX_JOBS=${MAX_JOBS:-128} python3 setup_rocm.py install) > "$LOGD/sgl_kernel_build.log" 2>&1
  EGG=$(ls -d "$SP"/sglang_kernel-*-linux-x86_64.egg | tail -1)
  rm -rf "$SP/sgl_kernel" && cp -a "$EGG/sgl_kernel" "$SP/sgl_kernel"   # the egg is shadowed otherwise
  has_sort && say "sgl-kernel rebuilt OK" || { say "FAIL sgl-kernel rebuild, see $LOGD/sgl_kernel_build.log"; exit 1; }
fi

# 4. model + datasets
test -f /shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash/config.json && say "model OK" || say "WARN model missing"
say "done"
