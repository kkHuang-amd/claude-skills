#!/usr/bin/env bash
# Run a command inside the unpacked rocm/atom-dev rootfs (atom_image_fetch.py) via chroot, with the image ENV.
#   ROOT=/shared_nfs/kk/atom_image/rootfs GPUS=4,5 bash atom_chroot_run.sh <cmd...>
#   MODE=server CONC=1 GPUS=4,5 PROBE=1 LOG=/shared_nfs/kk/atom_run/x.log bash atom_chroot_run.sh   (TP2 recipe server)
# PROBE=1 prepends scripts/atom_pathprobe (sitecustomize counters -> $ATOM_PATHPROBE_OUT).
set -eo pipefail
ROOT=${ROOT:-/shared_nfs/kk/atom_image/rootfs}
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for m in dev dev/pts dev/shm proc sys tmp shared_nfs workspace; do
  mkdir -p "$ROOT/$m"
  mountpoint -q "$ROOT/$m" || mount --rbind "/$m" "$ROOT/$m"
done
cp -f /etc/resolv.conf "$ROOT/etc/resolv.conf" 2>/dev/null || true
MODEL_PATH=${MODEL_PATH:-/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash}
ENVV=(PATH=/root/.cargo/bin:/usr/local/go/bin:/opt/venv/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  LD_LIBRARY_PATH=/opt/rocm/lib HOME=/root TERM=dumb GPU_ARCH_LIST="gfx942;gfx950" PYTORCH_ROCM_ARCH="gfx942;gfx950"
  HSA_ENABLE_IPC_MODE_LEGACY=1 MOONCAKE_DISABLE_HIP_DMABUF=1 HIP_VISIBLE_DEVICES=${GPUS:-4,5}
  OMP_NUM_THREADS=4 ATOM_NUMA_BIND=0 ATOM_DISABLE_MMAP=true AITER_LOG_LEVEL=WARNING HF_HUB_OFFLINE=1
  ATOM_PATHPROBE_OUT=${ATOM_PATHPROBE_OUT:-/tmp/atom_pathprobe.jsonl})
[ "${PROBE:-0}" = 1 ] && ENVV+=(PYTHONPATH=$D/atom_pathprobe)
if [ "${MODE:-cmd}" = server ]; then
  CONC=${CONC:-1}; CAP="[1,2,3,4,5,6,7,8,16,32,48,64,128]"
  [ "$CONC" -eq 32 ] && CAP="[$(seq -s, 1 32),48,64,128]"
  set -- python3 -u -m atom.entrypoints.openai_server --model "$MODEL_PATH" --trust-remote-code \
    --host 0.0.0.0 --server-port ${PORT:-8000} --tensor-parallel-size 2 \
    --kv_cache_dtype bf16 --index-cache-dtype fp8 --gpu-memory-utilization 0.9 --max-num-seqs 128 \
    --max-num-batched-tokens 16384 --attn-prefill-chunk-size 16384 --enable_prefix_caching --block-size 16 \
    --state-checkpoint-interval-tokens 8192 --level 3 --cudagraph-mode FULL --cudagraph-capture-sizes "$CAP" \
    --method dspark --num-speculative-tokens 5 --spec-decode-acceptance-length ${ACC_LEN:-3.51} \
    --tool-call-parser dsml_v41 ${ATOM_EXTRA_ARGS:-}
fi
exec chroot "$ROOT" /usr/bin/env -i "${ENVV[@]}" "$@"
