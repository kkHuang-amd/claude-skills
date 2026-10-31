# Kimi-K3.1 runtime env (gfx950). Usage: source /workspace/claude-skills/kimi-k31/env.sh
# /etc/bash.bashrc prepends /sgl-workspace/aiter (old golden tree) to PYTHONPATH,
# which shadows the editable aiter-k31 install. Override it here.
export PYTHONPATH=/sgl-workspace/mori:/sgl-workspace/aiter-k31
export GPU_ARCHS=gfx950 PYTORCH_ROCM_ARCH=gfx950 GPU_ARCH_LIST=gfx950
export AITER_USE_SYSTEM_TRITON=1
export K31_MODEL=/shared_nfs/Kimi-K3.1
export K31_OUT=/shared_nfs/kimi-k31
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
