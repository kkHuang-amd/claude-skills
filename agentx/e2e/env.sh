# Env for the inferencex-e2e layout (InferenceX >= 2026-09, srt-slurm recipes).
# Every path is overridable so the same file works on any node/image; see
# REGRESSION_RUNBOOK.md. Defaults are the crsuse2-m2m-075 layout.
# Separate runtime dir from agentx_env.sh: install_agentic_deps rm -rf's the venv.
export INFX_REPO="${INFX_REPO:-/workspace/InferenceX-agentx}"
export INFMAX_CONTAINER_WORKSPACE="$INFX_REPO/inferencex-e2e"
export AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-/workspace/agentx-runtime-e2e}"
export AIPERF_VENV="$AIPERF_RUNTIME_DIR/venv"
export AIPERF_UV_INSTALL_DIR="$AIPERF_RUNTIME_DIR/uv/bin"
export AIPERF_UV_CACHE_DIR="$AIPERF_RUNTIME_DIR/uv-cache"
export AIPERF_PYTHON_VERSION=3.11
export HF_HOME="${HF_HOME:-/shared_nfs/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export MODEL_DIR="${MODEL_DIR:-/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro-0813}"
# venv is built once by setup; srt_agentic.sh would otherwise rebuild it every run.
[[ -x "$AIPERF_VENV/bin/aiperf" ]] && export AIPERF_DEPS_READY=1
