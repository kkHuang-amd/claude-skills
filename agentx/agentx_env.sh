# AgentX test-environment for InferenceX single-node agentic benchmarks.
# source this before running any benchmarks/single_node/agentic/*.sh
export INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX
export AIPERF_RUNTIME_DIR=/workspace/agentx-runtime
export AIPERF_VENV="$AIPERF_RUNTIME_DIR/venv"
export AIPERF_UV_INSTALL_DIR="$AIPERF_RUNTIME_DIR/uv/bin"
export AIPERF_UV_CACHE_DIR="$AIPERF_RUNTIME_DIR/uv-cache"
export HF_HOME="${HF_HOME:-/shared_nfs/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
