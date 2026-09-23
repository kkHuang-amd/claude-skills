#!/usr/bin/env bash
# Runbook section 4 AIPerf step only, against an already-running server+router.
# Deviation: --tokenizer points at the local checkpoint; the runbook relies on
# --model DeepSeek-V4-Pro resolving on HuggingFace, which 401s on this node.
set -euo pipefail
export INFX=/workspace/InferenceX
export AIPERF_RUNTIME_DIR=/workspace/agentx-runtime
export HF_HOME=/shared_nfs/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export INFMAX_CONTAINER_WORKSPACE="$INFX"
export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=1000000
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
RESULT_DIR="${RESULT_DIR:-/workspace/results/runbook-agentx-c64-notbo}"
mkdir -p "$RESULT_DIR/aiperf_artifacts"

PYTHONPATH="$INFX" "$AIPERF_RUNTIME_DIR/venv/bin/aiperf" profile \
  --scenario inferencex-agentx-mvp \
  --url http://localhost:31046 \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model DeepSeek-V4-Pro \
  --tokenizer /shared_nfs/deepseek-ai/DeepSeek-V4-Pro \
  --concurrency 64 --benchmark-duration 900 --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --agentic-cache-warmup-duration 600 \
  --warmup-grace-period 1800 \
  --use-server-token-count --no-gpu-telemetry \
  --tokenizer-trust-remote-code \
  --num-dataset-entries 393 --slice-duration 1.0 \
  --output-artifact-dir "$RESULT_DIR/aiperf_artifacts" \
  --public-dataset semianalysis_cc_traces_weka_with_subagents \
  --server-metrics http://127.0.0.1:31047/metrics \
  2>&1 | tee "$RESULT_DIR/aiperf.log"
echo "DONE $RESULT_DIR"
