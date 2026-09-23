#!/usr/bin/env bash
# AgentX c64 following SGLANG_MEGAMOE_FIXED_AGENTX_RUNBOOK_20260911.md section 4:
# direct server + sglang_router + direct aiperf, no InferenceX launcher.
#
# Deviations from the runbook, all forced:
#   - paths/model moved to this node
#   - --attention-backend compressed -> dsv4 (compressed is not in this sglang's registry)
#   - single arm: this node has one aiter (main ffa945f93), not the main-vs-#5001 pair
set -euo pipefail

export INFX=/workspace/InferenceX
export AIPERF_RUNTIME_DIR=/workspace/agentx-runtime
export MODEL_PATH=/shared_nfs/deepseek-ai/DeepSeek-V4-Pro
export HF_HOME=/shared_nfs/hf_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export RESULT_DIR="${RESULT_DIR:-/workspace/results/runbook-agentx-c64-$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$RESULT_DIR/aiperf_artifacts" "$RESULT_DIR/eplb" "$RESULT_DIR/traces"

export GPU_ARCHS=gfx950 PYTORCH_ROCM_ARCH=gfx950
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1
export AITER_BF16_FP8_MOE_BOUND=0

export SGLANG_USE_AITER=1
export SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1
export SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192
export MORI_SHMEM_HEAP_SIZE=42949672960
export SGLANG_SHARED_EXPERT_TP1=0
export SGLANG_DP_SHARED_EXPERT_LOCAL=0
export SGLANG_DP_USE_GATHERV=0
export SGLANG_DP_USE_REDUCE_SCATTER=0
export SGLANG_AITER_MEGA_RANK_SYNC=1
export SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY=1
export SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD=1
export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="$RESULT_DIR/eplb"
export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR/traces"

export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_INT4_WEIGHT=0
export SGLANG_MOE_PADDING=1
export SGLANG_ROCM_DISABLE_LINEARQUANT=0
export SGLANG_ROCM_FUSED_DECODE_MLA=1
export SGLANG_SET_CPU_AFFINITY=1

export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=1000000
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export INFMAX_CONTAINER_WORKSPACE="$INFX"

setsid stdbuf -oL -eL python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name DeepSeek-V4-Pro \
  --host 127.0.0.1 --port 31047 \
  --tp 8 --ep-size 8 --dp-size 8 --enable-dp-attention \
  --enable-prefill-delayer \
  --moe-a2a-backend megamoe --moe-dense-tp-size 1 --enable-dp-lm-head \
  --load-balance-method round_robin \
  --enable-eplb --eplb-rebalance-num-iterations 200 \
  --expert-distribution-recorder-mode stat \
  --speculative-algorithm EAGLE --speculative-num-steps 3 \
  --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
  --attention-backend dsv4 \
  --cuda-graph-max-bs-decode 64 --max-running-requests 128 \
  --mem-fraction-static 0.65 --swa-full-tokens-ratio 0.10 \
  --page-size 256 --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 65536 --disable-shared-experts-fusion \
  --enable-hierarchical-cache --hicache-ratio 4 \
  --hicache-write-policy write_through \
  --hicache-io-backend direct --hicache-mem-layout page_first_direct \
  --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4 \
  --chat-template "$INFX/benchmarks/single_node/chat_templates/deepseek_v4_thinking.jinja" \
  --enable-metrics --watchdog-timeout 1800 \
  > "$RESULT_DIR/server.log" 2>&1 &
export SERVER_PID=$!
echo "SERVER_PID=$SERVER_PID RESULT_DIR=$RESULT_DIR"

until curl -fsS http://127.0.0.1:31047/health >/dev/null; do
  kill -0 "$SERVER_PID" || { echo "server died"; tail -n 40 "$RESULT_DIR/server.log" | cut -c1-200; exit 1; }
  sleep 5
done
echo "server healthy"

setsid stdbuf -oL -eL python3 -m sglang_router.launch_router \
  --worker-urls http://127.0.0.1:31047 \
  --policy consistent_hashing \
  --request-id-headers x-correlation-id \
  --dp-aware --host 127.0.0.1 --port 31046 \
  --prometheus-host 127.0.0.1 --prometheus-port 41046 \
  --connect-timeout-secs 900 --request-timeout-secs 14400 \
  --tcp-keepalive-secs 30 \
  --disable-health-check --disable-retries \
  > "$RESULT_DIR/router.log" 2>&1 &
export ROUTER_PID=$!
echo "ROUTER_PID=$ROUTER_PID"

until curl -fsS http://127.0.0.1:31046/health >/dev/null; do
  kill -0 "$ROUTER_PID" || { echo "router died"; tail -n 40 "$RESULT_DIR/router.log" | cut -c1-200; exit 1; }
  sleep 2
done

for attempt in $(seq 1 120); do
  if curl -fsS http://127.0.0.1:31046/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"DeepSeek-V4-Pro","messages":[{"role":"user","content":"Return only the number equal to 1+1."}],"temperature":0,"max_tokens":16}' \
    -o "$RESULT_DIR/smoke.json"; then
    python3 -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("choices")' "$RESULT_DIR/smoke.json" && break
  fi
  kill -0 "$ROUTER_PID" || { echo "router died"; exit 1; }
  test "$attempt" -lt 120 || { echo 'router did not return a valid completion'; exit 1; }
  sleep 1
done
echo "smoke ok"

PYTHONPATH="$INFX" "$AIPERF_RUNTIME_DIR/venv/bin/aiperf" profile \
  --scenario inferencex-agentx-mvp \
  --url http://localhost:31046 \
  --endpoint /v1/chat/completions --endpoint-type chat --streaming \
  --model DeepSeek-V4-Pro \
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

kill -TERM -- "-$ROUTER_PID" 2>/dev/null || true
kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
echo "DONE $RESULT_DIR"
