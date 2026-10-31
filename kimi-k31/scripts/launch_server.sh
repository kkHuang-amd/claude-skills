#!/bin/bash
# Launch the K3.1 A/B base server (DCP8 + DSPARK, sim acc 3.84, no radix) on :30000.
# Usage: [REPLAYSSM=0] [SIM=0] launch_server.sh <tag> [extra server args...]
#   extra env: prefix the call, e.g. SGLANG_SIMULATE_ACC_GREEDY=1 launch_server.sh tag
# Writes $K31_OUT/logs/server_<tag>.log and prints the python PID (also in .pid).
set -u
source /workspace/claude-skills/kimi-k31/env.sh
TAG=${1:?tag}; shift
LOG=$K31_OUT/logs/server_$TAG.log
SPEC_KDA="--enable-linear-replayssm-spec"; [ "${REPLAYSSM:-1}" = 0 ] && SPEC_KDA="--mamba-ssm-dtype float32"
SIMENV=""; [ "${SIM:-1}" = 1 ] && SIMENV="SGLANG_SIMULATE_ACC_LEN=3.84 SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token"
cd /tmp
env $SIMENV SGLANG_PROFILE_V2=0 SGLANG_PROFILE_WITH_STACK=false SGLANG_PROFILE_RECORD_SHAPES=false \
  SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 \
  SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_USERPTR_FOR_PAGED_MEM=0 \
  SGLANG_ROCM_RUNTIME_RESERVE_GB=10 SGLANG_ENABLE_KIMI_K31_PREFILL_BCG=1 \
  PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95 \
  nohup python -m sglang.launch_server --model-path $K31_MODEL --trust-remote-code \
  --tp-size 8 --dcp-size 8 --dcp-comm-backend a2a \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --mamba-full-memory-ratio 0.65 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests 48 --cuda-graph-max-bs-decode 48 \
  --speculative-algorithm DSPARK --speculative-draft-model-path /shared_nfs/Kimi-K3-DSpark \
  --speculative-draft-attention-backend triton $SPEC_KDA \
  --disable-radix-cache \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --port 30000 "$@" \
  > "$LOG" 2>&1 &
sleep 5
P=$(pgrep -f "python -m sglang.launch_server" | head -1)
echo "$P" > "$K31_OUT/logs/server_$TAG.pid"; echo "server pid=$P log=$LOG"
