#!/usr/bin/env bash
# Run INSIDE the vLLM container (image vllm/vllm-openai-rocm:nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984),
# started with --network=host, /shared_nfs mounted, GPUs 4-7. Benchmarks are driven from the SGLang container
# (run_pr_style_c1.sh / run_throughput.sh with BACKEND=vllm, run_gsm8k_openai.sh ENGINE=vllm).
# Base: InferenceX benchmarks/single_node/agentic/dsv41flash_fp4_mi355x_vllm_mtp.sh, plus knobs to match SGLang:
#   DSPARK=1       DSpark 5 draft tokens (block rejection = real acceptance); 0 = no spec
#   SIM_AL=3.51    synthetic acceptance (throughput only; never GSM8K)
#   EP=1           --enable-expert-parallel (SGLang runs TP4+EP4)      KV_FP8=1  --kv-cache-dtype fp8   BLOCK_SIZE=(unset)
#   PREFIX_CACHE=0 (default, = SGLang radix off)   AITER_MLA=1 (0 disables AITER MLA/OPUS prefill)
#   GPUS=4,5,6,7 PORT=8000 MODEL=/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash
# Log: /shared_nfs/kk/dsv41/vllm/server_<TAG>.log ; image versions -> /shared_nfs/kk/dsv41/vllm/versions.txt
set -euo pipefail
OUT=/shared_nfs/kk/dsv41/vllm; mkdir -p "$OUT"; TAG=${TAG:-$(date +%m%d_%H%M)}
# never clobber a previous run: a reused TAG gets a time suffix
[ -e "$OUT/server_${TAG}.log" ] && TAG=${TAG}_$(date +%H%M%S)
MODEL=${MODEL:-/shared_nfs/models/deepseek-ai/DeepSeek-V4.1-Flash}
{ date; pip list 2>/dev/null | grep -iE '^(vllm|torch|triton|amd-aiter|aiter|flydsl|amdsmi) '; cat /opt/rocm/.info/version 2>/dev/null; } > "$OUT/versions.txt" || true   # ROCm10 image has no .info/version
export HIP_VISIBLE_DEVICES=${GPUS:-4,5,6,7}
export VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1 AITER_TRITON_LOG_LEVEL=ERROR
export VLLM_USE_BREAKABLE_CUDAGRAPH=1 OMP_NUM_THREADS=1 VLLM_ENGINE_READY_TIMEOUT_S=3600 PYTHONUNBUFFERED=1
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}; NSPEC=5; CAP=1
while (( CAP < MAX_NUM_SEQS * (1 + NSPEC) && CAP < 2048 )); do CAP=$((CAP * 2)); done
ARGS=(--host 0.0.0.0 --port "${PORT:-8000}" --tensor-parallel-size 4 --language-model-only
      --tokenizer-mode deepseek_v41 --reasoning-parser deepseek_v41 --tool-call-parser deepseek_v41 --enable-auto-tool-choice
      --moe-backend aiter --gpu-memory-utilization 0.9 --max-model-len ${MAX_MODEL_LEN:-1048576}
      --max-num-seqs "$MAX_NUM_SEQS" --max-cudagraph-capture-size "$CAP" --max-num-batched-tokens 16384
      --disable-uvicorn-access-log)
[ "${EP:-1}" = 1 ] && ARGS+=(--enable-expert-parallel)
# PREFIX_CACHE=0 (default) matches SGLang --disable-radix-cache. With prefix caching ON, GSM8K (shared 5-shot
# prefix, 82% hit) crashed all 4 GPUs with HSA_STATUS_ERROR_MEMORY_FAULT on the first AITER OPUS sparse-MLA
# prefill (module_mla_v4_prefill_opus, >=1024 queries), DSPARK=0, 2026-09-24.
[ "${PREFIX_CACHE:-0}" = 1 ] || ARGS+=(--no-enable-prefix-caching)
# AITER_MLA=0 -> VLLM_ROCM_USE_AITER_MLA=0: disables AITER MLA incl. the OPUS prefill path (fallback if it still faults)
[ "${AITER_MLA:-1}" = 0 ] && export VLLM_ROCM_USE_AITER_MLA=0
[ "${KV_FP8:-1}" = 1 ] && ARGS+=(--kv-cache-dtype fp8)
# --block-size 256 FAILS on DSV4.1 with the default BLHNC KV layout ("manager block cannot be split into 4 kernel
# blocks of 64 tokens"); vLLM default block size is used unless BLOCK_SIZE is set (64 is the documented safe value).
[ -n "${BLOCK_SIZE:-}" ] && ARGS+=(--block-size "$BLOCK_SIZE")
if [ "${DSPARK:-1}" = 1 ]; then
  if [ -n "${SIM_AL:-}" ]; then
    SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","synthetic_acceptance_length":'"$SIM_AL"',"enable_adaptive_verification":false}'
  else
    SPEC='{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}'
  fi
  ARGS+=(--speculative-config "$SPEC")
fi
printf '%q ' vllm serve "$MODEL" "${ARGS[@]}" > "$OUT/cmd_${TAG}.txt"; echo >> "$OUT/cmd_${TAG}.txt"
echo "[vllm_launch] GPUs=$HIP_VISIBLE_DEVICES port=${PORT:-8000} log=$OUT/server_${TAG}.log (foreground; Ctrl-C stops the server)"
echo "[vllm_launch] ready when log shows 'Application startup complete' (typically 10-20 min)"
# `vllm` may be missing from PATH (e.g. a docker exec shell); the console script is vllm.entrypoints.cli.main:main
if command -v vllm >/dev/null 2>&1; then VLLM=(vllm); else VLLM=(python3 -m vllm.entrypoints.cli.main); echo "[vllm_launch] vllm not on PATH, using: ${VLLM[*]}"; fi
exec "${VLLM[@]}" serve "$MODEL" "${ARGS[@]}" > "$OUT/server_${TAG}.log" 2>&1
