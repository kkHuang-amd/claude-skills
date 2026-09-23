#!/usr/bin/env bash
set -eo pipefail
set -x

# Agentic trace replay benchmark for DeepSeek-V4-Pro FP4 on MI355X using SGLang
# with EAGLE/MTP speculative decoding.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../../benchmark_lib.sh"

check_env_vars MODEL TP CONC KV_OFFLOADING TOTAL_CPU_DRAM_GB RESULT_DIR DURATION EP_SIZE DP_ATTENTION

if [[ -n "$SLURM_JOB_ID" ]]; then
    echo "JOB $SLURM_JOB_ID running on $SLURMD_NODENAME"
fi

# ROCR/HIP visibility under slurm cgroups.
if [ -n "$ROCR_VISIBLE_DEVICES" ]; then
    export HIP_VISIBLE_DEVICES="$ROCR_VISIBLE_DEVICES"
fi

if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" || -z "$(ls -A "$MODEL_PATH" 2>/dev/null)" ]]; then
        hf download "$MODEL" --local-dir "$MODEL_PATH"
    fi
else
    hf download "$MODEL"
    export MODEL_PATH="$MODEL"
fi
rocm-smi || true
amd-smi || true

# A server killed on this node minutes earlier (previous job, crashed run)
# can still be draining its HBM: KFD reclaim takes minutes, and booting into a
# half-drained node fails RCCL init with HIP 'unhandled cuda error' /
# 'invalid argument'. DeepSeek-V4-Pro is an 805 GiB checkpoint, so the drain
# window here is at the long end. Wait for the GPUs to come back before
# launching. Per-GPU threshold: idle nodes hold a small driver/firmware VRAM
# baseline (observed up to ~4%/GPU), while a draining or occupied GPU sits at
# 50-90%. Require every GPU <= 10%.
GPU_CLEAN=false
for i in $(seq 1 90); do
    VRAM_MAX=$(rocm-smi --showmemuse 2>/dev/null | grep -oE "GPU Memory Allocated \(VRAM%\): [0-9]+" | awk '{if ($NF > m) m = $NF} END {print m+0}')
    if [ "${VRAM_MAX:-0}" -le 10 ]; then echo "GPUs clean (vram%max=$VRAM_MAX after $((i*10))s)"; GPU_CLEAN=true; break; fi
    echo "waiting for prior-job GPU memory reclaim: vram%max=$VRAM_MAX"; sleep 10
done
[ "$GPU_CLEAN" = "true" ] || { echo "Error: GPUs still draining prior job's memory after 15min" >&2; exit 1; }

# ---- Resolve traces and install deps ----------------------------------------
resolve_trace_source
install_agentic_deps

SERVER_LOG="$RESULT_DIR/server.log"
ROUTER_LOG="$RESULT_DIR/router.log"
mkdir -p "$RESULT_DIR"

# ---- Client config ----------------------------------------------------------
export PYTHONNOUSERSITE=1
# Agentic warmup dispatches hundreds of large prompts at once; allow up to
# 15 minutes of TCP progress before AIPerf declares a connection dead.
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
# AIPerf pins one pooled keep-alive connection per session (client-side
# keep-alive 300s) while uvicorn's default SGLANG_TIMEOUT_KEEP_ALIVE is 5s;
# inter-turn idle gaps can reuse a socket exactly as the server closes it.
# Outlast the client pool so the race cannot occur.
export SGLANG_TIMEOUT_KEEP_ALIVE=900

# ---- DSv4 kernel routing / thinking mode ------------------------------------
# Mirrors the deleted spec-none sibling plus the DSv4 block in
# benchmarks/multi_node/amd_utils/env.sh. AgentX measures the thinking-on
# regime, which is also the golden-AL curve committed for this model.
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
# aiter batched GEMM for the absorbed MLA projections, carried by the v0.5.18
# image and off by default in environ.py.
export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1

# ---- Aiter MegaMoEv2 (sgl-project/sglang#35619, UNMERGED) --------------------
# MTPR is validated at startup against the per-rank chunked-prefill budget in
# arg_groups/deepseek_v4_hook.py. Under DP attention the launcher widens
# CHUNKED_PREFILL_SIZE to 8192*TP and SGLang divides by dp_size, so the
# effective per-rank budget is 8192 again -> MTPR 8192 exactly covers it and is
# a power of two, which _mtpr() requires.
export SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1
export SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR="${SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR:-8192}"
export SGLANG_AMD_FLYDSL_MEGA_QUANT="${SGLANG_AMD_FLYDSL_MEGA_QUANT:-a8w4}"
# Off by default: neither exists in the validated megamoe bring-up
# (run_sgl_dsv4_unified.sh MODE=megamoe). RANK_SYNC additionally trips an
# IndexError in forward_batch_info.adjust_num_token_non_padded_for_attn_tp -
# its idle path sets global_num_tokens=[1] then indexes it by attn_dp_rank.
export SGLANG_AITER_MEGA_RANK_SYNC="${SGLANG_AITER_MEGA_RANK_SYNC:-0}"
export SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY="${SGLANG_AITER_MEGA_EPLB_PREFILL_ONLY:-0}"
export SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD="${SGLANG_AITER_MEGA_EPLB_FUSED_MAP_RECORD:-0}"
# mori symmetric heap is allocated OUTSIDE sglang's mem-fraction-static budget.
# Default is 4 GiB (include/mori/shmem/internal.hpp); MegaMoEV2's dispatch/combine
# buffers need ~3.1 GiB EACH, so the default overflows during decode graph capture.
export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-40G}"

# Unified radix tree: per-component (full-attn / SWA) cache management for
# hybrid-attention models, plus proactive release of out-of-window SWA KV
# slots during chunked prefill. Without the latter, in-flight requests pin SWA
# KV for their whole context and the trailing window of cached sessions gets
# flushed under LRU, collapsing the effective prefix-cache hit rate on
# multi-turn agentic workloads.
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1

# ---- HiCache (host DRAM KV tier) --------------------------------------------
# Per-arm L2 sizing: host pinned memory is roughly
# HICACHE_RATIO * (per-rank device KV pool) * TP, which must stay under the
# node's ~2.7 TB of DRAM. The deleted spec-none sibling used ratio 4 with a
# smaller device pool; at TP8 with mem-fraction-static 0.85 that would
# oversubscribe host DRAM, so this recipe starts from 1.5 (the value validated
# on this cluster by glm5.2_fp4_mi355x_sglang_mtp.sh) and leaves every knob
# overridable for tuning.
CACHE_ARGS=()
if agentic_kv_offload_enabled; then
    case "$KV_OFFLOAD_BACKEND" in
        hicache)
            HICACHE_RATIO="${HICACHE_RATIO:-1.5}"
            HICACHE_WRITE_POLICY="${HICACHE_WRITE_POLICY:-write_through}"
            HICACHE_IO_BACKEND="${HICACHE_IO_BACKEND:-direct}"
            HICACHE_MEM_LAYOUT="${HICACHE_MEM_LAYOUT:-page_first_direct}"
            echo "HiCache DSv4 CPU tier: ratio=$HICACHE_RATIO, write_policy=$HICACHE_WRITE_POLICY, io_backend=$HICACHE_IO_BACKEND, mem_layout=$HICACHE_MEM_LAYOUT, dram_budget=${TOTAL_CPU_DRAM_GB} GB, tp=$TP"
            CACHE_ARGS=(
                --enable-hierarchical-cache
                --hicache-ratio "$HICACHE_RATIO"
                --hicache-write-policy "$HICACHE_WRITE_POLICY"
                --hicache-io-backend "$HICACHE_IO_BACKEND"
                --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
            )
            ;;
        *)
            echo "Error: unsupported KV_OFFLOAD_BACKEND '$KV_OFFLOAD_BACKEND' (expected: hicache)" >&2
            exit 1
            ;;
    esac
fi

# ---- Parallelism ------------------------------------------------------------
# NOTE: the DP-attention path below is currently DORMANT (no dp-attn arms in
# amd-master.yaml for this key). It is kept so a future arm can enable it
# without rebuilding the router plumbing: sglang-router fronts the DP ranks
# with consistent hashing on the AIPerf correlation id, keeping multi-turn
# sessions on the DP rank that holds their radix/hicache prefix.
USE_SGLANG_ROUTER=false
SGLANG_BACKEND_PORT="$PORT"
# Small prefill chunks interleave long-context agentic prefills across
# requests instead of letting one ~100K-token prefill monopolize the engine
# (the conc>=16 queue-saturation / decode-stall failure mode). 8192 = 32*256,
# a page-size multiple well under the dsv4 compressor kernel's uint16 token
# cap; same value the multi-node DeepSeek-V4-Pro-AgentX no_dp profile uses.
CHUNKED_PREFILL_SIZE=8192
# MTP adds a draft KV pool and extra graph captures on top of the spec-none
# footprint, which ran at 0.90. 0.89 recovers most of that: the DSv4 compressor
# state pools are sized from the full-attention pool and allocated after it,
# outside this budget, so the remainder has to stay large enough to cover them.
# Lowered from the 0.89 baseline to leave HBM for the mori symmetric heap.
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.65}"
PARALLEL_ARGS=(--tensor-parallel-size "$TP")
# MegaMoE / a2a cannot run with the shared expert fused into the routed list:
# forward_mega_moe sizes MegaMoEV2 with topk = num_experts_per_tok +
# num_fused_shared_experts, and the FlyDSL dispatch path assumes an unfused
# shared expert. The B200 megamoe arm disables it for the same reason.
SHARED_EXPERTS_ARGS=(--enforce-shared-experts-fusion)
if [ "$DP_ATTENTION" = "true" ]; then
    USE_SGLANG_ROUTER=true
    export AIPERF_HTTP_X_SMG_ROUTING_KEY_FROM_CORRELATION_ID=true
    SGLANG_BACKEND_PORT=$((PORT + 1))
    SGLANG_ROUTER_METRICS_PORT=$((PORT + 10000))
    SGLANG_ROUTER_CMD=(python3 -m sglang_router.launch_router)

    # MoE comm is handled by mori a2a under MegaMoE, so the DP gatherv /
    # reduce-scatter path must be OFF (run_sgl_dsv4_unified.sh MODE=megamoe).
    export SGLANG_SHARED_EXPERT_TP1=0
    export SGLANG_DP_SHARED_EXPERT_LOCAL=0
    export SGLANG_DP_USE_GATHERV=0
    export SGLANG_DP_USE_REDUCE_SCATTER=0
    export GPU_MAX_HW_QUEUES=5

    # Chunked prefill is a whole-engine budget, so widen it by the DP degree.
    CHUNKED_PREFILL_SIZE=$((8192 * TP))
    SHARED_EXPERTS_ARGS=(--disable-shared-experts-fusion)
    PARALLEL_ARGS+=(
        --dp "$TP"
        --enable-dp-attention
        --moe-a2a-backend megamoe
        --moe-dense-tp-size 1
        --enable-dp-lm-head
        --load-balance-method round_robin
    )
    # EPLB is absent from the validated megamoe bring-up; opt in explicitly.
    # NOTE: redundant experts RAISE peak device memory (the ATOM runbook needed
    # util 0.90 instead of 0.85 for 64 of them), so raising MEM_FRACTION_STATIC
    # and enabling EPLB at the same time pushes HBM from both sides.
    if [ "${ENABLE_EPLB:-0}" = "1" ]; then
        # Matches the config that reproduced sglang#35619's 42,382 tok/s:
        # the distribution recorder is REQUIRED (without it EPLB measured -6.5%,
        # not +8.4%), and ep-num-redundant-experts stays at its default 0.
        export SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR="${SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR:-/tmp}"
        PARALLEL_ARGS+=(
            --enable-eplb
            --eplb-rebalance-num-iterations "${EPLB_REBALANCE_NUM_ITERATIONS:-200}"
            --expert-distribution-recorder-mode stat
        )
        [ -n "${EPLB_ALGORITHM:-}" ] && PARALLEL_ARGS+=(--eplb-algorithm "$EPLB_ALGORITHM")
        [ -n "${EP_NUM_REDUNDANT_EXPERTS:-}" ] && PARALLEL_ARGS+=(--ep-num-redundant-experts "$EP_NUM_REDUNDANT_EXPERTS")
    fi
fi

if [ "$EP_SIZE" -gt 1 ]; then
    PARALLEL_ARGS+=(--ep-size "$EP_SIZE")
fi

# AgentX concurrency counts live session trees, not individual requests.
# Subagent fan-out can push instantaneous request concurrency above CONC, so
# leave 2x headroom rather than clipping those bursts at the scheduler.
MAX_RUNNING_REQUESTS=$((2 * CONC))
[ "$MAX_RUNNING_REQUESTS" -gt 256 ] && MAX_RUNNING_REQUESTS=256
CUDA_GRAPH_MAX_BS=$MAX_RUNNING_REQUESTS
[ "$CUDA_GRAPH_MAX_BS" -gt 128 ] && CUDA_GRAPH_MAX_BS=128

# Saturation arms carry a larger in-flight working set than the 30-minute
# default warmup drain allows.
if [ "$CONC" -ge 32 ]; then
    export AGENTIC_WARMUP_GRACE_PERIOD=3600
fi

# ---- Speculative decoding ---------------------------------------------------
# DeepSeek-V4 ships a built-in MTP head, loaded through the EAGLE spec path
# with eagle-topk 1 (a single MTP chain); NOT NEXTN, whose V3/R1 loader
# crashes on the V4 architecture. Depth 3 matches the vLLM agentic sibling
# (dsv4-fp4-mi355x-vllm-agentic-mtp) and the fixed-seq-len SGLang MTP recipe.
SPEC_ARGS=(
    --speculative-algorithm EAGLE
    --speculative-num-steps 3
    --speculative-eagle-topk 1
    --speculative-num-draft-tokens 4
)

# Throughput runs pin acceptance to the committed golden AL for this model,
# thinking mode, and draft length (golden_al_distribution/dsv4_mtp.yaml:
# thinking_on, 3 -> 2.49). Eval-only runs keep real target verification so
# accuracy stays meaningful.
if [ "${EVAL_ONLY:-false}" != "true" ]; then
    export SGLANG_SIMULATE_ACC_LEN=2.49
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
fi

# ---- Launch -----------------------------------------------------------------
# NO --chat-template. sglang already renders DSv4 prompts with its NATIVE
# encoder, not a jinja template: resolve_chat_encoding_spec() returns "dsv4" for
# tool_call_parser == "deepseekv4" (entrypoints/openai/chat_encoding.py:112),
# which we pass, so entrypoints/openai/encoding_dsv4.py owns thinking
# (<think>/</think>), tool calls, tool results, EOS and reasoning history.
# Passing --chat-template OVERRIDES that native path with the 9-line
# chat_templates/deepseek_v4_thinking.jinja, which handles none of it.
# Measured 2026-08-28: with the template, generation collapsed to ~1 output
# token per request (server /metrics: 62.7M prompt tokens vs 331 generated)
# against 919-956 tok/req on the untemplated reference arms -- the arm was
# producing garbage. The startup log line "No chat template found, defaulting
# to 'string' content format" is MISLEADING: the native encoder does the work,
# so that line is not evidence of a defect.

SGLANG_CMD=(
    python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --served-model-name "$MODEL"
    --host 0.0.0.0
    --port "$SGLANG_BACKEND_PORT"
    --trust-remote-code
    "${PARALLEL_ARGS[@]}"
    --attention-backend dsv4
    --page-size 256
    --swa-full-tokens-ratio 0.10
    --kv-cache-dtype fp8_e4m3
    "${SHARED_EXPERTS_ARGS[@]}"
    --tool-call-parser deepseekv4
    --reasoning-parser deepseek-v4
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --max-running-requests "$MAX_RUNNING_REQUESTS"
    --cuda-graph-max-bs-decode "$CUDA_GRAPH_MAX_BS"
    "${SPEC_ARGS[@]}"
    "${CACHE_ARGS[@]}"
    # MTP draft-token forward passes under long-context agentic load block the
    # scheduler long enough to trip the 1800s watchdog mid-warmup.
    --watchdog-timeout 3600
    --enable-metrics
)

printf '%q ' "${SGLANG_CMD[@]}" | tee "$RESULT_DIR/sglang_command.txt"
printf '\n' | tee -a "$RESULT_DIR/sglang_command.txt"

{
    echo "=== SGLANG_* env vars at launch ==="
    env | grep -E '^SGLANG_' | sort
    echo "==================================="
} | tee "$SERVER_LOG"

echo "Starting SGLang server for MI355X..."
"${SGLANG_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

wait_for_server_ready --port "$SGLANG_BACKEND_PORT" --server-log "$SERVER_LOG" --server-pid "$SERVER_PID"

if [ "$USE_SGLANG_ROUTER" = "true" ]; then
    echo "Starting SGLang router on port $PORT for $TP DP ranks..."
    "${SGLANG_ROUTER_CMD[@]}" \
        --worker-urls "http://localhost:$SGLANG_BACKEND_PORT" \
        --policy consistent_hashing \
        --request-id-headers x-correlation-id \
        --dp-aware \
        --host 0.0.0.0 \
        --port "$PORT" \
        --prometheus-host 127.0.0.1 \
        --prometheus-port "$SGLANG_ROUTER_METRICS_PORT" \
        --connect-timeout-secs 900 \
        --request-timeout-secs 14400 \
        --disable-health-check \
        --disable-retries > "$ROUTER_LOG" 2>&1 &
    ROUTER_PID=$!
    echo "Router PID: $ROUTER_PID"
    wait_for_server_ready --port "$PORT" --server-log "$ROUTER_LOG" --server-pid "$ROUTER_PID"
fi

if [ "${EVAL_ONLY}" = "true" ]; then
    run_eval --port "$PORT"
else
    build_replay_cmd "$RESULT_DIR"
    REPLAY_CMD+=" --server-metrics http://localhost:$SGLANG_BACKEND_PORT/metrics"
    run_agentic_replay_and_write_outputs "$RESULT_DIR"
fi
