#!/usr/bin/env bash
# S22.4 item 2: does cache-aware routing recover DP's per-rank cache split?
#
#   agentx_router.sh serve            # rung B config on 8889 + dp-aware router on 8888
#   agentx_router.sh bench <p1|nokey|key>
#   agentx_router.sh stop
#
# Rung B (DP8, no delayer, no TBO) is the arm that showed +35% at low cache hit
# and -2.9% at high hit, because DP splits the radix cache per rank: 83.2% hit
# vs TP8's 91.8%, 2x the prefill tokens (S22.3 item 5). The router hashes a
# request-id header to a DP rank, so all prompts sharing a prefix land on the
# rank that already holds it. bench_serving --gsp-send-routing-key sends
# X-SMG-Routing-Key = one distinct key per gsp group, so the router is pointed
# at that header instead of the recipe's x-correlation-id (which aiperf sends).
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SKILL_DIR/agentx_env.sh"

MODEL_PATH=/shared_nfs/models/DeepSeek-V4-Pro
MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
CONC=64
BACKEND_PORT=8889
ROUTER_PORT=8888
D=/workspace/results/ladder/rungB-router
PIDFILE=/tmp/agentx_router.pids

_assert_clean() {
    local v n pb
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{if($2>m)m=$2}END{print m+0}')
    n=$(ps -eo cmd | grep -cE '[s]glang\.launch_server|[s]glang::|[s]glang_router' || true)
    pb=$(ss -lntp 2>/dev/null | grep -cE ':(8888|8889)\b' || true)
    if [ "$n" != "0" ] || [ "${v:-0}" -gt 5 ] || [ "$pb" != "0" ]; then
        echo "REFUSING: vram%max=$v procs=$n ports=$pb -- run '$0 stop'" >&2; exit 1
    fi
}

cmd_serve() {
    _assert_clean; mkdir -p "$D"; : > "$PIDFILE"
    # rung B env: common base + the recipe's DP bundle.
    export PYTHONNOUSERSITE=1
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 SGLANG_DEFAULT_THINKING=1
    export SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_DSV4_REASONING_EFFORT=high
    export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
    export SGLANG_INT4_WEIGHT=0 SGLANG_MOE_PADDING=1
    export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
    export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1 SGLANG_ROCM_DISABLE_LINEARQUANT=0
    export SGLANG_ROCM_FUSED_DECODE_MLA=1 SGLANG_SET_CPU_AFFINITY=1
    export SGLANG_TIMEOUT_KEEP_ALIVE=900 SGLANG_USE_AITER=1 SGLANG_USE_ROCM700A=0
    export AITER_BF16_FP8_MOE_BOUND=0 GPU_MAX_HW_QUEUES=5
    export SGLANG_SIMULATE_ACC_LEN=2.49 SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
    export SGLANG_SHARED_EXPERT_TP1=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1 SGLANG_DP_USE_REDUCE_SCATTER=1

    local cmd=(python3 -m sglang.launch_server
        --model-path "$MODEL_PATH" --served-model-name "$MODEL_NAME"
        --host 0.0.0.0 --port "$BACKEND_PORT" --trust-remote-code
        --tensor-parallel-size 8 --dp 8 --enable-dp-attention
        --attention-backend dsv4 --page-size 256
        --swa-full-tokens-ratio 0.15 --kv-cache-dtype fp8_e4m3
        --disable-shared-experts-fusion
        --tool-call-parser deepseekv4 --reasoning-parser deepseek-v4
        --chunked-prefill-size 65536 --mem-fraction-static 0.90
        --max-running-requests 128 --cuda-graph-max-bs 128
        --speculative-algorithm EAGLE --speculative-num-steps 3
        --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
        --enable-hierarchical-cache --hicache-ratio 1.5
        --hicache-write-policy write_through --hicache-io-backend direct
        --hicache-mem-layout page_first_direct
        --watchdog-timeout 3600 --enable-metrics)
    printf '%q ' "${cmd[@]}" > "$D/sglang_command.txt"; printf '\n' >> "$D/sglang_command.txt"
    { echo "=== SGLANG_* env vars at launch ==="; env | grep -E '^SGLANG_' | sort;
      echo "==================================="; } > "$D/server.log"
    ( cd /tmp && "${cmd[@]}" >> "$D/server.log" 2>&1 ) &
    echo "server $!" >> "$PIDFILE"
    echo "backend pid=$! port=$BACKEND_PORT"
    for i in $(seq 1 400); do
        grep -q 'ready to roll' "$D/server.log" 2>/dev/null && { echo "BACKEND READY after $((i*15))s"; break; }
        [ "$i" = "400" ] && { echo "TIMEOUT"; return 1; }
        sleep 15
    done

    local rcmd=(python3 -m sglang_router.launch_router
        --worker-urls "http://localhost:$BACKEND_PORT"
        --policy consistent_hashing
        --request-id-headers x-smg-routing-key
        --dp-aware --host 0.0.0.0 --port "$ROUTER_PORT"
        --connect-timeout-secs 900 --request-timeout-secs 14400
        --disable-health-check --disable-retries)
    printf '%q ' "${rcmd[@]}" > "$D/router_command.txt"; printf '\n' >> "$D/router_command.txt"
    ( cd /tmp && "${rcmd[@]}" > "$D/router.log" 2>&1 ) &
    echo "router $!" >> "$PIDFILE"
    for i in $(seq 1 40); do
        curl -sf "http://127.0.0.1:$ROUTER_PORT/health" >/dev/null 2>&1 && { echo "ROUTER READY after $((i*5))s"; return 0; }
        sleep 5
    done
    echo "ROUTER TIMEOUT; tail:"; tail -6 "$D/router.log" | cut -c1-160; return 1
}

cmd_bench() {
    local name="${1:?usage: bench <p1|nokey|key>}" args
    local gsp="--dataset-name generated-shared-prefix --gsp-num-groups 8
               --gsp-prompts-per-group 64 --gsp-system-prompt-len 7936
               --gsp-question-len 256 --gsp-output-len 1024 --gsp-range-ratio 1.0
               --gsp-num-turns 1 --num-prompts 512"
    case "$name" in
        p1)    args="--dataset-name random --random-input-len 8192 --random-output-len 1024
                     --random-range-ratio 1.0 --num-prompts 512" ;;
        nokey) args="$gsp" ;;
        key)   args="$gsp --gsp-send-routing-key" ;;
        *) echo "bad name $name" >&2; exit 2 ;;
    esac
    curl -s -X POST "http://127.0.0.1:$BACKEND_PORT/flush_cache" >/dev/null || true
    sleep 5
    echo "WINDOW_START $name $(date +%H:%M:%S)" | tee -a "$D/windows.txt"
    ( cd /tmp && python3 -m sglang.benchmark.serving \
        --backend sglang --host 127.0.0.1 --port "$ROUTER_PORT" \
        --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
        $args --max-concurrency $CONC --request-rate inf \
        --seed 42 --warmup-requests $CONC \
        --output-file "$D/$name.json" --output-details ) > "$D/$name.bench.log" 2>&1
    local rc=$?
    echo "WINDOW_END $name $(date +%H:%M:%S)" | tee -a "$D/windows.txt"
    echo "exit=$rc"
    grep -E 'Successful requests|Benchmark duration|Request throughput|Total token throughput|Median TTFT|Mean TPOT' \
        "$D/$name.bench.log" | tr -s ' ' | cut -c1-55
}

cmd_stop() {
    local pat='[s]glang\.launch_server|[s]glang::|[s]glang_router'
    ps -eo pid,cmd | grep -E "$pat" | awk '{print $1}' \
        | while read -r q; do kill -TERM "$q" 2>/dev/null && echo "TERM $q"; done
    for i in $(seq 1 120); do
        local v n pb
        v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{if($2>m)m=$2}END{print m+0}')
        n=$(ps -eo cmd | grep -cE "$pat" || true)
        pb=$(ss -lntp 2>/dev/null | grep -cE ':(8888|8889)\b' || true)
        if [ "$n" = "0" ] && [ "${v:-0}" -le 5 ] && [ "$pb" = "0" ]; then echo "CLEAN after $((i*5))s"; return 0; fi
        if [ "$i" = "24" ]; then
            ps -eo pid,cmd | grep -E "$pat" | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
            rocm-smi --showpids 2>/dev/null | awk '$1 ~ /^[0-9]+$/ && $2 != "gpuagent" {print $1}' \
                | xargs -r kill -9 2>/dev/null || true
            echo "escalated to SIGKILL at 120s"
        fi
        sleep 5
    done
    echo "still draining (vram=$v procs=$n ports=$pb)"; return 1
}

case "${1:-}" in
    serve) cmd_serve ;;
    bench) shift; cmd_bench "$@" ;;
    stop)  cmd_stop ;;
    *) sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
