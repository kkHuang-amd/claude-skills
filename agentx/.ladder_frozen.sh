#!/usr/bin/env bash
# S21 config ladder: isolate DP and TBO on fixed-seq-len sglang.benchmark.serving.
#
#   agentx_ladder.sh serve <A|B|C|D> [port]
#   agentx_ladder.sh bench <A|B|C|D> <1|5> [port]
#   agentx_ladder.sh stop
#
# Rungs (each adds exactly one thing):
#   A  TP8 baseline                                   (no dp, no delayer, no tbo)
#   B  + --dp 8 --enable-dp-attention  (+ DP env bundle, as the recipe couples it)
#   C  + --enable-prefill-delayer
#   D  + --enable-two-batch-overlap    (== on-file tbo-tp8-c64 config)
#
# HELD CONSTANT across all four rungs (deliberate deviations from the published
# TP8 arm, to keep each step single-variable -- see S16 "confounds"):
#   --disable-shared-experts-fusion   (TP8 arm uses --enforce-...; TBO+fusion untested)
#   --chunked-prefill-size 65536      (TP8 arm uses 8192)
#   --mem-fraction-static 0.90, --swa-full-tokens-ratio 0.15
# No sglang_router in any rung (the recipe puts one in front of DP arms for
# session affinity; bench_serving sends no routing key, so it would be a no-op
# that differs between rungs).
set -eo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
source "$SKILL_DIR/agentx_env.sh"

MODEL_PATH=/shared_nfs/models/DeepSeek-V4-Pro
MODEL_NAME=deepseek-ai/DeepSeek-V4-Pro
CONC=64
OUT_ROOT=/workspace/results/ladder
PIDFILE=/tmp/agentx_ladder.pid

_common_env() {
    export PYTHONNOUSERSITE=1
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export SGLANG_DEFAULT_THINKING=1
    export SGLANG_DISABLE_CUDNN_CHECK=1
    export SGLANG_DSV4_REASONING_EFFORT=high
    export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
    export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
    export SGLANG_INT4_WEIGHT=0
    export SGLANG_MOE_PADDING=1
    export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
    export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1
    export SGLANG_ROCM_DISABLE_LINEARQUANT=0
    export SGLANG_ROCM_FUSED_DECODE_MLA=1
    export SGLANG_SET_CPU_AFFINITY=1
    export SGLANG_TIMEOUT_KEEP_ALIVE=900
    export SGLANG_USE_AITER=1
    export SGLANG_USE_ROCM700A=0
    export AITER_BF16_FP8_MOE_BOUND=0
    export GPU_MAX_HW_QUEUES=5
    # MTP acceptance simulation, same as every agentic arm.
    export SGLANG_SIMULATE_ACC_LEN=2.49
    export SGLANG_SIMULATE_ACC_METHOD=match-expected
    export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
}

_dp_env() {
    # The recipe's DP bundle -- enabled iff dp-attention is on, exactly as
    # dsv4_fp4_mi355x_sglang_tbo_mtp.sh:150-155 does it.
    export SGLANG_SHARED_EXPERT_TP1=1
    export SGLANG_DP_SHARED_EXPERT_LOCAL=1
    export SGLANG_DP_USE_GATHERV=1
    export SGLANG_DP_USE_REDUCE_SCATTER=1
}

_rung_args() {
    case "$1" in
        A) ;;
        B) echo "--dp 8 --enable-dp-attention" ;;
        C) echo "--dp 8 --enable-dp-attention --enable-prefill-delayer" ;;
        D) echo "--dp 8 --enable-dp-attention --enable-prefill-delayer --enable-two-batch-overlap" ;;
        *) echo "bad rung $1" >&2; exit 2 ;;
    esac
}

_assert_clean() {
    local v n pb
    v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{if($2>m)m=$2}END{print m+0}')
    n=$(ps -eo cmd | grep -cE '[s]glang.launch_server|[s]glang::router' || true)
    pb=$(ss -lntp 2>/dev/null | grep -cE ':(8888|8889)\b' || true)
    if [ "$n" != "0" ] || [ "${v:-0}" -gt 5 ] || [ "$pb" != "0" ]; then
        echo "REFUSING: vram%max=$v sglang_procs=$n ports=$pb -- run '$0 stop'" >&2; exit 1
    fi
}

cmd_serve() {
    local rung="${1:?usage: serve <A|B|C|D> [port]}" port="${2:-8888}"
    _assert_clean
    _common_env
    [ "$rung" = "A" ] || _dp_env

    local d="$OUT_ROOT/rung$rung"; mkdir -p "$d"
    local log="$d/server.log"
    local cmd=(python3 -m sglang.launch_server
        --model-path "$MODEL_PATH" --served-model-name "$MODEL_NAME"
        --host 0.0.0.0 --port "$port" --trust-remote-code
        --tensor-parallel-size 8 $(_rung_args "$rung")
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

    printf '%q ' "${cmd[@]}" > "$d/sglang_command.txt"; printf '\n' >> "$d/sglang_command.txt"
    { echo "=== SGLANG_* env vars at launch ==="; env | grep -E '^SGLANG_' | sort;
      echo "==================================="; } > "$log"
    cd /tmp && "${cmd[@]}" >> "$log" 2>&1 &
    echo $! > "$PIDFILE"
    echo "rung $rung pid=$(cat $PIDFILE) port=$port log=$log"
    for i in $(seq 1 400); do
        grep -q 'ready to roll' "$log" 2>/dev/null && { echo "READY after $((i*15))s"; return 0; }
        kill -0 "$(cat $PIDFILE)" 2>/dev/null || { echo "SERVER DIED; tail:"; tail -12 "$log" | cut -c1-180; return 1; }
        sleep 15
    done
    echo "TIMEOUT waiting for readiness"; return 1
}

cmd_bench() {
    local rung="${1:?usage: bench <A|B|C|D> <1|5> [port]}" pt="${2:?}" port="${3:-8888}"
    local d="$OUT_ROOT/rung$rung"; mkdir -p "$d"
    local args
    case "$pt" in
        # Point 1: PR condition -- near-zero prefix cache hit, fixed 8K/1K.
        1) args="--dataset-name random --random-input-len 8192 --random-output-len 1024
                 --random-range-ratio 1.0 --num-prompts 512" ;;
        # Point 5: same shape, ~95% prefix hit. 7936 shared (31 x page 256)
        #          + 256 unique = 8192 ISL; 8 groups x 64 = 512 prompts.
        5) args="--dataset-name generated-shared-prefix --gsp-num-groups 8
                 --gsp-prompts-per-group 64 --gsp-system-prompt-len 7936
                 --gsp-question-len 256 --gsp-output-len 1024 --gsp-range-ratio 1.0
                 --gsp-num-turns 1 --num-prompts 512" ;;
        *) echo "bad point $pt" >&2; exit 2 ;;
    esac
    curl -s -X POST "http://127.0.0.1:$port/flush_cache" >/dev/null || true
    sleep 5
    # Full lines, untruncated: cache_hit_rate carries a long label set and a
    # 140-col cut ate the value on the first ladder attempt. pre/post lets the
    # per-point hit rate be differenced out of the cumulative gauge.
    curl -s "http://127.0.0.1:$port/metrics" 2>/dev/null \
        | grep -E '^sglang:(cache_hit_rate|prompt_tokens|generation_tokens|num_requests)' > "$d/point$pt.metrics.pre.txt" || true
    local json="$d/point$pt.json" blog="$d/point$pt.bench.log"
    ( cd /tmp && python3 -m sglang.benchmark.serving \
        --backend sglang --host 127.0.0.1 --port "$port" \
        --model "$MODEL_PATH" --tokenizer "$MODEL_PATH" \
        $args --max-concurrency $CONC --request-rate inf \
        --seed 42 --warmup-requests $CONC \
        --output-file "$json" --output-details ) > "$blog" 2>&1
    echo "exit=$?"
    curl -s "http://127.0.0.1:$port/metrics" 2>/dev/null \
        | grep -E '^sglang:(cache_hit_rate|prompt_tokens|generation_tokens|num_requests)' > "$d/point$pt.metrics.post.txt" || true
    grep -E 'Successful requests|Benchmark duration|Request throughput|Input token throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median E2E|P99 TTFT' "$blog" | cut -c1-100
}

cmd_stop() {
    # DP8 servers spawn sglang::scheduler_* children that the launch_server /
    # router pattern misses, and an 8-rank server can take >200s to release
    # VRAM after SIGTERM. Both cost the first ladder attempt rungs C and D.
    local pat='[s]glang\.launch_server|[s]glang::'
    ps -eo pid,cmd | grep -E "$pat" | awk '{print $1}' \
        | while read -r q; do kill -TERM "$q" 2>/dev/null && echo "TERM $q"; done
    rm -f "$PIDFILE"
    for i in $(seq 1 120); do
        local v n pb
        v=$(rocm-smi --showmemuse 2>/dev/null | grep -oE '\(VRAM%\): [0-9]+' | awk '{if($2>m)m=$2}END{print m+0}')
        n=$(ps -eo cmd | grep -cE "$pat" || true)
        pb=$(ss -lntp 2>/dev/null | grep -cE ':(8888|8889)\b' || true)
        if [ "$n" = "0" ] && [ "${v:-0}" -le 5 ] && [ "$pb" = "0" ]; then
            echo "CLEAN after $((i*5))s"; return 0
        fi
        # escalate to SIGKILL, including anything still holding a GPU.
        if [ "$i" = "24" ]; then
            # `set -e` + a nonzero xargs/kill aborted the whole script here on
            # the second attempt, so stop gave up at 120s instead of 600s.
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
    serve) shift; cmd_serve "$@" ;;
    bench) shift; cmd_bench "$@" ;;
    stop)  cmd_stop ;;
    *) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
