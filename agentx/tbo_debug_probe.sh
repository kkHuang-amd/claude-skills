#!/usr/bin/env bash
# TBO structural probe -- answers ONE question, cheaply:
#
#   with --enable-two-batch-overlap on, does SGLang split DSV4's *decode* and
#   *target-verify* batches into two ubatches, or only its prefill batches?
#
# Why it cannot be read off the source: batch_overlap/two_batch_overlap.py:92
# halves decode and target-verify batches and decode_cuda_graph_runner.py
# {1178,1375} drives the TBO plugin through capture and replay, while DSV4's HIP
# radix backend sets tbo_supports_cuda_graph=False whose docstring claims decode
# and target-verify stay NON-TBO. Both cannot be true at runtime, and our arms
# log nothing: `rg -c tbo server.log` == 1 (the args line).
#
# Why not just run a short agentic arm: of the 92 min a c128 arm takes, server
# startup is 3 (15:35:50 -> 15:38:50 in dptbo-c128); the rest is aiperf dataset
# configuration, warmup and certification. None of that is needed to observe a
# decode batch, so this drives gsm8k straight at the backend instead. ~8 min.
#
# Fidelity: the server command is REPLAYED VERBATIM from the c128 baseline arm's
# own sglang_command.txt, so every server flag matches the arm being explained.
# The env below is the launcher's DP branch (b200align_mtp.sh lines 58-100,
# 171-175, 283-285) plus the arm's HSA_NO_SCRATCH_RECLAIM. The MTP acceptance
# pin is KEPT, unlike gsm8k_arm.sh which sets NO_ACC_PIN=1: this probe is about
# the target-verify forward mode, so the spec-decode path must behave as it does
# in the arm. Probe accuracy is therefore meaningless BY DESIGN -- do not read
# the gsm8k score.
#
# Reads the tree but does not write it. Safe to run only when NO arm is running.
set -u

PROBE_DIR=/workspace/results/tbo-debug-probe
CMD_SRC=/workspace/results/dptbo-c128/sglang_command.txt
BACKEND_PORT=8889

test -f "$CMD_SRC" || { echo "FATAL: no recorded server command at $CMD_SRC"; exit 2; }
mkdir -p "$PROBE_DIR"

# Same three process shapes every arm kills. A clean rocm-smi is NOT evidence
# the node is free: the tokenizer workers hold ports without holding VRAM.
for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server|[s]glang_router" | awk '{print $1}'); do
    kill -9 "$p" 2>/dev/null
done
sleep 12

source /workspace/claude-skills/agentx/agentx_env.sh

# THE VARIABLE UNDER TEST: two_batch_overlap.py:59 reads this at import time.
export SGLANG_TBO_DEBUG=1

# Launcher env, DP branch.
export PYTHONNOUSERSITE=1
export SGLANG_TIMEOUT_KEEP_ALIVE=900
export SGLANG_DEFAULT_THINKING=1
export SGLANG_DSV4_REASONING_EFFORT=high
export SGLANG_USE_ROCM700A=0
export SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
export AITER_BF16_FP8_MOE_BOUND=0
export SGLANG_OPT_USE_AITER_BATCHED_GEMM=1
export SGLANG_ENABLE_UNIFIED_RADIX_TREE=1
export SGLANG_OPT_UNIFIED_CACHE_FREE_OUT_OF_WINDOW_SLOTS=1
export SGLANG_SHARED_EXPERT_TP1=1
export SGLANG_DP_SHARED_EXPERT_LOCAL=1
export SGLANG_DP_USE_GATHERV=1
export SGLANG_DP_USE_REDUCE_SCATTER=1
export GPU_MAX_HW_QUEUES=5
export SGLANG_SIMULATE_ACC_LEN=2.49
export SGLANG_SIMULATE_ACC_METHOD=match-expected
export SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token
export HSA_NO_SCRATCH_RECLAIM=0

# Dual purpose: proves vram_sampler.sh works before the 1.5 h arm depends on it.
bash /workspace/claude-skills/agentx/vram_sampler.sh "$PROBE_DIR/vram.csv" 15 &
SAMPLER_PID=$!
echo "sampler PID: $SAMPLER_PID"

SERVER_LOG="$PROBE_DIR/server.log"
: >"$SERVER_LOG"
# shellcheck disable=SC2046
setsid $(cat "$CMD_SRC") >>"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server PID: $SERVER_PID  log: $SERVER_LOG"

cleanup() {
    kill -9 "$SAMPLER_PID" 2>/dev/null
    for p in $(ps -eo pid,args | grep -E "[s]glang::|[s]glang\.launch_server" | awk '{print $1}'); do
        kill -9 "$p" 2>/dev/null
    done
}
trap cleanup EXIT

# The launcher's own gate, reimplemented: a dead server keeps answering
# GET /metrics with HTTP 200, so wait on the log line, not on the port.
for _ in $(seq 1 90); do
    grep -q "ready to roll" "$SERVER_LOG" && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FATAL: server died during startup"
        grep -oE "Error|RuntimeError: .{0,80}|Unsupported kernel config" "$SERVER_LOG" | tail -5
        exit 3
    fi
    sleep 10
done
grep -q "ready to roll" "$SERVER_LOG" || { echo "FATAL: server never became ready"; exit 3; }
echo "server ready after $(grep -c '' "$SERVER_LOG") log lines"

# Load only has to keep all 8 DP ranks in decode simultaneously: can_run_tbo
# requires forward_mode agreement across ranks (two_batch_overlap.py:463-467),
# so a trickle of requests would leave ranks IDLE and prove nothing.
cd /sgl-workspace/sglang/python
timeout 900 python3 -m sglang.test.few_shot_gsm8k \
    --num-questions 128 --max-new-tokens 512 --parallel 64 \
    --port "$BACKEND_PORT" >"$PROBE_DIR/load.log" 2>&1
echo "load exit=$?"

cleanup
trap - EXIT
sleep 5

echo "=== forward modes TBO actually prepared (count, mode, split index) ==="
# Fields only. The raw line carries extend_seq_lens=[...], which is unbounded.
rg -o 'tbo_split_seq_index=(\S+).*bs=(\S+) forward_mode=(\S+)' -r 'mode=$3 idx=$1' \
    "$SERVER_LOG" | sort | uniq -c | sort -rn | head -20
echo "=== control: did the debug hook fire at all? ==="
echo "TboForwardBatchPreparer lines: $(rg -c 'TboForwardBatchPreparer' "$SERVER_LOG" || echo 0)"
echo "=== vram sampler ==="
echo "rows: $(wc -l <"$PROBE_DIR/vram.csv"), peak used_gb: $(awk -F, 'NR>1 && $4>m{m=$4} END{print m}' "$PROBE_DIR/vram.csv")"
