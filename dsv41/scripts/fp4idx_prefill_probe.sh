#!/usr/bin/env bash
# P0.3b (FP4_INDEX_PLANE_PORT.md): share of the V4.1 HIP low-ratio indexer (FP4 scorer) in prefill steps.
# Same TP4 server config as tp4_decode_profile.sh (TP4_GAP_1006.md lane: Replay, EP1, OPUS sparse prefill,
# --fp8-gemm-backend aiter --enforce-shared-experts-fusion, chunk 16384, mem 0.70), SERVER_ONLY, but SRC = the
# instrumented worktree (spans in low_ratio_backend_hip.py / deepseek_v4.py, env SGLANG_DSV41_IDX_TIMING=<dir>).
# Load: prefill_cost_probe.py (max_new_tokens=1, cached prefix + NEW fresh tokens), then fp4idx_timing_summary.py.
#   bash fp4idx_prefill_probe.sh      # GPUs 0-3, port 8888
#   SRC=/sgl-workspace/sglang-fp4idx/python GPUS=0,1,2,3 PORT=8888 TAG=<name> PROBE="16384:0,16 4096:0"
#   AITER=/sgl-workspace/aiter-i4 puts that aiter worktree ahead of the editable install (I4 bf16 logits).
# Output: /shared_nfs/kk/dsv41/fp4_index_port/<TAG>/ (server.log, load.out, idx_timing_rank0.jsonl, summary.txt).
# Refuses while any sglang server / lane runs.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:-p03b_$(hostname | sed 's/.*-//')_$(date +%m%d_%H%M)}
OUT=/shared_nfs/kk/dsv41/fp4_index_port/$TAG; mkdir -p "$OUT"
if pgrep -f 'agentx_lane.sh|tp4_moe_tune.sh|gemm_moe_tune.py' >/dev/null || pgrep -f '^sglang::' >/dev/null; then
    echo "REFUSING: lane / MoE tuning / sglang server running"; exit 1
fi
export PYTHONPATH=/sgl-workspace/mori${AITER:+:$AITER} SRC=${SRC:-/sgl-workspace/sglang-fp4idx/python} SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0
export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
export TP=4 EP_SIZE=1 GPUS=${GPUS:-0,1,2,3} PORT=${PORT:-8888} CONC=1
export PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 MEM_FRACTION_STATIC=${MEM:-0.70}
export SERVER_ONLY=1 TAG=$TAG SGLANG_DSV41_IDX_TIMING=$OUT
setsid bash "$D/agentx_colleague_run.sh" > "$OUT/server.log" 2>&1 < /dev/null &
SPID=$!
echo "server PID/PGID $SPID (kill -- -$SPID); log $OUT/server.log"
for _ in $(seq 1 180); do
    grep -q 'ready to roll' "$OUT/server.log" && break
    grep -qE 'Traceback|Initialization failed' "$OUT/server.log" && { echo "server failed"; grep -m3 -E 'Traceback|Error' "$OUT/server.log" | cut -c1-200; kill -- -$SPID; exit 1; }
    sleep 10
done
grep -q 'ready to roll' "$OUT/server.log" || { echo "server not ready after 30 min"; kill -- -$SPID; exit 1; }
sleep 15   # TP4_GAP_1006.md: let the post-ready request finish before any load
: > "$OUT/load.out"
for p in ${PROBE:-16384:0,16,48,112 4096:0,28,124}; do   # PROBE="<new tokens>:<prefix k list> ..."
    python3 -I "$D/prefill_cost_probe.py" --port "$PORT" --new "${p%%:*}" --prefix-k "${p#*:}" --reps 3 >> "$OUT/load.out" 2>&1
done
echo "load exit=$?"; grep -E '^prefix|Error|Traceback' "$OUT/load.out" | cut -c1-200
kill -- -$SPID 2>/dev/null; sleep 5
for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 "$p"; done
python3 -I "$D/fp4idx_timing_summary.py" "$OUT/idx_timing_rank0.jsonl" > "$OUT/summary.txt" 2>&1
echo "summary exit=$?"; head -c 4000 "$OUT/summary.txt"
