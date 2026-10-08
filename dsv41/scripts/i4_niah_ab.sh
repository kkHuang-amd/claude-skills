#!/usr/bin/env bash
# I4 long-context accuracy A/B: same TP4 server config as fp4idx_prefill_probe.sh, sglang-i4 + aiter-i4,
# SGLANG_DSV41_PREFILL_LOGITS_BF16 = 1 then 0, i4_niah_eval.py on each. Prefix cache off so every prompt prefills.
#   bash i4_niah_ab.sh                       # GPUs 0-3, port 8888
#   SIDES="1 0" GPUS=0,1,2,3 PORT=8888 NIAH_ARGS="--lengths 32000,64000,120000 --n 20"
# Output: /shared_nfs/kk/dsv41/i4_niah/<TAG>_bf16<v>/ (server.log, niah.out, niah.jsonl).
# Refuses while any sglang server / lane runs.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAG=${TAG:-i4niah_$(hostname | sed 's/.*-//')_$(date +%m%d_%H%M)}
for v in ${SIDES:-1 0}; do
    OUT=/shared_nfs/kk/dsv41/i4_niah/${TAG}_bf16$v; mkdir -p "$OUT"
    if pgrep -f 'agentx_lane.sh|tp4_moe_tune.sh|gemm_moe_tune.py' >/dev/null || pgrep -f '^sglang::' >/dev/null; then
        echo "REFUSING: lane / MoE tuning / sglang server running"; exit 1
    fi
    export PYTHONPATH=/sgl-workspace/mori:${AITER:-/sgl-workspace/aiter-i4} SRC=${SRC:-/sgl-workspace/sglang-i4/python}
    export SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0 SGLANG_DSV41_PREFILL_LOGITS_BF16=$v
    export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion --disable-radix-cache"
    export TP=4 EP_SIZE=1 GPUS=${GPUS:-0,1,2,3} PORT=${PORT:-8888} CONC=1
    export PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 MEM_FRACTION_STATIC=${MEM:-0.70}
    export SERVER_ONLY=1 TAG=${TAG}_bf16$v
    setsid bash "$D/agentx_colleague_run.sh" > "$OUT/server.log" 2>&1 < /dev/null &
    SPID=$!
    echo "bf16=$v server PID/PGID $SPID (kill -- -$SPID); log $OUT/server.log"
    for _ in $(seq 1 180); do
        grep -q 'ready to roll' "$OUT/server.log" && break
        grep -qE 'Traceback|Initialization failed' "$OUT/server.log" && break
        sleep 10
    done
    if grep -q 'ready to roll' "$OUT/server.log"; then
        sleep 15
        python3 -I "$D/i4_niah_eval.py" --port "$PORT" --out "$OUT/niah.jsonl" ${NIAH_ARGS:-} > "$OUT/niah.out" 2>&1
        echo "bf16=$v niah exit=$?"; grep -E '^len|^total|Error|Traceback' "$OUT/niah.out" | cut -c1-200
    else
        echo "bf16=$v server failed"; grep -m3 -E 'Traceback|Error' "$OUT/server.log" | cut -c1-200
    fi
    kill -- -$SPID 2>/dev/null; sleep 5
    for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 "$p"; done
    sleep 10
done
