#!/usr/bin/env bash
# MI355X TP4 decode profile at the B200_REQUEST_1006.md D64 shape (TP4_GAP_1006.md): same server config as the AgentX
# rerun (Replay, EP1, engram host table, DSpark sim AL, c8 PDI/chunk/mem), SERVER_ONLY, then decode_load_profile.py.
#   CONC=8 ISL=65536 OSL=1024 bash tp4_decode_profile.sh      # GPUs 0-3, port 8888
# Output: /shared_nfs/kk/dsv41/profile_tp4/<TAG>/ (server.log, load.out, torch traces). Refuses while GPUs are busy.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CONC=${CONC:-8} ISL=${ISL:-65536} OSL=${OSL:-1024} STEPS=${STEPS:-40}
TAG=${TAG:-m255_tp4_d$((ISL / 1024))k_c${CONC}}
OUT=/shared_nfs/kk/dsv41/profile_tp4/$TAG; mkdir -p "$OUT"
if pgrep -f 'agentx_lane.sh|tp4_moe_tune.sh|gemm_moe_tune.py' >/dev/null || pgrep -f '^sglang::' >/dev/null; then
    echo "REFUSING: lane / MoE tuning / sglang server running"; exit 1
fi
export PYTHONPATH=/sgl-workspace/mori SRC=/sgl-workspace/sglang/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0
export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
export TP=4 EP_SIZE=1 GPUS=${GPUS:-0,1,2,3} PORT=${PORT:-8888} CONC=$CONC
export PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 MEM_FRACTION_STATIC=${MEM:-0.70}
export SERVER_ONLY=1 TAG=$TAG SGLANG_TORCH_PROFILER_DIR=$OUT
setsid bash "$D/agentx_colleague_run.sh" > "$OUT/server.log" 2>&1 < /dev/null &
SPID=$!
echo "server lane PID/PGID $SPID (kill -- -$SPID); log $OUT/server.log"
for _ in $(seq 1 180); do
    grep -q 'ready to roll' "$OUT/server.log" && break
    grep -qE 'Traceback|Initialization failed' "$OUT/server.log" && { echo "server failed"; grep -m3 -E 'Traceback|Error' "$OUT/server.log" | cut -c1-200; kill -- -$SPID; exit 1; }
    sleep 10
done
grep -q 'ready to roll' "$OUT/server.log" || { echo "server not ready after 30 min"; kill -- -$SPID; exit 1; }
grep -o "'enable_decoder_swa_bounded_replay': [A-Za-z]*\|'tp_size': [0-9]*\|'ep_size': [0-9]*" "$OUT/server.log" | head -3 | paste -sd' '
python3 -I "$D/decode_load_profile.py" --port "$PORT" --conc "$CONC" --isl "$ISL" --osl "$OSL" \
    --profile-dir "$OUT" --profile-steps "$STEPS" > "$OUT/load.out" 2>&1
echo "load exit=$?"; grep '^\[' "$OUT/load.out" | cut -c1-220
sleep 20   # trace flush
kill -- -$SPID 2>/dev/null; sleep 5
for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 "$p"; done
echo "traces:"; find "$OUT" -maxdepth 2 -name '*.json*' -newer "$OUT/load.out" -printf '  %s %p\n' 2>/dev/null | head -8
find "$OUT" -maxdepth 2 \( -name '*.trace.json*' -o -name '*.pt.trace*' \) -printf '  %s %p\n' 2>/dev/null | head -8
