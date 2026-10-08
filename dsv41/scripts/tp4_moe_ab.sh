#!/usr/bin/env bash
# TP4 MoE tuning A/B (TP4_GAP_1006.md): run after the baseline decode profile (m255_tp4_d64k_c8, MoE untuned).
#   1. DEPLOY=1 tp4_moe_tune.sh (backs up + merges the 23 TP4 rows into aiter model_configs)
#   2. decode profile again at the same D64 c8 shape -> TAG m255_tp4_d64k_c8_tuned
#   3. GSM8K 5-shot 1319 on an EVAL_ONLY server (real draft acceptance, CONC=32 eval config) -> results/gsm8k.md
# Undo the deploy: copy /shared_nfs/kk/results/DeepSeek-V4.1-Flash/moe_tune_tp4/backup_<ts>/*.csv back into aiter model_configs.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_tp4
busy(){ pgrep -f 'agentx_lane.sh|tp4_moe_tune.sh|gemm_moe_tune.py|tp4_decode_profile.sh' >/dev/null || pgrep -f '^sglang::' >/dev/null; }
busy && { echo "REFUSING: GPUs busy (lane / tuning / profile / sglang server)"; exit 1; }

echo "== 1. deploy $(date +%T)"
DEPLOY=1 bash "$D/tp4_moe_tune.sh" || { echo "deploy failed"; exit 1; }

echo "== 2. tuned decode profile $(date +%T)"
TAG=m255_tp4_d64k_c8_tuned bash "$D/tp4_decode_profile.sh"
grep -c 'no tuned FlyDSL config' "$OUT/m255_tp4_d64k_c8_tuned/server.log" | sed 's/^/untuned-MoE warnings (expect 0 for 640 shapes): /'

echo "== 3. GSM8K $(date +%T)"
sleep 20
export PYTHONPATH=/sgl-workspace/mori SRC=/sgl-workspace/sglang/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0
export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
export TP=4 EP_SIZE=1 GPUS=0,1,2,3 PORT=8888 CONC=32 PREFILL_DECODE_INTERVAL=4 CHUNKED_PREFILL_SIZE=16384
export MEM_FRACTION_STATIC=0.80 SERVER_ONLY=1 EVAL_ONLY=true TAG=m255_tp4_gsm8k_tuned
G=$OUT/m255_tp4_gsm8k_tuned; mkdir -p "$G"
setsid bash "$D/agentx_colleague_run.sh" > "$G/server.log" 2>&1 < /dev/null &
SPID=$!
for _ in $(seq 1 180); do
    grep -q 'ready to roll' "$G/server.log" && break
    grep -qE 'Traceback|Initialization failed' "$G/server.log" && break
    sleep 10
done
if grep -q 'ready to roll' "$G/server.log"; then
    grep -c 'SGLANG_SIMULATE_ACC' "$G/server.log" | sed 's/^/sim-acceptance mentions (expect 0): /'
    TAG=m255_tp4_tuned PORT=8888 bash "$D/run_gsm8k.sh"
else
    echo "GSM8K server failed"; grep -m3 -E 'Traceback|Error' "$G/server.log" | cut -c1-200
fi
kill -- -$SPID 2>/dev/null; sleep 5
for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 "$p"; done
echo "== done $(date +%T)"
