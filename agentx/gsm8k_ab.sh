#!/usr/bin/env bash
# GSM8K A/B for the HCA split-K change.
#
#   ./gsm8k_ab.sh <splits>        # 4 = patch on, 0 = heuristic (baseline)
#
# Uses the EXACT server command the c128 arm ran (recovered from its
# sglang_command.txt), so the only difference between the two arms is
# SGLANG_MLA_HCA_KV_SPLITS.
#
# Two things deliberately NOT set:
#   - SGLANG_SIMULATE_ACC_LEN. AgentX pins it to 3.77 to make MTP acceptance
#     deterministic for throughput work. It fakes acceptance, so it would
#     invalidate any accuracy number. Real DSPARK speculation runs here.
#   - SGLANG_MLA_FAKE_KVLEN, which is the garbage-output counterfactual.
#
# --max-new-tokens 8192, not the harness default: the DSv4 launch env sets
# SGLANG_DSV4_REASONING_EFFORT=max, and a long reasoning CoT truncated mid-way
# leaves no final answer and scores 0. At 2048 this reads ~0.88 for reasons
# that have nothing to do with the kernel.
set -uo pipefail
SPLITS="${1:?usage: gsm8k_ab.sh <kv_splits, 4=on 0=off>}"
PORT=8889
OUT="/shared_nfs/kk/gsm8k_split${SPLITS}"

export PYTHONPATH="/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori${PYTHONPATH:+:$PYTHONPATH}"
export MORI_SHMEM_HEAP_SIZE=16G
export SGLANG_MLA_HCA_KV_SPLITS="$SPLITS"
export SGLANG_MLA_FAKE_KVLEN=0
export SGLANG_MLA_KVLEN_STATS=0

CMD=$(cat /workspace/results/megamoe-eplb-c128-hcasplit4/sglang_command.txt)
echo "[gsm8k] splits=$SPLITS  starting server"
$CMD > "$OUT.server.log" 2>&1 &
SRV=$!
echo "[gsm8k] server pid=$SRV"

for _ in $(seq 1 180); do
    code=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/health_generate" || echo 000)
    [ "$code" = "200" ] && break
    kill -0 $SRV 2>/dev/null || { echo "[gsm8k] FAIL: server died"; exit 1; }
    sleep 10
done
echo "[gsm8k] server ready at $(date -u +%H:%M:%S)"

python3 -m sglang.test.few_shot_gsm8k \
    --num-questions 200 --num-shots 5 --max-new-tokens 8192 \
    --parallel 64 --port "$PORT" > "$OUT.log" 2>&1
echo "[gsm8k] splits=$SPLITS exit=$?"
rg -i 'accuracy|invalid|latency' "$OUT.log" | tail -5

kill -9 $SRV 2>/dev/null
sleep 5
pgrep '^sglang::' > /tmp/gsm_s.txt; xargs -r kill -9 < /tmp/gsm_s.txt
echo "[gsm8k] done, server killed"
