#!/usr/bin/env bash
# Engine-neutral GSM8K: vLLM's tests/evals/gsm8k/gsm8k_eval.py (5-shot, /v1/completions, temp 0, seed 42)
# run from THIS container against either server, so SGLang and vLLM use the identical client.
#   ENGINE=sglang|vllm TAG=<name> PORT=30000 NQ=1319 MAXTOK=256 CONC=128
# NEVER run with a server that has simulated acceptance on (SGLANG_SIMULATE_ACC_LEN / vLLM synthetic): accuracy
# is meaningless there. Log -> /shared_nfs/kk/dsv41/gsm8k_oai_<TAG>.log ; row -> results/gsm8k.md
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ENGINE=${ENGINE:-sglang}; TAG=${TAG:-$(date +%m%d_%H%M)}; PORT=${PORT:-30000}; NQ=${NQ:-1319}
EVAL=${VLLM_SRC:-/sgl-workspace/vllm-src}/tests/evals/gsm8k/gsm8k_eval.py
LOG=/shared_nfs/kk/dsv41/gsm8k_oai_${TAG}.log
PYTHONPATH=$HERE/scripts/_stub python3 "$EVAL" --port "$PORT" --num-questions "$NQ" --num-shots 5 \
  --max-tokens "${MAXTOK:-256}" --max-concurrency "${CONC:-128}" --save-results "${LOG%.log}.json" > "$LOG" 2>&1
rc=$?
acc=$(grep -oE 'Accuracy: [0-9.]+' "$LOG" | tail -1); inv=$(grep -oE 'Invalid responses: [0-9.]+' "$LOG" | tail -1)
lat=$(grep -oE 'Total latency: [0-9.]+' "$LOG" | tail -1)
echo "rc=$rc engine=$ENGINE $acc $inv $lat log=$LOG"
echo "| $(date +%F) | $ENGINE:$TAG (oai client) | $NQ | 5 | ${acc#Accuracy: } | ${inv#Invalid responses: } | ${lat#Total latency: } s | $LOG |" >> "$HERE/results/gsm8k.md"
exit $rc
