#!/usr/bin/env bash
# GSM8K few-shot accuracy against a running server (default: full 1319, 5-shot, greedy),
# matching the PR's reference: TP4/EP4 5-shot all 1319 -> 90.45% (DSpark off) / 90.22% (on).
#   TAG=<name> NQ=1319 SHOTS=5 PORT=30000 PARALLEL=128 MAXTOK=512
# Full log -> /shared_nfs/kk/dsv41/gsm8k_<TAG>.log ; summary line appended to results/gsm8k.md
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${TAG:-$(date +%m%d_%H%M)}; NQ=${NQ:-1319}; SHOTS=${SHOTS:-5}; PORT=${PORT:-30000}
LOG=/shared_nfs/kk/dsv41/gsm8k_${TAG}.log; mkdir -p "$(dirname "$LOG")"
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
python3 -m sglang.test.few_shot_gsm8k --port "$PORT" --num-questions "$NQ" --num-shots "$SHOTS" \
  --parallel "${PARALLEL:-128}" --max-new-tokens "${MAXTOK:-512}" > "$LOG" 2>&1
rc=$?
acc=$(grep -oE 'Accuracy: [0-9.]+' "$LOG" | tail -1); inv=$(grep -oE 'Invalid: [0-9.]+' "$LOG" | tail -1)
lat=$(grep -oE 'Latency: [0-9.]+ s' "$LOG" | tail -1)
echo "rc=$rc $acc $inv $lat log=$LOG"
[ -f "$HERE/results/gsm8k.md" ] || printf '| date | tag | nq | shots | accuracy | invalid | latency | log |\n|---|---|---|---|---|---|---|---|\n' > "$HERE/results/gsm8k.md"
echo "| $(date +%F) | $TAG | $NQ | $SHOTS | ${acc#Accuracy: } | ${inv#Invalid: } | ${lat#Latency: } | $LOG |" >> "$HERE/results/gsm8k.md"
exit $rc
