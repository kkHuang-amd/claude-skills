#!/usr/bin/env bash
# Serving throughput sweep matching the PR table: ISL 4096 / OSL 1024, fixed lengths,
# concurrency 1/8/32, two datasets: random-ids ("random tokens") and random (ShareGPT text, "real text").
#   BACKEND=sglang|sglang-oai|vllm (use sglang-oai vs vllm for cross-engine: same /v1/completions client)
#   TAG=<name> PORT=30000 CONCS="1 8 32" DATASETS="random-ids random" PROMPTS_PER_CONC=4
# Logs/JSONL -> /shared_nfs/kk/dsv41/perf_<TAG>/ ; rows appended to results/perf.md
# PR reference (4xMI350X, output tok/s, DSpark off -> on): random bs1 155.73->641.88;
#   real bs1 155.88->335.11, bs8 951.67->1446.53, bs32 2182.16->2241.87
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${TAG:-$(date +%m%d_%H%M)}; PORT=${PORT:-30000}; K=${PROMPTS_PER_CONC:-4}
OUT=/shared_nfs/kk/dsv41/perf_${TAG}; mkdir -p "$OUT"
SHAREGPT=${SHAREGPT:-/shared_nfs/hf-hub-cache/datasets--anon8231489123--ShareGPT_Vicuna_unfiltered/snapshots/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json}
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
R=$HERE/results/perf.md
[ -f "$R" ] || printf '| date | tag | dataset | conc | out tok/s | mean TPOT ms | mean TTFT ms | accept len | log |\n|---|---|---|---|---|---|---|---|---|\n' > "$R"
for ds in ${DATASETS:-random-ids random}; do for c in ${CONCS:-1 8 32}; do
  n=$(( c * K )); [ $n -lt 8 ] && n=8
  log=$OUT/${ds}_c${c}.log
  python3 -m sglang.bench_serving --backend "${BACKEND:-sglang}" --port "$PORT" --dataset-name "$ds" --dataset-path "$SHAREGPT" \
    --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 \
    --num-prompts $n --max-concurrency $c --output-file "$OUT/${ds}_c${c}.jsonl" > "$log" 2>&1
  tp=$(grep -oE 'Output token throughput \(tok/s\):\s+[0-9.]+' "$log" | grep -oE '[0-9.]+$')
  tpot=$(grep -oE 'Mean TPOT \(ms\):\s+[0-9.]+' "$log" | grep -oE '[0-9.]+$')
  ttft=$(grep -oE 'Mean TTFT \(ms\):\s+[0-9.]+' "$log" | grep -oE '[0-9.]+$')
  acc=$(grep -ioE 'accept length:\s+[0-9.]+' "$log" | grep -oE '[0-9.]+$' | tail -1)
  echo "$TAG $ds c=$c out_tok/s=${tp:-FAIL} tpot=${tpot:-} acc=${acc:--}"
  echo "| $(date +%F) | $TAG | $ds | $c | ${tp:-FAIL} | ${tpot:-} | ${ttft:-} | ${acc:--} | $log |" >> "$R"
done; done
