#!/usr/bin/env bash
# PR #39857 measurement method for bs1: random-ids 4096/1024, ignore_eos, seed 42, 1 discarded warm-up +
# RUNS timed runs, /flush_cache before each, metric = decode tok/s "first to last streamed event" = 1000/TPOT.
#   BACKEND=sglang|sglang-oai|vllm (cross-engine: sglang-oai vs vllm)
#   TAG=<name> PORT=30000 RUNS=6 ; logs -> /shared_nfs/kk/dsv41/prstyle_<TAG>/ ; row -> results/perf_prstyle.md
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
TAG=${TAG:-$(date +%m%d_%H%M)}; PORT=${PORT:-30000}; RUNS=${RUNS:-6}
OUT=/shared_nfs/kk/dsv41/prstyle_${TAG}; mkdir -p "$OUT"
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
vals=()
for i in $(seq 0 "$RUNS"); do
  curl -s -X POST "http://127.0.0.1:$PORT/flush_cache" > /dev/null
  python3 -m sglang.bench_serving --backend "${BACKEND:-sglang}" --port "$PORT" --dataset-name random-ids \
    --random-input-len 4096 --random-output-len 1024 --random-range-ratio 1 --seed 42 \
    --num-prompts 1 --max-concurrency 1 --warmup-requests 0 > "$OUT/run$i.log" 2>&1
  t=$(grep -oE 'Mean TPOT \(ms\):\s+[0-9.]+' "$OUT/run$i.log" | grep -oE '[0-9.]+$')
  [ "$i" = 0 ] && continue   # warm-up discarded
  vals+=("$(python3 -c "print(f'{1000/$t:.1f}')" 2>/dev/null || echo NaN)")
done
med=$(printf '%s\n' "${vals[@]}" | python3 -c "import sys,statistics as s; v=[float(x) for x in sys.stdin if x.strip()!='NaN']; print(f'{s.median(v):.1f}' if v else 'FAIL')")
echo "$TAG c1 decode_tok/s median=$med runs=${vals[*]}"
R=$HERE/results/perf_prstyle.md
[ -f "$R" ] || printf '| date | tag | bs | median decode tok/s | runs | logs |\n|---|---|---|---|---|---|\n' > "$R"
echo "| $(date +%F) | $TAG | 1 | $med | ${vals[*]} | $OUT |" >> "$R"
