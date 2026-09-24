#!/usr/bin/env bash
# Torch-profile ONE big prefill batch on a running SGLang server and print GPU time by kernel category.
#   PORT=30000 NREQ=4 ISL=4096 (NREQ*ISL = batch tokens; 4x4096 = one 16384 chunk)  TAG=<name>
# Trace -> /shared_nfs/kk/dsv41/prof_<TAG>/ ; summary via scripts/trace_summary.py
set -uo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PORT=${PORT:-30000}; NREQ=${NREQ:-4}; ISL=${ISL:-4096}; TAG=${TAG:-prefill$((NREQ*ISL))}
OUT=/shared_nfs/kk/dsv41/prof_${TAG}; rm -rf "$OUT"; mkdir -p "$OUT"
export PYTHONPATH=${SRC:-/sgl-workspace/sglang-dsv41/python}${PYTHONPATH:+:$PYTHONPATH}
curl -s -X POST localhost:$PORT/flush_cache >/dev/null
curl -s -X POST localhost:$PORT/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\":\"$OUT\",\"num_steps\":2,\"activities\":[\"GPU\",\"CPU\"],\"profile_id\":\"$TAG\"}" | head -c 200; echo
python3 -m sglang.bench_serving --backend sglang --port "$PORT" --dataset-name random-ids --random-input-len "$ISL" \
  --random-output-len 2 --random-range-ratio 1 --num-prompts "$NREQ" --max-concurrency "$NREQ" --warmup-requests 0 \
  > "$OUT/bench.log" 2>&1
for i in $(seq 1 60); do ls "$OUT"/*.trace.json.gz >/dev/null 2>&1 && break; sleep 2; done; sleep 5
T=$(ls -S "$OUT"/*TP-0*.trace.json.gz 2>/dev/null | head -1); [ -z "$T" ] && T=$(ls -S "$OUT"/*.trace.json.gz | head -1)
echo "trace=$T"; python3 "$HERE/trace_summary.py" "$T"
