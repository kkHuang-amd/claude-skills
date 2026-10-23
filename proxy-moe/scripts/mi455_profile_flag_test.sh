#!/usr/bin/env bash
# Does `bench_serving --profile` (torch profiler in the sglang schedulers) record GPU kernels here?
# Prints the event categories of the TP0 trace.
set -u
OUT=/shared_nfs/kk/results/proxy_12l/${1:-profile_flag_test_$(hostname -s)_$(date +%m%d_%H%M)}
MODEL=${MODEL:-/workspace/proxy_work/proxy_12l}
PORT=18030
mkdir -p "$OUT"; cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 SGLANG_USE_AITER=1 PYTHONUNBUFFERED=1
export SGLANG_TORCH_PROFILER_DIR=$OUT/trace

python3 -m sglang.launch_server --model-path "$MODEL" \
  --tp-size 8 --ep-size 8 --kv-cache-dtype fp8_e4m3 \
  --context-length 32768 --mem-fraction-static 0.80 \
  --max-running-requests 2 --cuda-graph-max-bs-decode 2 \
  --chunked-prefill-size 8192 --attention-backend triton \
  --watchdog-timeout 3600 --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 &
SPID=$!
for i in $(seq 1 360); do
  curl -sf -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; exit 1; }
  sleep 10
done
echo "server ready after ~$((i*10))s"

python3 -m sglang.bench_serving --backend sglang \
  --host 127.0.0.1 --port $PORT --tokenizer "$MODEL" \
  --dataset-name random-ids --tokenize-prompt \
  --random-input-len 1024 --random-output-len 256 --random-range-ratio 1.0 \
  --num-prompts 4 --max-concurrency 2 --warmup-requests 2 --request-rate inf \
  --profile --profile-num-steps 20 > "$OUT/bench.log" 2>&1
echo "bench exit=$?"
sleep 20   # let the trace export finish
kill $SPID; wait $SPID 2>/dev/null

python3 - "$OUT/trace" <<'EOF'
import gzip, json, collections, glob, sys
fs = sorted(glob.glob(sys.argv[1] + "/**/*TP-0*.json.gz", recursive=True))
print("traces:", len(glob.glob(sys.argv[1] + "/**/*.json.gz", recursive=True)))
for f in fs[:1]:
    ev = json.load(gzip.open(f))["traceEvents"]
    c = collections.Counter(e.get("cat") for e in ev if e.get("ph") == "X")
    print(f.split("/")[-1], dict(c.most_common(8)))
    print("GPU kernel events:", c.get("kernel", 0))
EOF
