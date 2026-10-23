#!/usr/bin/env bash
# Start the proxy server (same flags as mi455_point.sh), dump greedy tokens + logprobs for fixed prompts, kill it.
# Usage: greedy_dump.sh <out.json>   (env passes through: EP, AITER_*, SGLANG_OPT_*, ...)
# Compare two dumps with: python3 greedy_compare.py a.json b.json
set -u
OUT=$1; MODEL=${MODEL:-/workspace/proxy_work/proxy_12l}; PORT=18031; LOG=${OUT%.json}.server.log
cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 SGLANG_USE_AITER=1 PYTHONUNBUFFERED=1
python3 -m sglang.launch_server --model-path "$MODEL" \
  --tp-size 8 --ep-size ${EP:-8} --kv-cache-dtype fp8_e4m3 \
  --context-length 32768 --mem-fraction-static 0.80 \
  --max-running-requests 8 --cuda-graph-max-bs-decode 8 \
  --chunked-prefill-size 8192 --attention-backend triton \
  --watchdog-timeout 3600 ${EXTRA_ARGS:-} --host 127.0.0.1 --port $PORT > "$LOG" 2>&1 &
SPID=$!
for i in $(seq 1 360); do
  curl -sf -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "server died"; grep -E 'Error|Traceback' "$LOG" | tail -10 | cut -c1-200; exit 1; }
  sleep 10
done
python3 - "$OUT" $PORT <<'EOF'
import json, random, sys, urllib.request
out, port = sys.argv[1], sys.argv[2]
rng = random.Random(0)
# Lengths cover short, a sliding-window crossing (window 2048) and a long prompt.
prompts = [[rng.randrange(1000, 100000) for _ in range(n)] for n in (16, 100, 512, 1500, 2100, 3000, 64, 700)]
body = {"input_ids": prompts, "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        "return_logprob": True}
req = urllib.request.Request(f"http://127.0.0.1:{port}/generate", json.dumps(body).encode(), {"Content-Type": "application/json"})
res = json.load(urllib.request.urlopen(req, timeout=600))
dump = [{"ids": [t[1] for t in r["meta_info"]["output_token_logprobs"]],
         "lp": [t[0] for t in r["meta_info"]["output_token_logprobs"]]} for r in res]
json.dump(dump, open(out, "w"))
print("dumped", len(dump), "requests")
EOF
kill $SPID; wait $SPID 2>/dev/null
