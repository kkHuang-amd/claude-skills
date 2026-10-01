#!/usr/bin/env bash
# Short AgentX (aiperf agentx-mvp, c8) replay against an EAGER server running moe_route_probe, then print mean
# distinct routed experts per decode-sized MoE call over the replay window (probe snapshot diff).
#   PORT=8000 MODEL=/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash PROBE_GLOB='/shared_nfs/kk/atom_run/moe_probe/atom.*' \
#   OUT=/shared_nfs/kk/atom_run/agentx_route DURATION=600 bash agentx_route.sh
set -eo pipefail
: "${PORT:?}" "${MODEL:?}" "${PROBE_GLOB:?}" "${OUT:?}"
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
AIPERF=${AIPERF:-/workspace/agentx-runtime/venv_p8888/bin/aiperf}
mkdir -p "$OUT"
snap() { python3 - "$PROBE_GLOB" "$1" <<'EOF'
import glob, json, os, sys
files = [f for f in glob.glob(sys.argv[1]) if f.rsplit(".", 1)[1].isdigit()]
f = min(files, key=lambda p: int(p.rsplit(".", 1)[1]))  # one TP rank
json.dump(json.load(open(f))["stats"], open(sys.argv[2], "w"))
EOF
}
snap "$OUT/before.json"
"$AIPERF" profile --scenario inferencex-agentx-mvp --url "http://localhost:$PORT" --endpoint /v1/chat/completions \
  --endpoint-type chat --streaming --model "$MODEL" --tokenizer deepseek-ai/DeepSeek-V4.1-Flash --concurrency 8 \
  --benchmark-duration "${DURATION:-600}" --stats-interval 30 --random-seed 42 --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane "${WARMUP:-1}" \
  --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --use-server-token-count --no-gpu-telemetry \
  --tokenizer-trust-remote-code --num-dataset-entries 393 --slice-duration 1.0 \
  --output-artifact-dir "$OUT/aiperf_artifacts" --public-dataset semianalysis_cc_traces_weka_062126 --unsafe-override \
  > "$OUT/aiperf.log" 2>&1 || echo "aiperf exit=$?"
sleep 5
snap "$OUT/after.json"
python3 - "$OUT/before.json" "$OUT/after.json" <<'EOF'
import json, sys
b, a = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
rows = []
for k, v in a.items():
    E, T = map(int, k.split(":"))
    o = b.get(k, [0, 0, 0, 0])
    c, s = v[0] - o[0], v[1] - o[1]
    if c > 0 and T <= 64 and E >= 384:
        rows.append((T, c, s / c))
tot_c = sum(c for _, c, _ in rows)
print(f"target-MoE decode calls={tot_c}  weighted mean distinct={sum(c * m for _, c, m in rows) / max(tot_c, 1):.1f}")
for T, c, m in sorted(rows):
    if c >= 40:
        print(f"  T={T:3d} calls={c:6d} mean_distinct={m:5.1f}  (= {m / T:.2f} experts/token)")
EOF
