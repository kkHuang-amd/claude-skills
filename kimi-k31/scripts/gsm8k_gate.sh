#!/bin/bash
# Accuracy gate: gsm8k200 (sgl-eval, thinking, T=1.0, top_p 0.95, seed 42, 64 threads) against a
# server on :30000 launched WITHOUT SGLANG_SIMULATE_ACC_* (real draft), plus mean accept len.
# Usage: [TEMP=0] gsm8k_gate.sh <tag> <server_log>   -> $K31_OUT/results/acc/<tag>/{metrics.json,accept_len.txt}
# TEMP=0 (greedy) is the low-noise A/B mode: at T=1.0 a single long-tail generation (seen: 3k-12k
# tokens) dominates wall time and the time-weighted accept len.
# Baseline (T2, 2026-10-30): score 0.985, accept len mean 4.06 (req-weighted 4.90).
set -u
source /workspace/claude-skills/kimi-k31/env.sh
cd /tmp
TAG=$1; SLOG=$2; R=$K31_OUT/results/acc/$TAG; mkdir -p "$R"
until rg -q 'ready to roll' "$SLOG"; do rg -q 'Traceback|Initialization failed' "$SLOG" && { echo SERVER-FAILED; exit 1; }; sleep 15; done
S=$(date -u '+%H:%M:%S')
python3 - "$R/metrics.json" "${TEMP:-1.0}" > "$R/eval.log" 2>&1 <<'EOF'
import json, sys, time
from types import SimpleNamespace
from sglang.test.sgl_eval_utils import run_sgl_eval
t = time.time()
temp = float(sys.argv[2])
m = run_sgl_eval(SimpleNamespace(base_url="http://127.0.0.1:30000", model="/shared_nfs/Kimi-K3.1", eval_name="gsm8k",
    num_examples=200, num_threads=64, max_tokens=32768, temperature=temp, top_p=0.95 if temp > 0 else 1.0,
    seed=42, sgl_eval_thinking=True))
m["temperature"] = temp
m["wall_s"] = round(time.time() - t, 1)
json.dump(m, open(sys.argv[1], "w"), indent=1, default=str)
EOF
echo "eval exit=$?"
E=$(date -u '+%H:%M:%S')
python3 -c "import json;m=json.load(open('$R/metrics.json'));print('score',m['score'],'truncated',m.get('truncated_rate'),'error',m.get('error_rate'),'wall_s',m['wall_s'])"
rg 'TP0\] Decode batch' "$SLOG" | python3 -c "
import sys,re
n=acc=w=wa=0
for l in sys.stdin:
  t=re.search(r' (\d\d:\d\d:\d\d) TP0',l); a=re.search(r'accept len: ([0-9.]+)',l); r=re.search(r'#running-req: (\d+)',l)
  if not(t and a and r) or not('$S'<=t.group(1)<='$E'): continue
  n+=1; acc+=float(a.group(1)); w+=int(r.group(1)); wa+=float(a.group(1))*int(r.group(1))
print(f'accept len: lines={n} mean={acc/max(n,1):.3f} req_weighted={wa/max(w,1):.3f}')" | tee "$R/accept_len.txt"
python3 - "$R/metrics.json" <<'EOF' | tee -a "$R/accept_len.txt"
import json, sys
p = json.load(open(sys.argv[1]))["sgl_eval_metrics_path"].rsplit("/", 1)[0]
import glob, os
f = [x for x in glob.glob(p + "/*") if "pred" in os.path.basename(x).lower() or x.endswith(".jsonl")][0]
ls = sorted(json.loads(l)["num_generated_tokens"] for l in open(f))
print(f"output tokens: sum={sum(ls)} p50={ls[len(ls)//2]} p90={ls[int(len(ls)*.9)]} max={ls[-1]}")
EOF
