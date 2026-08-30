#!/usr/bin/env bash
# Run one accuracy arm: serve <ref_dir>, gsm8k 1319, record, stop.
# Usage: gsm8k_arm.sh <ref_dir> <label>
# Appends one line to $STATUS and leaves the full log at /tmp/gsm8k_<label>.log
set -uo pipefail
SKILL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REF="${1:?usage: gsm8k_arm.sh <ref_dir> <label>}"; LABEL="${2:?}"
STATUS="${STATUS:-/workspace/results/accuracy/STATUS.txt}"
mkdir -p "$(dirname "$STATUS")"
say() { echo "[$(date '+%F %T')] $LABEL  $*" | tee -a "$STATUS"; }

# An accuracy run must NEVER inherit the throughput acceptance pin (see
# agentx_debug.sh). Without this every arm scores ~0.72 regardless of config.
export NO_ACC_PIN=1
bash "$SKILL_DIR/agentx_debug.sh" stop >/dev/null 2>&1
say "START ref=$REF (NO_ACC_PIN=1)"
cd /workspace
: > /tmp/agentx_debug_server.log   # cmd_serve REUSES this log; without the
# truncate the loop below matches the PREVIOUS server's "ready to roll",
# declares ready seconds after launch, and gsm8k dies Connection refused.
setsid bash "$SKILL_DIR/agentx_debug.sh" serve "$REF" 8889 </dev/null \
  > "/tmp/serve_${LABEL}.out" 2>&1 &
sleep 20
for _ in $(seq 1 220); do
    grep -q "ready to roll" /tmp/agentx_debug_server.log 2>/dev/null && break
    sleep 10
done
if ! grep -q "ready to roll" /tmp/agentx_debug_server.log 2>/dev/null; then
    say "FAILED server never ready -- $(grep -oE 'Unsupported kernel config|RuntimeError: .{0,60}' \
        /tmp/agentx_debug_server.log | tail -1)"
    bash "$SKILL_DIR/agentx_debug.sh" stop >/dev/null 2>&1
    exit 1
fi
say "server ready"
cd /sgl-workspace/sglang/python
timeout 3600 python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 \
    --max-new-tokens 8192 --parallel 64 --port 8889 > "/tmp/gsm8k_${LABEL}.log" 2>&1
rc=$?
acc=$(grep -oE '^Accuracy: [0-9.]+' "/tmp/gsm8k_${LABEL}.log" | tail -1)
inv=$(grep -oE '^Invalid: [0-9.]+'  "/tmp/gsm8k_${LABEL}.log" | tail -1)
say "DONE rc=$rc ${acc:-Accuracy:?} ${inv:-Invalid:?}"
[ "${KEEP_SERVER:-0}" = "1" ] || bash "$SKILL_DIR/agentx_debug.sh" stop >/dev/null 2>&1
