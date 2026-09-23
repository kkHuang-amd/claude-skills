#!/usr/bin/env bash
# SMOKE TEST — fp8 two-pool unified_kv (sgl-project/sglang#37413), c128.
#
#   setsid nohup bash ./smoke_kvfp8.sh > /shared_nfs/kk/smoke_kvfp8_run.log 2>&1 &
#
# `SGLANG_DSV4_UNIFIED_KV_FP8=1` splits the unified pool into an fp8 nope pool
# plus a bf16 rope pool, 640 B/token instead of 1024. Two consequences, and the
# second is the reason to care:
#   1. KV capacity ~1.5x (#37413 measured 13.6M -> 20.4M tokens at TP8+DP8+MTP).
#   2. Decode STOPS using the Triton kernel we have spent today measuring: the
#      backend switches on `q_rope is not None` and sends the fp8 two-pool path
#      to aiter's `mla_a8w8` v4 nm ASM kernel instead
#      (deepseek_v4_backend_hip_radix.py, runtime.decode_fp8_2buff).
#
# Note `--kv-cache-dtype fp8_e4m3` does NOT do this: environ.py:1466 records
# that the unified pool takes no dtype and this switch is the only way to ask.
#
# EXPECTATION, from #37413's own tables: accuracy flat (0.946-0.949 fp8 vs
# 0.940-0.952 bf16) and ITL slightly WORSE at every concurrency (+3 % to +9 %).
# So this is a CAPACITY experiment, not an MLA speedup -- read
# max_total_num_tokens first, step time second. Baseline to compare against:
# 12,077,312 tokens (megamoe-eplb-c128-hcasplit4).
#
# What this smoke test answers, in order:
#   a. does it start at all on this tree/image (gfx950 gate, pool rewrite)?
#   b. what is max_total_num_tokens vs 12,077,312?
#   c. is decode actually on the asm kernel (no _paged_decode_split_kernel)?
#   d. gsm8k 1319 -- same gate the segment plan passed (0.937 baseline).
set -uo pipefail

KK=/shared_nfs/kk
SKILL_DIR=/workspace/claude-skills/agentx
STATUS="$KK/smoke_kvfp8_status.txt"
ALOG="$KK/smoke_kvfp8_arm.log"
GLOG="$KK/smoke_kvfp8_gsm8k.log"
RESULT_DIR=/workspace/results/smoke-kvfp8
VRAM_BASELINE=400000000

log() { echo "[kvfp8 $(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }

vram_used() {
    rocm-smi --showmeminfo vram 2>/dev/null \
      | rg -o 'Used Memory \(B\): [0-9]+' | rg -o '[0-9]+$' | sort -rn | head -1
}
cleanup() {
    ps -eo pid,args | rg 'launch_serve[r]' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
    sleep 5
    pgrep '^sglang::' | xargs -r kill -9 2>/dev/null
    sleep 15
}

cleanup
log "waiting for the VRAM cliff"
for i in $(seq 1 120); do
    u=$(vram_used); p=$(pgrep -c '^sglang::')
    if [ "${u:-999999999999}" -lt "$VRAM_BASELINE" ] && [ "$p" = "0" ]; then
        sleep 60; u2=$(vram_used)
        [ "${u2:-999999999999}" -lt "$VRAM_BASELINE" ] && break
    fi
    sleep 30
done

log "launching with SGLANG_DSV4_UNIFIED_KV_FP8=1"
(
    export EVAL_ONLY=true
    export SGLANG_DSV4_UNIFIED_KV_FP8=1
    export RESULT_DIR="$RESULT_DIR"
    bash "$SKILL_DIR/agentx_c128_hcasplit.sh"
) > "$ALOG" 2>&1 &
WRAP=$!

for _ in $(seq 1 180); do
    rg -q 'ready to roll' "$ALOG" 2>/dev/null && break
    sleep 10
done
if ! rg -q 'ready to roll' "$ALOG" 2>/dev/null; then
    log "FAILED to start: $(rg -o 'Memory access fault|OutOfResources.*|[A-Za-z]*Error: .{0,70}|Unsupported.*|assert.*' "$ALOG" | tail -2 | tr '\n' ' ')"
    kill "$WRAP" 2>/dev/null; cleanup
    log "=== SMOKE TEST FAILED (a: does it start) -- see $ALOG"
    exit 1
fi
log "server ready"
log "capacity: $(rg -o 'max_total_num_tokens=[0-9]*' "$ALOG" | tail -1)  (bf16 baseline max_total_num_tokens=12077312)"

PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python timeout 3600 \
  python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 \
  --max-new-tokens 8192 --parallel 64 --port 8889 > "$GLOG" 2>&1
log "gsm8k rc=$? $(rg -o '^Accuracy: [0-9.]+' "$GLOG" | tail -1) $(rg -o '^Invalid: [0-9.]+' "$GLOG" | tail -1)  (bf16 baseline 0.937)"

# Which decode kernel ran? The Triton one should be ABSENT on the fp8 path.
log "decode kernel: triton_split_lines=$(rg -c 'paged_decode' "$RESULT_DIR/server.log" 2>/dev/null || echo 0) asm_v4_nm=$(rg -c 'v4_nm|mla_a8w8' "$ALOG" 2>/dev/null || echo 0)"

kill "$WRAP" 2>/dev/null
cleanup
log "=== SMOKE TEST COMPLETE"
