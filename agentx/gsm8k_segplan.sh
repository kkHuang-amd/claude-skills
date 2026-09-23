#!/usr/bin/env bash
# GSM8K A/B for the segment-plan port: correctness BEFORE performance.
#
#   setsid nohup ./gsm8k_segplan.sh > /shared_nfs/kk/gsm8k_segplan.log 2>&1 &
#
# Runs two arms back to back on the same launcher, same flags, same tree, with
# only SGLANG_MLA_SEG_PLAN differing, and records gsm8k 1319 for each.
#
# WHY THE LAUNCHER AND NOT gsm8k_ab.sh. Hand-rolling the server from
# sglang_command.txt does not reproduce a MegaMoE run: MegaMoE is turned on by
# launcher ENV (SGLANG_AMD_USE_FLYDSL_MEGA_MOE, ..._MEGA_QUANT=a8w4,
# SGLANG_USE_AITER, SGLANG_MOE_PADDING), not by --moe-a2a-backend alone, and
# without them the MoE falls into the Triton path and dies at
# fused_moe_triton_kernels.py:863. Going through the arm script inherits all of
# it by construction.
#
# EVAL_ONLY=true is the launcher's own accuracy switch
# (dsv4_fp4_mi355x_sglang_mtp.sh:313): it skips the aiperf replay AND, crucially,
# does not pin SGLANG_SIMULATE_ACC_LEN to the golden 3.77 -- with that pin the
# MTP acceptance is faked and any accuracy number is meaningless.
#
# --max-new-tokens 8192, not the default: DSv4-Pro's reasoning CoT truncates at
# 2048 and scores ~0.
set -uo pipefail

KK=/shared_nfs/kk
SKILL_DIR=/workspace/claude-skills/agentx
STATUS="$KK/gsm8k_segplan_status.txt"
PORT_BACKEND=8889
READY_TIMEOUT_MIN=${READY_TIMEOUT_MIN:-30}
VRAM_BASELINE=400000000

log() { echo "[gsm8k $(date -u +%H:%M:%S)] $*" | tee -a "$STATUS"; }

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

wait_vram() {
    for i in $(seq 1 120); do
        u=$(vram_used); p=$(pgrep -c '^sglang::')
        if [ "${u:-999999999999}" -lt "$VRAM_BASELINE" ] && [ "$p" = "0" ]; then
            sleep 60
            u2=$(vram_used)
            [ "${u2:-999999999999}" -lt "$VRAM_BASELINE" ] && return 0
        fi
        sleep 30
    done
    return 1
}

run_arm() {
    local label="$1" seg="$2"
    local alog="$KK/gsm8k_arm_${label}.log"
    local glog="$KK/gsm8k_${label}.log"

    cleanup
    wait_vram || { log "$label SKIPPED: VRAM never returned to baseline"; return 1; }

    log "$label launching (SGLANG_MLA_SEG_PLAN=$seg)"
    (
        export EVAL_ONLY=true
        export SGLANG_MLA_SEG_PLAN="$seg"
        export SGLANG_MLA_SEG_MAX=16
        export SGLANG_MLA_SEG_WG_MULT=4
        export RESULT_DIR="/workspace/results/gsm8k-segplan-${label}"
        bash "$SKILL_DIR/agentx_c128_hcasplit.sh"
    ) > "$alog" 2>&1 &
    local wrapper=$!

    for _ in $(seq 1 $((READY_TIMEOUT_MIN * 6))); do
        rg -q 'ready to roll' "$alog" 2>/dev/null && break
        sleep 10
    done
    if ! rg -q 'ready to roll' "$alog" 2>/dev/null; then
        log "$label FAILED: server never ready -- $(rg -o 'Memory access fault|OutOfResources.*|[A-Za-z]*Error: .{0,60}' "$alog" | tail -1)"
        kill "$wrapper" 2>/dev/null; cleanup
        return 1
    fi
    log "$label server ready"

    PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python timeout 3600 \
      python3 -m sglang.test.few_shot_gsm8k --num-questions 1319 \
      --max-new-tokens 8192 --parallel 64 --port "$PORT_BACKEND" \
      > "$glog" 2>&1
    local rc=$?
    local acc inv
    acc=$(rg -o '^Accuracy: [0-9.]+' "$glog" | tail -1)
    inv=$(rg -o '^Invalid: [0-9.]+' "$glog" | tail -1)
    log "$label DONE rc=$rc ${acc:-Accuracy:?} ${inv:-Invalid:?}"

    kill "$wrapper" 2>/dev/null
    cleanup
}

log "=== GSM8K A/B for the segment plan, 1319 questions, max-new-tokens 8192"
run_arm "planoff" 0
run_arm "planon" 1
log "=== COMPLETE. Status: $STATUS"
