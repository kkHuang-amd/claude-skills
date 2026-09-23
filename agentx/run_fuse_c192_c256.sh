#!/usr/bin/env bash
# Drive the fusion series at c192 then c256, sequentially.
#
# SEQUENTIAL, NOT PARALLEL: one node, and each arm wants the whole machine.
# Each arm's own process + VRAM gates protect it, so the driver only has to
# order them and wait out the post-arm VRAM drain in between.
#
# THE DRAIN IS REAL AND IT IS NOT A FOREIGN JOB (findings, node trap corrected
# 2026-09-03). After the c128 fusion arm the node showed ~119 GB held evenly
# across all 8 GPUs with ZERO KFD processes. It drained on its own in ~13 min:
# ~1.2 GB/min for the first two minutes, then ~22.5 GB/min. Sampling only the
# first minute looks like a stall, which is how an earlier session concluded it
# "did not drain". So: wait, do not escalate, and never kill blindly. The next
# arm's own 30-min VRAM gate would cover this anyway; the explicit wait here
# just keeps the gate log clean.
set -u

HERE=/workspace/claude-skills/agentx
held_gb() {
    rocm-smi --showmeminfo vram 2>/dev/null \
        | grep -oE 'Total Used Memory \(B\): [0-9]+' \
        | awk '{s += $NF} END {printf "%.0f", s / 1e9}'
}

for C in 192 256; do
    echo "############################################################"
    echo "$(date '+%F %T') waiting for VRAM to drain before c$C (now $(held_gb) GB)"
    for _ in $(seq 1 40); do          # up to 20 min
        [ "$(held_gb)" -lt 10 ] && break
        sleep 30
    done
    echo "$(date '+%F %T') VRAM $(held_gb) GB -- starting c$C"
    echo "############################################################"

    LOG=/shared_nfs/kk/logs/hicache-fp4-int20-c$C-fuse
    mkdir -p "$LOG"
    CONC_TARGET=$C bash "$HERE/hicache_fp4_int20_fuse_conc.sh" >"$LOG/run.log" 2>&1
    echo "$(date '+%F %T') c$C driver rc=$?  log=$LOG/run.log"
    rg -n 'WINNER:|NO ATTEMPT|hit OOR|NOT comparable|IS board-comparable' "$LOG/run.log" \
        | tail -4 | cut -c1-140
done

echo "############################################################"
echo "$(date '+%F %T') series done. Regenerating the summary table."
python3 "$HERE/summary_table.py" | tail -8
