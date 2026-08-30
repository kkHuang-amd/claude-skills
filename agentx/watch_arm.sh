#!/usr/bin/env bash
# Stall/death detector for a running arm. Emits ONLY on: stall, server death,
# phase change, completion. Heartbeat every 30 min so silence is never ambiguous.
# Usage: watch_arm.sh <arm-name>
A=${1:?arm name}; LOG=/workspace/results/$A.log; SRV=/workspace/results/$A/server.log
prev=""; same=0; started=0; beat=0; phase=""
while true; do
  sleep 300
  # warmup prints returned=N/M; profiling prints done=N ok=N err=N -- track both,
  # or a phase change looks identical to a stall (false STALL, 2026-08-30).
  cur=$(grep -oE "returned=[0-9,]+/[0-9,]+|done=[0-9,]+ ok=[0-9,]+" "$LOG" 2>/dev/null | tail -1)
  ph=$(grep -oE "Phase [a-z]+ (started|complete)" "$LOG" 2>/dev/null | tail -1)
  alive=$(pgrep -cf "sglang.launch_server")
  [ "${alive:-0}" -gt 0 ] && started=1
  [ -n "$ph" ] && [ "$ph" != "$phase" ] && { echo "PHASE: $ph  ($cur)"; phase=$ph; }
  # Completion must be recognised, or a finished arm looks exactly like a stall
  # (false STALL after every run, 2026-08-30). The result json is the ground truth.
  if ls /workspace/results/$A/*_c[0-9]*.json >/dev/null 2>&1; then
    echo "COMPLETE: result json written; last progress $cur"; exit 0; fi
  if [ "$started" = 1 ] && [ "${alive:-0}" = 0 ]; then echo "SERVER GONE -- last progress $cur"; exit 0; fi
  if [ -n "$cur" ] && [ "$cur" = "$prev" ]; then same=$((same+1)); else same=0; fi
  prev=$cur
  if [ "$same" -ge 3 ]; then
    echo "STALL: no progress for ~20 min at $cur; server procs=$alive; last scheduler line: $(grep -E '#running-req' "$SRV" 2>/dev/null | tail -1 | cut -c1-110)"
    same=0
  fi
  beat=$((beat+1))
  [ $((beat % 6)) = 0 ] && echo "heartbeat: $cur  phase=${phase:-?}  procs=$alive"
done
