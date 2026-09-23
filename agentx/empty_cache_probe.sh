#!/usr/bin/env bash
# Distinguish allocator fragmentation from genuine memory use.
# Waits until device VRAM crosses THRESH%, then pauses the scheduler in place and
# calls torch.cuda.empty_cache(). Large "freed" => the PyTorch allocator was
# hoarding (fragmentation, an upstream/ROCm problem). Small "freed" => the memory
# is genuinely in use (sglang-side: preallocate MoE workspace / resize the pool).
# Usage: empty_cache_probe.sh <server.log> [port] [thresh]
LOG=${1:?server.log}; PORT=${2:-8889}; TH=${3:-95}
maxv(){ rocm-smi --showmemuse 2>/dev/null | grep -oE "VRAM%\): [0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1; }
echo "waiting for VRAM >= ${TH}% ..."
started=0
while true; do
  v=$(maxv); v=${v:-0}
  # "not started yet" must not read as "died" -- the launcher spends minutes on
  # venv install and weight load before any sglang process exists (2026-08-30).
  if pgrep -f "sglang.launch_server" >/dev/null; then started=1
  elif [ "$started" = 1 ]; then echo "SERVER GONE before threshold (last VRAM ${v}%)"; exit 1; fi
  [ "$v" -ge "$TH" ] && break
  sleep 5
done
echo "TRIGGER at VRAM ${v}% -- pausing in place"
mark=$(wc -l < "$LOG")
curl -s -m 60 -X POST "http://localhost:$PORT/pause_generation" -H 'Content-Type: application/json' -d '{"mode":"in_place"}'; echo " <- pause rc=$?"
curl -s -m 300 -X POST "http://localhost:$PORT/continue_generation" -H 'Content-Type: application/json' -d '{"torch_empty_cache":true}'; echo " <- continue rc=$?"
sleep 5
echo "=== empty_cache result (per rank) ==="
tail -n +"$mark" "$LOG" | grep -E "continue_generation.*empty_cache" | cut -c1-160
echo "=== VRAM after: $(maxv)% (was ${v}%) ==="
