#!/usr/bin/env bash
# Wait for GPU VRAM to be reclaimed after a run (or after a SIGKILL, which can
# leave 8x133 GB attributed to no live process for several minutes).
# Exits 0 when max VRAM% < threshold, 1 on timeout.
TH=${TH:-10}; MAX=${MAX:-1800}; t=0
maxv(){ rocm-smi --showmemuse 2>/dev/null | grep -oE "VRAM%\): [0-9]+" | grep -oE "[0-9]+$" | sort -n | tail -1; }
while [ $t -lt $MAX ]; do
  v=$(maxv); v=${v:-100}
  [ "$v" -lt "$TH" ] && { echo "VRAM reclaimed after ${t}s (max ${v}%)"; exit 0; }
  sleep 20; t=$((t+20))
done
echo "TIMEOUT after ${MAX}s -- VRAM still ${v}%, no live sglang proc: driver may need manual attention"; exit 1
