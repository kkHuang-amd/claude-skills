#!/usr/bin/env bash
# Exit 1 unless every socket's GFXCLK (sysfs freq1_input) looks locked at TARGET. Run before any benchmark on a node that is
# supposed to be clock-locked (h13-17 from 2026-10-29: 1000 MHz via agt_clock.sh). sysfs is a smoothed value: locked at 1000 it
# reads 998-1008 idle, 963-975 in decode, 938-952 in 70k prefill (all sockets together), so accept [TARGET-LOW, TARGET+HIGH] AND a cross-socket
# spread <= SPREAD (DPM spreads sockets by 100+ MHz). Usage: check_clk_lock.sh [target=1000] [low=100] [high=20] [spread=30]
T=${1:-1000}; LOW=${2:-100}; HIGH=${3:-20}; SPREAD=${4:-30}; bad=0; vals=()
for f in /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input; do
  v=$(( $(cat "$f") / 1000000 )); vals+=($v)
  [ $v -ge $((T-LOW)) ] && [ $v -le $((T+HIGH)) ] || { echo "NOT LOCKED: $f = $v MHz (want $((T-LOW))..$((T+HIGH)))"; bad=1; }
done
[ ${#vals[@]} = 0 ] && { echo "no freq1_input found"; exit 1; }
mn=$(printf '%s\n' "${vals[@]}" | sort -n | head -1); mx=$(printf '%s\n' "${vals[@]}" | sort -n | tail -1)
[ $((mx-mn)) -le $SPREAD ] || { echo "NOT LOCKED: cross-socket spread $((mx-mn)) MHz > $SPREAD (${vals[*]})"; bad=1; }
[ $bad = 0 ] && echo "clock lock OK: ${#vals[@]} sockets ${vals[*]} MHz" || exit 1
