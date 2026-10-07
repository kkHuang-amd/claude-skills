#!/usr/bin/env bash
# usage: watch_until_done.sh <server.log> <crash_pattern> <timeout_s>
L=$1; P=$2; T=${3:-3600}; t0=$(date +%s)
while :; do
  if rg -q "$P" "$L"; then echo CRASH; rg -n "$P" "$L" | head -3 | cut -c1-250; exit 2; fi
  if ! ps -eo args | rg -q '^bash .*agentx_colleague_mi355x_sglang\.sh'; then echo DONE; exit 0; fi
  [ $(( $(date +%s) - t0 )) -ge "$T" ] && { echo timeout; exit 1; }
  sleep 30
done
