#!/usr/bin/env bash
# fixed_seq_len sweep of the current build (2026-10-22 night, crsuse2-m2m-231):
#   mono : all 50 points (TP8 24 layers, then TP4 12 layers), --monokernel-backend amd
#   base : the points where mono can act (CONC 2..16, decode batch <= 16), same env, no mono
# Both with the same code (mono_env.sh: newbase heads + the sgl-amd-mono clone) and SP=${SP:-1}.
# START_UTC (HH:MM): wait until then before the first server.
set -u
S="$(cd "$(dirname "$0")/.." && pwd)/fixed_seq_len/sweep_proxy.sh"
M=/shared_nfs/kk/results/proxy_24l/fixed_seq_len_mono_1022
B=/shared_nfs/kk/results/proxy_24l/fixed_seq_len_base_1022
export ENV_FILE=/data/proxy_moe/mono_env.sh
if [ "${SP:-1}" = 1 ]; then export SGLANG_OPT_LAGUNA_PREFILL_SP=1; else unset SGLANG_OPT_LAGUNA_PREFILL_SP; fi
if [ -n "${START_UTC:-}" ]; then
  now=$(date -u +%s); start=$(date -u -d "today ${START_UTC}" +%s)
  [ "$start" -gt "$now" ] && { echo "waiting until ${START_UTC} UTC"; sleep $((start - now)); }
fi
echo "=== $(date -u +%H:%M) start (SP=${SP:-1}, plugin $(git -C /data/proxy_moe/ext/sgl-amd-mono rev-parse --short HEAD))"
EXTRA_ARGS="--monokernel-backend amd" OUT=$M RECIPES=tp8ep1 bash $S
SGLANG_AMD_MONO_LAGUNA_TP=4 EXTRA_ARGS="--monokernel-backend amd" OUT=$M RECIPES=tp4ep1 bash $S
EXTRA_ARGS="" OUT=$B CONCS="2 4 8 16" CONCS_70K="2 4 8 16" RECIPES="tp8ep1 tp4ep1" bash $S
echo "=== $(date -u +%H:%M) all done"
