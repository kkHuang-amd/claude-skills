#!/usr/bin/env bash
# RFC refactor B (structure as kevin-mii's branch) + C (K2i in the plugin, srt = 7 files; DSpark split into its own
# commits) on perf/v41-mono-rfc 2d998c2496 = snap_rfc_c_1014:
# 1. the K2i GPU test (test_dsv41_k2a_indexer.py)  2. A/B snap_rfc_s5_1014 vs snap_rfc_c_1014 (GSM8K 1319 + NIAH + TPOT)
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
say() { echo "[$(date +%F' '%T)] $*"; }
say "1. K2i test"
(cd /sgl-workspace/sglang_v41mono && HIP_VISIBLE_DEVICES=4 PYTHONPATH=/sgl-workspace/snap_rfc_c_1014/python \
  timeout 1800 python3 -m pytest -q test/registered/mono/test_dsv41_k2a_indexer.py > /tmp/rfc_c_k2i.log 2>&1)
say "k2i test: $(grep -E 'passed|failed' /tmp/rfc_c_k2i.log | tail -1)"
say "2. A/B s5 vs c"
VARIANTS="s5:/sgl-workspace/snap_rfc_s5_1014:MONO_ARG=--enable-monokernel c:/sgl-workspace/snap_rfc_c_1014:MONO_ARG=--enable-monokernel" \
  PHASES="gsm ab" NQ=1319 REPS=3 TAGP=rfcc bash "$D/fw_ab.sh"
say "RFC C DONE"
