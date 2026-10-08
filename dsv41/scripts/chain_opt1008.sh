#!/usr/bin/env bash
# OPT_SWEEP_1008.md chain: wait for the aiter AOT build -> import check -> TP2 + TP4 GSM8K smokes (parallel, all opts on,
# EVAL_ONLY, no simulated acceptance) with QR INT8 -> if either GSM8K < GSM_MIN (0.895) redo the smokes with QR NONE
# (user 2026-10-08) -> launch both sweep lanes with the QR setting that passed (TAGP opt1008 / opt1008qrnone).
# Progress -> /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/chain_opt1008.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); R=/shared_nfs/kk/results/DeepSeek-V4.1-Flash
S=$R/agentx/chain_opt1008.txt; BL=$R/repo_backup_1008/aiter_aot_build.log; GSM_MIN=${GSM_MIN:-0.895}
say(){ echo "[$(date +%F' '%T)] $*" >> "$S"; }
say "chain start (pid $$)"
until grep -q 'BUILD_EXIT=' "$BL"; do sleep 30; done
rc=$(grep -oE 'BUILD_EXIT=[0-9]+' "$BL" | tail -1); say "aiter build $rc, jit .so: $(ls /sgl-workspace/aiter/aiter/jit/*.so 2>/dev/null | wc -l)"
[ "$rc" = BUILD_EXIT=0 ] || { say "STOP: build failed"; exit 1; }
python3 -c "import aiter" > /tmp/aiter_import.log 2>&1 || { say "STOP: import aiter failed: $(tail -1 /tmp/aiter_import.log | cut -c1-200)"; exit 1; }
smoke(){  # $1=QR_QUANT $2=TAGP ; returns 0 if both TPs >= GSM_MIN
  local qr=$1 tp ok=1 o acc inv; export QR_QUANT=$qr TAGP=$2
  say "smoke start QR=$qr TAGP=$2 (tp2 GPUs 0,1 :8888, tp4 GPUs 4-7 :8890)"
  MODE=smoke LANE=tp2 bash "$D/sweep_ci37423_opt.sh" > $R/agentx/${2}_smoke_tp2.out 2>&1 &
  MODE=smoke LANE=tp4 bash "$D/sweep_ci37423_opt.sh" > $R/agentx/${2}_smoke_tp4.out 2>&1 &
  wait
  for tp in 2 4; do
    o=$R/agentx/${2}_smoke_tp$tp.out
    say "tp$tp $(grep -E '^SMOKE' "$o" | cut -c1-500 | tr '\n' ' ')"
    acc=$(grep -oE 'Accuracy: [0-9.]+' $R/gsm8k_${2}_tp${tp}_smoke.log 2>/dev/null | tail -1 | awk '{print $2}')
    inv=$(grep -oE 'Invalid: [0-9.]+' $R/gsm8k_${2}_tp${tp}_smoke.log 2>/dev/null | tail -1 | awk '{print $2}')
    say "tp$tp QR=$qr GSM8K accuracy=${acc:-NONE} invalid=${inv:-?}"
    python3 -c "import sys; sys.exit(0 if float('${acc:-0}') >= $GSM_MIN else 1)" || ok=0
  done
  sleep 60   # let both smoke servers release VRAM
  [ "$ok" = 1 ]
}
if smoke INT8 opt1008; then say "GSM8K PASS with QR INT8"
elif smoke NONE opt1008qrnone; then say "GSM8K PASS with QR NONE (INT8 failed)"
else say "STOP: GSM8K below $GSM_MIN with QR INT8 and NONE, sweep NOT started"; exit 1; fi
say "sweep start: both lanes QR=$QR_QUANT TAGP=$TAGP"
LANE=tp2 bash "$D/sweep_ci37423_opt.sh" > $R/agentx/opt1008_lane_tp2.out 2>&1 &
LANE=tp4 bash "$D/sweep_ci37423_opt.sh" > $R/agentx/opt1008_lane_tp4.out 2>&1 &
wait
say "sweep done (lanes: agentx/lane_8888.txt, lane_8890.txt)"
