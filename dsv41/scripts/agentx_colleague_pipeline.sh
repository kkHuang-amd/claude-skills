#!/usr/bin/env bash
# Unattended "reproduce colleague numbers, then OPUS" pipeline (user plan 2026-09-26), colleague recipe, TP2 EP1.
#   1. Wait for the already-running series (PREFIX colleague_noopus_tp2ep1, our opus-prefill code, OPUS off) to finish
#      c16 PDI16; reproduced if TTT >= REPRO_FRAC * 45518.84 (colleague c16 PDI16).
#   2a. Reproduced: let it finish c64, then OPUS=1 on the same code, RUNS "16:16 64:4".
#   2b. Not: kill it, rerun with the colleague's code (kevin-mii dsv41-amd-4-model 3e8187fa88 + #41159) OPUS off;
#       if that reproduces, OPUS=1 on colleague-4model-opus; otherwise stop for a human.
#   RUNNING_SERIES_PID=<pid of the running agentx_series.sh>  REPRO_FRAC=0.85
# Decisions -> /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/series.txt ("PIPELINE ...").
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx; S=$OUT/series.txt
say(){ echo "[$(date +%F' '%T)] PIPELINE $*" >> "$S"; }
TARGET=45518.84; FRAC=${REPRO_FRAC:-0.85}; RUNS="16:16 64:4"
COMMON=(SCRIPT=agentx_colleague_run.sh TP=2 EP_SIZE=1 GPUS=4,5 DURATION=3600)

ttt(){ python3 -c "import json,sys,glob; f=glob.glob(sys.argv[1]+'/dsv41flash*.json'); d=json.load(open(f[0])) if f else {}; print(d.get('request_metrics',{}).get('throughput',{}).get('per_gpu',{}).get('total_tput_tps',0))" "$1" 2>/dev/null || echo 0; }
ok(){ python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2])*float(sys.argv[3]) else 1)" "$1" "$FRAC" "$TARGET"; }
stop_all(){
  local p; p=$(ps -eo pid,args | awk '($2=="bash" && $3 ~ /agentx_(series|colleague_run|colleague_mi355x_sglang)\.sh/) || ($2=="python3" && $3=="-m" && $4=="sglang.launch_server") || ($2=="aiperf") || ($3=="-m" && $4 ~ /infx.bench_serving.server_watch/) || ($2 ~ /^sglang::/) {print $1}')
  [ -n "$p" ] && kill $p 2>/dev/null; sleep 30
  p=$(ps -eo pid,args | awk '($2 ~ /^sglang::/) || ($2=="python3" && $4=="sglang.launch_server") {print $1}'); [ -n "$p" ] && kill -9 $p 2>/dev/null; sleep 20
}
series(){ env "${COMMON[@]}" "$@" RUNS="$RUNS" bash "$D/agentx_series.sh"; }

until grep -q "END colleague_noopus_tp2ep1_c16_pdi16" "$S"; do sleep 60; done
t=$(ttt "$OUT/colleague_noopus_tp2ep1_c16_pdi16")
if ok "$t"; then
  say "ours c16 TTT=$t >= ${FRAC}x$TARGET -> REPRODUCED with opus-prefill code; finishing c64 then OPUS=1"
  while kill -0 "${RUNNING_SERIES_PID:?}" 2>/dev/null; do sleep 60; done
  series OPUS=1 PREFIX=colleague_opus_tp2ep1
  say "DONE (ours: noopus + opus)"; exit 0
fi
say "ours c16 TTT=$t < ${FRAC}x$TARGET -> NOT reproduced; switching to colleague code (sglang-dsv41-4model)"
kill "${RUNNING_SERIES_PID:?}" 2>/dev/null; stop_all
series OPUS=0 SRC=/sgl-workspace/sglang-dsv41-4model/python PREFIX=colleague4m_noopus_tp2ep1
t=$(ttt "$OUT/colleague4m_noopus_tp2ep1_c16_pdi16")
if ok "$t"; then
  say "colleague code c16 TTT=$t -> REPRODUCED; running OPUS=1 on colleague-4model-opus"
  series OPUS=1 SRC=/sgl-workspace/sglang-dsv41-4model-opus/python PREFIX=colleague4m_opus_tp2ep1
  say "DONE (colleague code: noopus + opus)"
else
  say "colleague code c16 TTT=$t -> STILL NOT reproduced; stopping for a human"
fi
