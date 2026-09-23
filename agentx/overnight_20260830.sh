#!/usr/bin/env bash
# Detached overnight driver (2026-08-30). Survives the session that started it.
# 1) waits for the ALREADY-RUNNING c128 arm to reach a verdict
# 2) then runs the c64 replicate regardless of the c128 outcome (user's order)
# Verdicts append to $ST so they are readable in the morning even if all else is gone.
SK=/workspace/claude-skills/agentx
ST=/workspace/results/OVERNIGHT_20260830.txt
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$ST"; }

verdict_of(){   # $1=name $2=conc $3=max_seconds
  DIR=/workspace/results/$1; C=$2; t=0
  while [ $t -lt ${3:-9000} ]; do
    sleep 30; t=$((t+30))
    ls $DIR/*_c${C}.json >/dev/null 2>&1 && { echo "DONE at ${t}s"; return 0; }
    if grep -qE "HSA_STATUS_ERROR|Fatal Python error" $DIR/server.log 2>/dev/null; then
      echo "CRASHED OOR at ${t}s ($(grep -oE 'Available Free mem : [0-9]+ MB' $DIR/server.log|tail -1), peak VRAM $(awk -F, 'NR>1{for(i=3;i<=10;i++) if($i>m) m=$i} END{print m}' $DIR/vram.csv 2>/dev/null)%)"; return 1; fi
    pgrep -f "b200align_mt[p]" >/dev/null || { echo "LAUNCHER EXITED at ${t}s without a result json"; return 1; }
  done
  echo "TIMEOUT"; return 1
}
cleanup(){      # $1=graceful|force
  if [ "$1" = graceful ]; then
    w=0; while pgrep -f "b200align_mt[p]" >/dev/null && [ $w -lt 1200 ]; do sleep 20; w=$((w+20)); done
  else
    kill -9 $(pgrep -f 'sglang|[a]iperf|b200align_mt[p]' | tr '\n' ' ') 2>/dev/null
  fi
  pkill -f "vram_sample[r].sh" 2>/dev/null
  bash $SK/wait_vram.sh >/dev/null
}

say "watching the running c128 arm"
v=$(verdict_of c128-chunk16384-newmain 128); rc=$?
say "c128-chunk16384-newmain: $v"
[ $rc -eq 0 ] && cleanup graceful || cleanup force

N=c64-chunk16384-newmain-rep; DIR=/workspace/results/$N
say "launching c64 replicate (confirms this morning's 20,131.6)"
mkdir -p $DIR
setsid nohup bash $SK/vram_sampler.sh $DIR/vram.csv 30 >/dev/null 2>&1 &
cd /workspace
setsid nohup env EP_SIZE=1 CONC=64 DURATION=3600 CHUNK_PER_RANK=16384 RESULT_DIR=$DIR \
  bash $SK/agentx_b200align.sh > /workspace/results/$N.log 2>&1 </dev/null &
v=$(verdict_of $N 64); rc=$?
say "$N: $v"
[ $rc -eq 0 ] && cleanup graceful || cleanup force
if [ $rc -eq 0 ]; then
  say "comparison vs this morning:"
  python3 $SK/arm_report.py $DIR /workspace/results/c64-chunk16384-newmain 2>&1 | tail -12 | tee -a "$ST"
fi
say "OVERNIGHT DONE"
