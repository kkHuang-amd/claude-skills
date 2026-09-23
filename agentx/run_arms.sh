#!/usr/bin/env bash
# Run full 3600 s arms back to back, with VRAM sampling and OOR detection.
# One verdict line per arm. Lets the launcher do its OWN reclaim on success
# (killing it mid-cleanup leaves VRAM attributed to no process -- 2026-08-30).
SK=/workspace/claude-skills/agentx
run_arm(){
  NAME=$1; C=$2
  DIR=/workspace/results/$NAME; LOG=/workspace/results/$NAME.log
  bash $SK/wait_vram.sh >/dev/null || { echo "$NAME: VRAM never freed"; return 1; }
  mkdir -p $DIR
  setsid nohup bash $SK/vram_sampler.sh $DIR/vram.csv 30 >/dev/null 2>&1 &
  cd /workspace
  setsid nohup env EP_SIZE=1 CONC=$C DURATION=3600 CHUNK_PER_RANK=16384 \
    RESULT_DIR=$DIR bash $SK/agentx_b200align.sh > $LOG 2>&1 </dev/null &
  echo "$NAME: launched (conc=$C, no fusion)"
  t=0; verdict="TIMEOUT after ${t}s"
  while [ $t -lt 9000 ]; do
    sleep 30; t=$((t+30))
    if ls $DIR/*_c${C}.json >/dev/null 2>&1; then verdict="DONE at ${t}s"; break; fi
    if grep -qE "HSA_STATUS_ERROR|Fatal Python error" $DIR/server.log 2>/dev/null; then
      verdict="CRASHED OOR at ${t}s ($(grep -oE 'Available Free mem : [0-9]+ MB' $DIR/server.log | tail -1), peak VRAM $(awk -F, 'NR>1{for(i=3;i<=10;i++) if($i>m) m=$i} END{print m}' $DIR/vram.csv)%)"; break
    fi
  done
  echo "$NAME: $verdict"
  case "$verdict" in
    DONE*) w=0; while pgrep -f "b200align_mt[p]" >/dev/null && [ $w -lt 1200 ]; do sleep 20; w=$((w+20)); done
           echo "$NAME: launcher finished its own cleanup (${w}s)" ;;
    *)     P=$(pgrep -f 'sglang|[a]iperf|b200align_mt[p]' | tr '\n' ' '); kill -9 $P 2>/dev/null ;;
  esac
  pkill -f "vram_sample[r].sh" 2>/dev/null
  return 0
}
run_arm c128-chunk16384-newmain   128
run_arm c64-chunk16384-newmain-rep 64
echo "ARMS DONE"
