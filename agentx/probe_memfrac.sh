#!/usr/bin/env bash
# Short fusion probe at a given MEM_FRACTION_STATIC. Prints exactly one verdict
# line: SURVIVED (reached the profiling phase) or CRASHED (HSA OOR during warmup).
# Both fusion failures so far happened inside warmup, so warmup is the gate.
SK=/workspace/claude-skills/agentx
probe(){
  F=$1; NAME="probe-fusion-mf${F/./}"; DIR=/workspace/results/$NAME; LOG=/workspace/results/$NAME.log; SRV=$DIR/server.log
  bash $SK/wait_vram.sh >/dev/null || { echo "$F: VRAM never freed, aborting"; return 1; }
  mkdir -p $DIR
  setsid nohup bash $SK/vram_sampler.sh $DIR/vram.csv 10 >/dev/null 2>&1 &
  cd /workspace
  setsid nohup env EP_SIZE=1 CONC=64 DURATION=300 CHUNK_PER_RANK=16384 FUSE_SHARED_EXPERTS=1 \
    MEM_FRACTION_STATIC=$F RESULT_DIR=$DIR \
    bash $SK/agentx_b200align.sh > $LOG 2>&1 </dev/null &
  echo "mem-fraction $F: launched"
  t=0; verdict="TIMEOUT"
  while [ $t -lt 2700 ]; do
    sleep 15; t=$((t+15))
    if grep -qE "HSA_STATUS_ERROR|Fatal Python error" $SRV 2>/dev/null; then
      verdict="CRASHED after ${t}s ($(grep -oE 'Available Free mem : [0-9]+ MB' $SRV | tail -1))"; break
    fi
    # NB: "Initialized 2 phase(s): ['Warmup','Profiling']" appears at startup --
    # match the phase START, not the phase list, or every run reads as survived.
    if grep -qE "Credit phase start: profiling|realtime [0-9:]+ profiling" $LOG 2>/dev/null; then
      verdict="SURVIVED warmup -> profiling at ${t}s"; break
    fi
  done
  echo "mem-fraction $F: $verdict"
  P=$(pgrep -f 'sglang|[a]iperf|b200align_mt[p]|vram_sample[r]' | tr '\n' ' '); kill -9 $P 2>/dev/null
  case "$verdict" in SURVIVED*) return 0;; *) return 1;; esac
}
probe 0.85 || { echo "--- 0.85 failed, trying 0.80 ---"; probe 0.80; }
echo "LADDER DONE"
