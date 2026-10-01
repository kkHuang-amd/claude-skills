#!/usr/bin/env bash
# Original DSV4.1 (sglang 3e8187fa88) on the image aiter (acf8fdf93 + its 2 local mods, copy in /sgl-workspace/aiter-image),
# image flydsl 0.3.2; colleague recipe as-is (no EXTRA_ARGS, no OPUS prefill, chunk 4096). GPUs 4,5.
#
#   1) EVAL_ONLY CONC=32 server: GSM8K 1319 x3 (reported, no fallback)
#   2) AgentX c1, c2 (PDI 16), PREFIX orig41
# Output: /shared_nfs/kk/dsv41/atomport/orig41/summary.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); O=/shared_nfs/kk/dsv41/atomport/orig41; A=/shared_nfs/kk/dsv41/agentx
AITER=/sgl-workspace/aiter-image; SGL=/sgl-workspace/sglang-orig41/python
mkdir -p $O; S=$O/summary.txt; say(){ echo "[$(date +%T)] $*" >> $S; }
export PYTHONPATH=$AITER:/sgl-workspace/mori
export EXTRA_ARGS=""
export OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 SRC=$SGL

kill_servers(){
  for p in $(ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'); do kill -9 $p 2>/dev/null; done
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 60); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{print int(s/1e9)}'); [ "$u" -lt 10 ] && return; sleep 5
  done
  say "WARN vram still ${u}G after kill"
}
start(){ # $1 tag, rest env
  local tag=$1; shift
  (cd $A && env "$@" SERVER_ONLY=1 TAG=$tag setsid nohup bash $D/agentx_colleague_run.sh > $A/$tag.nohup 2>&1 < /dev/null &)
  for i in $(seq 1 120); do grep -qE 'SERVER_ONLY: ready|Traceback|Initialization failed' $A/$tag.nohup 2>/dev/null && break; sleep 20; done
  grep -qE 'SERVER_ONLY: ready' $A/$tag.nohup && say "ready $tag" || { say "FAILED start $tag (see $A/$tag.nohup)"; return 1; }
}
gsm8k_x3(){ # $1 tag -> prints mean
  local tag=$1 accs=""
  for r in 1 2 3; do
    SRC=$SGL PORT=8888 TAG=${tag}_r$r bash $D/run_gsm8k.sh >> $S 2>&1
    a=$(grep -oE 'Accuracy: [0-9.]+' /shared_nfs/kk/dsv41/gsm8k_${tag}_r$r.log | tail -1 | awk '{print $2}'); accs="$accs ${a:-0}"
  done
  echo $accs | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.4f", s/NF}'
}

say "START aiter $(git -C $AITER rev-parse --short HEAD) sglang $(git -C $SGL/.. rev-parse --short HEAD)"
kill_servers
start orig41_eval EVAL_ONLY=true CONC=32 || { kill_servers; say "STOP: server failed"; exit 1; }
mean=$(gsm8k_x3 orig41); say "GSM8K mean $mean (base scope)"; kill_servers
prefix=orig41
say "AgentX c1,c2 start ($prefix)"
SCRIPT=agentx_colleague_run.sh RUNS="1:16 2:16" PREFIX=$prefix bash $D/agentx_series.sh
say "DONE"
