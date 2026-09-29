#!/usr/bin/env bash
# Follow-up to atomport_draft_raw_ab.sh: (1) knob=1 + SGLANG_DEBUG_DSPARK_RAW_PARITY=1 on a real-acceptance c1 server,
# run decode traffic, grep the in-graph vs eager metadata parity log; (2) knob=0 simulated c1 baseline, same proxy
# (3 reps OSL 1024 + OSL 8192) as the knob=1 run. Output: /shared_nfs/kk/dsv41/atomport/draft_raw_ab/summary4.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); O=/shared_nfs/kk/dsv41/atomport/draft_raw_ab; A=/shared_nfs/kk/dsv41/agentx
S=$O/summary4.txt; say(){ echo "[$(date +%T)] $*" >> $S; }
export PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori
export SRC=/sgl-workspace/sglang-rolao-opt/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1
export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 CONC=1 PREFILL_DECODE_INTERVAL=16
kill_servers(){ for p in $(ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'); do kill -9 $p; done; sleep 20; }
start(){ local tag=$1; shift
  (cd $A && env "$@" TAG=$tag setsid nohup bash $D/agentx_colleague_run.sh > $A/$tag.nohup 2>&1 < /dev/null &)
  for i in $(seq 1 90); do grep -qE 'SERVER_ONLY: ready|Traceback|Initialization failed' $A/$tag.nohup 2>/dev/null && break; sleep 20; done
  grep -qE 'SERVER_ONLY: ready' $A/$tag.nohup && say "ready $tag" || { say "FAILED start $tag"; return 1; }; }
kill_servers
start draftraw_parity3 EVAL_ONLY=true SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA=1 SGLANG_DEBUG_DSPARK_RAW_PARITY=1 || exit 1
python3 $D/greedy_parity_dump.py dump --out $O/parity.json --n 8 >> $S 2>&1
python3 $D/atomport_proxy_bench.py --port 8888 --ctx 8192 --osl 512 --conc 1 --repeat 1 --tag parity_ctx8k >> $S 2>&1
grep -E 'DSPARK_RAW_PARITY' $A/draftraw_parity3/server.log | tail -n 5 >> $S
say "parity mismatch lines: $(grep -E 'DSPARK_RAW_PARITY.*mismatch=\[.' $A/draftraw_parity3/server.log | wc -l)"
SRC=/sgl-workspace/sglang-rolao-opt/python PORT=8888 TAG=draftraw_parity_gsm8k bash $D/run_gsm8k.sh >> $S 2>&1
kill_servers
say "DONE"
