#!/usr/bin/env bash
# A/B for SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA (DSpark draft metadata built inside the draft CUDA graph).
#   1) EVAL_ONLY c1 server, knob=1: greedy dump twice (run-to-run determinism)
#   2) EVAL_ONLY c1 server, knob=0: greedy dump once -> cmp vs 1)
#   3) simulated-acceptance c1 server, knob=1: proxy 3 reps + py-spy 10 s + GPU span profile
# Output: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/draft_raw_ab/ ; progress/verdict lines in summary.txt there.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); O=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/draft_raw_ab; A=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
mkdir -p $O; S=$O/summary.txt; say(){ echo "[$(date +%T)] $*" >> $S; }
export PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori
export SRC=/sgl-workspace/sglang-rolao-opt/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1
export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 CONC=1 PREFILL_DECODE_INTERVAL=16
kill_servers(){ for p in $(ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'); do kill -9 $p; done; sleep 20; }
start(){ # $1 tag, rest env
  local tag=$1; shift
  (cd $A && env "$@" TAG=$tag setsid nohup bash $D/agentx_colleague_run.sh > $A/$tag.nohup 2>&1 < /dev/null &)
  for i in $(seq 1 90); do grep -qE 'SERVER_ONLY: ready|Traceback|Initialization failed' $A/$tag.nohup 2>/dev/null && break; sleep 20; done
  grep -qE 'SERVER_ONLY: ready' $A/$tag.nohup && say "ready $tag" || { say "FAILED start $tag"; return 1; }
}
kill_servers
start draftraw_eval_on EVAL_ONLY=true SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA=1 || exit 1
python3 $D/greedy_parity_dump.py dump --out $O/on1.json >> $S 2>&1
python3 $D/greedy_parity_dump.py dump --out $O/on2.json >> $S 2>&1
python3 $D/greedy_parity_dump.py cmp $O/on1.json $O/on2.json 2>&1 | sed 's/^/on1 vs on2: /' >> $S
kill_servers
start draftraw_eval_off EVAL_ONLY=true SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA=0 || exit 1
python3 $D/greedy_parity_dump.py dump --out $O/off.json >> $S 2>&1
python3 $D/greedy_parity_dump.py cmp $O/on1.json $O/off.json 2>&1 | sed 's/^/on1 vs off: /' >> $S
kill_servers
start draftraw_c1_on SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA=1 || exit 1
python3 $D/atomport_proxy_bench.py --port 8888 --ctx 2048 --osl 1024 --conc 1 --repeat 3 --tag draftraw_on >> $S 2>&1
PID=$(ps -eo pid,args | grep 'sglang::scheduler_TP0' | grep -v grep | awk '{print $1}' | head -1)
(python3 $D/atomport_proxy_bench.py --port 8888 --ctx 2048 --osl 8192 --conc 1 --repeat 1 --tag draftraw_on_pyspy > $O/pyspy_bench.log 2>&1 &)
sleep 6; py-spy record --nonblocking -r 1000 -d 10 --format raw -o $O/pyspy_on.txt --pid $PID > /dev/null 2>&1; sleep 15
tail -n 1 $O/pyspy_bench.log >> $S
python3 $D/atomport_proxy_bench.py --port 8888 --ctx 2048 --conc 1 --repeat 1 --tag prof_draftraw_on --profile-dir $O/prof_on >> $S 2>&1
python3 $D/atomport_step_spans.py "tuned=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/sef_tuned_prof/*TP-0*" "raw_on=$O/prof_on/*TP-0*" >> $S 2>&1
kill_servers
say "DONE"
