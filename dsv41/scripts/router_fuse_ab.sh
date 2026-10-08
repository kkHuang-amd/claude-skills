#!/usr/bin/env bash
# Router fusion (rocm_router_gate + shared append) validation, GPUs 4,5, all sequential:
#   1) base tree SIM c1 (chunk 16384, PDI 16): proxy temp 1.0 x3 + GPU-only profile
#   2) fused tree with SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY=0 (router fusion only), same
#   2b) fused tree default policy 2 (router fusion + multi-phase moe_sorting), same; step spans of all three
#   3) fused tree (both) EVAL_ONLY CONC=32: GSM8K 1319 x3 (scripts/run_gsm8k.sh)
#   4) fused tree (both) AgentX c1, c2 (PDI 16, chunk 16384) via agentx_series.sh, PREFIX rf_c16k
# Output: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/router_fuse/summary.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); O=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/atomport/router_fuse; A=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
mkdir -p $O; S=$O/summary.txt; say(){ echo "[$(date +%T)] $*" >> $S; }
BASE=/sgl-workspace/sglang-rolao-opt/python; FUSE=/sgl-workspace/sglang-router-fuse/python
export PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori
export SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
export OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 CHUNKED_PREFILL_SIZE=16384

kill_servers(){
  for p in $(ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'); do kill -9 $p 2>/dev/null; done
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 $p 2>/dev/null; done
  for i in $(seq 1 60); do
    u=$(rocm-smi --showmeminfo vram 2>/dev/null | awk '/Used/{s+=$NF} END{print int(s/1e9)}'); [ "$u" -lt 10 ] && return; sleep 5
  done
  say "WARN vram still ${u}G after kill"
}
start(){ # $1 tag, $2 src, rest env
  local tag=$1 src=$2; shift 2
  (cd $A && env "$@" SRC=$src SERVER_ONLY=1 TAG=$tag setsid nohup bash $D/agentx_colleague_run.sh > $A/$tag.nohup 2>&1 < /dev/null &)
  for i in $(seq 1 90); do grep -qE 'SERVER_ONLY: ready|Traceback|Initialization failed' $A/$tag.nohup 2>/dev/null && break; sleep 20; done
  grep -qE 'SERVER_ONLY: ready' $A/$tag.nohup && say "ready $tag" || { say "FAILED start $tag"; return 1; }
}
proxy_and_prof(){ # $1 label
  python3 $D/atomport_proxy_bench.py --port 8888 --ctx 2048 --osl 1024 --conc 1 --repeat 3 --temperature 1.0 --tag rf_$1 >> $S 2>&1
  python3 $D/atomport_proxy_bench.py --port 8888 --ctx 2048 --conc 1 --repeat 1 --temperature 1.0 --tag rf_prof_$1 --profile-dir $O/prof_$1 >> $S 2>&1
}

say "START"
kill_servers
start rf_sim_base $BASE CONC=1 PREFILL_DECODE_INTERVAL=16 && proxy_and_prof base
kill_servers
start rf_sim_fuse $FUSE CONC=1 PREFILL_DECODE_INTERVAL=16 SGLANG_AITER_MOE_SORTING_DISPATCH_POLICY=0 && proxy_and_prof fuse
kill_servers
start rf_sim_fusemp $FUSE CONC=1 PREFILL_DECODE_INTERVAL=16 && proxy_and_prof fusemp
kill_servers
python3 $D/atomport_step_spans.py "base=$O/prof_base/*TP-0*" "fuse=$O/prof_fuse/*TP-0*" "fuse+sortMP=$O/prof_fusemp/*TP-0*" >> $S 2>&1
if start rf_eval_fuse $FUSE EVAL_ONLY=true CONC=32; then
  for r in 1 2 3; do SRC=$FUSE PORT=8888 TAG=rf_fuse_r$r bash $D/run_gsm8k.sh >> $S 2>&1; grep -oE 'Accuracy: [0-9.]+' /shared_nfs/kk/results/DeepSeek-V4.1-Flash/gsm8k_rf_fuse_r$r.log | tail -1 | sed "s/^/gsm8k r$r /" >> $S; done
fi
kill_servers
say "AgentX c1,c2 start"
SRC=$FUSE SCRIPT=agentx_colleague_run.sh RUNS="1:16 2:16" PREFIX=rf_c16k bash $D/agentx_series.sh
say "DONE"
