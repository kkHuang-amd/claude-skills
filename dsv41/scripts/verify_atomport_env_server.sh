#!/usr/bin/env bash
# Functional check of an environment built by setup_atomport_env.sh in a separate directory: wait for WAIT_PID
# (e.g. a running series) to exit, start an EVAL_ONLY CONC=32 best-config server from that copy (first start
# JIT-builds its aiter modules), run GSM8K 1319 once, stop the server (incl. sglang:: children).
#   V=/sgl-workspace/verify_ap WAIT_PID=<pid> bash verify_atomport_env_server.sh
# Output: /shared_nfs/kk/dsv41/atomport/verify_env/summary.txt (+ server under agentx/verify_atomport_env/).
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); V=${V:-/sgl-workspace/verify_ap}
O=/shared_nfs/kk/dsv41/atomport/verify_env; A=/shared_nfs/kk/dsv41/agentx; mkdir -p $O; S=$O/summary.txt
say(){ echo "[$(date +%F' '%T)] $*" >> $S; }
stop(){ for p in $(ps -eo pid,comm,args | awk '$2=="python3" && /sglang.launch_server/{print $1}'); do kill -9 $p; done
        for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 $p; done; sleep 20; }
if [ -n "${WAIT_PID:-}" ]; then say "waiting for PID $WAIT_PID"; while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done; fi
stop; t0=$(date +%s); say "start server from $V"
(cd $A && PYTHONPATH=$V/pydeps-flydsl-0341:$V/aiter-5750:/sgl-workspace/mori SRC=$V/sglang-rolao-opt/python \
  SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" \
  EVAL_ONLY=true SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 CONC=32 PREFILL_DECODE_INTERVAL=16 TAG=verify_atomport_env \
  setsid nohup bash $D/agentx_colleague_run.sh > $A/verify_atomport_env.nohup 2>&1 < /dev/null &)
for i in $(seq 1 360); do grep -qE 'SERVER_ONLY: ready|Traceback|Initialization failed' $A/verify_atomport_env.nohup 2>/dev/null && break; sleep 20; done
if ! grep -q 'SERVER_ONLY: ready' $A/verify_atomport_env.nohup; then
  say "FAILED server start: $(grep -E 'Error|Traceback' $A/verify_atomport_env/server.log 2>/dev/null | tail -2 | cut -c1-200)"; stop; exit 1; fi
say "ready after $(( $(date +%s) - t0 )) s; aiter from: $(grep -oE 'import \[module_aiter_core\] under [^ ]+' $A/verify_atomport_env/server.log | head -1)"
say "untuned 385/129 warnings: $(grep -cE "no tuned FlyDSL config for \('gfx950', 256, [0-9]+, 5120, 1152, (385|129)" $A/verify_atomport_env/server.log)"
SRC=$V/sglang-rolao-opt/python PORT=8888 TAG=verify_atomport_env bash $D/run_gsm8k.sh >> $S 2>&1
stop; say "DONE"
