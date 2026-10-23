#!/usr/bin/env bash
# Catch the intermittent warmup hang (NCCL_CUMEM_ENABLE=1) and dump every sglang process's stack.
# Starts the server up to TRIES times; a start that is not ready within WAIT seconds counts as hung.
# Usage: mi455_hang_dump.sh [run-tag]   env: TRIES (6), WAIT (300), MAX_BS (64)
set -u
OUT=/shared_nfs/kk/results/proxy_12l/${1:-hang_dump_$(hostname -s)_$(date +%m%d_%H%M)}
MODEL=${MODEL:-/workspace/proxy_work/proxy_12l}
PORT=18030; MAX_BS=${MAX_BS:-64}
mkdir -p "$OUT"; cd /tmp
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 SGLANG_USE_AITER=1 PYTHONUNBUFFERED=1 NCCL_CUMEM_ENABLE=1

stop() {  # kill the server tree
  pkill -TERM -f "sglang.launch_server.*--port $PORT" 2>/dev/null; sleep 10
  pkill -9 -f "sglang" 2>/dev/null; sleep 5
}

for t in $(seq 1 ${TRIES:-6}); do
  python3 -m sglang.launch_server --model-path "$MODEL" \
    --tp-size 8 --ep-size 8 --kv-cache-dtype fp8_e4m3 \
    --context-length 32768 --mem-fraction-static 0.80 \
    --max-running-requests $MAX_BS --cuda-graph-max-bs-decode $MAX_BS \
    --chunked-prefill-size 8192 --attention-backend triton \
    --watchdog-timeout 3600 --host 127.0.0.1 --port $PORT > "$OUT/server_try$t.log" 2>&1 &
  SPID=$!
  ready=0
  for i in $(seq 1 $(( ${WAIT:-300} / 10 ))); do
    curl -sf -m 5 http://127.0.0.1:$PORT/health_generate >/dev/null && { ready=1; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 10
  done
  if [ $ready = 1 ]; then echo "try$t: ready after ~$((i*10))s, no hang"; stop; continue; fi
  kill -0 $SPID 2>/dev/null || { echo "try$t: server exited before ready (not a hang)"; grep -E 'Error|Traceback' "$OUT/server_try$t.log" | tail -3 | cut -c1-200; stop; continue; }
  echo "try$t: HUNG (not ready after ${WAIT:-300}s); dumping stacks"
  for p in $(pgrep -f 'sglang' | grep -v $$); do
    name=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-80)
    timeout 90 py-spy dump --native --pid $p > "$OUT/pyspy_try${t}_$p.txt" 2>&1
    echo "  dumped $p ($name): $(wc -l < "$OUT/pyspy_try${t}_$p.txt") lines"
  done
  if [ "${ROCGDB:-0}" = 1 ]; then  # which GPU kernels are still resident, per scheduler
    for p in $(pgrep -f 'sglang::scheduler'); do
      name=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-40)
      timeout 180 rocgdb -q -nx -batch -p $p -ex 'set pagination off' -ex 'info agents' \
        -ex 'info queues' -ex 'info dispatches' -ex 'info threads' \
        > "$OUT/rocgdb_try${t}_$p.txt" 2>&1
      echo "  rocgdb $p ($name): $(grep -c 'AMDGPU Wave' "$OUT/rocgdb_try${t}_$p.txt") waves, $(grep -cE '^\s*\*?\s*[0-9]+\s+\(' "$OUT/rocgdb_try${t}_$p.txt") dispatch lines"
    done
  fi
  stop
  exit 0
done
echo "no hang in ${TRIES:-6} tries"
