#!/usr/bin/env bash
# Launch one proxy-MoE server on MI455 (gfx1250, h13-17) for a fixed_seq_len point, wait until ready, exit 0.
# Writes $SERVER_PID_FILE; does not run a client or stop the server (the driver kills by PID).
# Required: CONC.  Optional: MODEL TP EP PORT CONTEXT_LEN CHUNK MEM KV SERVER_LOG SERVER_PID_FILE
#   READY_TIMEOUT EXTRA_ARGS.  MoE/comm env comes from the caller (see sweep_mi455.sh recipe_env).
set -uo pipefail
: "${CONC:?CONC is required (bound to --max-running-requests and --cuda-graph-max-bs-decode)}"
MODEL=${MODEL:-/workspace/proxy_work/proxy_12l}
TP=${TP:-8}; EP=${EP:-1}; PORT=${PORT:-30000}
CONTEXT_LEN=${CONTEXT_LEN:-73728}; CHUNK=${CHUNK:-8192}; MEM=${MEM:-0.80}; KV=${KV:-fp8_e4m3}
SERVER_LOG=${SERVER_LOG:-/tmp/proxy_server.log}; SERVER_PID_FILE=${SERVER_PID_FILE:-/tmp/proxy_server.pid}
READY_TIMEOUT=${READY_TIMEOUT:-1800}; EXTRA_ARGS=${EXTRA_ARGS:-}
export SGLANG_USE_AITER=1 HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 PYTHONUNBUFFERED=1

ARGS=(--model-path "$MODEL" --tp-size "$TP" --ep-size "$EP" --kv-cache-dtype "$KV"
      --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM"
      --max-running-requests "$CONC" --cuda-graph-max-bs-decode "$CONC"
      --chunked-prefill-size "$CHUNK" --attention-backend triton --watchdog-timeout 3600
      --host 127.0.0.1 --port "$PORT")
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

# /sgl-workspace shadows the sglang package: launch from /tmp.
cd /tmp
echo "launch: python3 -m sglang.launch_server ${ARGS[*]}" > "$SERVER_LOG"
env | grep -E '^(NCCL_CUMEM_ENABLE|RCCL_DDA_ENABLE|SGLANG_OPT_|AITER_)' | sort >> "$SERVER_LOG"
python3 -m sglang.launch_server "${ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
SPID=$!
echo "$SPID" > "$SERVER_PID_FILE"
echo "server pid $SPID (log $SERVER_LOG)"
t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health_generate" > /dev/null; do
  if ! kill -0 "$SPID" 2> /dev/null; then echo "server $SPID exited"; tail -n 20 "$SERVER_LOG" | cut -c1-200; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt "$READY_TIMEOUT" ]; then echo "not ready after ${READY_TIMEOUT}s"; exit 1; fi
  sleep 5
done
owner=$(ss -ltnpH "sport = :$PORT" 2> /dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "$owner" ] && ! { [ "$owner" = "$SPID" ] || pgrep -P "$SPID" | grep -qx "$owner"; }; then
  echo "port $PORT is served by pid $owner, not by our server $SPID"; exit 1
fi
echo "ready after $(( $(date +%s) - t0 ))s"
