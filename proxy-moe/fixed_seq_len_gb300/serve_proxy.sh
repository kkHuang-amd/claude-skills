#!/usr/bin/env bash
# Launch one proxy-MoE SGLang server on GB300 for a fixed_seq_len point, wait until ready, exit 0.
# Writes $SERVER_PID_FILE; does NOT run a client and does NOT stop the server (kill by PID).
# Needs the Quark-MXFP4-on-CUDA SGLang patch (../patches/sglang_quark_mxfp4_cuda.patch).
#
# Required: CONC.  Optional (defaults = TP4 EP1, 12 layers on GPUs 0-3):
#   MODEL (/sgl-workspace/proxy_moe_work/proxy_12l)  TP (4)  EP (1)  PORT (30000)
#   CONTEXT_LEN (73728)  CHUNK (8192)  MEM (0.88)  KV (fp8_e4m3)
#   ARF (1 = --enable-flashinfer-allreduce-fusion)  MOE_BACKEND (flashinfer_mxfp4)
#   GPUS (CUDA_VISIBLE_DEVICES)  SERVER_LOG  SERVER_PID_FILE  READY_TIMEOUT (1800 s)  EXTRA_ARGS
set -uo pipefail
: "${CONC:?CONC is required (bound to --max-running-requests and --cuda-graph-max-bs-decode)}"
MODEL=${MODEL:-/sgl-workspace/proxy_moe_work/proxy_12l}
TP=${TP:-4}; EP=${EP:-1}; PORT=${PORT:-30000}
CONTEXT_LEN=${CONTEXT_LEN:-73728}; CHUNK=${CHUNK:-8192}; MEM=${MEM:-0.88}; KV=${KV:-fp8_e4m3}
ARF=${ARF:-1}; MOE_BACKEND=${MOE_BACKEND:-flashinfer_mxfp4}
SERVER_LOG=${SERVER_LOG:-/tmp/proxy_server.log}; SERVER_PID_FILE=${SERVER_PID_FILE:-/tmp/proxy_server.pid}
READY_TIMEOUT=${READY_TIMEOUT:-1800}; EXTRA_ARGS=${EXTRA_ARGS:-}

[ -n "${GPUS:-}" ] && export CUDA_VISIBLE_DEVICES=$GPUS

ARGS=(--model-path "$MODEL" --tp-size "$TP" --ep-size "$EP" --kv-cache-dtype "$KV"
      --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM"
      --max-running-requests "$CONC" --cuda-graph-max-bs-decode "$CONC"
      --chunked-prefill-size "$CHUNK" --moe-runner-backend "$MOE_BACKEND"
      --watchdog-timeout 3600 --host 127.0.0.1 --port "$PORT")
[ "$ARF" = 1 ] && ARGS+=(--enable-flashinfer-allreduce-fusion)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

# /sgl-workspace shadows the sglang package: launch from /tmp.
cd /tmp
echo "launch: python3 -m sglang.launch_server ${ARGS[*]}" > "$SERVER_LOG"
python3 -m sglang.launch_server "${ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
SPID=$!
echo "$SPID" > "$SERVER_PID_FILE"
echo "server pid $SPID (log $SERVER_LOG)"

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health_generate" > /dev/null; do
  if ! kill -0 "$SPID" 2> /dev/null; then echo "server $SPID exited"; tail -n 20 "$SERVER_LOG" | cut -c1-300; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt "$READY_TIMEOUT" ]; then echo "not ready after ${READY_TIMEOUT}s"; exit 1; fi
  sleep 5
done
# A stale server from an earlier point answering on the same port must not count as ready.
owner=$(ss -ltnpH "sport = :$PORT" 2> /dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "$owner" ] && ! { [ "$owner" = "$SPID" ] || pgrep -P "$SPID" | grep -qx "$owner"; }; then
  echo "port $PORT is served by pid $owner, not by our server $SPID"; exit 1
fi
echo "ready after $(( $(date +%s) - t0 ))s"
