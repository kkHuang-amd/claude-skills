#!/usr/bin/env bash
# Launch one proxy-MoE SGLang server for a fixed_seq_len point, wait until ready, exit 0.
# Writes $SERVER_PID_FILE; does NOT run a client and does NOT stop the server (kill by PID).
#
# Required: CONC.  Optional (defaults = TP8 EP1 best config on the pushed branches):
#   MODEL (/data/proxy_moe/proxy_24l)  TP (8)  EP (1)  PORT (30000)  CONTEXT_LEN (73728)
#   CHUNK (8192)  MEM (0.88)  KV (fp8_e4m3)  ARF (1 = --enable-aiter-allreduce-fusion)
#   ENV_FILE (/data/proxy_moe/newbase_env.sh)  ROUTER_BF16X3 (1)  GPUS (ROCR_VISIBLE_DEVICES)
#   SERVER_LOG  SERVER_PID_FILE  READY_TIMEOUT (1800 s)  EXTRA_ARGS
set -uo pipefail
: "${CONC:?CONC is required (bound to --max-running-requests and --cuda-graph-max-bs-decode)}"
MODEL=${MODEL:-/data/proxy_moe/proxy_24l}
TP=${TP:-8}; EP=${EP:-1}; PORT=${PORT:-30000}
CONTEXT_LEN=${CONTEXT_LEN:-73728}; CHUNK=${CHUNK:-8192}; MEM=${MEM:-0.88}; KV=${KV:-fp8_e4m3}
ARF=${ARF:-1}; ROUTER_BF16X3=${ROUTER_BF16X3:-1}
ENV_FILE=${ENV_FILE:-/data/proxy_moe/newbase_env.sh}
SERVER_LOG=${SERVER_LOG:-/tmp/proxy_server.log}; SERVER_PID_FILE=${SERVER_PID_FILE:-/tmp/proxy_server.pid}
READY_TIMEOUT=${READY_TIMEOUT:-1800}; EXTRA_ARGS=${EXTRA_ARGS:-}

# shellcheck disable=SC1090
[ -n "$ENV_FILE" ] && source "$ENV_FILE"
export SGLANG_USE_AITER=1
[ "$ROUTER_BF16X3" = 1 ] && export SGLANG_OPT_LAGUNA_ROUTER_BF16X3=1 || unset SGLANG_OPT_LAGUNA_ROUTER_BF16X3
[ -n "${GPUS:-}" ] && export ROCR_VISIBLE_DEVICES=$GPUS

ARGS=(--model-path "$MODEL" --tp-size "$TP" --ep-size "$EP" --kv-cache-dtype "$KV"
      --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM"
      --max-running-requests "$CONC" --cuda-graph-max-bs-decode "$CONC"
      --chunked-prefill-size "$CHUNK" --watchdog-timeout 3600
      --host 127.0.0.1 --port "$PORT")
[ "$ARF" = 1 ] && ARGS+=(--enable-aiter-allreduce-fusion)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

# The image WORKDIR (/sgl-workspace) shadows the sglang package: launch from /tmp.
cd /tmp
echo "launch: python3 -m sglang.launch_server ${ARGS[*]}" > "$SERVER_LOG"
python3 -m sglang.launch_server "${ARGS[@]}" >> "$SERVER_LOG" 2>&1 &
SPID=$!
echo "$SPID" > "$SERVER_PID_FILE"
echo "server pid $SPID (log $SERVER_LOG)"

t0=$(date +%s)
until curl -sf "http://127.0.0.1:$PORT/health_generate" > /dev/null; do
  if ! kill -0 "$SPID" 2> /dev/null; then echo "server $SPID exited"; tail -n 20 "$SERVER_LOG"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt "$READY_TIMEOUT" ]; then echo "not ready after ${READY_TIMEOUT}s"; exit 1; fi
  sleep 5
done
# A stale server from an earlier point answering on the same port must not count as ready.
owner=$(ss -ltnpH "sport = :$PORT" 2> /dev/null | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2)
if [ -n "$owner" ] && ! { [ "$owner" = "$SPID" ] || pgrep -P "$SPID" | grep -qx "$owner"; }; then
  echo "port $PORT is served by pid $owner, not by our server $SPID"; exit 1
fi
echo "ready after $(( $(date +%s) - t0 ))s"
