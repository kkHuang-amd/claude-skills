#!/usr/bin/env bash
# Start ONE rank of the 2-node proxy-MoE SGLang server (TP8 across two GB300 trays, MNNVL) inside
# the container on this node, then return immediately (server keeps running under setsid).
# Needs the Quark-MXFP4-on-CUDA SGLang patch (../patches/sglang_quark_mxfp4_cuda.patch).
#
# Required: CONC NODE_RANK DIST_INIT (node0_ip:port, identical on both nodes).
# Optional (defaults = TP8 EP1, 24 layers):
#   MODEL (/local/proxy_24l = node-local NVMe copy)  TP (8)  EP (1)  NNODES (2)  PORT (30000)
#   IFNAME (bond0.113)  CONTEXT_LEN (73728)  CHUNK (8192)  MEM (0.85)  KV (fp8_e4m3)
#   ARF (1 = --enable-flashinfer-allreduce-fusion)  MOE_BACKEND (flashinfer_mxfp4)
#   SERVER_LOG  SERVER_PID_FILE  EXTRA_ARGS  EXTRA_ENV (space-separated K=V, e.g. "NCCL_DEBUG=INFO")
set -uo pipefail
: "${CONC:?CONC is required}" "${NODE_RANK:?NODE_RANK is required}" "${DIST_INIT:?DIST_INIT is required}"
MODEL=${MODEL:-/local/proxy_24l}
TP=${TP:-8}; EP=${EP:-1}; NNODES=${NNODES:-2}; PORT=${PORT:-30000}; IFNAME=${IFNAME:-bond0.113}
# 0.88 (the 1-tray value) OOMs in prefill CUDA graph capture at TP8 across 2 trays.
CONTEXT_LEN=${CONTEXT_LEN:-73728}; CHUNK=${CHUNK:-8192}; MEM=${MEM:-0.85}; KV=${KV:-fp8_e4m3}
ARF=${ARF:-1}; MOE_BACKEND=${MOE_BACKEND:-flashinfer_mxfp4}
SERVER_LOG=${SERVER_LOG:-/tmp/proxy_server_n$NODE_RANK.log}
SERVER_PID_FILE=${SERVER_PID_FILE:-/tmp/proxy_server_n$NODE_RANK.pid}
EXTRA_ARGS=${EXTRA_ARGS:-}; EXTRA_ENV=${EXTRA_ENV:-}

export NCCL_SOCKET_IFNAME=$IFNAME GLOO_SOCKET_IFNAME=$IFNAME
# shellcheck disable=SC2086
[ -n "$EXTRA_ENV" ] && export $EXTRA_ENV

ARGS=(--model-path "$MODEL" --tp-size "$TP" --ep-size "$EP" --kv-cache-dtype "$KV"
      --context-length "$CONTEXT_LEN" --mem-fraction-static "$MEM"
      --max-running-requests "$CONC" --cuda-graph-max-bs-decode "$CONC"
      --chunked-prefill-size "$CHUNK" --moe-runner-backend "$MOE_BACKEND"
      --nnodes "$NNODES" --node-rank "$NODE_RANK" --dist-init-addr "$DIST_INIT"
      --watchdog-timeout 3600 --host 127.0.0.1 --port "$PORT")
[ "$ARF" = 1 ] && ARGS+=(--enable-flashinfer-allreduce-fusion)
# shellcheck disable=SC2206
[ -n "$EXTRA_ARGS" ] && ARGS+=($EXTRA_ARGS)

# /sgl-workspace shadows the sglang package: launch from /tmp.
cd /tmp
{ echo "host: $(hostname)  env: NCCL_SOCKET_IFNAME=$IFNAME $EXTRA_ENV"
  echo "launch: python3 -m sglang.launch_server ${ARGS[*]}"; } > "$SERVER_LOG"
setsid nohup python3 -m sglang.launch_server "${ARGS[@]}" >> "$SERVER_LOG" 2>&1 < /dev/null &
echo $! > "$SERVER_PID_FILE"
echo "rank $NODE_RANK pid $! (log $SERVER_LOG)"
