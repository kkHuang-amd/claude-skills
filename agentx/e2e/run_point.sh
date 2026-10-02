#!/usr/bin/env bash
# Single-node replay of one InferenceX CI AgentX point (DSV4-Pro, MI355X).
# The point is defined by $POINTS_DIR/<POINT>/ (built by extract_point.sh from
# the CI job log + a hand-written client_env); the client is srt_agentic.sh, as
# CI runs it. See README.md.
#
# usage: POINT=c4 RESULT_DIR=/workspace/results/<name> bash run_point.sh
#   POINT_SET=ci36401947630 (default) or POINTS_DIR=<dir> picks the point set
#   SMOKE=1  300 s plumbing check (1 warmup/lane, submission_valid=false)
# Launch through matrix.sh (SKIP_SMOKE=1 POINTS=<pt> for one point): it runs a
# snapshot, so editing this file under a live run is safe.
set -uo pipefail
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${POINT:?set POINT (dir under the point set)}" "${RESULT_DIR:?set RESULT_DIR}"
P="${POINTS_DIR:-$SKILL_DIR/points/${POINT_SET:-ci36401947630}}/$POINT"
[[ -f $P/server_args && -f $P/server_env && -f $P/client_env ]] || { echo "incomplete point $P"; exit 2; }

source "$SKILL_DIR/env.sh"
source "$INFMAX_CONTAINER_WORKSPACE/benchmarks/runtime_settings.sh"
export MODEL=deepseek-ai/DeepSeek-V4-Pro-0813
mkdir -p "$RESULT_DIR"
cp -r "$P" "$RESULT_DIR/point"
bash "$SKILL_DIR/stack_fingerprint.sh" > "$RESULT_DIR/stack.txt" 2>&1

# ---- server ----
mapfile -t SERVER_ENV < "$P/server_env"
mapfile -t SERVER_ARGS < "$P/server_args"
for i in "${!SERVER_ARGS[@]}"; do       # CI loads by HF id; use the local copy
  [[ ${SERVER_ARGS[$i]} == --model-path ]] && SERVER_ARGS[$((i + 1))]="$MODEL_DIR"
  [[ ${SERVER_ARGS[$i]} == --port ]] && SERVER_PORT=${SERVER_ARGS[$((i + 1))]}
done
echo "python3 -m sglang.launch_server ${SERVER_ARGS[*]}" > "$RESULT_DIR/sglang_command.txt"
env "${SERVER_ENV[@]}" python3 -m sglang.launch_server "${SERVER_ARGS[@]}" \
  > "$RESULT_DIR/server.log" 2>&1 &
PIDS=($!)
echo "server pid=${PIDS[0]} port=$SERVER_PORT log=$RESULT_DIR/server.log"
trap 'kill "${PIDS[@]}" 2>/dev/null' EXIT

wait_health() {  # url pid
  until curl -sf -m 5 "$1/health" >/dev/null; do
    kill -0 "$2" 2>/dev/null || { echo "process $2 died before $1 was healthy"; exit 1; }
    sleep 10
  done
}
wait_health "http://127.0.0.1:$SERVER_PORT" "${PIDS[0]}"
echo "server ready $(date +%T)"

# ---- optional router (CI frontend.type=sglang-router) ----
export PORT=$SERVER_PORT
if [[ -f $P/router_args ]]; then
  mapfile -t ROUTER_ARGS < "$P/router_args"
  for i in "${!ROUTER_ARGS[@]}"; do
    [[ ${ROUTER_ARGS[$i]} == --port ]] && PORT=${ROUTER_ARGS[$((i + 1))]}
  done
  python3 -m sglang_router.launch_router "${ROUTER_ARGS[@]}" > "$RESULT_DIR/router.log" 2>&1 &
  PIDS+=($!)
  echo "router pid=${PIDS[1]} port=$PORT"
  # Router is started with --disable-health-check; poll its models endpoint.
  until curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" >/dev/null; do
    kill -0 "${PIDS[1]}" 2>/dev/null || { echo "router died"; exit 1; }
    sleep 5
  done
  echo "router ready $(date +%T)"
fi

# ---- client ----
set -a; source "$P/client_env"; set +a      # after runtime_settings: recipe wins
export DURATION="${DURATION:-3600}" PORT
export SRT_AGG_ENDPOINTS="127.0.0.1:$SERVER_PORT"   # engine metrics, not the router's
export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
if [[ "${SMOKE:-0}" == 1 ]]; then
  export DURATION=300 AIPERF_WARMUP_REQUESTS_PER_LANE=1 AIPERF_UNSAFE_OVERRIDE=true
fi
bash "$INFMAX_CONTAINER_WORKSPACE/benchmarks/srt_agentic.sh" > "$RESULT_DIR/benchmark.log" 2>&1
rc=$?
echo "client exit=$rc"
exit $rc
