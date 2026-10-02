#!/usr/bin/env bash
# Turn one InferenceX CI job log into a runnable point definition:
#   points/<name>/server_env    KEY=VAL per line   (srtctl "Env:" line)
#   points/<name>/server_args   one argv token per line (sglang.launch_server)
#   points/<name>/router_args   one token per line, only if CI started sglang-router
# client_env is NOT derivable from the log alone (recipe benchmark.env + CI job
# env); write it by hand -- see REGRESSION_RUNBOOK.md §5.
#
# usage: extract_point.sh <name> <job_log_file>
#   job log: gh api repos/SemiAnalysisAI/InferenceX/actions/jobs/<id>/logs > f
set -euo pipefail
name=$1 log=$2
out="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/points/${POINT_SET:?set POINT_SET, e.g. ci<run id>}/$name"
mkdir -p "$out"

# Cluster plumbing (etcd/dynamo/head node) has no meaning on a single node.
grep -m1 '\[INFO\] Env: ' "$log" | sed 's/^.*Env: //' | tr ' ' '\n' \
  | grep -vE '^(DYN_|ETCD_|HEAD_NODE_IP=|NATS_)' | grep . > "$out/server_env"

grep -m1 'Command: python3 -m sglang.launch_server' "$log" \
  | sed 's/^.*sglang\.launch_server //' | tr ' ' '\n' | grep . > "$out/server_args"

if grep -q 'Starting sglang-router 0' "$log"; then
  # CI points the router at the worker's node IP; single node uses localhost.
  grep -m1 'Starting sglang-router 0' "$log" | sed 's/^.*launch_router //' | tr ' ' '\n' \
    | grep . | sed -E 's#http://[0-9.]+:#http://127.0.0.1:#' > "$out/router_args"
fi
echo "$name: env=$(wc -l < "$out/server_env") args=$(wc -l < "$out/server_args") router=$([ -f "$out/router_args" ] && wc -l < "$out/router_args" || echo none)"
