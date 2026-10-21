# Shared helpers for the 2-node proxy-MoE runs. Sourced by sweep_proxy_2node.sh / smoke_2node.sh,
# which run on the node0 HOST (not in a container): node0's container via `docker exec`, node1's via
# `ssh <node1> docker exec` (needs passwordless ssh node0 -> node1).
# Paths passed to the helpers are CONTAINER paths (/shared_nfs/..., /workspace/...).
NODE0_IP=${NODE0_IP:-10.13.113.15}; NODE0_C=${NODE0_C:-kk_proxy-moe_gb300-2-tray05-f3_1021_2node}
NODE1_IP=${NODE1_IP:-10.13.113.11}; NODE1_C=${NODE1_C:-kk_proxy-moe_gb300-2-tray01-f3_1021_2node}
PORT=${PORT:-30000}; DIST_PORT=${DIST_PORT:-5000}
# Loading 1.36 TB from Ceph takes ~16 min per launch; each node reads its own NVMe copy instead.
MODEL=${MODEL:-/local/proxy_24l}
G_C=${G_C:-/workspace/claude-skills/proxy-moe/fixed_seq_len_gb300_2node}

n0() { docker exec "$NODE0_C" bash -c "$1"; }
n1() { ssh -o BatchMode=yes "$NODE1_IP" "docker exec $NODE1_C bash -c $(printf %q "$1")"; }

# Container path -> host path (the host sees /shared_nfs and /workspace under /SFS-aGqda6ct).
host_path() { local p=${1/#\/shared_nfs/\/SFS-aGqda6ct\/models}; echo "${p/#\/workspace/\/SFS-aGqda6ct\/amd\/kk}"; }

KILL_SGLANG='pkill -f "[s]glang.launch_server"; for _ in $(seq 1 30); do pgrep -f "[s]glang" > /dev/null || break; sleep 2; done; pkill -9 -f "[s]glang"; true'
GPU_BUSY="nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '\$1>2048{c++}END{print c+0}'"

# Kill every sglang process in both containers and wait until all 8 GPUs are drained.
teardown_2node() {
  n0 "$KILL_SGLANG"; n1 "$KILL_SGLANG"
  local b0 b1
  for _ in $(seq 1 60); do
    b0=$(bash -c "$GPU_BUSY"); b1=$(ssh -o BatchMode=yes "$NODE1_IP" "$GPU_BUSY")
    [ "${b0:-1}" -eq 0 ] && [ "${b1:-1}" -eq 0 ] && return 0
    sleep 5
  done
  echo "  WARN: GPUs not drained (node0 busy=$b0 node1 busy=$b1)"
}

# launch_2node RD CONC [EXTRA_ENV]: start rank 1 then rank 0, return 0 once node0 serves
# /health_generate. Server logs/pids go to RD/server_n{0,1}.{log,pid}.
launch_2node() {
  local rd=$1 conc=$2 envs=${3:-} t0
  local common="CONC=$conc MODEL=$MODEL PORT=$PORT MEM=${MEM:-0.85} DIST_INIT=$NODE0_IP:$DIST_PORT EXTRA_ENV='$envs' EXTRA_ARGS='${EXTRA_ARGS:-}'"
  n1 "$common NODE_RANK=1 SERVER_LOG=$rd/server_n1.log SERVER_PID_FILE=$rd/server_n1.pid bash $G_C/rank_launch.sh" || return 1
  n0 "$common NODE_RANK=0 SERVER_LOG=$rd/server_n0.log SERVER_PID_FILE=$rd/server_n0.pid bash $G_C/rank_launch.sh" || return 1
  t0=$(date +%s)
  until curl -sf "http://127.0.0.1:$PORT/health_generate" > /dev/null; do
    n0 "kill -0 \$(cat $rd/server_n0.pid) 2> /dev/null" || { echo "  node0 server exited"; return 1; }
    n1 "kill -0 \$(cat $rd/server_n1.pid) 2> /dev/null" || { echo "  node1 server exited"; return 1; }
    if [ $(( $(date +%s) - t0 )) -gt "${READY_TIMEOUT:-1800}" ]; then echo "  not ready after ${READY_TIMEOUT:-1800}s"; return 1; fi
    sleep 10
  done
  echo "  ready after $(( $(date +%s) - t0 ))s"
}
