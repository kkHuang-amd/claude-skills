#!/usr/bin/env bash
# 2-node smoke (run on the node0 HOST): one TP8 server with NCCL_DEBUG=INFO -> package smoke_test.py
# + MNNVL check, then a 3-point mini sweep (1k/1k c2, c64; 70000/300 c2; NP_MULT=2).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib_2node.sh
source "$HERE/lib_2node.sh"
S=${S:-/shared_nfs/kk/results/proxy_24l/fixed_seq_len_gb300_2node_smoke}
PKG=${PKG:-/workspace/models/proxy_moe_sglang/proxy_moe_sglang}
RD=$S/pkg_smoke; RD_H=$(host_path "$RD"); mkdir -p "$RD_H"

teardown_2node
launch_2node "$RD" 16 "NCCL_DEBUG=INFO" > "$RD_H/launch.log" 2>&1
echo "launch rc=$? ($(tail -n 1 "$RD_H/launch.log"))"
for n in 0 1; do
  echo "node$n NCCL transport lines:"
  grep -Ei 'MNNVL|NVLS|via P2P|NET/Socket|Using network' "$RD_H/server_n$n.log" | sort | uniq -c | sort -rn | head -8 | cut -c1-200
done
n0 "cd /tmp && python3 $PKG/tools/smoke_test.py --port $PORT --out $RD/smoke.json > $RD/smoke.log 2>&1"
echo "smoke rc=$?"
teardown_2node

OUT=$S SHAPES="1024/1024 70000/300" CONCS="2 64" CONCS_70K="2" NP_MULT=2 bash "$HERE/sweep_proxy_2node.sh"
echo "SMOKE DONE"
