#!/usr/bin/env bash
# Proxy-MoE fixed_seq_len sweep, TP8 across 2 GB300 trays (4 GPUs each, MNNVL), one server per point.
# Same shapes / CONC / client as ../fixed_seq_len_gb300/sweep_proxy.sh; recipe tp8ep1 (24 layers).
# Run on the node0 HOST; see lib_2node.sh for node / container names.
#   SHAPES  : "1024/1024 8192/1024 70000/300"
#   CONCS   : 2..1024 for 1k/1k and 8k/1k; 2..32 for 70000/300 (override CONCS / CONCS_70K)
#   client  : sglang.bench_serving --backend sglang (in node0's container), random-ids + --tokenize-prompt,
#             range-ratio 1.0, np = CONC*8 (CONC*4 for ISL >= 32768), warmup = CONC*2, request-rate inf
# Resumable: a point with DONE is skipped.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=lib_2node.sh
source "$HERE/lib_2node.sh"
OUT=${OUT:-/shared_nfs/kk/results/proxy_24l/fixed_seq_len_gb300_2node}   # container path
SHAPES=${SHAPES:-"1024/1024 8192/1024 70000/300"}
CONCS=${CONCS:-"2 4 8 16 32 64 128 256 512 1024"}
CONCS_70K=${CONCS_70K:-"2 4 8 16 32"}
RECIPE=${RECIPE:-tp8ep1}
NP_MULT=${NP_MULT:-}          # override np = CONC * NP_MULT (smoke tests)
OUT_H=$(host_path "$OUT")
mkdir -p "$OUT_H"

echo "=== $(date +%H:%M:%S) preflight teardown ==="
teardown_2node

for SHAPE in $SHAPES; do
  ISL=${SHAPE%/*}; OSL=${SHAPE#*/}
  if [ "$ISL" -ge 32768 ]; then CL=$CONCS_70K; MULT=${NP_MULT:-4}; else CL=$CONCS; MULT=${NP_MULT:-8}; fi
  for CONC in $CL; do
    TAG="isl${ISL}-osl${OSL}-c${CONC}"
    RD="$OUT/$RECIPE/$TAG"; RD_H=$(host_path "$RD"); mkdir -p "$RD_H"
    NP=$((CONC * MULT)); WU=$((CONC * 2))
    echo "=== $(date +%H:%M:%S) $RECIPE $TAG (np=$NP warmup=$WU) ==="
    if [ -f "$RD_H/DONE" ]; then echo "  already done, skip"; continue; fi

    if ! launch_2node "$RD" "$CONC" > "$RD_H/launch.log" 2>&1; then
      echo "  SERVER FAILED - see $RD_H/launch.log, server_n0.log, server_n1.log"
      tail -n 3 "$RD_H/launch.log" | cut -c1-200; teardown_2node; continue
    fi
    echo "  servers up ($(tail -n 1 "$RD_H/launch.log" | tr -s ' ')), client"
    n0 "cd /tmp && python3 -m sglang.bench_serving --backend sglang \
        --host 127.0.0.1 --port $PORT --tokenizer $MODEL \
        --dataset-name random-ids --tokenize-prompt \
        --random-input-len $ISL --random-output-len $OSL --random-range-ratio 1.0 \
        --num-prompts $NP --max-concurrency $CONC --warmup-requests $WU \
        --request-rate inf --output-file $RD/result.jsonl --output-details > $RD/client.log 2>&1"
    crc=$?
    if [ $crc -ne 0 ] || [ ! -s "$RD_H/result.jsonl" ]; then
      echo "  CLIENT FAILED (rc=$crc) - see $RD_H/client.log"
    else
      python3 - "$RD_H/result.jsonl" <<'PY'
import json, sys
d = [json.loads(l) for l in open(sys.argv[1]) if l.strip()][-1]
print(f"  -> completed {d['completed']}  TTT {d['total_throughput']:.0f} tok/s  "
      f"TTFT med {d['median_ttft_ms']:.0f} ms  ITL med {d['median_itl_ms']:.2f} ms")
PY
      touch "$RD_H/DONE"
    fi
    teardown_2node
  done
done
echo "=== sweep complete: $OUT_H  (report: python3 $HERE/../fixed_seq_len/report.py $OUT_H) ==="
