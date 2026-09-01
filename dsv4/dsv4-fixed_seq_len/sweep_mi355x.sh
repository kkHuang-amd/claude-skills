#!/usr/bin/env bash
# DSV4 fixed-seq-len sweep on 8x MI355X.
#   ISL/OSL : 1024/1024 and 8192/1024
#   CONC    : 2 4 8 16 32 | 64 128 | 256 512 1024
#   recipe  : conc<=32 -> tp4 (no DP); 64,128 -> tp4+dp4; 256+ -> tp8+dp8
#   client  : python3 -m sglang.bench_serving (range-ratio 1.0 = exact ISL,
#             ignore_eos on by default, np=CONC*8, warmup=CONC*2)
# One server per CONC point: upstream binds --cuda-graph-max-bs and
# --max-running-requests to CONC.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${MODEL:-/dockerx/data/models/DeepSeek-V4-Pro}"
OUT="${OUT:-/workspace/results/dsv4-fixed_seq_len/runs}"
PORT="${PORT:-8888}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
CONCS="${CONCS:-2 4 8 16 32 64 128 256 512 1024}"
ISLS="${ISLS:-1024 8192}"
OSL="${OSL:-1024}"
mkdir -p "$OUT"
CSV="$OUT/summary.csv"
[ -f "$CSV" ] || echo "ts,isl,osl,conc,tp,dp_attn,status,output_throughput,total_throughput,mean_ttft_ms,p99_ttft_ms,mean_tpot_ms,p99_e2el_ms,result_json" > "$CSV"

recipe() {  # conc -> "TP DP_ATTENTION"
    case "$1" in
        2|4|8|16|32) echo "4 false" ;;
        64|128)      echo "4 true"  ;;
        *)           echo "8 true"  ;;
    esac
}

# Trap 1: setproctitle rewrites argv, so pkill -f 'sglang::' does NOT work.
# Trap 4: the launcher leaks its server. Kill by PID, then verify.
teardown() {
    local pid="$1"
    [ -n "$pid" ] && kill "$pid" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
    kill -9 "$pid" 2>/dev/null
    pids=$(ps -eo pid,args | grep -E '[s]glang::|[s]glang_router' | awk '{print $1}')
    [ -n "$pids" ] && { echo "  leftover pids: $pids -> KILL"; kill -9 $pids 2>/dev/null; sleep 5; }
    for _ in $(seq 1 30); do
        busy=$(rocm-smi --showmemuse 2>/dev/null | grep -oE 'VRAM%\): [0-9]+' | awk '$NF>2{c++}END{print c+0}')
        [ "${busy:-0}" -eq 0 ] && break; sleep 5
    done
}

for ISL in $ISLS; do
for CONC in $CONCS; do
    read -r TP DPA <<<"$(recipe "$CONC")"
    TAG="isl${ISL}-osl${OSL}-c${CONC}-tp${TP}$([ "$DPA" = true ] && echo "-dp${TP}")"
    RD="$OUT/$TAG"; mkdir -p "$RD"
    echo "=== $(date +%H:%M:%S) $TAG (np=$((CONC*8)) warmup=$((CONC*2))) ==="
    if [ -f "$RD/DONE" ]; then echo "  already done, skip"; continue; fi

    MODEL="$MODEL" TP="$TP" DP_ATTENTION="$DPA" EP_SIZE=1 ISL="$ISL" CONC="$CONC" \
      MAX_MODEL_LEN="$MAX_MODEL_LEN" PORT="$PORT" \
      SERVER_LOG="$RD/server.log" SERVER_PID_FILE="$RD/server.pid" \
      READY_TIMEOUT="${READY_TIMEOUT:-3600}" \
      bash "$HERE/serve_mi355x.sh" > "$RD/launch.log" 2>&1
    rc=$?; SPID=$(cat "$RD/server.pid" 2>/dev/null)
    if [ $rc -ne 0 ]; then
        echo "  SERVER FAILED (rc=$rc) -- see $RD/server.log"
        echo "$(date -Is),$ISL,$OSL,$CONC,$TP,$DPA,server_fail,,,,,,," >> "$CSV"
        teardown "$SPID"; continue
    fi
    echo "  server up (pid $SPID), running client"

    ( cd /tmp && python3 -m sglang.bench_serving \
        --backend sglang-oai \
        --base-url "http://0.0.0.0:$PORT" \
        --model "$MODEL" --tokenizer "$MODEL" \
        --dataset-name random \
        --random-input-len "$ISL" --random-output-len "$OSL" \
        --random-range-ratio 1.0 \
        --num-prompts $((CONC*8)) \
        --max-concurrency "$CONC" \
        --warmup-requests $((CONC*2)) \
        --request-rate inf \
        --output-file "$RD/result.jsonl" \
        --output-details ) > "$RD/client.log" 2>&1
    crc=$?

    if [ $crc -ne 0 ]; then
        echo "  CLIENT FAILED (rc=$crc) -- see $RD/client.log"
        echo "$(date -Is),$ISL,$OSL,$CONC,$TP,$DPA,client_fail,,,,,,," >> "$CSV"
    else
        python3 - "$RD/result.jsonl" "$CSV" "$ISL" "$OSL" "$CONC" "$TP" "$DPA" <<'PY'
import json,sys,datetime
f,csv,isl,osl,conc,tp,dpa = sys.argv[1:8]
d = [json.loads(l) for l in open(f) if l.strip()][-1]
g = lambda k: d.get(k,"")
row = [datetime.datetime.now().isoformat(timespec="seconds"),isl,osl,conc,tp,dpa,"ok",
       g("output_throughput"),g("total_token_throughput"),g("mean_ttft_ms"),
       g("p99_ttft_ms"),g("mean_tpot_ms"),g("p99_e2el_ms"),f]
open(csv,"a").write(",".join(str(x) for x in row)+"\n")
print("  ->", g("output_throughput"), "out tok/s |", g("mean_ttft_ms"), "ms ttft")
PY
        touch "$RD/DONE"
    fi
    teardown "$SPID"
    echo "  torn down"
done
done
echo "=== sweep complete: $CSV ==="
