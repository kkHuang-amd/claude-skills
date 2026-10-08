#!/usr/bin/env bash
# GSM8K 5-shot 1319 A/B on an EVAL_ONLY TP4 server (the tp4_moe_ab.sh step-3 config: CONC=32 eval, PDI 4, chunk 16384,
# mem 0.80, --fp8-gemm-backend aiter --enforce-shared-experts-fusion), one server per SRC, sequentially.
#   RUNS="i1:/sgl-workspace/sglang-i1/python base:/sgl-workspace/sglang/python" bash tp4_gsm8k_ab.sh
# Results: results/gsm8k.md rows (via run_gsm8k.sh), server logs /shared_nfs/kk/results/DeepSeek-V4.1-Flash/gsm8k_ab/<tag>/. GPUs 0-3, port 8888.
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
if pgrep -f '^sglang::' >/dev/null; then echo "REFUSING: sglang server running"; exit 1; fi
for run in ${RUNS:?}; do
    tag=${run%%:*}; src=${run#*:}
    G=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/gsm8k_ab/$tag; mkdir -p "$G"
    export PYTHONPATH=/sgl-workspace/mori SRC=$src SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 OPUS=0
    export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
    export TP=4 EP_SIZE=1 GPUS=0,1,2,3 PORT=8888 CONC=32 PREFILL_DECODE_INTERVAL=4 CHUNKED_PREFILL_SIZE=16384
    export MEM_FRACTION_STATIC=0.80 SERVER_ONLY=1 EVAL_ONLY=true TAG=gsm8k_ab_$tag
    setsid bash "$D/agentx_colleague_run.sh" > "$G/server.log" 2>&1 < /dev/null &
    SPID=$!
    for _ in $(seq 1 180); do
        grep -q 'ready to roll' "$G/server.log" && break
        grep -qE 'Traceback|Initialization failed' "$G/server.log" && break
        sleep 10
    done
    if grep -q 'ready to roll' "$G/server.log"; then
        sleep 15
        SRC=$src TAG=$(hostname | sed 's/.*-//')_gsm8k_$tag PORT=8888 bash "$D/run_gsm8k.sh"
    else
        echo "$tag: server failed"; grep -m3 -E 'Traceback|Error' "$G/server.log" | cut -c1-200
    fi
    kill -- -$SPID 2>/dev/null; sleep 5
    for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/{print $1}'); do kill -9 "$p"; done
    sleep 20
done
