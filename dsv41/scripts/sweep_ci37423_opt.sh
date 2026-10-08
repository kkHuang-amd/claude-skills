#!/usr/bin/env bash
# AgentX sweep with the InferenceX run 37423021942 recipe (PR #3696 agentic.yaml @ 47adb59a, image 20261006 settings)
# plus TP4_GAP_1006.md table items 0-6 on (sglang/aiter branch dsv41-opt-1008 in /sgl-workspace).
#   LANE=tp2 bash sweep_ci37423_opt.sh   # GPUs 0,1  port 8888: c1 2 4 8 16 32 64 (c64: max-running/graph bs 128)
#   LANE=tp4 bash sweep_ci37423_opt.sh   # GPUs 4-7  port 8890: c1 2 4 8 16 (c16 mem 0.70, as in CI)
#   MODE=smoke LANE=tp4 bash ...         # server only (CONC 8 config, EVAL_ONLY=true -> no SGLANG_SIMULATE_ACC_LEN)
#                                         # + scheduler env check + GSM8K 1319, then stop the server
#   Sweep lanes serialize on /tmp/opt1008_sweep.lock (tp2 then tp4; one server at a time).
#   TAGP=opt1008 (tag prefix). Items 0 (index-Q fuse) and 2 (block-max) have no flag; 4 and 6 are default on.
# Out: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/<TAGP>_tp<TP>_c<C>/, progress agentx/lane_<PORT>.txt
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
TAGP=${TAGP:-opt1008}
case ${LANE:?LANE=tp2|tp4} in
  tp2) export TP=2 GPUS=${GPUS:-0,1} PORT=${PORT:-8888}; CONCS="1 2 4 8 16 32"; C64=1 ;;
  tp4) export TP=4 GPUS=${GPUS:-4,5,6,7} PORT=${PORT:-8890}; CONCS="1 2 4 8 16"; C64=0 ;;
  *) echo "bad LANE"; exit 1 ;;
esac
export SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 QR_QUANT=${QR_QUANT:-INT8}   # user 2026-10-08: OPUS sparse prefill on, quick-reduce INT8 (CI: off / NONE)
export EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion"
export SGLANG_DSV41_PREFILL_LOGITS_BF16=1 SGLANG_ROCM_MHC_ALL_REDUCE_STATS=1 SGLANG_AITER_SMALL_MOE_SORT_MAX_PAIRS=64 \
       SGLANG_ROCM_MXFP8_AITER_PRESHUFFLE=1 SGLANG_HIP_WO_A_MXFP8=1
# CI per-point args: pdi 16 (c<32) / 4; chunk 16384 (c<64) / 4096; mem 0.70, 0.80 at TP2 c16/c32, 0.85 at c64.
mem(){ local c=$1; if ((c>=64)); then echo 0.85; elif ((TP==2 && c>=16)); then echo 0.80; else echo 0.70; fi; }
pt(){ local c=$1; echo "${TAGP}_tp${TP}_c$c:$c:$((c>=32?4:16)):$((c>=64?4096:16384)):$(mem $c)"; }
if [ "${MODE:-sweep}" = smoke ]; then
  T=${TAGP}_tp${TP}_smoke; OUT=/shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx
  TAG=$T CONC=8 PREFILL_DECODE_INTERVAL=16 CHUNKED_PREFILL_SIZE=16384 MEM_FRACTION_STATIC=0.70 SERVER_ONLY=1 EVAL_ONLY=true \
    setsid bash "$D/agentx_colleague_run.sh" > "$OUT/$T.nohup" 2>&1 &
  LP=$!
  for i in $(seq 1 360); do grep -q 'SERVER_ONLY: ready' "$OUT/$T.nohup" && break; kill -0 $LP 2>/dev/null || break; sleep 10; done
  if grep -q 'SERVER_ONLY: ready' "$OUT/$T.nohup"; then
    # Accuracy is only valid without simulated acceptance; also confirm every opt flag reached the scheduler.
    sp=$(for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::schedul/{print $1}'); do
      tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && echo $p; done | head -1)
    envs=$(tr '\0' '\n' < /proc/$sp/environ)
    if grep -q '^SGLANG_SIMULATE_ACC_LEN=' <<< "$envs"; then echo "SMOKE ABORT: SGLANG_SIMULATE_ACC_LEN set"; else
      echo "SMOKE env: $(grep -E '^SGLANG_(SIMULATE_ACC|DSV41_PREFILL_LOGITS_BF16|ROCM_MHC_ALL_REDUCE_STATS|AITER_SMALL_MOE_SORT|ROCM_MXFP8_AITER_PRESHUFFLE|HIP_WO_A_MXFP8|OPT_HIP_OPUS)|^ROCM_QUICK_REDUCE_QUANTIZATION=' <<< "$envs" | tr '\n' ' ')"
    TAG=$T PORT=$PORT SRC=/sgl-workspace/sglang/python bash "$D/run_gsm8k.sh"; fi
  else echo "SMOKE: server not ready"; grep -E 'Traceback|Error|error' "$OUT/$T.nohup" "$OUT/$T/server.log" 2>/dev/null | tail -5 | cut -c1-200; fi
  kill -- -$LP 2>/dev/null
  for p in $(ps -eo pid,comm | awk '$2 ~ /^sglang::/ || $2=="python3"{print $1}'); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "HIP_VISIBLE_DEVICES=$GPUS" && kill -9 "$p"; done
  exit 0
fi
# User 2026-10-08: never run TP2 and TP4 sweeps at the same time, one lane only -> serialize on a lock (tp2 first).
[ "$TP" = 4 ] && sleep 60
exec 9>/tmp/opt1008_sweep.lock; flock 9
echo "[$(date +%F' '%T)] sweep lock acquired: LANE=$LANE QR=$QR_QUANT TAGP=$TAGP" >> /shared_nfs/kk/results/DeepSeek-V4.1-Flash/agentx/chain_opt1008.txt
POINTS=""; for c in $CONCS; do POINTS+="$(pt $c) "; done
POINTS="$POINTS" bash "$D/tp4_gap_lane.sh"
[ "$C64" = 1 ] && POINTS="$(pt 64)" CUDA_GRAPH_MAX_BS=128 bash "$D/tp4_gap_lane.sh"
