#!/bin/bash
# e2e_speed.sh <tag> <sglang_python_dir> <aiter_dir> [ENV=VAL ...]
# Launch DSV4-Pro TP8 + attn-DP8 + EAGLE (single node, cookbook MI355X decode env),
# run bench_serving random-ids ${ISL:-8192}/${OSL:-1024} at ${CONCS:-16 64} (${REPS:-2} reps, 3*conc prompts),
# kill the server. One summary line per (conc, rep) to speed/results.md.
set -u
TAG=$1 SRC=$2 AITER=$3; shift 3
OUT=/shared_nfs/kk/results/DeepSeek-V4-Pro/upstream-prs-1013/speed/$TAG; mkdir -p $OUT
MODEL=${MODEL:-/shared_nfs/models/DeepSeek-V4-Pro}; PORT=${PORT:-30000}
env_list=(PYTHONPATH=$SRC:$AITER HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  HSA_NO_SCRATCH_RECLAIM=0 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton
  SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=1 SGLANG_OPT_USE_AITER_BATCHED_GEMM=1
  AITER_BF16_FP8_MOE_BOUND=0 TORCH_BLAS_PREFER_HIPBLASLT=1 GPU_MAX_HW_QUEUES=5
  SGLANG_DSV4_UNIFIED_KV_FP8=1 SGLANG_SHARED_EXPERT_TP1=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
  SGLANG_DP_USE_GATHERV=1 "$@")
printf '%s\n' "${env_list[@]}" > $OUT/env.txt
cd /tmp
setsid env "${env_list[@]}" python3 -m sglang.launch_server --model-path $MODEL --trust-remote-code \
  --tp 8 --attn-dp-size 8 --enable-dp-lm-head --enable-dp-attention-local-control-broadcast \
  --enable-deepseek-v4-fp4-indexer --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 --page-size 256 \
  --swa-full-tokens-ratio 0.15 --mem-fraction-static 0.88 --enforce-shared-experts-fusion \
  --load-balance-method round_robin --context-length ${CTX:-65536} --watchdog-timeout 3600 \
  --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 --max-running-requests 256 \
  --cuda-graph-bs-decode $(seq -s ' ' 1 32) ${EXTRA_ARGS:-} --host 127.0.0.1 --port $PORT > $OUT/server.log 2>&1 &
SPID=$!; echo $SPID > $OUT/server.pid
st=FAIL_START
for i in $(seq 1 720); do
  grep -q 'ready to roll' $OUT/server.log && { st=READY; break; }
  kill -0 $SPID 2>/dev/null || break
  grep -qE 'Initialization failed|Traceback' $OUT/server.log && { sleep 20; kill -0 $SPID 2>/dev/null || break; }
  sleep 5
done
RES=/shared_nfs/kk/results/DeepSeek-V4-Pro/upstream-prs-1013/speed/results.md
if [ $st = READY ]; then
  for c in ${CONCS:-16 64}; do for r in $(seq 1 ${REPS:-2}); do
    L=$OUT/bench_c${c}_r$r.log
    PYTHONPATH=$SRC timeout 3600 python3 -m sglang.bench_serving --backend sglang --port $PORT --dataset-name random-ids \
      --random-input-len ${ISL:-8192} --random-output-len ${OSL:-1024} --random-range-ratio 1 \
      --num-prompts $((3*c)) --max-concurrency $c > $L 2>&1
    m() { grep -m1 "$1" $L | grep -oE '[0-9.]+ *$' | tr -d ' '; }
    echo "| $(date +%F' '%H:%M) | $(hostname) | $TAG | c$c r$r | $(m 'Output token throughput') | $(m 'Median TPOT') | $(m 'Median ITL') | $(m 'Median TTFT') | $* |" >> $RES
    kill -0 $SPID 2>/dev/null || { st=DIED; break 2; }
  done; done
fi
kill -- -$SPID 2>/dev/null; sleep 15; kill -9 -- -$SPID 2>/dev/null
for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+ /{print $1}'); do kill -9 $p 2>/dev/null; done; sleep 10
echo "$TAG $st"
