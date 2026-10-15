#!/usr/bin/env bash
# PR #44405 (DSpark / spec-v2 host-sync skip on HIP) on DeepSeek-V4-Pro: main (d4daf52369) vs pr1 (0a603c0670).
# V4-Pro TP8 + attn-DP8 + EAGLE, cookbook MI355X decode env (dsv4-upstream-prs/scripts/e2e_one.sh), GPUs 0-7.
# Per variant: GSM8K 1319 (5-shot, parallel 128), then decode TPOT 8k c1 / c4 x3 (decode_load_profile.py).
set -uo pipefail
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
R=/shared_nfs/kk/results/DeepSeek-V4-Pro/pr44405_1015
MODEL=/shared_nfs/deepseek-ai/DeepSeek-V4-Pro PORT=30000 AITER=/sgl-workspace/aiter
ts() { date '+%F %T'; }
for v in main pr1; do
  SRC=/sgl-workspace/snap_v4_${v}_1015/python OUT=$R/$v; mkdir -p "$OUT"
  echo "[$(ts)] start $v"
  cd /tmp
  setsid env PYTHONPATH=$SRC:$AITER HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    HSA_NO_SCRATCH_RECLAIM=0 SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton \
    SGLANG_OPT_FP8_WO_A_FUSED_INVROPE=1 SGLANG_OPT_USE_AITER_BATCHED_GEMM=1 \
    AITER_BF16_FP8_MOE_BOUND=0 TORCH_BLAS_PREFER_HIPBLASLT=1 GPU_MAX_HW_QUEUES=5 \
    SGLANG_DSV4_UNIFIED_KV_FP8=1 SGLANG_SHARED_EXPERT_TP1=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1 SGLANG_DP_USE_GATHERV=1 \
    python3 -m sglang.launch_server --model-path $MODEL --trust-remote-code \
    --tp 8 --attn-dp-size 8 --enable-dp-lm-head --enable-dp-attention-local-control-broadcast \
    --enable-deepseek-v4-fp4-indexer --attention-backend dsv4 --kv-cache-dtype fp8_e4m3 --page-size 256 \
    --swa-full-tokens-ratio 0.15 --mem-fraction-static 0.88 --enforce-shared-experts-fusion \
    --load-balance-method round_robin --context-length 65536 --watchdog-timeout 3600 \
    --speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4 --max-running-requests 256 \
    --cuda-graph-bs-decode $(seq -s ' ' 1 32) --host 127.0.0.1 --port $PORT > "$OUT/server.log" 2>&1 < /dev/null &
  SPID=$!
  st=FAIL_START
  for _ in $(seq 1 720); do
    grep -q 'ready to roll' "$OUT/server.log" && { st=READY; break; }
    kill -0 $SPID 2>/dev/null || break
    sleep 5
  done
  echo "[$(ts)] $v server: $st"
  if [ $st = READY ]; then
    PYTHONPATH=$SRC timeout 3600 python3 -m sglang.test.few_shot_gsm8k --port $PORT --num-questions 1319 \
      --num-shots 5 --parallel 128 --max-new-tokens 512 > "$OUT/gsm8k.log" 2>&1
    echo "[$v] $(grep -oE 'Accuracy: [0-9.]+|Invalid: [0-9.]+' "$OUT/gsm8k.log" | tail -2 | paste -sd' ')"
    for r in 1 2 3; do
      timeout 1500 python3 -I "$D/decode_load_profile.py" --port $PORT --conc 1,4 --isl 8192 --osl 1024 --seed $r \
        > "$OUT/load_r$r.out" 2>&1
      grep '^\[clean' "$OUT/load_r$r.out" | sed "s/^/[$v r$r] /" | cut -c1-200
    done
    echo "[$v] tracebacks: $(grep -c Traceback "$OUT/server.log")  faults: $(grep -c 'Memory access fault' "$OUT/server.log")"
    echo "[$v] overlap: $(grep -oE "'disable_overlap_schedule': [A-Za-z]+" "$OUT/server.log" | head -1)"
  fi
  kill -- -$SPID 2>/dev/null; sleep 15; kill -9 -- -$SPID 2>/dev/null
  for p in $(rocm-smi --showpids 2>/dev/null | awk '/^[0-9]+ /{print $1}'); do kill -9 $p 2>/dev/null; done; sleep 10
done
echo "V4PRO AB DONE $(ts)"
