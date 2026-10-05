#!/bin/bash
# DeepSeek-R1-0528 FP8, 8xMI355X TP8, MTP (EAGLE 3/1/4)
# Base = cookbook mi355x/fp8/low-latency Pareto config + cookbook MTP flags.
export SGLANG_USE_AITER=1
export RCCL_MSCCL_ENABLE=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4

python3 -m sglang.launch_server \
  --model-path /shared_nfs/deepseek-ai/DeepSeek-R1-0528 \
  --trust-remote-code \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --attention-backend aiter \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.8 \
  --disable-radix-cache \
  --chunked-prefill-size 196608 \
  --max-prefill-tokens 196608 \
  --cuda-graph-max-bs-decode 128 \
  ${MRR:+--max-running-requests $MRR} \
  --host 0.0.0.0 --port ${PORT:-30000}
