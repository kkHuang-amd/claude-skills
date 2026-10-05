#!/bin/bash
# DeepSeek-R1-0528 FP8, 8xB200 TP8, MTP (EAGLE 3/1/4)
# Base = cookbook b200/fp8/8gpu/low-latency Pareto config + cookbook MTP flags.
# Aligned with MI355X run: --disable-radix-cache added; stream-interval left at default (1);
# no --enable-symm-mem.
export SGLANG_ENABLE_JIT_DEEPGEMM=false

python3 -m sglang.launch_server \
  --model-path ${MODEL:-deepseek-ai/DeepSeek-R1-0528} \
  --trust-remote-code \
  --tp 8 \
  --speculative-algorithm EAGLE \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --kv-cache-dtype fp8_e4m3 \
  --mem-fraction-static 0.82 \
  --disable-radix-cache \
  --chunked-prefill-size ${CHUNK:-32768} \
  --max-prefill-tokens ${CHUNK:-32768} \
  --cuda-graph-max-bs-decode 128 \
  --scheduler-recv-interval 10 \
  --fp8-gemm-backend flashinfer_trtllm \
  ${MRR:+--max-running-requests $MRR} \
  --host 0.0.0.0 --port ${PORT:-30000}
