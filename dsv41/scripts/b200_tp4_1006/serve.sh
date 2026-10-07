#!/bin/bash
# usage: serve.sh <conc_index: 0=c1 2=c8 3=c16 per recipe zip_override_tp4> <tag>
# env: WORK (outputs, traces), MODEL_PATH. Args = InferenceX 5ff11ab20 b200-fp4-mtp/agentic.yaml (copied here) base + zip_override_tp4[idx],
# speculative rejection replaced by synthetic AL 3.51 (as the throughput harness does).
set -e
IDX=${1:-2}; TAG=${2:-c8}
S=$(cd "$(dirname "$0")" && pwd); D=${WORK:-/shared_nfs/kk/dsv41_b200}; MODEL_PATH=${MODEL_PATH:-/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash}; R=$S/recipe_b200-fp4-mtp_agentic_5ff11ab20.yaml
read MNS MNBT CAP < <(python3 -c "
import yaml;t=yaml.safe_load(open('$R'))['zip_override_tp4']['roles']['agg']['args'];i=$IDX
print(t['max-num-seqs'][i],t['max-num-batched-tokens'][i],t['max-cudagraph-capture-size'][i])")
CC=$(python3 -c "
import yaml;print(yaml.safe_load(open('$R'))['zip_override_tp4']['roles']['agg']['args']['compilation-config'][$IDX])")
mkdir -p $D/traces/$TAG
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1 PYTHONUNBUFFERED=1 VLLM_ENGINE_READY_TIMEOUT_S=7200
PROF_ARGS=()
[ "${NO_PROFILER:-0}" = 1 ] || PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$D/traces/$TAG\",\"torch_profiler_with_stack\":false,\"torch_profiler_record_shapes\":true,\"torch_profiler_use_gzip\":true,\"ignore_frontend\":true,\"max_iterations\":${PROF_ITERS:-40}}")
echo "NO_PROFILER=${NO_PROFILER:-0} IDX=$IDX TAG=$TAG MNS=$MNS MNBT=$MNBT CAP=$CAP"
exec vllm serve $MODEL_PATH \
  --served-model-name deepseek-ai/DeepSeek-V4.1-Flash \
  --port ${PORT:-8000} \
  --language-model-only \
  --tokenizer-mode deepseek_v41 \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --engram-config '{"cpu_offload":true,"use_thp":true}' \
  --kernel-config '{"enable_flashinfer_autotune":true}' \
  --attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","enable_adaptive_verification":false,"synthetic_acceptance_length":3.51}' \
  --max-model-len 1048576 \
  --max-num-batched-tokens $MNBT \
  --gpu-memory-utilization 0.97 \
  --disable-uvicorn-access-log \
  --tensor-parallel-size 4 \
  --max-num-seqs $MNS \
  --compilation-config "$CC" \
  --max-cudagraph-capture-size $CAP \
  ${PROF_ARGS[@]+"${PROF_ARGS[@]}"}
