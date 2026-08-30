#!/usr/bin/env bash
# usage: sweep_driver.sh RESULT_DIR "1k_conc_list" "8k_conc_list"
RD="$1"; CONC1K="$2"; CONC8K="$3"
mkdir -p "$RD"
B="python3 /workspace/useful-scripts/benchmarking/dsv4/bench_dsv4.py"
run(){ local isl=$1 osl=$2 c=$3 np=$4 nw=$5
  local name="sglangClient_dsv4_isl${isl}_osl${osl}_c${c}"
  echo "### START $name np=$np conc=$c $(date +%T)"
  if $B --backend sglang-oai --base-url http://127.0.0.1:8000 \
       --model /shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro/ \
       --dataset-name random --random-input-len $isl --random-output-len $osl \
       --random-range-ratio 0.8 --num-prompts $np --max-concurrency $c \
       --request-rate inf --warmup-requests $nw \
       --output-file "${RD}/${name}.jsonl" >"${RD}/${name}.log" 2>&1; then
    echo "### DONE  $name $(grep -oE 'Output token throughput \(tok/s\): +[0-9.]+' ${RD}/${name}.log | tail -1) $(date +%T)"
  else
    echo "### FAIL  $name (see ${RD}/${name}.log)"
  fi
}
for c in $CONC1K; do np=$((c*8)); nw=$((c*2)); run 1024 1024 $c $np $nw; done
for c in $CONC8K; do np=$((c*8)); nw=$((c*2)); run 8192 1024 $c $np $nw; done
echo "ALL DONE $RD $(date +%T)"
