#!/bin/bash
# usage: decode_run.sh <conc> <clean|prof>
# clean: D64 bench (in 65536 / out 1024, ignore-eos, 3*conc prompts).
# prof : conc prompts with out 8192 so all conc requests decode together; once all have their first
#        token, POST /start_profile (server stops itself after max_iterations), then /stop_profile.
C=$1; MODE=$2
D=${WORK:-/shared_nfs/kk/dsv41_b200}; M=${MODEL_PATH:-/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash}; U=http://127.0.0.1:8000
mkdir -p $D/bench
metric() { curl -s $U/metrics | awk -v k="$1" '$1 ~ "^"k"({|$)" {s+=$2} END {print s+0}'; }
if [ "$MODE" = clean ]; then N=$((3*C)); OUT=1024; else N=$C; OUT=8192; fi
T0=$(metric vllm:time_to_first_token_seconds_count)
vllm bench serve --backend vllm --base-url $U --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer $M \
  --dataset-name random --random-input-len 65536 --random-output-len $OUT --ignore-eos \
  --num-prompts $N --max-concurrency $C --percentile-metrics ttft,tpot,itl --metric-percentiles 50,90 \
  --save-result --result-dir $D/bench --result-filename d64_c${C}_${MODE}.json > $D/bench/d64_c${C}_${MODE}.log 2>&1 &
BP=$!
if [ "$MODE" = prof ]; then
  while kill -0 $BP 2>/dev/null; do
    T=$(metric vllm:time_to_first_token_seconds_count)
    R=$(metric vllm:num_requests_running)
    if [ $(python3 -c "print(int($T-$T0>=$C and $R>=$C))") = 1 ]; then break; fi
    sleep 0.5
  done
  sleep 2
  echo "start_profile running=$(metric vllm:num_requests_running) waiting=$(metric vllm:num_requests_waiting)"
  curl -s -X POST $U/start_profile; sleep 15; curl -s -X POST $U/stop_profile; echo "stop_profile"
fi
wait $BP; echo "bench exit=$?"
rg 'Successful requests|Output token throughput|Mean TTFT|Median TTFT|P90 TTFT|Mean TPOT|Median TPOT|P90 TPOT|Mean ITL|Median ITL|P90 ITL|Error|Traceback' $D/bench/d64_c${C}_${MODE}.log | cut -c1-150
