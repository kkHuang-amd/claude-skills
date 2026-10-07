#!/bin/bash
# AgentX replay, same aiperf command as InferenceX run 37070984585 (tp4 ep1), against a local server.
# env: CONC (8), PORT (8000), RUN (output subdir, default c${CONC}_run1)
# also: W (aiperf venv + outputs), L (server log), SP (server PID), HF_HOME (cc-traces dataset cache)
W=${W:-/shared_nfs/kk/dsv41_b200/agentx}; L=${L:?server log}; SP=${SP:?server pid}
CONC=${CONC:-8}; PORT=${PORT:-8000}; RUN=${RUN:-c${CONC}_run1}; U=http://localhost:$PORT
until rg -q 'OpenAI server is ready' $L; do kill -0 $SP 2>/dev/null || { echo SERVER_DIED; exit 1; }; sleep 10; done
echo "ready $(date -u +%T)"; sleep 15
curl -s -m 120 $U/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-ai/DeepSeek-V4.1-Flash","prompt":"hi","max_tokens":4}' | head -c 120; echo
export HF_HOME=${HF_HOME:-/workspace/agentx/hf_cache}
OUT=$W/$RUN; mkdir -p $OUT
echo "aiperf start $(date -u +%T)"
$W/venv/bin/aiperf profile --scenario agentx --url $U --endpoint /v1/chat/completions \
  --model deepseek-ai/DeepSeek-V4.1-Flash --tokenizer deepseek-ai/DeepSeek-V4.1-Flash --concurrency $CONC \
  --benchmark-duration 3600 --failed-request-threshold 0.10 --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 --warmup-grace-period 1800 --tokenizer-trust-remote-code \
  --server-metrics $U/metrics --output-artifact-dir $OUT \
  --public-dataset semianalysis_cc_traces_weka_062126 > $OUT/aiperf.log 2>&1
echo "aiperf exit=$? $(date -u +%T)"
python3 $(dirname "$0")/ttt_p90.py $OUT/profile_export_aiperf.json
rg 'SpecDecoding metrics' $L | tail -1 | rg -o 'Mean acceptance length: [0-9.]+'
echo AGENTX_DONE $RUN
