# Request to the B200 node: vLLM TP4 profile + fixed-shape microbench for DSV4.1-Flash (2026-10-06, rev 3)

Requested by: crsuse2-m2m-255 (MI355X side, see TP4_GAP_1006.md). Executor: B200 node agent.
Write your results ONLY into `dsv41/results/b200_tp4_profile_1006.md` (new file, you own it; `Owner node: <hostname>`
at top). Do not edit other dsv41 docs. Commit small, `git pull --rebase` first, never force-push.
rev 2 (after the MI355X reruns): adds the cached-prefix TTFT sweep (P1), drops D128, splits attention in the kernel table.

## Why -- what the MI355X side already knows

MI355X SGLang vs B200 vLLM on the AgentX trace, TP4 no DP (InferenceX run 37070984585), from per-request records:
- **Decode**: MI355X TPOT is ~1.4x B200 at c1/c8/c16 (c8 p50 2.80 vs 1.92 ms) and flat in context length on BOTH sides
  -> a per-step cost, not KV length. Need: B200 decode step kernel breakdown.
- **Prefill on a cached prefix**: TTFT ~= a + b x ISL. B200 58 ms + 0.39 ms per 1k ISL, MI355X 200 ms + 1.89 ms per 1k
  (c8; c16 same slope ratio). Trace turns add only ~700 new tokens (p50; p90 5.2k), so the slope is work over the
  WHOLE cached prefix, not the new tokens. Need: B200 TTFT vs prefix length with a fixed small extend, plus kernel
  breakdown at two prefix lengths to see which kernels scale.

## Environment -- must match InferenceX run 37070984585 (TP4 ep1 point)

- Image `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7`, 4x B200, model `deepseek-ai/DeepSeek-V4.1-Flash`.
- Recipe: InferenceX a8504a430 `srt-slurm-recipes/dsv41flash/vllm/b200-fp4-mtp/agentic.yaml`, TP4 entry. Key args:
  `--tensor-parallel-size 4 --language-model-only --kv-cache-dtype fp8 --max-model-len 1048576`
  `--max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 --max-num-seqs 8|16 (c8|c16; use 8 for c1)`
  `--engram-config '{"cpu_offload":true,"use_thp":true}'`
  `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
  `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","enable_adaptive_verification":false,"synthetic_acceptance_length":3.51}'`
  `--compilation-config` and `--max-cudagraph-capture-size` exactly as in the recipe for that conc.
  env `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1`. Prefix caching ON (default).
- Keep synthetic acceptance 3.51 -- MI355X uses the same simulated AL; real acceptance would make the numbers incomparable.
- Record: image digest, `vllm --version`, driver, GPU clocks/power cap (`nvidia-smi -q -d CLOCK,POWER | head`).

## Priority and time budget

Do P1 first; stop after P1 if the machine time runs out and say so in the doc. One server (max-num-seqs 16) can serve
everything except where noted; restart only if the recipe differs per conc.
- **P1a** cached-prefix TTFT sweep (Deliverable 1b) -- ~15 min.
- **P1b** decode profiles D64 c8 + c16 and their clean numbers (Deliverables 1a, 2.1) -- ~30 min.
- **P2** extend profiles at prefix 64k and 256k (Deliverable 2.2) -- ~15 min.
- **P3** D64 c1 numbers + profile; kernel tables for everything (Deliverable 3).

## Deliverable 1 -- clean numbers (no profiler)

**1a. Decode, D64:** `vllm bench serve --dataset-name random --random-input-len 65536 --random-output-len 1024
--ignore-eos --seed 0 --num-prompts <conc> --max-concurrency <conc>`, conc 8 and 16 (P1b), conc 1 (P3). Run it
TWICE with the same --seed: the first run cold-prefills the prompts into the prefix cache; report and profile the
SECOND run, where every prefill is a cache hit and all `conc` requests decode together. (rev 3: with cold 64k
prefills the TTFTs spread over seconds and early requests finish before the last one starts -- the MI355X profile
caught bs=4 instead of 8 that way.) Report per row: mean/p50/p90 TTFT, TPOT, ITL (ms), output tok/s, and the server's
last `SpecDecoding metrics` line (must show mean AL ~3.5).

**1b. Cached-prefix TTFT sweep (conc 1):** for each prefix length L in 32k, 64k, 128k, 256k and new length N in 512,
4096: send prompt P_L (random token ids, max_tokens 1) to warm the prefix cache, then P_L + N fresh random tokens
(max_tokens 1) and record its TTFT; repeat the second request 3x with different N-token suffixes, report the median.
Token-id prompts via `/v1/completions` (`"prompt": [ids...]`, `"max_tokens": 1`, `"stream": true`); confirm the cache hit
from the server's `Prefix cache hit rate` / `prompt_tokens_details.cached_tokens`. Report a table L x N -> TTFT ms
and the linear fit TTFT = a + b x L per N. MI355X runs the identical sweep.

## Deliverable 2 -- torch profiler traces

Enable the profiler the way this image supports it (check `vllm serve --help | grep -i profil`; either env
`VLLM_TORCH_PROFILER_DIR=<dir>` or `--profiler-config`). Keep traces SHORT (bounded iterations or ~1-2 s windows),
with_stack off, record_shapes on. Use `POST /start_profile` / `POST /stop_profile`.

1. **Decode steady state**, D64 at conc 8 and 16 (P1b), conc 1 (P3): on the SECOND (cache-warm) run of 1a, wait
   until every request has its first token, then start_profile, ~30-40 decode steps, stop_profile. Check in the trace
   that the decode batch size equals conc (e.g. from the attention / MoE kernel shapes) and state it in the doc.
   Summarize the rank whose trace actually contains the CUDA-graph kernels (on MI355X only one rank's trace did).
2. **Extend over a cached prefix** (P2), conc 1: warm P_L (max_tokens 1), then profile P_L + 512 new tokens
   (max_tokens 1) for L = 64k and L = 256k -- two traces, so per-kernel time can be compared across prefix length.

## Deliverable 3 -- kernel summary (so nobody has to open GBs of traces)

For each trace above, from the GPU (device) events of rank 0:
- decode: mean ms per step (one step = target verify forward + DSpark draft) and the step count used; extend: total
  GPU ms of the request.
- top-30 kernels by total GPU time as CSV `kernel_name(<=120 chars),calls,total_us,us_per_step,pct`.
- grouped into: sparse MLA attention, indexer (logits + top-k), KV compressor / compressed-KV, MoE (incl. routing),
  dense GEMM, all-reduce/comm, engram, norm/rope/elementwise, sampling/draft-specific, other; plus GPU idle % within
  the window (CPU/launch overhead).
- for the two extend traces: the same grouping side by side (L=64k vs 256k) -- which groups grow with L.

Put the tables in `results/b200_tp4_profile_1006.md`. Compress traces (`*.json.gz`), keep them on the B200 node, and
write their path + sizes in the doc; the user will relay them.

## Do not

- Do not change the recipe args beyond what is listed (no tuning) -- we want the InferenceX config as-is.
- Do not run the full AgentX trace again; run 37070984585 already has it.
- No hostnames / internal paths in anything that goes upstream.
