# Request to the B200 node: vLLM TP4 profile + fixed-shape microbench for DSV4.1-Flash (2026-10-06)

Requested by: crsuse2-m2m-255 (MI355X side, see TP4_GAP_1006.md). Executor: B200 node agent.
Write your results ONLY into `dsv41/results/b200_tp4_profile_1006.md` (new file, you own it; `Owner node: <hostname>`
at top). Do not edit other dsv41 docs. Commit small, `git pull --rebase` first, never force-push.

## Why

MI355X SGLang trails B200 vLLM on the AgentX trace at TP4 (no DP): c1 -18.5% / -30.3%, c8 -7.3% / -25.8%,
c16 -4.3% / -30.7% (TTT tok/s/GPU / P90 interactivity; InferenceX run 37070984585). We already have that run's
per-request records and server logs. What we lack is a kernel-level picture of B200 vLLM and clean fixed-shape numbers
we can reproduce 1:1 on MI355X.

## Environment -- must match InferenceX run 37070984585 (TP4 ep1 point)

- Image `vllm/vllm-openai:nightly-dev-x86_64-cu130-ac9126e58aa7`, 4x B200, model `deepseek-ai/DeepSeek-V4.1-Flash`.
- Recipe: InferenceX a8504a430 `srt-slurm-recipes/dsv41flash/vllm/b200-fp4-mtp/agentic.yaml`, TP4 entry. Key args:
  `--tensor-parallel-size 4 --language-model-only --kv-cache-dtype fp8 --max-model-len 1048576`
  `--max-num-batched-tokens 8192 --gpu-memory-utilization 0.97 --max-num-seqs 8|16 (c8|c16; use 8 for c1)`
  `--engram-config '{"cpu_offload":true,"use_thp":true}'`
  `--attention-config '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}'`
  `--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"synthetic","enable_adaptive_verification":false,"synthetic_acceptance_length":3.51}'`
  `--compilation-config` and `--max-cudagraph-capture-size` exactly as in the recipe for that conc.
  env `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_RUST_FRONTEND=1`.
- Keep synthetic acceptance 3.51 -- MI355X uses the same simulated AL; real acceptance would make the numbers incomparable.
- Record: image digest, `vllm --version`, driver, GPU clocks/power cap (`nvidia-smi -q -d CLOCK,POWER | head`).

## Deliverable 1 -- clean fixed-shape numbers (no profiler)

AgentX shape on this run: ISL median ~109k (p10 45k, p90 423k), mostly prefix-cached; OSL mean ~1.1k.
`vllm bench serve --dataset-name random --ignore-eos`, prefix caching ON (default), one server per conc:

| case | ISL | OSL | conc | num-prompts |
|---|---|---|---|---|
| D64 | 65536 | 1024 | 1 / 8 / 16 | 3 x conc (min 4) |
| D128 | 131072 | 1024 | 1 / 8 | 3 x conc (min 4) |

Report per row: mean/p50/p90 TTFT, TPOT, ITL (ms), output tok/s, total tok/s, and the server's last
`SpecDecoding metrics` line (must show mean AL ~3.5).

## Deliverable 2 -- torch profiler traces

Enable the profiler the way this image supports it (check `vllm serve --help | grep -i profil`; either env
`VLLM_TORCH_PROFILER_DIR=<dir>` or `--profiler-config`). Keep traces SHORT (bounded iterations or ~1-2 s windows),
with_stack off, record_shapes on. Use `POST /start_profile` / `POST /stop_profile`.

1. **Decode steady state**, D64 at conc 1, 8, 16: start the bench, wait until every request has its first token
   (all prefills done), then start_profile, ~30-40 decode steps, stop_profile.
2. **Prefill with cached prefix** (AgentX pattern), conc 1: send prompt P (100k tokens, max_tokens 1), then P + 4096 new
   tokens (max_tokens 1) under the profiler. That captures one ~4k extend over a 100k cached prefix (chunk <= 8192).
3. **Optional**: same as (1) at conc 16 with D128.

## Deliverable 3 -- kernel summary (so nobody has to open GBs of traces)

For each trace above, from the GPU (device) events of rank 0:
- per-step time: one decode step = target verify forward + DSpark draft; give mean ms/step and the step count used.
- top-30 kernels by total GPU time as CSV `kernel_name(<=120 chars),calls,total_us,us_per_step,pct`.
- grouped into: attention (sparse MLA + indexer), MoE (incl. routing), dense GEMM, all-reduce/comm, engram,
  norm/rope/elementwise, sampling/draft-specific, other; plus GPU idle % within the window (CPU/launch overhead).

Put the tables in `results/b200_tp4_profile_1006.md`. Compress traces (`*.json.gz`), keep them on the B200 node, and
write their path + sizes in the doc; the user will relay them.

## Do not

- Do not change the recipe args beyond what is listed (no tuning) -- we want the InferenceX config as-is.
- Do not run the full AgentX trace again; run 37070984585 already has it.
- No hostnames / internal paths in anything that goes upstream.
