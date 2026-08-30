# DeepSeek-V4-Pro serving perf — experiment log

Working log of the DeepSeek-V4-Pro serving-performance experiments on 8×MI355X.
For the reusable how-to see `SKILL.md`; this file records *what was run and what
we found*, for review.

> **Topic docs** (not daily logs):
> - `TRACE_PROFILING.md` — how-to for decode/prefill trace capture (wave-tail
>   method), the CUDA-graph per-kernel-timing caveat, trace analysis snippets,
>   collective microbench, TTFT attribution from batch logs, and gotchas.
> - `TBO_RESEARCH.md` — ATOM `--enable-tbo` vs sglang `--enable-two-batch-overlap`:
>   mechanisms, why sglang needs EP but ATOM doesn't, what ATOM overlaps in the
>   no-EP DP-attention case, and feasibility/effort of adding non-EP TP-MoE TBO to
>   sglang (DeepseekV2 ~1-2wk vs DSV4 ~3-5wk).

> **Daily split**: this file grew too large; newer days are logged in dated files
> `EXPERIMENT_LOG_YYYY-MM-DD.md` in this folder.
> - `EXPERIMENT_LOG_2026-06-23.md` — ATOM decode trace; root-cause of decode
>   all_reduce vs reduce_scatter (gatherv/ROCM700A gating); ROCM700A 0-vs-1 A/B;
>   collective microbench; **aiter reduce_scatter PoC** (`SGLANG_DP_USE_REDUCE_SCATTER`):
>   1k/1k c512 +3.07%, gsm8k 0.9507, TTFT side-effect = admission timing.

- **Date**: 2026-06-09
- **Hardware**: 8× AMD Instinct MI355X (gfx950)
- **Model**: `/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro/` (FP8 checkpoint, 64 shards)
- **Bench client**: ATOM `atom.benchmarks.benchmark_serving` (`--backend vllm` → `/v1/completions`)
- **Common bench params**: `random-range-ratio=0.8`, `--ignore-eos` (exact OSL, verified
  `total_generated == num_prompts*OSL`), `request-rate inf` (closed loop),
  `num-prompts = conc*8`, `num-warmups = conc*2`.
- **interactivity** = output tokens/s per user = `1000 / median_TPOT_ms`.

Scripts: `useful-scripts/benchmarking/dsv4/` — `run_atom_dsv4.sh` (orig),
`run_atom_dsv4_aligned.sh`, `run_sgl_dsv4.sh` (orig), `run_sgl_dsv4_aligned.sh`,
`sweep_dsv4_atom_client.sh`, `sweep_dsv4_sglang_client.sh`, `summarize_dsv4.py`.
Raw results: `/workspace/bench_results_dsv4_*`.

---

## Index (experiments by date)


### [2026-06-09](EXPERIMENT_LOG_2026-06-09.md)

- Experiment 1 — ATOM full sweep (two parallelism configs)
- Experiment 2 — SGLang vs ATOM, apples-to-apple (8k/1k, conc 64)
- Experiment 3 — ATOM multi-stream vs single-stream (8k/1k, conc 64)
- Code changes / patches made
- Caveats / notes
- Experiment 4 — Root-cause of the tp8+dp8 gap (SGLang vs ATOM, both single-stream)
- Experiment 5 — SGLang scheduling knobs to close the gap
- Experiment 6 — Re-profile SGLang WITH delayer; isolate the residual gap
- Experiment 7 — SGLANG_USE_ROCM700A=0 (best settings) — no effect
- Experiment 8 — bench_serving --profile capture (CPU+GPU) → Python attribution
- Known Issues
- Experiment 9 — Full SGLang tp8+dp8 sweep (delayer + ROCM700A=0) vs ATOM
- Experiment 10 — Matched stage-separated profile (8k/64, c64): SGLang vs ATOM
- Experiment 11 — all-reduce gap: TP-MoE collective pattern (NOT EP vs TP)
- Experiment 12 — ROOT CAUSE of the decode moe1 1.7× gap = MoE routing spread
- Open follow-ups

### [2026-06-10](EXPERIMENT_LOG_2026-06-10.md)

- Exp: apples-to-apple MoE routing (active-expert gap) — 2026-06-10

### [2026-06-11](EXPERIMENT_LOG_2026-06-11.md)

- Experiment 13 — ATOM multi-stream vs single-stream, 8k/1k dp8 high-conc sweep — 2026-06-11
- Experiment 14 — c256 profiling: DECODE is equal, the gap is PREFILL — 2026-06-11
- Experiment 15 — c256 PREFILL profiling: chunking + per-layer all-reduce — 2026-06-11
- Experiment 16 — chunked-prefill-size lever: helps prefill-heavy, NOT the real gap — 2026-06-11
- METHODOLOGY NOTES (apply to all subsequent SGLang experiments) — 2026-06-11
- Exp 17 — Two-lever throughput check @ real 8k/1k c256 — 2026-06-11
- Exp 18 — CORRECTION: all-reduce-count theory REFUTED; param semantics — 2026-06-11

### [2026-06-12](EXPERIMENT_LOG_2026-06-12.md)

- Exp 19 — Apple-to-apple PREFILL trace: methodology failure + corrections — 2026-06-12
- Exp 20 — prefill-delayer ON/OFF A/B: delayer is ESSENTIAL, not the gap — 2026-06-12
- Exp 21 — ATOM scheduler code study vs SGLang (mechanism comparison) — 2026-06-12
- Exp 21b — CORRECTION: ATOM does NOT use EP; option A premise was WRONG — 2026-06-12
- Exp 22 — ROOT CAUSE FOUND: variable-length vs MAX-pad DP-MoE gather — 2026-06-12
- Exp 23 — DP-MoE gather comm microbench (isolates the primitive) — 2026-06-12
- Exp 24 — Padding-mode confirmation: SGLang ALWAYS uses SUM_LEN — 2026-06-12
- Exp 25 — Port ATOM all_gatherv+reduce_scatterv into SGLang (WIP, 2 bugs hit) — 2026-06-12
- Exp 26 — gatherv Bug 2 FIXED (zero-pad buffer) + gsm8k correctness PASS — 2026-06-12
- Exp 27 — gatherv throughput A/B: NO improvement (≈0%) — 2026-06-12
- Exp 28 — gatherv TRACE verify: the gather was never the expensive collective — 2026-06-12
- Exp 29 — ATOM vs SGLang per-layer MoE output reduce OP (code comparison) — 2026-06-12
- Exp 30 — Symmetric reduce_scatterv combine: WRONG for SGLang (gsm8k 95%→43%) — 2026-06-12
- Exp 31 — CORRECTION: Exp 30 was a DOUBLE-REDUCE bug, not a semantic mismatch — 2026-06-12
- Exp 32 — Symmetric pair throughput: +2.8% (real, small win) — 2026-06-12
- Exp 33 — Concurrency sweep: symmetric pair win GROWS with concurrency — 2026-06-12

### [2026-06-15](EXPERIMENT_LOG_2026-06-15.md)

- Exp 34 — A-fix: gatherv prefill path was silently falling back to all_reduce — 2026-06-15

### [2026-06-16](EXPERIMENT_LOG_2026-06-16.md)

- Exp 35 — c256 gap localized to prefill per-token COMPUTE (+8%), symmetric measurement — 2026-06-16
- Exp 36 — prefill +8% broken down: it is NOT MoE/comm (shared), points to attn(MLA) — 2026-06-16
- Exp 37 — split the attn block; out_proj wo_a einsum is NOT slow (correction) — 2026-06-16
- Exp 38 — re-evaluate the 8x-redundant gate/router GEMM (C2 corrected) — 2026-06-16

### [2026-06-17](EXPERIMENT_LOG_2026-06-17.md)

- Exp 39 — full re-benchmark on updated codebases (2026-06-17)
- Exp 40 — re-added ATOM_DISABLE_SIDE_STREAMS flag + ATOM single vs multi-stream (2026-06-17)
- Exp 41 — SGLang c512 stability (3 repeats) (2026-06-17)
- Exp 42 — c512 gap root-cause + levers tried (swa-ratio, mixed-chunk) (2026-06-17)
- Exp 43 — client validation: SGLang client vs ATOM client, same server (2026-06-17)

### [2026-06-18](EXPERIMENT_LOG_2026-06-18.md)

- Exp 44 — c512 levers: conservativeness (neutral) + chunk-size (helps) + CORRECTED mechanism (2026-06-18)
- Exp 45 — chunk-size sweep: 8192 stability + 4096 + 8k validation (2026-06-18)
- Exp 46 — old vs new ATOM at c512: is ATOM's speed a recent change? (2026-06-18)
- Exp 47 — pure-prefill (OSL=1) compute: is SGLang's per-step prefill slower? (2026-06-18)
- Exp 48 — prefill TRACE: split the ~20% into raw-kernel vs overhead (2026-06-18)
- Exp 50 — shared-expert-local PoC: +6–7% prefill (2026-06-18)
- Exp 51 — ALL 3 prefill levers stacked: 84% → 97–98% of ATOM (2026-06-18)
- Exp 52 — ALL3 c512 END-TO-END A/B (OSL=1024, with decode) (2026-06-18)
- Exp 53 — A2: output inverse-RoPE full-fuse (contiguous kernel) (2026-06-18)
- Exp 54 — make CK-GEMM + batched/contig-RoPE DEFAULT-ON for DSV4 (no env) (2026-06-18)
- Exp 49 — FIX the prefill kernel gap: Triton→CK GEMM + batched RoPE (2026-06-18)

### [2026-06-21](EXPERIMENT_LOG_2026-06-21.md)

- Exp 55 — Port shared-expert-local to sglang-upstream + A/B; debunk the "upstream is slower" / BLOCK_M scare (2026-06-21)
- Exp 56 — C5: extend shared-expert-local to DECODE → fixes the c512 1k/1k regression (2026-06-21)

### [2026-06-22](EXPERIMENT_LOG_2026-06-22.md)

- Exp 57 — flat-row RoPE kernel: attention-output rope 168us -> 59us, BELOW ATOM (2026-06-22)
- Exp 58 — gatherv MoE-gather: 2 redundant 940MB DtoD copies removed (2026-06-22)

### [2026-06-23](EXPERIMENT_LOG_2026-06-23.md)

- ATOM decode trace; decode all_reduce-vs-reduce_scatter root cause; ROCM700A 0-vs-1 A/B; collective microbench; aiter reduce_scatter PoC (SGLANG_DP_USE_REDUCE_SCATTER)

### [2026-06-24](EXPERIMENT_LOG_2026-06-24.md)

- reduce_scatter PoC -> shippable: gatherv SUM_LEN gating + helper routing; env rename SGLANG_DP_USE_REDUCE_SCATTER + platform-conditional default; no-regression validation (8k/1k 37,701 / 1k/1k 18,666, gsm8k 0.9545); PR #29103; TBO research + TRACE_PROFILING docs

### [2026-06-25](EXPERIMENT_LOG_2026-06-25.md)

- NEW regime 70k/300 low-conc (2–32) sweep: TP8 vs DP-attention, chunk, swa, prefill-delayer. Best-of: c2/4 TP8 (13.9k/17.7k), c8/16/32 DP-attn (29.2k/36.8k/42.1k). Findings: TP8 saturates ~conc8; DP-attn scales + 0 retract but needs mem-fraction 0.80 (else MoE OOM on gathered global batch); chunk size neutral (16384/rank chosen); DSV4 swa sub-pool binds TP8 at c32 (mem-edge); **prefill-delayer OFF wins +6–16%** (opposite of 8k/1k high-conc). sglang-upstream pinned via PYTHONPATH (namespace-shadow fix).
- (PM) **DSV4 EP+TBO implemented** (prefill two-batch-overlap on mori a2a): op-decomposed `DeepseekV4DecoderLayer` (reuse DeepseekV2MoE op_*; add op_mhc_* + MQALayer.op_attn; disable fused-mHC under TBO), registered in `init_new_tbo`, DSV4 TBO driver in model.forward, `TboAttnBackend.__getattr__`→primary, hash op_select_experts fix. **gsm8k EP+TBO 0.9600/0.9567** (correct). OPEN: intermittent `HSA_STATUS_ERROR_OUT_OF_RESOURCES` under sustained load (not VRAM, not aiter JIT).
