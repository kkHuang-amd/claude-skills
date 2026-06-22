# DeepSeek-V4-Pro serving perf — experiment log

Working log of the DeepSeek-V4-Pro serving-performance experiments on 8×MI355X.
For the reusable how-to see `SKILL.md`; this file records *what was run and what
we found*, for review.

- **Date**: 2026-06-09
- **Hardware**: 8× AMD Instinct MI355X (gfx950)
- **Model**: `/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/` (FP8 checkpoint, 64 shards)
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

## Experiment 1 — ATOM full sweep (two parallelism configs)

Server: `run_atom_dsv4.sh` env (`ATOM_DISABLE_MMAP=true ATOM_MOE_GU_ITLV=1
AITER_BF16_FP8_MOE_BOUND=0`), `-tp 8 --kv_cache_dtype fp8`, ATOM defaults
otherwise. **No `ATOM_USE_TRITON_MOE=1`** (see caveat at bottom).

- **tp8** = plain TP8 (no dp-attention)
- **tp8+dp8** = TP8 + `--enable-dp-attention`

### tp8 (no dp-attention)
| workload | conc | total tok/s | tok/s/gpu | out tok/s | TTFT ms | TPOT ms | ITL ms | interact | E2E ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1k/1k | 2  | 241   | 30   | 121  | 207.7 | 16.23 | 16.06 | 61.6 | 15,042 |
| 1k/1k | 4  | 453   | 57   | 228  | 207.4 | 16.65 | 16.14 | 60.1 | 15,944 |
| 1k/1k | 8  | 840   | 105  | 420  | 209.5 | 18.01 | 16.83 | 55.5 | 16,872 |
| 1k/1k | 16 | 1,485 | 186  | 739  | 208.8 | 21.18 | 18.66 | 47.2 | 19,386 |
| 1k/1k | 32 | 2,435 | 304  | 1,214| 209.3 | 25.69 | 20.54 | 38.9 | 23,398 |
| 1k/1k | 64 | 3,628 | 453  | 1,819| 215.1 | 34.87 | 25.33 | 28.7 | 32,028 |
| 8k/1k | 4  | 1,909 | 239  | 215  | 332.2 | 17.99 | 17.12 | 55.6 | 17,355 |
| 8k/1k | 8  | 3,364 | 421  | 379  | 332.6 | 19.78 | 17.75 | 50.6 | 18,724 |
| 8k/1k | 16 | 5,717 | 715  | 634  | 341.6 | 23.96 | 19.51 | 41.7 | 22,380 |
| 8k/1k | 32 | 9,028 | 1,128| 1,001| 367.9 | 30.39 | 21.23 | 32.9 | 28,359 |
| 8k/1k | 64 | 12,523| 1,565| 1,397| 376.5 | 44.94 | 26.16 | 22.3 | 42,012 |

### tp8+dp8 (--enable-dp-attention)
| workload | conc | total tok/s | tok/s/gpu | out tok/s | TTFT ms | TPOT ms | ITL ms | interact | E2E ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1k/1k | 64   | 3,702  | 463   | 1,856 | 821.3   | 32.67  | 29.55 | 30.6 | 30,788 |
| 1k/1k | 128  | 6,368  | 796   | 3,181 | 629.0   | 38.15  | 33.91 | 26.2 | 35,536 |
| 1k/1k | 256  | 11,093 | 1,387 | 5,543 | 499.2   | 44.02  | 37.26 | 22.7 | 40,860 |
| 1k/1k | 512  | 16,759 | 2,095 | 8,381 | 511.9   | 59.43  | 44.75 | 16.8 | 54,800 |
| 1k/1k | 1024 | 23,158 | 2,895 | 11,583| 632.0   | 83.81  | 56.28 | 11.9 | 78,560 |
| 8k/1k | 64   | 13,044 | 1,631 | 1,455 | 1,671.3 | 41.77  | 30.12 | 23.9 | 40,331 |
| 8k/1k | 128  | 19,938 | 2,492 | 2,212 | 1,761.3 | 54.36  | 33.92 | 18.4 | 52,313 |
| 8k/1k | 256  | 27,809 | 3,476 | 3,085 | 1,710.7 | 79.37  | 38.22 | 12.6 | 74,977 |
| 8k/1k | 512  | 34,036 | 4,255 | 3,783 | 2,682.0 | 128.07 | 45.31 | 7.8  | 122,687|

**Takeaways**: dp8 scales throughput much higher at heavy concurrency (8k/1k c512
= 34k tok/s, 4,255 tok/s/gpu) but with higher TTFT; plain tp8 keeps low/stable
TTFT (~0.2–0.4s) but saturates by c64.

---

## Experiment 2 — SGLang vs ATOM, apples-to-apple (8k/1k, conc 64)

Goal: same engine knobs, isolate engine differences. Aligned both servers:
`fp8 KV`, `chunked-prefill/max-num-batched = 16384`, `cuda-graph max = 512`,
`max-running/max-num-seqs = 512`, `mem-fraction/gpu-util = 0.90`,
`page/block size = 256`, **prefix/radix cache OFF** (ATOM auto-disables prefix
cache for DSV4 — "SWA buffer is not cacheable" — so it already matched SGLang's
`--disable-radix-cache`). Scripts: `run_atom_dsv4_aligned.sh`,
`run_sgl_dsv4_aligned.sh` (both `DP_MODE=tp8|tp8dp8`).

| config | engine | total tok/s | tok/s/gpu | out tok/s | TTFT ms | TPOT ms | ITL ms | E2E ms |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| tp8      | ATOM   | 12,483 | 1,560 | 1,392 | 363.4   | 44.79 | 26.12 | 41,935 |
| tp8      | SGLang | 11,666 | 1,458 | 1,301 | 484.5   | 47.60 | 26.38 | 44,693 |
| tp8+dp8  | ATOM   | 12,983 | 1,623 | 1,448 | 1,656.4 | 42.03 | 30.27 | 40,224 |
| tp8+dp8  | SGLang | 8,550  | 1,069 | 954   | 1,183.0 | 67.42 | 30.11 | 63,018 |

**Takeaways**:
- **tp8**: SGLang ≈ 93% of ATOM. Close.
- **tp8+dp8**: SGLang only ≈ 66% of ATOM, and *slower than its own tp8* (8,550 <
  11,666). At conc 64 with dp8 each rank only sees ~8 reqs; SGLang's dp-attention
  overhead doesn't pay off at this low per-rank batch, while ATOM's still helps.
- Sanity: aligning ATOM's block-size (128→256) and explicit prefix-off changed
  ATOM by <0.5% (12,483 vs prior 12,523; 12,983 vs 13,044) → low-impact knobs;
  the meaningful alignment (fp8 KV etc.) was already in place.

---

## Experiment 3 — ATOM multi-stream vs single-stream (8k/1k, conc 64)

Motivation: SGLang runs single-stream; ATOM uses 3 streams (main + alt +
compress). Added env gate `ATOM_DISABLE_SIDE_STREAMS=1` in
`atom/models/deepseek_v4.py` (sets `alt_stream`/`compress_stream = None`,
forcing compressor + shared-experts onto the main stream). All other knobs =
Experiment 2 aligned config.

| config | engine / mode | total tok/s | tok/s/gpu | TTFT ms | TPOT ms | ITL ms | E2E ms |
|---|---|--:|--:|--:|--:|--:|--:|
| tp8     | ATOM multi-stream  | 12,483 | 1,560 | 363.4   | 44.79 | 26.12 | 41,935 |
| tp8     | ATOM single-stream | 11,686 | 1,461 | 358.0   | 47.98 | 29.31 | 44,990 |
| tp8     | SGLang             | 11,666 | 1,458 | 484.5   | 47.60 | 26.38 | 44,693 |
| tp8+dp8 | ATOM multi-stream  | 12,983 | 1,623 | 1,656.4 | 42.03 | 30.27 | 40,224 |
| tp8+dp8 | ATOM single-stream | 12,299 | 1,537 | 1,708.6 | 44.38 | 32.78 | 42,487 |
| tp8+dp8 | SGLang             | 8,550  | 1,069 | 1,183.0 | 67.42 | 30.11 | 63,018 |

**Takeaways**:
1. Multi-stream is worth **~5–6%** throughput on ATOM (tp8 −6.4%, tp8+dp8 −5.3%
   when disabled), plus better TPOT.
2. **ATOM single-stream tp8 ≈ SGLang tp8** (11,686 vs 11,666) — i.e. ATOM's tp8
   lead over SGLang was *essentially all from multi-stream overlap*. True
   single-vs-single, they're on par.
3. **dp8 gap is NOT a stream issue**: ATOM single-stream dp8 (12,299) is still
   ~44% faster than SGLang dp8 (8,550). SGLang's dp-attention inefficiency at
   conc 64 is a separate problem.

---

## Code changes / patches made

1. **`sglang/python/sglang/srt/configs/cohere2_moe.py`** — fix import crash
   (`StrictDataclassDefinitionError`) with huggingface_hub ≥1.x: drop
   `huggingface_hub`'s `@strict`, add `@dataclass`. Patch file saved at
   `/sgl-workspace/sglang/cohere2_moe_strict_import_fix.patch`. (Reapply with
   `git apply` if reverted.)
2. **`atom/models/deepseek_v4.py`** — env-gated side-stream disable:
   `ATOM_DISABLE_SIDE_STREAMS=1` → `alt_stream/compress_stream = None`. Default
   behavior unchanged. Applied to BOTH the runtime copy
   (`/opt/venv/lib/python3.10/site-packages/atom/...`, the one actually imported)
   and the source tree (`/sgl-workspace/ATOM/atom/...`).
3. **Bench client shim** (`dsv4/bench_dsv4.py`) — wraps `sglang.bench_serving`
   so it tolerates ATOM's usage-only final SSE chunk (no `choices` key). Only
   needed when driving an ATOM server via the *SGLang* client; the ATOM client
   doesn't need it.

## Caveats / notes
- **`ATOM_USE_TRITON_MOE=1` NOT set** in these runs. The ROCm/ATOM recipe says
  it's required for V4-Pro correctness (else GSM8K ~0.95→~0.6). Perf comparisons
  here are internally consistent (all runs same config), but absolute numbers /
  accuracy may not reflect the "correct" MoE path. Validate with `lm-eval.sh`
  before trusting absolutes.
- All bench points completed fully (e.g. `512/512`, `8192/8192`).
- Remaining tiny known diff before alignment: page-size 256 (SGLang) vs ATOM KV
  block 128 — aligned to 256 in Exp 2/3; impact measured <0.5%.

## Experiment 4 — Root-cause of the tp8+dp8 gap (SGLang vs ATOM, both single-stream)

Both single-stream, dp8, aligned config. Goal: why SGLang ≈66% of ATOM at
8k/1k c64. Used a fast gap config and torch traces.

### Fast config that reproduces the gap (no need for OSL=1k / prompts=conc*8)
Continuous, **conc 64, ISL 8192, OSL 256, num-prompts 192, ratio 0.8** (~1 min):

| engine (single-stream dp8) | output tok/s | TPOT ms | median ITL ms |
|---|--:|--:|--:|
| ATOM   | 812 | 65.6 | 32.7 |
| SGLang | 542 | 101.4 | 31.3 |

Key observations narrowing it down:
- **Pure decode is equal**: lockstep np64 → SGLang TPOT 43.6 ≈ ATOM; median ITL
  ~31–33 ms identical both. Decode kernels are equally fast.
- The gap appears only in the **mixed prefill+decode** (continuous) regime;
  SGLang TPOT(101) ≫ ITL(31) → decode periodically stalled by prefill.

### Trace evidence (GPU-only, OSL=32 frequent-prefill load, 915 forwards/rank)
Tools: `dsv4/step_timeline.py` (per-forward wall via `moe1` marker),
`dsv4/analyze_trace.py` (kernel time + GPU busy).

| metric (DP rank 0) | ATOM single-stream | SGLang |
|---|--:|--:|
| span for 915 forwards | **4011 ms** | **7050 ms** (1.76× slower) |
| GPU busy fraction | 41% | **77%** |
| total GPU kernel-time | 1643 ms | **5601 ms** (3.4× more) |
| `moe1` (routed gemm1) median dur | 60 µs | 776 µs |
| step structure | **separate** prefill (61 steps, ~22 ms) + **cheap decode** (853 steps, p50 **0.59 ms**) | **fused** prefill+decode every step (p50 **6.67 ms**) |

### Root cause
**Prefill/decode scheduling differs:**
- **ATOM** runs prefill in *dedicated* forwards (~22 ms bursts) and keeps
  pure-decode forwards extremely cheap (0.59 ms, GPU ~idle at 8 seqs/rank). GPU
  only 41% busy → finishes the same forwards in 4.0 s.
- **SGLang** *fuses a chunked-prefill slice into (almost) every forward*
  (mixed batching, `moe1` median 776 µs vs ATOM's 60 µs decode). Every decode
  token then "pays" prefill cost → each step ~6.7 ms, GPU 77% busy doing **3.4×
  more total kernel work**, span 7.05 s for the same 915 forwards → the ~1.76×
  (≈ the measured ~1.5×) throughput gap.

**Trade-off, not a pure bug**: SGLang's fusion gives *lower TTFT*
(1183 ms vs ATOM 1671 ms at c64) but much worse decode throughput on this
prefill-heavy 8k workload. With dp-attention the cost compounds: all 8 ranks run
the TP MoE in lockstep, so a heavy fused step on any rank inflates every step.

### Actionable hypotheses for closing the SGLang gap
- Tune SGLang prefill/decode scheduling so decode steps stay cheap (decode
  priority / smaller `--chunked-prefill-size` / avoid fusing large prefill
  chunks into decode steps).
- Re-profile after each knob change with the same fast config + `step_timeline.py`.

Traces: `/workspace/prof3_sgl/` (SGLang, 8×10 MB), `/workspace/prof3_atom/dp*_tp0/`
(ATOM). First decode-only capture: `/workspace/prof_{sgl,atom}_tp8dp8/`.

## Experiment 5 — SGLang scheduling knobs to close the gap

Hypothesis from Exp 4: SGLang interleaves/fuses prefill into decode forwards;
ATOM uses a **PrefillDelayer** by default (its log:
`PrefillDelayer ... max_delay_passes=30 max_delay_ms=5000`). SGLang has the SAME
feature but OFF by default: `--enable-prefill-delayer` (defaults
`max_delay_passes=30`, `max_delay_ms=None`).

Test config (single-stream dp8, fast gap config): conc 64, ISL 8192, OSL 256,
num-prompts 192, ratio 0.8.

| SGLang config | output tok/s | TPOT ms | TTFT ms | vs baseline |
|---|--:|--:|--:|--:|
| baseline (no delayer) | 542 | 101.4 | 1782 | — |
| **+ prefill-delayer** (passes 30, ms 5000) | **667** | 78.6 | 2172 | **+23%** |
| + delayer + num-continuous-decode-steps 4 | 666 | 78.7 | 2177 | no change |
| + stronger delayer (passes 120, ms 10000) + schedule-conservativeness 2.0 | 629 | 76.9 | 2499 | worse (over-throttled) |
| **ATOM single-stream (target)** | **812** | 65.6 | 1671 | — |

**Findings**
- **`--enable-prefill-delayer` is the key lever** — it's literally the same
  mechanism ATOM runs by default. With SGLang's defaults it recovers **~46% of
  the gap** (542→667; gap to ATOM 270→145 tok/s). Confirms the root cause is
  prefill/decode scheduling.
- `--num-continuous-decode-steps` had no effect here.
- Over-tuning (longer delay + higher conservativeness) *hurts* — prefill gets
  starved, TTFT rises, throughput drops. Default delayer is the sweet spot.
- **Remaining ~18% gap** (667 vs 812) is NOT scheduling — likely prefill/MoE
  kernel efficiency or dp-attention all-gather overhead (Exp 4 showed SGLang did
  3.4× more total GPU kernel work). Needs a separate kernel-level investigation.

**Recommendation**: run SGLang DSV4 dp8 with `--enable-prefill-delayer` (defaults).

## Experiment 6 — Re-profile SGLang WITH delayer; isolate the residual gap

Used SGLang's own profiling skills (`/sgl-workspace/sglang/.claude/skills/`):
`generate-profile` and `llm-torch-profiler-analysis` (unified analyzer
`scripts/analyze_llm_torch_profile.py`, gives kernel / overlap / fuse tables).

### (a) Decode forwards dropped to ~0.6 ms — VERIFIED
SGLang dp8 + `--enable-prefill-delayer`, decode-heavy load (OSL512), CPU+GPU
window:
- decode forward wall **p50 0.57 ms, p90 0.61 ms** (was 6.7 ms fused without
  delayer) — now matches ATOM (0.59 ms).
- **cuda graph now recorded**: trace contains `cuda_graph_runner.py:replay`,
  `can_run`, `is_cuda_graph`. The earlier `prof3_sgl` had "no cuda_graph"
  because those captured forwards were *eager prefill* (moe1 776 µs), not graphed
  decode; with the delayer the decode runs as graph replay.

### (b) Residual ~18% gap = SGLang's all-reduce, not decode/kernels
Unified-analyzer kernel tables (top GPU-time share):

| | SGLang (mixed, OSL32) | ATOM prefill | ATOM decode |
|---|---|---|---|
| dominant comm kernel | **`quickreduce::allreduce_twoshot` (Q8) 27.8%** (+ `cross_device_reduce_2stage` 3.3% ≈ **31%**) | `ncclDevKernel` 27.1% | **`allgather_vec` 6.6%** (under cuda graph) |
| MoE gemm1/gemm2 | 10.4% / 8.1% | 10.4% / 11.5% | 10.6% / 13.4% |

- **SGLang spends ~28–31% of GPU time in TP all-reduce** (`quickreduce`
  two-shot Q8 all-reduce, ~1.7 launches/forward). ATOM's **decode** all-reduce is
  only **6.6%** (`aiter allgather_vec` + `reduce_scatter`, fused into the cuda
  graph). The MoE/GEMM kernels themselves are comparable.
- This is the residual gap source: SGLang's TP/dp-attention collective strategy
  (`quickreduce` all-reduce) is much heavier than ATOM's allgather+reduce-scatter.

### Next levers to try for the residual gap (all-reduce)
- Switch SGLang all-reduce backend: `SGLANG_USE_AITER_AR=1` (aiter custom AR) or
  `--enable-fused-moe-sum-all-reduce`, or try `--disable-custom-all-reduce`
  (RCCL) vs the quickreduce path; re-profile and compare the comm-kernel share.
- Confirm whether `quickreduce` Q8 (quantized) AR is being used by default and if
  a non-quantized / different codec is faster here.

Traces: `/workspace/prof4_sgl_delayer/` (decode, cuda-graph visible),
`/workspace/prof4_sgl_pf/` (prefill-inclusive), `/workspace/prof3_atom/`.

## Experiment 7 — SGLANG_USE_ROCM700A=0 (best settings) — no effect

Best SGLang settings (aligned + `--enable-prefill-delayer`) with
`SGLANG_USE_ROCM700A=0` (default is 1), gap config conc 64 / ISL 8192 / OSL 256:

| setting | output tok/s | TPOT ms | GPU busy | top comm kernel |
|---|--:|--:|--:|---|
| ROCM700A=1 (best) | 667 | 78.6 | 91% | quickreduce twoshot Q8 27.8% |
| **ROCM700A=0** | **672** | 77.9 | 92% | quickreduce twoshot Q8 27.8% |

**Verdict: no meaningful difference.** Throughput within noise (672 vs 667),
identical step structure, and the all-reduce is *still* `quickreduce
allreduce_twoshot` (Q8) at 27.8% (+ `cross_device_reduce_2stage` 4.4%).
`SGLANG_USE_ROCM700A` is not the lever for the all-reduce overhead. Trace:
`/workspace/prof5_sgl_no700a/` (rank0 kept).

## Experiment 8 — bench_serving --profile capture (CPU+GPU) → Python attribution

Captured via SGLang's own `python -m sglang.bench_serving --profile
--profile-num-steps 40 --profile-activities CPU GPU` (best config, ROCM700A=0).
CPU activities give kernel→Python attribution the GPU-only traces lacked.

Gotchas:
- `--profile-steps` does NOT cap the capture (sends `num_steps=None` → profiles
  the WHOLE run; produced a 2.5 GB trace and slowed the run to 300 s). Use
  **`--profile-num-steps N`** (maps to server `num_steps`, auto-stops).
- CPU+GPU traces are ~6× larger than GPU-only (~270 MB/rank for 40 mixed steps).
- Throughput during a CPU-profiled run is not representative (overhead); read the
  kernel %, not absolute times.

### Result — confirms communication-bound, with exact call sites
Unified analyzer (stage auto-split):

DECODE forwards (graphed):
| kernel | category | share of decode GPU time | Python site |
|---|---|--:|---|
| `cross_device_reduce_2stage` | communication | **61.8%** | `parallel_state.py:573 all_reduce` → `outplace_all_reduce` |
| `mfma_moe1` | moe | 3.9% | `moe_runner/aiter.py:123` |

ALL (incl. prefill):
| kernel | category | share | Python site |
|---|---|--:|---|
| `quickreduce allreduce_twoshot` (Q8) | communication | **29.3%** | `custom_all_reduce_ops.py:150 qr_all_reduce` |
| `mfma_moe1` / `mfma_moe2` | moe | 10.8% / 8.5% | `moe_runner/aiter.py:123` |
| `pa_sparse_prefill` | attn | 5.6% | `dsv4/.../paged_prefill.py:283` |
| Memcpy DtoD | memory | 1.3% | `dp_attention.py:463 _dp_gather_via_all_reduce` |

**Conclusion**: SGLang DSV4 dp8 decode is **~62% all-reduce** (`cross_device_reduce_2stage`
via `parallel_state.all_reduce`); the mixed/prefill window is ~29% `quickreduce`
all-reduce. The dp-attention token gather itself is done via all-reduce
(`_dp_gather_via_all_reduce`). So the residual gap vs ATOM is squarely the
TP + dp-attention **collective strategy**, with exact call sites now known.
Trace (rank0 kept): `/workspace/prof6_benchprofile/`.

## Known Issues

### KI-1: prefill-delayer over-throttles at extreme concurrency (1k/1k c1024)
SGLang dp8 with `--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000`
at **1k/1k conc=1024**: median **TTFT explodes to ~63 s** and total throughput
plateaus (15,376 tok/s ≈ the c512 value of 15,026), i.e. it does NOT scale from
c512→c1024. 8k/1k c512 shows the same symptom milder (TTFT ~16 s).
- Cause (hypothesis): at very high concurrency the delayer (30 passes / 5 s)
  holds back prefill too aggressively → large prefill queue → huge TTFT, and the
  running batch can't grow enough to lift decode throughput.
- Status: **deferred** (not the current focus). Fix candidates to try later:
  shorter `--prefill-delayer-max-delay-ms` / fewer passes, a concurrency-based
  cutoff, or disable the delayer at extreme concurrency.
- Data: `/workspace/bench_results_dsv4_sgl_delayer_sweep/sglangClient_dsv4_isl1024_osl1024_c1024.jsonl`.

### KI-2: SGLang decode kernels hidden by raw-CUDA-graph replay (ATOM's are not)
SGLang **decode** runs each forward as a raw `torch.cuda.CUDAGraph` replay; ROCm's
torch profiler logs each replay as ONE opaque `hipGraphLaunch` and does NOT expand
the internal per-layer kernels. Evidence (prof9, by-stage GPU-only):
`EXTEND hipGraphLaunch=0` (eager prefill, moe1=610 clean) vs `DECODE
hipGraphLaunch=8` (only 8 opaque launches; moe1=183 came only from the eager
*tail* steps as batch shrinks 64→0).

**Why ATOM doesn't have this problem (it IS also cudagraph):** ATOM uses
torch.compile / inductor cudagraph (`use_inductor=True, level=3`; trace shows
`## Call CompiledFxGraph ## ` annotations). The profiler EXPANDS each replay into
per-kernel events: ATOM decode `hipGraphLaunch=135` → `moe1(t32)=7930`
(135 replays × ~58 MoE layers ≈ 7830). So ATOM's decode kernel detail is real and
its Exp 6/10 decode numbers are trustworthy; SGLang's raw-graph replays stay
opaque, so SGLang's Exp 8/10 decode numbers came from the eager tail (indicative
only). SGLang prefill (eager) numbers are solid.

- **Consequence**: decode comparisons so far are apples-to-oranges (ATOM = real
  graphed steady-state decode; SGLang = eager tail).
- **Fix**: profile SGLang decode with `--disable-cuda-graph` (eager) so every
  kernel is recorded — then the decode all-reduce/MoE breakdown is comparable to
  ATOM. (Graph-off changes decode perf but is correct for kernel attribution; this
  is the SGLang profiler skill's graph-OFF "mapping" trace.)
- **Additional evidence (tp8 capture attempt)**: with cuda graph ON, neither
  capture path yields a usable SGLang decode trace:
  - `bench --profile --profile-by-stage`: the DECODE stage *starts* (log:
    `Profiling starts for DECODE`) but never flushes a file (no
    `Stop profiling-DECODE`), even with a clean pure-decode phase (np=64).
  - HTTP `/start_profile`+`/stop_profile` (CPU+GPU) during decode: **`/stop_profile`
    HANGS the server** (had to kill it). The torch-profiler stop over CUDA-graph
    decode + CPU stalls.
  - Prefill (EXTEND) captures fine both ways (eager). → For ANY SGLang decode
    trace you MUST launch with `--disable-cuda-graph`.

## Experiment 9 — Full SGLang tp8+dp8 sweep (delayer + ROCM700A=0) vs ATOM

SGLang client (`sglang.bench_serving` via `bench_dsv4.py`), server = aligned +
`--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000` +
`SGLANG_USE_ROCM700A=0`. ratio 0.8, num-prompts=conc*8, warmups=conc*2. No profiler.
Summarizer: `dsv4/summarize_sgl_dsv4.py`. Results:
`/workspace/bench_results_dsv4_sgl_delayer_sweep/`.

| workload | conc | total tok/s | tok/s/gpu | out tok/s | TTFT ms | TPOT ms | ITL ms | interact | E2E ms |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 1k/1k | 64  | 3,617  | 452  | 1,805 | 816    | 33.7  | 31.1 | 29.7 | 31,623 |
| 1k/1k | 128 | 6,022  | 753  | 3,007 | 736    | 40.1  | 35.9 | 24.9 | 37,389 |
| 1k/1k | 256 | 9,775  | 1,222| 4,892 | 564    | 50.0  | 41.9 | 20.0 | 46,424 |
| 1k/1k | 512 | 15,026 | 1,878| 7,512 | 3,206  | 61.3  | 48.5 | 16.3 | 59,294 |
| 1k/1k | 1024| 15,376 | 1,922| 7,691 | **62,949** | 61.5 | 48.5 | 16.2 | 119,579 |
| 8k/1k | 64  | 11,222 | 1,403| 1,242 | 2,020  | 48.5  | 31.0 | 20.6 | 45,933 |
| 8k/1k | 128 | 16,788 | 2,099| 1,860 | 2,043  | 65.1  | 35.9 | 15.4 | 61,692 |
| 8k/1k | 256 | 22,664 | 2,833| 2,518 | 1,995  | 97.5  | 41.5 | 10.3 | 92,278 |
| 8k/1k | 512 | 27,239 | 3,405| 3,026 | 15,933 | 149.2 | 48.8 |  6.7 | 152,702|

### vs ATOM dp8 (Exp 1, ratio 0.8) — total tok/s
| workload | conc | SGLang | ATOM | SGLang/ATOM |
|---|--:|--:|--:|--:|
| 1k/1k | 64  | 3,617  | 3,702  | 98% |
| 1k/1k | 128 | 6,022  | 6,368  | 95% |
| 1k/1k | 256 | 9,775  | 11,093 | 88% |
| 1k/1k | 512 | 15,026 | 16,759 | 90% |
| 1k/1k | 1024| 15,376 | 23,158 | **66%** |
| 8k/1k | 64  | 11,222 | 13,044 | 86% |
| 8k/1k | 128 | 16,788 | 19,938 | 84% |
| 8k/1k | 256 | 22,664 | 27,809 | 82% |
| 8k/1k | 512 | 27,239 | 34,036 | 80% |

**Findings**
- With delayer + ROCM700A=0, SGLang reaches **80–98% of ATOM** on most points
  (vs ~66% pre-delayer at c64) — the delayer materially closes the gap.
- **Anomaly: 1k/1k c1024** — TTFT explodes to **63 s** and throughput plateaus
  (15,376 ≈ c512). At extreme concurrency the prefill delayer (30 passes / 5 s)
  over-throttles prefill → severe prefill queueing. 8k/1k c512 also shows
  elevated TTFT (16 s). Fix candidate: shorter delay cap or disable delayer at
  very high concurrency.
- Cross-engine caveat: ATOM column used the ATOM client; the two clients agreed
  within ~3% (Exp 2), so treat deltas as ±a few %.

## Experiment 10 — Matched stage-separated profile (8k/64, c64): SGLang vs ATOM

Goal: pin the 8k/1k c64 dp8 gap at kernel level with identical config + clean
prefill/decode separation. Config: ISL 8192, OSL 64, conc 64, num-prompts 128
(conc*2), warmup 64 (conc*1), ratio 0.8, GPU-only.
- SGLang (best: delayer + ROCM700A=0): `bench --profile --profile-by-stage
  --profile-num-steps 10 --profile-activities GPU` → separate EXTEND/DECODE.
- ATOM (single-stream dp8): `benchmark_serving --profile` whole short run.
Per-launch GPU time (rank0):

PREFILL:
| kernel | SGLang | ATOM |
|---|--:|--:|
| moe1 t128x256x256 | 784 µs ×610 | 10,563 µs ×610 |
| moe2 t64x256x256  | 619 µs ×610 | 3,327 µs ×610 |
| all-reduce | `quickreduce` 980 µs ×1220 (1196 ms) | `allgather_vec` 18 µs |

DECODE:
| kernel | SGLang | ATOM |
|---|--:|--:|
| moe1 t32x128x256 | 102 µs ×122 | 60 µs ×7686 |
| moe2 t32x256x256 |  90 µs ×61  | 71 µs ×7930 |
| all-reduce | `cross_device_reduce_2stage` 1244 µs ×308 (383 ms) | `allgather_vec` 18 µs |

**Findings**
1. The `moe1`/`moe2` per-launch gap is mostly **tokens-per-launch, not kernel
   speed** — same aiter kernels; ATOM concentrates prefill MoE into big launches
   (~16384 tok → 10.5 ms ≈ 0.65 µs/tok), SGLang does many smaller ones. (Minor
   genuine hint: decode `t32x128x256` SGLang ~102 µs vs ATOM ~60 µs at the same
   ~64-tok batch — possible ~1.7× decode-MoE diff, worth a closer look.)
2. **Dominant gap = the all-reduce / collective strategy.** ATOM uses only a
   cheap `allgather_vec` (~18 µs/launch) in both stages; SGLang uses heavy
   `quickreduce_twoshot` (~980 µs, prefill) and `cross_device_reduce_2stage`
   (~1244 µs, decode) — ~50–70× more expensive per collective op. This is the
   8k/1k c64 dp8 gap.

Traces: `/workspace/prof9_sgl/` (EXTEND/DECODE), `/workspace/prof9_atom/` (rank0).

## Experiment 11 — all-reduce gap: TP-MoE collective pattern (NOT EP vs TP)

**CORRECTION of an earlier mis-read.** The aiter log
`DP rank N, TP rank 0, EP rank N` is just aiter group *bookkeeping* (an EP rank is
always assigned per device); it does NOT mean ATOM's MoE runs expert-parallel.

Source of truth: `atom/model_ops/moe.py :: FusedMoEParallelConfig.make()`:
- `--enable-dp-attention` → `flatten_tp_across_dp()` flattens DP into the MoE
  sharding: `tp_size = dp_size*tp_size = 8` (MoE shards across all 8 devices).
- EP vs TP is gated by `enable_expert_parallel`, NOT by dp-attention:
  - `enable_expert_parallel=True`  → `use_ep=True`  → `ep_size=8, tp_size=1` (EP).
  - `enable_expert_parallel=False` → `use_ep=False` → `tp_size=8, ep_size=1` (TP).
- **ATOM ran with `enable_expert_parallel=False`** → MoE is **TP=8**, not EP.
- **SGLang** `ep_size=1` → MoE is **TP=8** too.

So **both engines run the MoE tensor-parallel across 8 devices.** The real gap is
the **collective pattern for dp-attention + TP-MoE output redistribution**:
- **ATOM**: all-gather tokens + **reduce-scatter** output (`allgather_vec` +
  `reduce_scatter`) — each rank gets back only its DP slice → cheaper.
- **SGLang**: all-gather + **all-reduce** (`quickreduce`) + separate dp gather
  (`_dp_gather_via_all_reduce`) → moves more data → heavier collective.

**Fixes to try (both still valid levers, reframed):**
1. SGLang `--ep-size 8` (+ maybe `--moe-a2a-backend deepep`) → switch MoE to true
   EP (dispatch/combine), avoiding the TP all-reduce entirely.
2. OR change SGLang's TP-MoE redistribution toward reduce-scatter/all-gather
   (e.g. `--enable-fused-moe-sum-all-reduce`, or AR backend), to mimic ATOM.

## Experiment 12 — ROOT CAUSE of the decode moe1 1.7× gap = MoE routing spread

Goal: why SGLang decode `mfma_moe1...t32x128x256` is ~1.7–2× slower than ATOM at
bs=64, same aiter kernel/lib. Method: dump the exact moe1 inputs from one bs=64
decode step in each engine (tp8, eager, single-stream, ratio 1.0), then replay
through the real flydsl kernel in isolation. Hooks: env-gated `DUMP_MOE1_DIR` at
SGLang `moe_runner/aiter.py` + ATOM `model_ops/moe.py`; microbench needs
`AITER_BF16_FP8_MOE_BOUND=0` to select the flydsl path (else CK mxgemm).

**#1 layout: identical.** w13/w2/w13_scale/w2_scale/hidden/topk all same
shape/stride/dtype (only cosmetic scale dtype label e8m0fnu vs uint8); gate_mode
interleave, swiglu 10, per_1x32, a1/a2_scale=None (dynamic) — both. Only kwarg
diff: `intermediate_pad` SGLang=0 vs ATOM=128.

**#2 microbench (isolated flydsl moe1, µs/launch):**
| dump | ipad | moe1 |
|---|--:|--:|
| SGLang | 0 (orig) | 126 |
| SGLang | 128 | 104 |
| ATOM | 128 (orig) | 55 |
| ATOM | 0 | 70 |
`intermediate_pad` matters a little but does NOT close the gap (same ipad → ATOM
still ~2× faster). So it's the **dumped tensors**, not kwargs/kernel.

**Cross-swap (decisive):** SGLang weights + **ATOM routing** → moe1 126.8→**70.1 µs**.

**Root cause = MoE expert routing spread.** From `topk_ids` (64 tok × 6):
- SGLang: **216** active experts (of 384), max 7/expert → padded M(bm32) = **6912**
- ATOM:   **103** active experts,        max 26/expert → padded M       = **3296**
SGLang spreads tokens over ~2× more experts → ~2.1× padded M → the grouped moe1
GEMM (memory-bound, time ∝ padded rows) does ~2× work → ~2× slower. (moe2 less
affected; sorting fine.) ATOM's per-launch was stable across the whole run →
this is **structural** (router/topk behavior), not random per-step.

**Why the routing differs (CORRECTED — shared-experts hypothesis REJECTED):**
Config: `n_routed_experts=384, n_shared_experts=1 (NOT fused), top_k=6`. Neither
engine fuses shared experts (double-confirmed). So routing spread is NOT a
shared-expert effect.
Real mechanism: the 64 decode hidden states are **all distinct in both** dumps
(64/64), but ATOM's route to far fewer experts — in each topk column ONE expert
is picked by **26 of 64** tokens (only ~20 distinct/col) vs SGLang ~48 distinct/col
(max ×5). So ATOM's distinct hidden states are **clustered in the router subspace**
→ concentrated routing → small M → fast moe1; SGLang's are diverse → spread → big M.
**Leading explanation:** ATOM ran WITHOUT `ATOM_USE_TRITON_MOE=1` (recipe: this is
the *numerically incorrect* MoE path, GSM8K 0.95→0.6). A degraded MoE produces
collapsed/clustered decode hidden states → concentrated routing → artificially
small M → "fast" moe1. ATOM's per-launch was stable across the run → systematic
(consistent with collapse, not a one-off). If so, the moe1 "1.7×" is largely a
**numerics artifact**, not a kernel/engine difference (kernel is identical and
equally fast at equal M — proven by the cross-swap).

**Next:** re-run ATOM WITH `ATOM_USE_TRITON_MOE=1` (correct numerics), re-dump,
and check whether its routing spreads toward SGLang's (active experts ↑, M ↑,
moe1 time ↑→ matches SGLang). That would confirm the gap is a numerics/clustering
artifact, not a real per-kernel difference. (Also: feed identical tokens to both
for a truly controlled routing comparison.)

Artifacts: dumps `/workspace/moe1_dump_{sgl,atom}/full.pt` (2.2 GB each),
`/workspace/microbench_moe2.py`, `/workspace/microbench_cross.py`. Hooks left in
place but env-gated (no-op unless `DUMP_MOE1_DIR` set).

## Open follow-ups
- TEST SGLang `--ep-size 8` (+ `--moe-a2a-backend deepep`): EP MoE should avoid
  the TP-MoE all-reduce; re-measure 8k/1k c64 + re-profile the collective.
- Also try `--enable-fused-moe-sum-all-reduce` (cheaper TP-MoE combine).
  `--disable-custom-all-reduce` (RCCL), `--enable-fused-moe-sum-all-reduce`;
  ATOM's `allgather_vec` is ~50-70x cheaper per op.
- Closer look at decode-MoE `t32x128x256` (SGLang 102 vs ATOM 60 µs).
- Fix the 1k/1k c1024 delayer over-throttle (shorter `--prefill-delayer-max-delay-ms`
  / passes, or disable at extreme conc); re-measure.
- Attack the all-reduce (Exp 8: `parallel_state.all_reduce` / `qr_all_reduce` /
  `_dp_gather_via_all_reduce`): try `SGLANG_USE_AITER_AR=1`,
  `--disable-custom-all-reduce` (RCCL), or `--enable-fused-moe-sum-all-reduce`;
  ATOM uses `allgather_vec` + `reduce_scatter` (decode AR 6.6% vs SGLang 62%).
- SGLang tp8+dp8 higher-concurrency points (128/256/512) vs ATOM scaling.
- Optional: GSM8K accuracy check with/without `ATOM_USE_TRITON_MOE=1`.

---

## Exp: apples-to-apple MoE routing (active-expert gap) — 2026-06-10

Question: ATOM moe1 decode dump had **103 active experts** vs SGLang **216** (over
64 tokens x top_k=6 = 384 selections, n_routed=384). Is ATOM's topk systematically
concentrating?

Method: added env/sentinel-gated **router-stage dump** in both engines (one full
T==64 decode pass, all 61 layers): hidden, router_logits, topk_ids/weights,
e_score_correction_bias, config, layer_id.
- SGLang hook: `TopK.forward_cuda` STANDARD branch (NOT the BYPASSED branch — aiter
  resolves output_format=STANDARD) + `MoERunnerConfig.layer_id`.
- ATOM hook: after `FusedMoE.select_experts` in `model_ops/moe.py` apply(); layer_id
  parsed from `layer.prefix`.
- Gate is a SENTINEL FILE `/workspace/router_dump/.enable` (NOT env) because
  `sglang serve` does NOT propagate inline env / PYTHONPATH to worker procs.
  `router_dump.py` installed into site-packages so workers can import it.
- Fed the **same 64 fixed prompts** to both engines (`/workspace/router_probe.py`,
  temp 0; auto-detects model id from /v1/models; needs long max_tokens so a 64-wide
  decode batch forms).

Findings (`/workspace/analyze_router.py`, `/workspace/cross_hidden.py`):
- Both DSv4 routers are **identical config**: ungrouped `sqrtsoftplus`, top_k=6,
  renormalize=True, routed_scaling_factor=2.5, bias-for-selection.
- A reference `sqrt(softplus(logits))+bias` top-6 selector reproduces BOTH engines'
  actual topk with overlap ~1.000 -> selection algorithm is identical, no kernel bug.
- Cross-engine `router_logits` cosine ~0.997 at every layer.
- NON-HASH active experts essentially EQUAL given same input: ATOM avg 22.5 vs
  SGL 20.5, tracking layer-by-layer (19/19, 23/23, 16/15, ...). (Low ~20 because the
  64 probe prompts were homogeneous; both engines collapse equally.)
- Hash layers 0-2 differ (ATOM ~29 vs SGL ~52) because they route by current
  token-id and the first 64-wide decode step had already diverged generated tokens.

Conclusion: the topk ALGORITHM is identical; ATOM does not intrinsically pick fewer
experts from the same hidden. The original 103-vs-216 came from two INDEPENDENT
dumps on different random data. BUT (user's sharp point) the moe1 time scales ~linearly
with active experts via per-expert M-padding in the sorted grouped GEMM, so a
*reproducible* moe-time gap implies a *systematic* active-expert gap whose root cause,
if real, is UPSTREAM hidden-state homogeneity (ATOM decode hidden more collapsed),
NOT topk. Decisive test pending: same diverse prompts, active-expert DISTRIBUTION
across many real decode steps, + identical-routing replay to subtract the structural part.

### BUG found (user): SGLang intermediate_pad passed as 0 (should be 128)
- `fp8.py process_weights_after_loading_block_quant` (line ~1151, `_use_aiter and
  is_fp4_expert`) DOES pad intermediate 384->512 (like ATOM), but the pad amount is
  never stored on the layer.
- `fp8.py maybe_get_hip_aiter_quant_info` builds `AiterMoeQuantInfo` WITHOUT
  intermediate_pad -> defaults to 0 -> `fused_moe(..., intermediate_pad=0)` while
  ATOM passes 128. SGLang therefore computes the full padded 512 intermediate
  (wasteful, routing-independent structural slowdown). Fix: store
  `layer.intermediate_pad = padded_inter - inter_per_part` and pass it through.

Router-stage hook artifacts: `/workspace/router_dump/` (router_{sgl,atom}.pt,
112-118 MB), `/workspace/moe_dump/router_dump.py` (+ site-packages copy),
`/workspace/router_probe.py`, `/workspace/analyze_router.py`, `/workspace/cross_hidden.py`.

### BUG FIXED + decisive routing test — 2026-06-10 (cont.)
Fix applied in `sglang/python/sglang/srt/layers/quantization/fp8.py`:
- `process_weights_after_loading_block_quant`: store `layer.intermediate_pad =
  padded_inter - inter_per_part` (and `layer.hidden_pad = 0`).
- `maybe_get_hip_aiter_quant_info`: pass `intermediate_pad`/`hidden_pad` into
  `AiterMoeQuantInfo`.
Verified: SGLang now sends `intermediate_pad=128` to fused_moe (was 0) -> no longer
computes the padded 512 intermediate; matches ATOM. (router_dump meta confirms 128.)

Decisive active-expert distribution test (same 64 diverse prompts, 256 decode steps,
58 non-hash layers, 14,840 samples/engine; lightweight per-step `count_step`,
sentinel `/workspace/router_dump/.count`, rank-0 only):
  overall mean active experts:  SGL = 182.1   ATOM = 182.6   (diff +0.5)
  per-layer diffs small & bidirectional (-7.6 .. +6.9).
=> CONCLUSION: ATOM does NOT systematically pick fewer active experts. Given the same
input both engines route to ~the same expert count. The 103-vs-216 was purely a
data artifact (different benchmark data in the two original independent dumps). The
moe1 timing gap is therefore STRUCTURAL (the intermediate_pad=0 bug + scale dtype /
kernel), not routing. Hooks left in code but sentinel-gated (inert; no .enable/.count).
Artifacts: /workspace/router_dump/counts_{sgl_saved,atom}.json.

### Controlled moe1/moe2 timing matrix (resolves routing-vs-structural) — 2026-06-10
`/workspace/microbench_matrix.py` (same aiter.fused_moe; vary weights-engine/routing/pad,
common hidden, AITER_BF16_FP8_MOE_BOUND=0):
                                    moe1      moe2     active
  SGL-w  R216 pad=0  (orig/BUG)    127.1us   98.3us   216
  SGL-w  R216 pad=128 (FIXED)      104.4us   79.7us   216
  ATOM-w R216 pad=128             103.9us   79.5us   216   == SGL-w
  SGL-w  R103 pad=128              55.2us   73.4us   103   == ATOM-w
  ATOM-w R103 pad=128 (ATOM base)  55.1us   73.2us   103
Conclusions:
1. intermediate_pad fix: SGL moe1 127->104us (-18%), moe2 98->80us (-19%).
2. NO engine/kernel structural diff: at matched routing+pad, SGL-w == ATOM-w
   (104 vs 104; 55 vs 55). scale dtype e8m0 vs uint8 costs nothing.
3. routing dominates: R216 vs R103 (same w/pad) = 104 vs 55us (~1.9x).
Reconciliation: the original ATOM55-vs-SGL127 gap = ~23us real bug (pad, fixed) +
~49us DATA artifact (SGL data 216-active vs ATOM data 103-active). With identical data
both route ~182 -> both ~89us moe1 -> EQUAL after fix. No contradiction with the
earlier microbench (which leveraged the 216-vs-103 data difference).

### End-to-end impact of intermediate_pad fix (tp8, cuda graph ON) — 2026-06-10
Valid A/B (pad value baked at graph-capture; live sentinel toggle does NOT work under
cuda graph, must restart). random data, ratio 1.0, num-prompts 128/192, conc 64:
                  pre-fix(pad0)   post-fix(pad128)   delta
  8k/1k c64 tok/s    1346            1413            +5.0%
            TPOT ms   37.30           35.09          -5.9%
            ITL  ms   27.98           26.02          -7.0%
  1k/1k c64 tok/s    2139            2274            +6.3%
            TPOT ms   28.22           26.36          -6.6%
            ITL  ms   27.66           25.48          -7.9%
=> the intermediate_pad fix buys ~5-6% decode throughput / ~6-8% TPOT-ITL at conc 64.
(Earlier "live toggle" A/B showed ~0% because cuda graph had already captured pad=128;
must restart server with sentinel to bake pad=0.) Fix is clean (no toggle) in fp8.py;
sentinel removed. Artifacts: /workspace/bench_fix/*.jsonl.

### Full re-sweep WITH intermediate_pad fix (SGLang) — 2026-06-10
random ratio 1.0; bench_dsv4.py; np=4*conc(1k)/2*conc(8k). Results in
/workspace/bench_fix/{tp8,tp8dp8}/ (summarize_sgl_dsv4.py).
tp8       1k/1k out tok/s: c2=138 c4=269 c8=508 c16=894 c32=1470 c64=2294
tp8       8k/1k out tok/s: c4=249 c8=444 c16=716 c32=1061 c64=1416
tp8dp8    1k/1k out tok/s: c64=2066 c128=3499 c256=5668 c512=8585 c1024=8575(plateau,TTFT67s)
tp8dp8    8k/1k out tok/s: c64=1494 c128=2141 c256=2795 c512=3344
tp8dp8    8k/1k total tok/s: c64=13445 c128=19273 c256=25154 c512=30098
(c1024 1k plateau = known extreme-conc delayer over-throttle, not the fix.)

### PR #27858 applied + GSM8K accuracy parity — 2026-06-11
Applied upstream PR sgl-project/sglang#27858 (same MoE fix; my intermediate_pad
change was incorporated verbatim). Added: drop shuffle_*_a16w4 imports, add
self.gu_intv = envs.SGLANG_USE_AITER_MOE_GU_ITLV.get() (default True == old
hardcoded), use shuffle_scale/shuffle_weight with self.gu_intv. Behavior at
default unchanged.
GSM8K 5-shot, 200 q, TP8, same config, only intermediate_pad toggled:
  before (pad=0):   Accuracy 0.970  Invalid 0.000
  after  (pad=128): Accuracy 0.965  Invalid 0.000
=> within noise (1 question; different GEMM M-tiling), NO accuracy regression.
PR description written to claude-skills/dsv4/PR_27858_description.md.

### Full re-sweep ratio 0.8 (np=conc*8, warm=conc*2), tp8dp8 with delayer+ROCM700A=0
tp8 (ROCM700A=1, no delayer):
  1k/1k out tok/s: c2=136 c4=256 c8=470 c16=801 c32=1254 c64=1867
  8k/1k out tok/s: c4=242 c8=424 c16=692 c32=1025 c64=1360
tp8dp8 (--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000, SGLANG_USE_ROCM700A=0):
  1k/1k out tok/s: c64=1944 c128=3261 c256=5306 c512=8073 c1024=8264(TTFT58s)
  8k/1k out tok/s: c64=1306 c128=1966 c256=2646 c512=3129
  8k/1k total tok/s: c64=11798 c128=17750 c256=23821 c512=28165
Results: /workspace/bench_r08/{tp8,tp8dp8}/.

### SGLang(+fix) vs ATOM-best, matched methodology (ratio 0.8, np=conc*8, warm=conc*2) — 2026-06-11
SGLang = bench_r08 (intermediate_pad fix; dp8 with delayer+ROCM700A=0).
ATOM = Experiment 1 best numbers (ATOM client, same ratio/np/warm). out tok/s, SGL/ATOM%:
tp8:
  1k/1k  c2 136/121 112% | c4 256/228 112% | c8 470/420 112% | c16 801/739 108% | c32 1254/1214 103% | c64 1867/1819 103%
  8k/1k  c4 242/215 112% | c8 424/379 112% | c16 692/634 109% | c32 1025/1001 102% | c64 1360/1397 97%
tp8+dp8:
  1k/1k  c64 1944/1856 105% | c128 3261/3181 103% | c256 5306/5543 96% | c512 8073/8381 96% | c1024 8264/11583 71%
  8k/1k  c64 1306/1455 90% | c128 1966/2212 89% | c256 2646/3085 86% | c512 3129/3783 83%
Takeaways: tp8 SGLang now matches/beats ATOM everywhere except 8k/1k c64 (97%); the
pad fix closed the old ~93% gap. dp8: SGLang wins low conc (c64-128) but ATOM scales
better high conc (dp-attention weakness, not MoE). Outliers 1k c1024 (TTFT 58.6s) &
8k c512 (TTFT 16.3s) are prefill-delayer admission at extreme conc, not decode.
Comparison script: /workspace/compare_atom.py

## Experiment 13 — ATOM multi-stream vs single-stream, 8k/1k dp8 high-conc sweep — 2026-06-11

Goal: quantify how much of ATOM's lead at high concurrency comes from its
side-streams (multi-stream MoE/compressor overlap), since SGLang is inherently
single-stream. Same `run_atom_dsv4_aligned.sh` (tp8+dp8), ratio 0.8,
np=conc*8, warmup=conc*2, ATOM client. Only variable = `ATOM_DISABLE_SIDE_STREAMS`
(single-stream when =1; verified active via `/proc/<pid>/environ`). All points
completed (succ == conc*8). Orchestrator: `/workspace/run_stream_ab.sh`.
Results: `/workspace/bench_atom_{multistream,singlestream}_8k/`.

| conc | multi total tok/s | single total tok/s | multi gain | multi TPOT | single TPOT | multi ITL | single ITL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64  | 12,866 | 12,162 | **+5.8%** | 42.6 | 44.8 | 30.3 | 32.9 |
| 128 | 19,436 | 18,651 | **+4.2%** | 56.4 | 59.0 | 34.7 | 37.0 |
| 256 | 27,881 | 27,023 | **+3.2%** | 79.1 | 81.4 | 38.0 | 40.9 |
| 512 | 33,779 | 33,126 | **+2.0%** | 132.2 | 133.4 | 47.6 | 51.9 |

(multi-stream matches Exp 1 within noise: 12,866 vs 13,044 at c64, etc.)

**Findings**
1. **Multi-stream benefit SHRINKS with concurrency: +5.8% (c64) → +2.0% (c512).**
   At high conc the GPU is already saturated by the large batch, so side-stream
   overlap finds less idle time to hide. So multi-stream is NOT the source of the
   growing high-conc gap.
2. **Fair single-vs-single (SGLang+pad-fix `bench_r08` vs ATOM single-stream here):**
   c64 97% | c128 95% | c256 88% | c512 85% — SGLang still falls behind and the
   gap still widens with conc even after removing multi-stream. The dominant
   high-conc gap is elsewhere: the dp-attention collective/all-reduce cost that
   scales with concurrency (Exp 8/10/11: SGLang decode ~62% all-reduce via
   `cross_device_reduce_2stage`; ATOM uses cheap `allgather_vec`+`reduce_scatter`).
3. vs ATOM multi-stream (the absolute best): SGLang c64 92% → c512 83%.

**Conclusion**: multi-stream accounts for only ~2% of the high-conc gap (and less
as conc rises). Next: profile c256 (ATOM single-stream vs SGLang) to pin the
remaining, concurrency-scaling collective overhead.

## Experiment 14 — c256 profiling: DECODE is equal, the gap is PREFILL — 2026-06-11

Config (both dp8, EAGER for kernel visibility, ATOM single-stream): ISL 8192,
OSL 128, conc 256, ratio 1.0; GPU-only `/start_profile`+`/stop_profile` window.
Gap reproduced first (mixed np512): SGLang out 548 tok/s vs ATOM-single 746
(73%), duration 119.6 vs 87.8s. **But pure-decode median ITL is EQUAL**
(SGLang 37.1 vs ATOM 38.7 ms, graph-on) → decode steps not the gap.

Both profile windows landed in a **pure-decode phase** (markers `step[DECODE bs=32]`
/ `eager_decode[bs=32]`; dp8 splits 256 → bs=32/rank). Per-decode-step (eager, rank0):

| per decode step (bs=32/rank) | SGLang | ATOM single |
|---|--:|--:|
| step wall (median) | 202.6 ms | 185.0 ms |
| all-reduce `cross_device_reduce_2stage` | 60.4 | — |
| all-gather `allgather_vec` | 71.9 | 109.0 |
| reduce-scatter `reduce_scatter_first_dim` | — | 26.1 |
| **collective subtotal** | **132.3** | **135.1** |
| moe1 / moe2 / attn / other | 7.6/5.0/3.0/20.2 | 6.8/4.4/3.3/18.3 |
| total GPU work (sum of medians) | ~168 | ~168 |

(Eager inflates per-step ~5× vs graph; use for RELATIVE comparison only.)

**Findings**
1. **DECODE is NOT the gap.** Per-step total GPU work is identical (~168 ms),
   collective subtotal nearly equal (132 vs 135 ms). The collective *pattern*
   differs (SGLang all-reduce 60 + all-gather 72; ATOM all-gather 109 +
   reduce-scatter 26) but nets the same. SGLang's slightly higher step wall
   (202 vs 185) is eager dispatch gap, not collective — and in graph-on
   production, pure-decode ITL was equal/SGLang-faster (37.1 vs 38.7).
2. **Decode at c256 is ~78% collective for BOTH** — but equal, so it doesn't
   explain the throughput gap. The gap must be in **PREFILL** (consistent with
   the reproduced TTFT gap: 27 vs 20 s, and Exp 10 c64 prefill which showed
   SGLang's `quickreduce` all-reduce 1196 ms vs ATOM `allgather_vec` 18 µs).
3. So the c256 (and growing-with-conc) gap is **prefill throughput**, not decode.

**Caveat / correction to earlier framing**: Exp 8/10/11 emphasized decode
all-reduce (62%), but at matched bs=32/rank decode the collective nets equal to
ATOM. The actionable gap is the **prefill** collective + MoE, to be captured next.

Traces: `/workspace/prof_c256_sgl/` (rank0, 205MB), `/workspace/prof_c256_atom/dp0_tp0/`.
Scripts: `/workspace/cmp_step_c256b.py`.

### Next: capture PREFILL (EXTEND) steps at c256
- SGLang: `bench --profile --profile-by-stage --profile-num-steps N --profile-activities GPU`
  (separate EXTEND trace) OR `/start_profile` right at load start (prefill-heavy).
- ATOM single-stream: `/start_profile` during the initial prefill burst.
- Compare prefill MoE (tokens-per-launch) + collective (all-reduce vs allgather/RS).

## Experiment 15 — c256 PREFILL profiling: chunking + per-layer all-reduce — 2026-06-11

Captured prefill (EXTEND) windows for both (eager, dp8, ISL8192/OSL128/conc256/
ratio1.0, GPU-only, `/start_profile` during the initial prefill burst).
Markers confirm the structural difference:
- **SGLang**: `step[EXTEND bs=1/2 toks=2048]` → prefills **2048 tokens/rank/step**
  (chunked-prefill 16384 ÷ dp8 = 2048 per DP rank).
- **ATOM**: `prefill[bs=2 tok=16384]` → prefills **16384 tokens/step** (2 whole
  8192-req, NOT chunked).
→ SGLang runs ~**8× more prefill steps** for the same work.

Per-launch (rank0/dp0):
| kernel | SGLang (2048-tok steps) | ATOM (16384-tok steps) |
|---|--:|--:|
| moe1 128x256x256 | 655 µs ×610 (0.32 µs/tok) | 4210 µs ×122 (0.26 µs/tok) |
| moe2 64x256x256 | 590 µs ×610 | 4496 µs ×122 |
| TP-MoE all-reduce | `quickreduce` 975 µs **×1220** (61/step = 1/layer) | nccl (wait-inflated, n/a) |
| moe1 launches/step | 30.5 | 30.5 (same #MoE layers) |

**Findings (prefill = the c256 gap)**
1. SGLang chunks prefill to **2048 tok/rank**; ATOM uses **16384 tok/step**. Same
   #MoE layers/step (30.5), but SGLang needs ~8× the steps → (a) smaller MoE
   GEMMs (per-token moe1 0.32 vs ATOM 0.26 µs/tok, ~20% worse — the tokens/launch
   effect from Exp 10/12), and (b) a per-layer TP-MoE `quickreduce` all-reduce
   paid on **every** 2048-tok chunk (1220 launches in the window, 61/step).
2. Collective ABSOLUTE cost not directly comparable: ATOM prefill uses
   `ncclDevKernel` whose trace duration includes blocking-wait (inflated). So we
   don't over-claim the all-reduce magnitude; the robust point is SGLang pays the
   MoE-TP all-reduce far more FREQUENTLY due to chunking.
3. Decode was already shown equal (Exp 14) → the c256 (and growing-with-conc) gap
   is prefill: chunk granularity + per-chunk MoE-TP all-reduce.

**Levers to TEST (conclusive via throughput, no trace ambiguity)**
- Raise SGLang `--chunked-prefill-size` so per-DP-rank chunk ≈ ATOM's 16384
  (fewer, larger prefill steps → better MoE GEMM + fewer all-reduces). Direct A/B.
- MoE-TP all-reduce → EP or reduce-scatter: `--ep-size 8` (+`--moe-a2a-backend
  deepep`) or `--enable-fused-moe-sum-all-reduce`.
Traces: `/workspace/prof_c256b_sgl/` (rank0), `/workspace/prof_c256b_atom/dp0_tp0/`.
Script: `/workspace/perlaunch_prefill.py`.

## Experiment 16 — chunked-prefill-size lever: helps prefill-heavy, NOT the real gap — 2026-06-11

Confirmed `server_args.py:3315`: with dp-attention, `chunked_prefill_size //=
dp_size`. So `--chunked-prefill-size 16384` → **2048 tok/rank**. Set
`--chunked-prefill-size 131072` → **16384 tok/rank** (runtime verified: prefill
batches now `#new-token: 8192` = whole request, vs 2048 before). SGLang dp8 best,
graph ON, ROCM700A=0.

| metric | baseline (2048/rank) | chunk lever (16384/rank) | ATOM single |
|---|--:|--:|--:|
| **fast cfg** (c256, OSL128, np512, ratio1.0) out tok/s | 548 | **585 (+6.7%)** | 746 |
| **REAL** 8k/1k c256 (np2048, ratio0.8) total tok/s | 23,821 | **23,882 (+0.3%, noise)** | 27,023 |
| REAL out tok/s | 2,646 | 2,653 | 2,997 |
| REAL TTFT ms | ~2,000 | 2,128 | ~1,710 |

**KEY CORRECTION / finding**
- The chunk lever helps ONLY the **prefill-heavy fast config** (OSL128, TTFT 27s)
  → +6.7%. In the **realistic** c256 8k/1k (OSL1024, TTFT ~2s, decode-dominated)
  it does **nothing** (23,882 vs 23,821).
- So the **fast-config (OSL128) over-weighted prefill and mis-attributed the real
  gap to chunk granularity.** The real c256 gap (SGLang 88% of ATOM) is NOT
  prefill-chunking.
- Reconciliation: decode steps are equal (Exp 14, ITL ~38 both); chunk size
  doesn't matter in the real regime (Exp 16); yet SGLang total is 88% of ATOM.
  The mechanism is **prefill THROUGHPUT** (SGLang TTFT 2128 vs ATOM ~1710): SGLang
  feeds the decode batch slower → smaller average running batch → lower output
  throughput *despite equal per-step ITL*. The remaining lever is the **per-layer
  TP-MoE all-reduce** (present in every forward regardless of chunk size), not the
  chunk granularity.

**Next**: attack the collective that's in EVERY step — `--ep-size 8`
(+`--moe-a2a-backend deepep`) or `--enable-fused-moe-sum-all-reduce` — and
re-measure REAL 8k/1k c256 total tok/s vs the 23,821 baseline / 27,023 ATOM.
Data: `/workspace/bench_chunk_c256/`.

### Verification (before lever): the bottleneck is prefill STALLING decode
SGLang real c256 server log: steady `#running-req` median = **32/rank = 256 total**
(full). ATOM trace markers also `bs=32/rank`. So **both run the full 256 batch** —
NOT a decode-batch-fill problem. The gap is TPOT vs ITL (how much decode is
stalled by interleaved prefill):

| | ITL (pure decode) | TPOT (with stalls) | TPOT/ITL | out tok/s |
|---|--:|--:|--:|--:|
| ATOM single | 40.9 | 81.4 | **1.99×** | 2,997 |
| SGLang | 37.9 | 91.8 | **2.42×** | 2,653 |

- Pure-decode speed equal (ITL 38–41), batch full (256) both. But SGLang's decode
  is stalled by prefill **more** (TPOT = 2.42× ITL vs ATOM 1.99×). `out ≈ 256/TPOT`
  (ATOM 256/.0814=3145, SGLang 256/.0918=2789) matches measured.
- This also explains why the chunk lever did nothing: bigger chunks = fewer-but-
  longer prefill stalls; TPOT net-unchanged.
- ⇒ Root: prefill steps are too expensive and stall decode. To cut TPOT, make each
  prefill step CHEAPER (not change chunk size). Prefill step cost is dominated by
  the per-layer `quickreduce` all-reduce → EP / fused-moe-sum-all-reduce lever now
  well-justified.

## METHODOLOGY NOTES (apply to all subsequent SGLang experiments) — 2026-06-11

### N1. quickreduce is ON via a DOCKER env, not a code default
The trace's `quickreduce::allreduce_twoshot Q8` is enabled by the **docker env
`ROCM_QUICK_REDUCE_QUANTIZATION=INT8`** (confirmed in the shell env). The code
default in `distributed/device_communicators/quick_all_reduce.py:176` is `"NONE"`
(which disables quick AR). Dispatch: `parallel_state.py` tries quick AR
(`should_quick_allreduce`) → else custom AR (`cross_device_reduce_2stage`,
AiterCustomAllreduce, `SGLANG_USE_AITER_AR` default true) → else RCCL.
- **To DISABLE quickreduce**: launch SGLang with `ROCM_QUICK_REDUCE_QUANTIZATION=NONE`
  (falls back to AiterCustomAllreduce `cross_device_reduce_2stage` / RCCL).
- Other knobs: `ROCM_QUICK_REDUCE_QUANTIZATION` ∈ {FP, INT8, INT6, INT4, NONE};
  `ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB`; `--disable-custom-all-reduce` (RCCL);
  `SGLANG_USE_AITER_AR=0`.

### N2. SGLang chunked-prefill MUST be set to 131072 for apple-to-apple
With dp-attention, `chunked_prefill_size //= dp_size` (server_args.py:3315). The
aligned script's `--chunked-prefill-size 16384` → **2048 tok/rank**, but ATOM
prefills **16384 tok/step**. For any apple-to-apple comparison (esp. PREFILL
traces), launch SGLang with **`--chunked-prefill-size 131072`** (→ 16384/rank,
matching ATOM). Pass via `SGL_EXTRA_ARGS="--chunked-prefill-size 131072"`.
- ⚠️ Exp 15's prefill trace used the DEFAULT 2048/rank (NOT matched) — so its
  prefill step-structure comparison is confounded by chunk size; re-capture
  PREFILL traces with 131072 for a clean comparison.
- (Exp 16 showed chunk size doesn't change real-regime throughput, but it DOES
  change the prefill trace structure, which matters for trace comparisons.)

## Exp 17 — Two-lever throughput check @ real 8k/1k c256 — 2026-06-11

Orchestrator `/workspace/run_lever_exp.sh`. Both servers brought up/down clean.

| config | total tok/s | out tok/s | TTFT ms | TPOT ms | ITL ms |
|---|---|---|---|---|---|
| Baseline SGLang (QR=INT8, default chunk) | 23,821 | — | — | — | — |
| ATOM baseline (multi-stream, chunk 16384) | **27,023** | — | — | — | — |
| SGLang QR=NONE (chunk131072 = 16384/rank) | 23,464 | 2,607 | 2,177 | 94.1 | 38.1 |
| ATOM single-stream chunk2048 | 18,309 | 2,031 | 1,236 | 127.6 | 43.4 |

### L1. Disabling quickreduce does NOT help SGLang (23,464 vs 23,821, −1.5%)
Falling back to AiterCustomAllreduce/RCCL doesn't make the collective cheaper.
⇒ bottleneck is NOT "quickreduce kernel is slow"; it is the STRUCTURAL count of
prefill all-reduces. Swapping the AR backend cannot fix it.

### L2. ATOM with small chunk (2048) collapses −32% (27,023 → 18,309) ✅
Direct answer to "does small chunk hurt ATOM?": YES, massively. TPOT 127.6ms,
total falls BELOW SGLang baseline. This is the reverse-direction confirmation
that large chunk (few all-reduces) is ATOM's main lead. Forced to small chunk,
ATOM becomes worse than SGLang.

### Conclusion (bidirectional proof)
prefill chunk size (→ #all-reduce calls) is the PRIMARY gap driver:
- SGLang bigger chunk: +6.7% in prefill-heavy, ~0 in real decode-bound c256.
- ATOM smaller chunk: −32%.
- AR backend swap (QR off): no effect → not kernel efficiency, it is
  (#calls × per-sync cost) structural.
Chunk lever is maxed in real c256 (decode-bound). Next must REDUCE #all-reduce
syncs themselves: EP (`--ep-size 8 --moe-a2a-backend deepep`) or fused-moe-sum-AR.

## Exp 18 — CORRECTION: all-reduce-count theory REFUTED; param semantics — 2026-06-11

### Verified both levers actually took effect (per-step prefill token counts)
- SGLang chunk131072: prefilled **16384 tok/step** (340× at 16384) — matched ATOM.
- ATOM max-num-batched 2048: prefilled **2048 tok/step** (8116× at 2048).

### 2×2 by per-step prefill size (real 8k/1k c256, SINGLE-STREAM)
| prefill/step | SGLang | ATOM |
|---|---|---|
| 2048/rank  | 23,821 | 18,309 |
| 16384/rank | 23,464 | 27,023 |

Lines CROSS. ATOM −32% shrinking chunk; SGLang ~flat enlarging chunk.
At 2048 SGLang WINS; at 16384 ATOM wins.

### ⇒ all-reduce-count is NOT the gap cause (RETRACTED Exp 15/17 claim)
If "small chunk → more all-reduces" were the driver, enlarging chunk would speed
up BOTH. It speeds up ATOM massively, SGLang not at all. Real differentiator:
**ATOM's prefill scales efficiently with chunk size (fewer/bigger prefill steps,
less decode disruption); SGLang cannot convert bigger prefill chunks into
throughput.** Consistent with SGLang decode being stalled by prefill more
(TPOT/ITL 2.42× vs ATOM 1.99×) AND enlarging chunk NOT fixing it.
⇒ Bottleneck = SGLang prefill↔decode scheduling/interleave efficiency, NOT the
all-reduce kernel/count. Next: investigate how SGLang interleaves prefill+decode.

### Param semantics (ATOM, code-confirmed)
- `max_num_batched_tokens` (def 16384): per-forward-step TOTAL token budget,
  SHARED by prefill+decode. Actual prefill chunk = min(remaining_prompt,
  budget - used). THIS is the real "prefill chunk / all-reduce frequency" knob
  and the correct analogue to SGLang `--chunked-prefill-size`.
  (scheduler.py:725-733, 817-824, 898)
- `attn_prefill_chunk_size` (def 16384): attention-internal MLA tiling that ONLY
  fires for CACHED-PREFIX path (`has_cached and total_kv > chunk_size`,
  aiter_mla.py:795-800). With prefix caching OFF (our bench) it is a NO-OP.
  ⇒ Exp 17 EXP2's `--attn-prefill-chunk-size 2048` did nothing; the −32% came
  entirely from the co-set `--max-num-batched-tokens 2048`. EXP2 was already
  single-stream, so the single-stream small-budget point = 18,309.

## Exp 19 — Apple-to-apple PREFILL trace: methodology failure + corrections — 2026-06-12

### What was attempted
Per-kernel prefill comparison SGLang(QR=INT8/NONE) vs ATOM single-stream, ALL at
16384 tok/step/rank (aligned launch scripts), DP0/TP0 torch trace, 6s window.
Goal: see if SGLang prefill spends more in collectives than ATOM.

### Trace settings were NOT fully apple-to-apple (root issue)
Server launch WAS aligned (tp8+dp8, fp8 KV, page/block256, max-seqs512,
mem0.90, prefix-cache off, single-stream, 16384/rank). BUT the **client load
differed**:
- SGLang traces: ISL8192 OSL8 c8 np128
- ATOM trace (1st): same c8/np128 → UNUSABLE (ATOM drained before window, GPU 3%)
- ATOM trace (rerun, the one analyzed): **c32 np1024 OSL1** to sustain prefill.
⇒ SGLang(c8) vs ATOM(c32) — different load. Re-captured SGLang at MATCHED
**c32/np1024/OSL1** (prof_aa_sgl_int8_matched, verified #new-token=16384/step).

### FINDING 1 — "collective fraction" is LOAD-SENSITIVE (kills Exp 15/17 claim)
Same SGLang, same 16384/rank, only load differs:
| SGLang load | collective / union-busy |
|---|---|
| c8/np128  | 71% |
| c32/np1024| 33% |
The "SGLang prefill ~70% in collective" headline was a **low-concurrency (c8)
artifact**: tiny batch → compute kernels finish fast → fixed collective latency
dominates. At realistic c32 it drops to 33%. ⇒ collective-fraction is NOT a
valid cross-config metric. Consistent with Exp 18 retraction.

### FINDING 2 — trace `dur` is SYNC-INFLATED → absolute cross-engine compare INVALID
On a SINGLE stream, summed kernel-time exceeds wall-span (impossible w/o
inflation):
- SGL tid6: 12171ms over 9094ms span = 134% of span
- ATOM tid3: 35314ms over 9602ms span = 368% of span
Longest single "kernels" are 2.7–4.4 SECONDS (mhc_pre_big_fuse_rmsnorm 4362ms,
dynamic_per_group_scaled_quant 4361ms, gemm_a16w16 4362ms). No GPU kernel runs
4s — the recorded `dur` absorbs **host-side launch-barrier / stream-sync wait**.
⇒ "sum of dur", "us/launch", and "us/moe-sort" are ALL meaningless for absolute
cross-engine cost. ATOM's apparent 5–9× per-kernel cost is an artifact, not real.
This is the same symptom as the earlier %busy>100% / 7234us-per-launch anomaly.

### Implication for the gap investigation
Per-kernel torch-trace cost attribution CANNOT fairly compare SGLang vs ATOM
here (sync-wait inflation + load sensitivity + different stream layout). The gap
must be pursued via end-to-end throughput/TPOT under matched real load (Exp 18
2×2), NOT via trace kernel-time totals. Exp 18 conclusion stands: bottleneck is
**SGLang prefill↔decode scheduling/interleave efficiency**, not all-reduce
count/kernel. Trace artifacts here do not support an all-reduce-cost story.

### Artifacts
- /workspace/prof_aa_sgl_int8_matched/  (SGLang c32 matched trace, 8 ranks)
- /workspace/prof_aa_sgl_int8|none/, /workspace/prof_aa_atom/ (earlier)
- /workspace/union_busy.py, union_busy2.py (analysis; note caveats above)
- /workspace/run_sgl_trace_matched.sh (matched-load capture orchestrator)

## Exp 20 — prefill-delayer ON/OFF A/B: delayer is ESSENTIAL, not the gap — 2026-06-12

### Setup (clean: chunk NOT a variable)
SGLang dp8, 8k/1k c256, ratio0.8, np2048, warm512. BOTH runs
`--chunked-prefill-size 131072` (=16384/rank, matched to ATOM; confirmed
#new-token=16384 dominant). Only `--enable-prefill-delayer` differs.
Delayer config (verified ON-run log): max_delay_passes=30,
token_usage_low_watermark=None, queue_min_ratio=None, max_delay_ms=5000.

### Result (decisive)
| metric | delayer ON | delayer OFF | OFF vs ON |
|---|---|---|---|
| total tok/s | **23,908** | 14,103 | **−41.0%** |
| output tok/s | 2,656 | 1,567 | −41.0% |
| median TPOT ms | 92.2 | 167.7 | +81.8% |
| median TTFT ms | 2,106 | 1,428 | −32% (OFF lower) |
| bench duration s | 711 | 1,206 | +69.5% |

⇒ **prefill-delayer is NOT the gap cause; it is CRITICAL to SGLang throughput.**
Turning it off makes SGLang far worse (−41%). KI-1 "delayer over-throttles" is
REFUTED for this workload. (delayer ON ≈ Exp 16's 23,882 → consistent.)

### Mechanism (why OFF is 2× slower per prefill)
- retract events: **0 in BOTH** runs. swa-usage-1.00: **0 in BOTH**. new_token
  _ratio collapse: none. ⇒ at chunk16384/rank + swa-ratio0.15 + c256, **swa pool
  is sufficient** (confirms user hypothesis: no need to raise swa-full-tokens-ratio).
  The earlier user-pasted retract+swa1.00 log was a DIFFERENT config (likely
  2048/rank fragmentation or higher conc).
- Same total prefill work both (19,311,616 tokens). Same decode batch fullness
  (median #running-req 32, mean 30 both). Same step counts (~2.1k prefill,
  ~1.5k decode).
- KEY: **prefill input throughput median 3814 tok/s (ON) vs 1828 tok/s (OFF)** —
  OFF runs each prefill batch ~2× SLOWER despite identical work.
- ⇒ The delayer's real job in dp-attention is **cross-rank prefill timing
  coordination**: it gates prefill so the 8 DP ranks enter prefill together.
  Without it, ranks desync → per-step all-gather/MoE waits on the slowest rank →
  effective prefill throughput halves. This is the dp8 "wait on slowest rank"
  tax (cf. dp_attn.py prepare_mlp_sync_batch + idle-batch padding).

### Conclusion + next direction
The remaining SGLang↔ATOM gap is NOT delayer, NOT all-reduce count, NOT chunk
size, NOT retract/swa. It is **dp-attention's per-step cross-rank synchronization
efficiency**: SGLang's 8 DP ranks must agree (all-gather) every forward on
prefill-vs-decode and pad with idle batches; throughput is bounded by the slowest
rank each step. ATOM evidently coordinates prefill across ranks more cheaply (or
does not pay the same per-step global sync). Next: (a) compare how ATOM's
scheduler handles multi-rank/dp prefill coordination (user-suggested mechanism
comparison), (b) measure the all-gather/idle-batch overhead per step in SGLang
dp8 directly (scheduler timing, not torch trace).

### Artifacts (Exp 20)
- /workspace/run_delayer_ab.sh  (A/B orchestrator)
- /workspace/bench_delayer_ab/  (jsonl + bench logs)
- /workspace/server_logs/delayer_delayerON|OFF.serverlog
- run_sgl_dsv4_aligned.sh: added DELAYER=on|off knob (default on, preserves prior)

## Exp 21 — ATOM scheduler code study vs SGLang (mechanism comparison) — 2026-06-12

ATOM path: /opt/venv/lib/python3.10/site-packages/atom/model_engine/

### HEADLINE: ATOM scheduler is architecturally the SAME as SGLang
- `scheduler.py:schedule()`: tries **prefill first**, returns a **pure prefill
  batch** if any (lines 859-888), else falls to decode. Identical prefill-XOR-
  decode design to SGLang's `get_next_batch_to_run`. NOT mixed-batch.
- `prefill_delayer.py`: docstring line 4 literally says **"Direct port of
  SGLang's PrefillDelayer"**. Same all/none/mixed → delay logic, same defaults
  (max_delay_passes=30, max_delay_ms=5000, watermark=None), **enabled by default**
  (`ATOM_ENABLE_PREFILL_DELAYER=1`, utils/envs.py:200). So both engines ran WITH
  the same delayer in all our experiments. The +41% delayer benefit (Exp 20) is
  the SAME mechanism on both sides.
⇒ The gap is NOT high-level scheduling policy, NOT priority, NOT the delayer.

### Real differences found (candidates for the remaining gap)
1. **Decode KV pressure handling: preempt (ATOM) vs retract+new_token_ratio (SGL)**
   - ATOM `preempt()` (scheduler.py:967): on `can_append` failure, pops the
     TAIL of running, frees its KV, requeues to waiting head. Simple LIFO
     preempt, NO upfront memory reservation.
   - SGLang reserves decode memory UPFRONT via `new_token_ratio` (×0.3 under
     dp-attn → init 0.21) which SHRINKS prefill admission budget every step,
     then retracts on OOM. ATOM does NOT pre-reserve → more KV free for prefill
     admission, fewer artificial prefill stalls.
   - BUT Exp 20 showed 0 retract at our setting, so this is not active here.
2. **MoE collective: all_to_all/EP (ATOM) vs all-reduce (SGLang dp-attn)**
   - ATOM delayer docstring (lines 26-32): mixed forwards cost = "MoE all_to_all
     bottlenecked by prefill rank's tokens, 81.7% pad_waste, 67% wall in
     moe.gather". ATOM MoE uses **expert-parallel all_to_all dispatch**.
   - SGLang aligned config uses TP-MoE with **per-layer all-reduce** (Exp 19
     trace: AR cross_device + quickreduce + allgather dominate). Different MoE
     comm primitive. This is the likely structural differentiator and matches
     Exp 20's "SGLang prefill input tput = half of... its own potential" via
     per-step dp-sync + AR.
3. **schedule_conservativeness ×0.3 auto-applied under dp-attn** (SGL
   server_args.py:3313) → init new_token_ratio 0.7×0.3=0.21. ATOM has no
   equivalent auto-shrink.

### SGLang tunable knobs to try (no code change; all are ServerArgs/env)
A. `--moe-a2a-backend deepep --ep-size 8`  → switch SGLang MoE from TP-all-reduce
   to **expert-parallel all_to_all**, matching ATOM's MoE comm. MOST LIKELY to
   close the gap if diff #2 is the cause. (Earlier Exp 15 "next step" never run.)
B. `--prefill-delayer-token-usage-low-watermark X` (default None) → enable the
   safety valve so prefill is force-allowed when KV underutilized; may reduce
   over-delay tail at high conc.
C. `--prefill-delayer-queue-min-ratio R` → opt-in adaptive queue trigger; ATOM
   intentionally did NOT port this (delayer docstring line 38), so SGL has a
   knob ATOM lacks — could help fragment-prefill workloads.
D. `--schedule-conservativeness 1.5..3` → counteract the ×0.3 dp-attn shrink,
   raise init new_token_ratio so prefill admission budget is less throttled.
   (Only matters if retract/admission-throttle is active — not at Exp 20 setting,
   but may matter at higher conc / smaller chunk.)

### Recommended next experiment
**A/B SGLang `--moe-a2a-backend deepep --ep-size 8` vs aligned baseline**, 8k/1k
c256 (+ maybe c128), chunk16384/rank, ratio0.8. This directly tests whether
ATOM's lead is its EP/all_to_all MoE comm vs SGLang's TP all-reduce — the one
remaining structural difference consistent with all data so far.

## Exp 21b — CORRECTION: ATOM does NOT use EP; option A premise was WRONG — 2026-06-12

### User caught the error
ATOM aligned launch (run_atom_dsv4_aligned.sh) = `-tp 8 --enable-dp-attention`
ONLY. NO `--enable-expert-parallel`. Code confirms (moe.py:85-86,108):
`use_all2all_kernels = dp_size>1 AND use_ep AND has(mori)`, and
`use_ep = ... AND parallel_config.enable_expert_parallel`. With EP off →
**use_ep=False → NO mori all2all**. So ATOM ran TP-MoE, NOT EP/all_to_all.
⇒ Exp 21's "ATOM uses EP all_to_all" claim is RETRACTED. The delayer docstring's
"moe.gather / all_to_all" describes ATOM's EP-mode path, not the mode we ran.

### What BOTH actually do under dp-attention + no-EP (now confirmed)
Both flatten DP into a larger TP group for the MoE and bridge attn(dp)→MoE(tp):
- ATOM (moe.py:104-117): enable_dp_attention → `flatten_tp_across_dp` →
  tp_size = dp_size×tp_size = 64; MoE comm = `all_gather_with_padding` hidden
  states across DP ranks (line 205) → expert GEMM → `reduce_scatter_with_unpadding`
  (line 215). No per-rank all-reduce; it's all_gather + reduce_scatter.
- SGLang (dp_attention.py:243-244): attn_tp_size = tp_size//attn_dp_size (=1 at
  tp8dp8 for attn), MoE at full tp=8; transition uses all_gather before MoE +
  all-reduce/reduce-scatter after. Exp 19 trace showed AR + allgather dominate.
⇒ Same FAMILY (gather → expert compute → scatter/reduce). NOT a TP-allreduce vs
EP-all2all dichotomy. Option A (force SGLang EP) would make SGLang DIFFERENT from
ATOM, not closer — so A is NOT the right apple-to-apple test. SHELVED.

### The MoE intermediate-padding angle (user's real lead — needs measuring)
moe_inter_dim = 3072 (deepseek_v4.py:262). ATOM pads intermediate per partition
up to a 256 multiple (moe.py:780-793, pad_align=256, tracks `intermediate_pad`).
User's point: 3072 is divisible by 256 → if NOT TP-sharded (EP/full expert),
no pad waste; but if TP-sharded, 3072/tp may not align to 256 and pads.
Earlier work already found "SGLang MoE lacked intermediate_pad → extra compute"
and fixed it, yet gap persisted. So padding alone is not the whole gap, but the
EXACT per-partition intermediate size on each engine (and resulting pad) is still
unconfirmed and worth a direct check.

### Corrected next steps (do NOT run EP A/B)
1. Determine the EFFECTIVE MoE sharding on EACH engine in OUR config (tp8dp8,
   no EP): is the routed-expert intermediate (3072) TP-sharded (→ small per-part,
   heavy 256-pad) or kept whole per rank? Read both weight-creation paths and/or
   dump per-expert w13/w2 shapes at load time. This decides whether pad/GEMM
   shape is the differentiator.
2. Compare the dp-bridge comm volume: ATOM all_gather+reduce_scatter vs SGLang
   all_gather+all_reduce — per-step bytes and kernel, measured from scheduler
   timing / a SMALL controlled microbench (NOT the unreliable full trace).
3. Only after (1)(2) pinpoint a concrete divergence, pick the matching SGLang
   knob (e.g. moe_dense_tp_size, ep, or a pad fix) for an A/B.

## Exp 22 — ROOT CAUSE FOUND: variable-length vs MAX-pad DP-MoE gather — 2026-06-12

### Source: ATOM PR #930 (merged 5/27) + #157 (merged 1/21), both IN our build
Verified our atom build contains PR #930 code: moe.py has `all_gatherv`,
`reduce_scatterv`, `dp_uniform_decode`, `use_dp_gather_scatter`,
`dp_gather_hidden_and_router`. PR #157 added the 3-mode DP-MoE split.

ATOM has THREE DP-MoE modes (moe.py:3112-3115, PR #157+#930):
  1. Pure DP (`-dp N`, no dp-attn): NO MoE all_gather/reduce at all.
  2. DP-attn + EP (mori all2all): `--enable-expert-parallel`.
  3. DP-attn + TP all_gather/reduce (our mode): `-tp8 --enable-dp-attention`.
PR #157 perf note (1k/1k c128): **Pure DP 20400 >> EP-mori 14000 > all_gather/
reduce TP ~lower**. So even ATOM considers our mode (3) the SLOWEST of its three.

### THE KEY MECHANISM DIFFERENCE (answers the whole investigation)
In DP-attn + TP-MoE (mode 3, what BOTH engines run in our config), the MoE must
gather hidden states across DP ranks, run experts, scatter back. The cost hinges
on HOW the gather handles UNEVEN per-rank token counts (mixed prefill/decode):

- **ATOM (PR #930)**: `dp_eager_mode = not dp_uniform_decode`. When any rank is
  mid-prefill (ranks have DIFFERENT token counts), uses **variable-length
  `all_gatherv` / `reduce_scatterv`** with per-rank `sizes` (moe.py:261-282).
  Gathers exactly sum(real tokens). Also fuses hidden+router into ONE gather.
  → NO padding waste in mixed steps.
- **SGLang**: `DpPaddingMode` (dp_attention.py:53-86). `MAX_LEN` mode pads EVERY
  rank to `max(global_num_tokens)` then `all_gather_into_tensor`. In a mixed step
  (1 rank prefills 16384, 7 ranks decode ~32), ALL 8 ranks are padded to 16384 →
  gathers 8×16384 instead of 16384+7×32 → **~8× comm + 8× wasted expert compute
  on padding**. SGLang only escapes to `SUM_LEN` (all_reduce) when
  `is_extend_in_batch and dp_size>1` — i.e. the WHOLE batch is extend; the
  worst case (MIXED prefill+decode across ranks) still hits MAX_LEN pad.

⇒ THIS is why ATOM converts a big prefill chunk into throughput and SGLang does
not (Exp 18/20): ATOM's variable-length gather keeps mixed-step MoE cheap; SGLang
pads mixed steps to max-len, inflating both the all-gather volume AND the expert
GEMM (padding tokens are computed). Matches Exp 20 "SGLang prefill input tput =
½" and the delayer's value (delayer aligns ranks → fewer mixed/max-pad steps).
This is consistent with ALL prior data and needs NO unreliable trace.

### What SGLang can tune / do about it (concrete)
- SGLang ALREADY has variable-length `all_gatherv`/`reduce_scatterv` BUT only on
  the `should_use_flashinfer_cutlass_moe_fp4_allgather()` path
  (token_dispatcher/standard.py:152,229) — NOT active for our ROCm/aiter fp8
  MoE. The general dp-attn path uses DpPaddingMode MAX_LEN/SUM_LEN padding.
- Knob candidates to test (in priority order):
  A. Force **SUM_LEN** mode (all_reduce, no max-pad) for mixed steps — check if a
     server arg / env selects DpPaddingMode, or if `--moe-dense-tp-size` /
     attention-dp config changes the gather. SUM_LEN avoids the 8× max-pad blow-up.
  B. **Prefill-delayer tuning** to MINIMIZE mixed steps: lower max_delay_ms is
     wrong (more mixed); instead ensure delayer keeps ranks aligned. Already ON
     and helping (+41%, Exp 20). Try `--prefill-delayer-queue-min-ratio` (SGL-only
     knob ATOM didn't port) to further reduce fragmentation.
  C. File/seek a SGLang feature: variable-length dp gather for the aiter MoE path
     (port the flashinfer all_gatherv path to the general path) — the real fix,
     mirrors ATOM PR #930.

### Recommended next experiment
Test (A): find how SGLang selects DpPaddingMode and force SUM_LEN (or confirm it's
already SUM_LEN at our setting via logging `get_dp_padding_mode`), then A/B vs
MAX_LEN at 8k/1k c256. If SGLang is stuck in MAX_LEN for mixed steps, that's the
quantified gap; if already SUM_LEN, the residual is the all_reduce-vs-gatherv +
fused-gather efficiency, pointing to feature (C).

Refs: ATOM PR #930 (github.com/ROCm/ATOM/pull/930), PR #157
(github.com/ROCm/ATOM/pull/157).

## Exp 23 — DP-MoE gather comm microbench (isolates the primitive) — 2026-06-12

### Why: user noted SUM_LEN uses all_reduce (not gather) — primitive differs from
ATOM's all_gatherv even when both avoid padding. So a MAX_LEN-vs-SUM_LEN model
A/B can't prove "SGLang can match ATOM". Instead measured the 3 primitives
directly (no model, no unreliable trace). /workspace/moe_comm_microbench.py,
torchrun 8×MI355X, hid=7168 bf16.

### SGLang SUM_LEN is all_reduce-as-gather (confirmed, dp_attention.py:463-495)
`_dp_gather_via_all_reduce`: zero a sum_len buffer → memcpy local slice into own
offset → all_reduce(SUM). Result == all_gatherv result (sum_len, no pad) BUT each
rank ships the FULL sum_len buffer (mostly zeros) through ring all_reduce (~2×
traffic). MAX_LEN → `_dp_gather_via_all_gather` (all_gather_into_tensor, padded).

### Results — MIXED step (rank0 prefill 16384, rank1..7 decode 32; sum=16608)
| primitive | ms/iter | gathered rows | buffer |
|---|---|---|---|
| MAX_LEN allgather (SGL worst) | 3.974 | 131072 (7.9× pad) | 1879 MB |
| SUM_LEN allreduce (SGL better)| 1.240 | 16608 | 238 MB |
| ATOM allgatherv               | **0.877** | 16608 | 238 MB |

### Results — ALIGNED step (every rank prefill 16384; sum=131072)
| primitive | ms/iter |
|---|---|
| MAX_LEN allgather | 4.513 |
| SUM_LEN allreduce | 8.807 |
| ATOM allgatherv   | 5.248 |

### Findings (quantified gap decomposition)
1. **MIXED step is where the gap lives.** MAX_LEN (3.97ms) is **4.5× slower than
   ATOM allgatherv (0.88ms)** — the 7.9× padding blow-up. If SGLang is stuck in
   MAX_LEN on mixed steps, that alone is a huge per-step penalty.
2. **SUM_LEN closes MOST but not all of it**: 1.24ms vs ATOM 0.88ms → SUM_LEN is
   3.2× faster than MAX_LEN, but still **~1.4× slower than ATOM allgatherv** on the
   mixed step. THIS is the residual "all_reduce vs all_gatherv" primitive gap the
   user predicted — real but small (~0.36ms/layer) vs the padding penalty (~3ms).
3. **ALIGNED step: all_reduce is the WORST (8.8ms).** When every rank has equal
   big tokens, SUM_LEN all_reduce (ships 131072 rows ×~2 ring) is ~1.7× slower
   than both all_gather variants (~4.5-5.2ms). ⇒ SUM_LEN is only good for the
   UNEVEN/mixed case; for aligned big batches all_gather wins.
   This is exactly why SGLang picks SUM_LEN only when communication is cheaper
   (DpPaddingMode.get_dp_padding_mode: sum_len*2 vs max_len*dp).

### Conclusion / what it means for SGLang
- Most of the SGLang↔ATOM gap = **MIXED-step MAX_LEN padding (4.5×)**, NOT the
  all_reduce-vs-allgatherv primitive (only ~1.4× residual).
- So the high-value SGLang lever is: **avoid MAX_LEN on mixed steps**. Two routes:
  (i) the delayer (aligns ranks → fewer mixed steps; already +41% Exp 20), and
  (ii) make mixed steps use a variable-length gather instead of MAX_LEN pad.
- SGLang's existing SUM_LEN (all_reduce) already captures ~75% of the available
  win on mixed steps (3.97→1.24 vs 0.88 floor). The remaining ~1.4× needs the
  ATOM-style all_gatherv path (the flashinfer dispatcher has it; aiter MoE path
  does not — porting it = ATOM PR #930 equivalent, the real fix C).
- IMPORTANT: SUM_LEN must NOT be forced globally — it's 1.7× WORSE on aligned
  steps. SGLang's adaptive MAX_LEN/SUM_LEN choice is correct in principle; the
  problem is MAX_LEN's pad cost on mixed steps, which all_gatherv would fix.

### Next: verify which mode SGLang actually uses on our mixed steps
Add logging of `forward_batch.dp_padding_mode` (or `get_dp_padding_mode`) over a
real 8k/1k c256 run and histogram mixed vs aligned steps + chosen mode. If mixed
steps hit MAX_LEN → the 4.5× penalty is live and the delayer + a future
all_gatherv port are the fixes. Refs: dp_attention.py:53-95,463-531.
Artifact: /workspace/moe_comm_microbench.py

## Exp 24 — Padding-mode confirmation: SGLang ALWAYS uses SUM_LEN — 2026-06-12

### Method
Added env-gated log (SGLANG_DP_PADDING_MODE_DEBUG) at forward_batch_info.py:1103
printing chosen DpPaddingMode + token skew per step. Real 8k/1k c256 dp8 run,
chunk16384/rank, delayer ON. Deduped to DP0/TP0 (global decision). 54 steps.

### Result — SGLang NEVER uses MAX_LEN at this workload
| step type | MAX_LEN | SUM_LEN | total |
|---|---:|---:|---:|
| MIXED (prefill+decode skew) | 0 | 18 | 18 |
| aligned-ish extend | 0 | 36 | 36 |
| pure-decode | 0 | 0 | 0 |
| **overall** | **0** | **54** | 54 |

ALL 18 mixed steps (skew up to 951×, e.g. tmin=32 tmax=16301) chose **SUM_LEN**.
maxpad blow-up of mixed steps: p50=1.8×, max=7.6× — but since SUM_LEN is used,
that padding is NOT paid. Why: get_dp_padding_mode returns SUM_LEN whenever
`is_extend_in_batch and dp_size>1` (dp_attention.py:76-77) — and every prefill-
containing step IS extend_in_batch. So MAX_LEN is essentially never selected
for our mixed prefill steps.

### Conclusion — the 4.5× MAX_LEN penalty is NOT live; gap is the all_reduce residual
- Exp 22/23's "MAX_LEN 4.5× padding penalty" is REAL in isolation but **NOT
  occurring** in our runs — SGLang already avoids it via SUM_LEN.
- Therefore the SGLang↔ATOM MoE-comm gap that IS live = the **SUM_LEN all_reduce
  vs ATOM all_gatherv residual**, which Exp 23 measured at only **~1.4×** on the
  mixed step (1.24ms vs 0.88ms), i.e. ~0.36ms/layer. The user's original concern
  ("SUM_LEN still all_reduce, not gather") is exactly the live difference — and
  it is SMALL, not the dominant gap.
- ⇒ MoE dp-gather comm is NOT the main remaining throughput gap. The per-step
  all_reduce overhead is modest. The larger throughput gap (Exp 18: SGLang ~88%
  of ATOM at 8k/1k c256) must come from ELSEWHERE — candidates not yet isolated:
  expert GEMM efficiency / quant (Exp 19 showed ATOM's fp8 quant kernel heavy but
  trace unreliable), attention(MLA) kernel cost, or scheduler step-rate (CPU
  overhead / sync frequency), NOT the dp-gather padding.

### Net takeaways for the whole DP investigation
1. delayer: essential (+41%), keep ON. (Exp 20)
2. chunk size: must be 16384/rank but doesn't close gap. (Exp 16)
3. retract/swa: not active at this setting. (Exp 20)
4. DP-MoE gather: SGLang uses SUM_LEN (good), avoids MAX_LEN pad; residual vs
   ATOM all_gatherv is only ~1.4×/layer. (Exp 23/24)
5. ⇒ Remaining gap is NOT comm-padding. Next look at COMPUTE (expert GEMM/quant,
   MLA) per-step cost or scheduler step throughput, with a RELIABLE method
   (controlled microbench or careful per-kernel timing, not the inflated trace).

Artifacts: forward_batch_info.py +SGLANG_DP_PADDING_MODE_DEBUG log,
/workspace/run_padmode_check.sh, /workspace/server_logs/padmode_sgl.serverlog

## Exp 25 — Port ATOM all_gatherv+reduce_scatterv into SGLang (WIP, 2 bugs hit) — 2026-06-12

### Goal
Wire variable-length DP-MoE gather (ATOM PR #930 style) into SGLang's general
(aiter, non-flashinfer) dp-attention path, env-gated SGLANG_DP_USE_GATHERV.
Infra already existed: GroupCoordinator.all_gatherv / reduce_scatterv (pynccl,
parallel_state.py:813,1003); only the flashinfer fp4 path + EP combine used them.

### Implementation (env-gated, default OFF; only attn_tp_size==1, tp==dp)
- dp_attention.py: `is_dp_gatherv_active()`, `_dp_gather_via_all_gatherv()`,
  `_dp_gatherv_sizes()`; new branch in `_dp_gather` (gather) and reduce_scatterv
  in `dp_reduce_scatter_tensor`.
- communicator.py: gatherv branch in `_scatter_hidden_states` (combine,
  reduce_scatterv) + `should_use_reduce_scatter` returns True for gatherv.

### Bug 1 (FIXED): logits path size mismatch
`dp_gather_replicate` is reused by logits_processor with DIFFERENT per-rank sizes
(global_num_tokens_for_logprob, not global_num_tokens). Hardcoding
get_dp_global_num_tokens() → `assert input.shape[0]==sizes[rank]` crash at startup.
Fix: `_dp_gatherv_sizes(obj)` reads sizes from the passed ForwardBatch/
LogitsMetadata (global_num_tokens_for_logprob_cpu else global_num_tokens_cpu),
+ guard: only take gatherv path when local rows >= sizes[rank] and
sum(sizes) <= global buffer rows; else fall back to all_reduce. Startup passed.

### Bug 2 (OPEN): aiter MoE GPU fault under real load
After the logits fix, server reached READY but crashed under real c256 traffic:
"Fatal Python error: Aborted" → GPU fault inside aiter `flydsl_moe_stage1`
(aiter/ops/flydsl/moe_kernels.py:631) via watchdog. The variable-length gathered
tensor has sum_len rows (e.g. 16608 — odd, unaligned) which the aiter fused MoE
flydsl kernel cannot handle. MAX_LEN (graph-aligned max_len*dp) and SUM_LEN
(pre-sized buffer + all_reduce) both fed the kernel a "safe" shape; raw sum_len
all_gatherv does not. ATOM avoids this via its own pad path (moe.py pad_align=256,
pad_for_all_gather) — the gathered token dim must be padded/aligned before the
flydsl MoE GEMM.
⇒ A correct SGLang port must ALSO pad the gathered buffer to the aiter MoE's
required token-block alignment (then unpad on reduce_scatterv), not just swap the
collective. This is the non-trivial part ATOM's PR #930 actually solved.

### Status / result
- OFF (baseline, np1024 8k/1k c256): 24,794 tok/s, TPOT 85.5 — clean.
- ON: crashes in aiter MoE; throughput not measurable yet.
- Code left in place but DEFAULT OFF (SGLANG_DP_USE_GATHERV unset) → zero impact
  on normal runs. Needs the token-alignment fix before it can be benchmarked.

### Next step to finish the port
Add token-dim padding around the gatherv MoE path: pad each rank's local rows (or
the gathered sum_len) up to the aiter flydsl MoE block alignment before
quant_method.apply, and slice back before reduce_scatterv. Mirror ATOM
moe.py:780-793 (pad_align=256) + pad_for_all_gather. Then re-run the A/B.

Artifacts: dp_attention.py + communicator.py (gatherv path, env-gated OFF),
/workspace/run_gatherv_ab.sh, /workspace/run_gatherv_on_only.sh,
/workspace/server_logs/gatherv_*.serverlog, /workspace/bench_gatherv_ab/

## Exp 26 — gatherv Bug 2 FIXED (zero-pad buffer) + gsm8k correctness PASS — 2026-06-12

### Bug 2 root cause (M=32768 memory fault)
deepseek_v4.py:1536-1548: MoE runs on the ENTIRE global_dp_buffer
(M = global_tokens.shape[0]), not just the valid rows. MAX_LEN/SUM_LEN fill the
whole buffer; my all_gatherv filled only sum(real per-rank) → unfilled tail =
garbage → aiter flydsl MoE read OOB → "Memory access fault" (saw M=32768).

### Fix
_dp_gather_via_all_gatherv now zero-pads each rank's local tensor up to
sizes[rank] so sum(sizes) == buffer rows and every row is initialized; guard in
_dp_gather tightened to require sum(sizes) == global_tokens.shape[0] exactly
(else fall back to all_reduce). dp_attention.py only; env-gated, default OFF.

### Validation
- Diag (--disable-cuda-graph, c16 np64): **0 errors, 64/64 successful**, no fault.
- gsm8k 5-shot correctness A/B (chunk16384/rank, delayer ON):
  | | flexible-extract | strict-match | server errors |
  |---|---|---|---|
  | gatherv OFF | 0.9515 | 0.9522 | 0 |
  | gatherv ON  | 0.9507 | 0.9515 | 0 |
  Δ = 0.08% / 0.07% — within gsm8k noise (±0.59%). ⇒ **gatherv preserves
  accuracy**; the variable-length gather/scatter is functionally correct.

### Status
gatherv path now CORRECT (accuracy verified) and crash-free. Throughput A/B at
full c256 was interrupted to run the accuracy check first; rerun pending.
OFF baseline (earlier, np1024 8k/1k c256): 24,794 tok/s. Next: measure gatherv ON
throughput at same config and compare.

Artifacts: /workspace/run_gatherv_gsm8k_ab.sh, /workspace/gsm8k_gatherv/,
/workspace/run_gatherv_diag.sh (diag), dp_attention.py (zero-pad fix).

## Exp 27 — gatherv throughput A/B: NO improvement (≈0%) — 2026-06-12

### Result (8k/1k c256, np1024, ratio0.8, chunk16384/rank, delayer ON, cuda-graph ON)
| metric | OFF (SUM_LEN all_reduce) | ON (all_gatherv+reduce_scatterv) | ON vs OFF |
|---|---|---|---|
| total tok/s | 24,314 | 24,312 | **−0.0%** |
| output tok/s | 2,693 | 2,693 | −0.0% |
| median TTFT ms | 2,254 | 2,272 | +0.8% |
| median TPOT ms | 87.1 | 86.7 | −0.4% |
| p99 TPOT ms | 118.6 | 117.6 | −0.8% |
| bench dur s | 349.9 | 349.9 | +0.0% |
both 1024/1024 completed, 0 server errors.

### Conclusion — the comm-primitive swap does NOT move end-to-end throughput
- Despite the microbench (Exp 23) showing all_gatherv ~1.4× cheaper than SUM_LEN
  all_reduce on a mixed step, the FULL-MODEL throughput is identical (±0.4%).
- Why: the DP-MoE gather/scatter is only a SMALL fraction of total per-layer
  time. Saving ~0.36ms/layer on the collective is washed out by the dominant
  cost (expert GEMM, MLA attention, fp8 quant, and the rest of the comm that the
  zero-padding re-introduces — gatherv pads each rank to sizes[rank] which is the
  cuda-graph-aligned per-rank size, so the gathered M ≈ the same as SUM_LEN's
  buffer; the theoretical "no padding" win is largely given back by aligning to
  the buffer the MoE requires).
- ⇒ Porting ATOM's all_gatherv into SGLang is CORRECT (gsm8k 95.07% ≈ 95.15%,
  Exp 26) but does NOT close the SGLang↔ATOM throughput gap. The gap is NOT the
  DP-MoE collective primitive.

### Net (final) for the DP throughput investigation
The 8k/1k c256 SGLang≈88%-of-ATOM gap is NOT explained by any communication
factor we tested: not delayer (essential, Exp 20), not chunk size (Exp 16), not
retract/swa (Exp 20), not MAX_LEN padding (never used, Exp 24), not all_reduce-
vs-all_gatherv primitive (no end-to-end effect, Exp 27). Remaining suspect =
COMPUTE per-step (expert GEMM / fp8 quant / MLA) or scheduler step-rate, to be
measured with a controlled compute microbench (not the inflated full trace).

### Decision on the gatherv code
Keep env-gated (SGLANG_DP_USE_GATHERV), DEFAULT OFF — it is correct but gives no
throughput win, so it should not be enabled by default. Can be removed or left as
dormant infra. The DP-MoE comm line of investigation is CLOSED.

Artifacts: /workspace/run_gatherv_ab.sh, /workspace/bench_gatherv_ab/
(gathervOFF/ON jsonl), /workspace/run_gatherv_ab2.master.log.

## Exp 28 — gatherv TRACE verify: the gather was never the expensive collective — 2026-06-12

### Captured gatherv ON prefill trace (DP0/TP0, 16384/rank, --disable-cuda-graph)
Communication kernels in the ON trace:
| kernel | ms | count |
|---|---|---|
| quickreduce::allreduce_twoshot | 1648.9 | 244 |
| cross_device_reduce_2stage | 543.6 | 122 |
| nccl/rccl | 16.1 | 7 |
| **allgather_vec (my gatherv path)** | **0.3** | **3** |
reduce_scatter kernels: 0 named; allgather_vec: 3 (gatherv IS active).

### KEY FINDING — gatherv changes the trace, but the gather was tiny all along
- My gatherv path IS taking effect (allgather_vec present, dp_gather_partial in
  deepseek_v4.py:1540 confirmed reached for tp8dp8). But it accounts for only
  ~0.3ms — the DP attn→MoE hidden-state **gather was never the expensive comm**.
- The dominant collective (2.2s: quickreduce 1649ms + cross_device 544ms) is a
  SEPARATE per-layer all-reduce (MoE reduce_results / attention output all-reduce
  via moe_tensor_model_parallel_all_reduce, communicator.py:542), present in BOTH
  OFF and ON, untouched by my change.
- ⇒ This DEFINITIVELY explains Exp 27's 0% throughput delta: swapping the DP
  gather primitive (all_reduce→all_gatherv) optimizes a ~0.3ms collective; the
  real comm cost is the per-layer quickreduce all-reduce, which my change does
  not affect. The earlier Exp 23 microbench compared the WRONG collective for
  this code path (it modeled the full hidden-state gather as if it were the heavy
  one; in DSV4 the heavy one is the per-layer AR, not the dp-gather).

### Corrected understanding of SGLang DSV4 dp8 MoE comm
1. attn→MoE hidden gather: dp_gather_partial (all_reduce SUM_LEN, or my
   all_gatherv) — SMALL (~0.3ms over the window here).
2. per-layer MoE/attn output all-reduce: quickreduce + cross_device_2stage —
   LARGE (~2.2s), the real comm cost. This is what Exp 19 trace also flagged.
⇒ To attack SGLang comm, the target is the per-layer quickreduce all-reduce
(count × cost), NOT the dp-gather. Candidate levers: --enable-fused-moe-sum-all-
reduce, --enable-aiter-allreduce-fusion, or reducing the number of per-layer ARs.
But note Exp 18 already showed AR-count is not the throughput differentiator vs
ATOM, and ATOM pays a similar per-layer AR — so this likely won't close the gap
either; compute remains the prime suspect.

### Decision
gatherv code stays env-gated DEFAULT OFF (correct but irrelevant to throughput,
now trace-proven). DP-MoE gather investigation fully CLOSED with trace evidence.

Artifacts: /workspace/run_gatherv_trace.sh, /workspace/prof_gatherv_ON/ (8 gz),
/workspace/prof_gatherv_OFF/ (truncated — server killed early; ON is sufficient).
token-padding code: dp_attention.py `_dp_gather_via_all_gatherv` lines ~557-585.

## Exp 29 — ATOM vs SGLang per-layer MoE output reduce OP (code comparison) — 2026-06-12

### SGLang (our config): per-layer MoE/attn output = ALL_REDUCE
Trace (Exp 28): quickreduce::allreduce_twoshot 1649ms + cross_device_reduce_2stage
544ms dominate. SGLang reduces the MoE/attn output across TP via
`(moe_)tensor_model_parallel_all_reduce` (communicator.py:542, the custom
all-reduce: quickreduce / cross_device_2stage). The dp-attn hidden gather is a
separate small collective.

### ATOM (our config tp8dp8, no-EP): per-layer MoE output = REDUCE_SCATTERV
atom/model_ops/moe.py `forward_impl_graph`:
- `use_dp_gather_scatter` path (dp_size>1, no mori/EP, enable_dp_attention — i.e.
  OUR config): MoE output reduced via **`reduce_scatterv(final_hidden_states,
  sizes, dp_group)`** (line 3178, variable-length reduce-scatter) — NOT all_reduce.
- `self.reduce_results` all_reduce (line 3190) is **default False** for the
  routed FusedMoE (deepseek_v4.py:2179,2198 construct experts with
  reduce_results=False) → skipped.
- `combine_outputs` (deepseek_v4.py:2316-2328) does a `tensor_model_parallel_
  all_reduce` ONLY for the shared-expert add when tp_size>1; in the dp-attn
  flattened config the heavy routed-expert reduce is the reduce_scatterv above.

### The actual per-layer comm-OP difference
| | attn→MoE gather | routed-MoE output reduce |
|---|---|---|
| SGLang | all_reduce(SUM_LEN) / all_gatherv | **all_reduce** (quickreduce, 2-shot, full hidden replicated to every rank) |
| ATOM   | all_gatherv | **reduce_scatterv** (each rank keeps only its 1/dp slice) |

⇒ ATOM uses GATHER+SCATTER symmetric pair (all_gatherv in, reduce_scatterv out),
so each rank only ever holds/communicates its own token slice. SGLang uses
all_gather-or-allreduce in and **all_reduce out**, where all_reduce moves the FULL
hidden state (every rank ends with all tokens, ~2x ring traffic) then a separate
scatter. ATOM's reduce_scatterv is the natural inverse of all_gatherv and moves
~half the bytes of an all_reduce for the same result.

### Why my SGLang gatherv port didn't help (now fully explained)
I replaced the GATHER side (all_reduce→all_gatherv, ~0.3ms, irrelevant) but the
heavy per-layer collective is the OUTPUT **all_reduce** (2.2s), which SGLang's
MoE layer does via tensor_model_parallel_all_reduce — I did NOT replace that.
To match ATOM, SGLang's routed-MoE output would need to use **reduce_scatterv**
(reduce_results=False + a dp reduce_scatter combine), i.e. the symmetric inverse
of the gather. That is the real change; the gather swap alone is cosmetic.

### Candidate next step (if pursuing comm)
Make SGLang's DSV4 routed-MoE output use reduce_scatterv instead of all_reduce
in the dp-attn path (pair it with the all_gatherv gather already added). SGLang
HAS the EP combine reduce_scatterv (should_use_dp_reduce_scatterv) but it's gated
to ep_size==dp_size; the no-EP TP-MoE path still all_reduces. Reusing that
reduce_scatterv for the no-EP dp-attn path = the ATOM-equivalent fix. NOTE: Exp
18 suggests comm is not the throughput differentiator vs ATOM, so expected upside
is uncertain; measure before investing.

## Exp 30 — Symmetric reduce_scatterv combine: WRONG for SGLang (gsm8k 95%→43%) — 2026-06-12

### What was tried
Paired the all_gatherv gather with a reduce_scatterv combine in DSV4
deepseek_v4.py:1557 (and communicator _scatter_hidden_states), to mirror ATOM's
symmetric all_gatherv + reduce_scatterv (Exp 29).

### Result — accuracy COLLAPSED (no crash, wrong output)
gsm8k 5-shot: OFF 0.95 vs ON **0.4306** (both strict+flexible). 0 server errors.
⇒ reduce_scatterv is SEMANTICALLY WRONG for SGLang's non-EP TP-MoE path.

### Root cause — SGLang vs ATOM MoE token/expert layout differs
- ATOM (use_dp_gather_scatter, dp→tp64 flatten): experts ARE sharded across the
  flattened ranks. Each rank all_gatherv's all tokens, computes its PARTIAL
  expert contribution for all tokens, then **reduce_scatterv SUMS partials across
  ranks + scatters** each rank its slice. Sum is REQUIRED.
- SGLang (_use_tp_moe_gather, no EP): the gather replicates tokens to every rank
  and each rank computes the FULL routed-expert set for those tokens (experts NOT
  sharded the ATOM way). The correct combine is therefore a SLICE (dp_scatter) —
  the per-token result is already complete on the rank that owns it. Applying
  reduce_scatterv SUMS across ranks → each kept token is corrupted (summed with
  other ranks' full results) → garbage logits → 43% accuracy.
- reduce_scatterv is correct ONLY for SGLang's EP path
  (should_use_dp_reduce_scatterv: ep_size==dp_size, experts sharded).

### Conclusion (final on the comm line)
SGLang and ATOM's DP-MoE are NOT the same gather+scatter pair at the semantic
level: ATOM = all_gatherv + reduce_scatterv (sum, sharded experts); SGLang no-EP
= gather(all_reduce/all_gatherv) + dp_scatter(slice, replicated experts). You
CANNOT just swap SGLang's combine to reduce_scatterv — it's a different expert-
parallelization scheme. Matching ATOM would require switching SGLang to the EP /
expert-sharded MoE (--moe-a2a-backend / --ep-size), which is a much larger change
and a different config than the aligned baseline.
REVERTED the reduce_scatterv combine. Kept ONLY the all_gatherv GATHER path
(env-gated SGLANG_DP_USE_GATHERV, default OFF, paired with dp_scatter slice —
verified correct gsm8k 95.07% in Exp 26, and 0% throughput in Exp 27/28).

### Net
- The all_gatherv gather alone: correct, no throughput win (gather isn't the
  bottleneck — Exp 28).
- The reduce_scatterv combine: incorrect for SGLang's non-EP MoE — cannot adopt
  ATOM's symmetric pair without adopting ATOM's expert sharding (EP).
- DP-MoE comm line CLOSED. The SGLang↔ATOM gap is not reachable via comm-op swaps
  in the no-EP TP-MoE config. Remaining suspect = compute (expert GEMM/quant/MLA)
  or EP itself.

Files reverted to safe state: deepseek_v4.py (dp_scatter for non-EP, import
trimmed), communicator.py (no gatherv reduce_scatterv branch). dp_attention.py
all_gatherv gather retained (env-gated OFF).

## Exp 31 — CORRECTION: Exp 30 was a DOUBLE-REDUCE bug, not a semantic mismatch — 2026-06-12

### User pushback (correct): ATOM is also TP, how does it shard experts?
Re-read ATOM weight loading (moe.py:2776-2794, non-EP path):
- w13_weight: `loaded_weight[:, 2*tp_rank_start:2*tp_rank_end]` — slices the
  INTERMEDIATE dim. w2_weight: `[..., tp_rank_start//2:tp_rank_end//2]` — also
  intermediate. ⇒ ATOM TP-MoE shards each expert by INTERMEDIATE (every rank
  holds ALL experts, a 1/tp slice of each) → each rank produces a PARTIAL output
  → post-experts reduce is a SUM. reduce_scatterv (sum+scatter) is CORRECT.
- SGLang is the SAME: layer.py:215-216 `intermediate_size_per_partition =
  intermediate_size // moe_tp_size`, and moe_tp_size = tp//ep//moe_dp = 8//1//1
  = **8** (parallel_state.py:2030). So SGLang ALSO shards experts by intermediate
  across TP=8. My Exp 30 "experts not sharded" claim was WRONG.

### Real cause of Exp 30's 95%→43%: DOUBLE REDUCE
SGLang's gather→MoE→combine for TP-sharded experts is:
  gather (all_reduce/all_gatherv → all tokens on all ranks)
  → MoE computes partial (intermediate-slice) outputs
  → **MoE-INTERNAL post-experts all_reduce** sums TP partials (deepseek_v2.py:927,
     gated by should_skip_post_experts_all_reduce(use_reduce_scatter=...))
  → dp_scatter slices each rank its tokens.
Exp 30 added reduce_scatterv at the combine but LEFT the MoE-internal all_reduce
on (use_reduce_scatter stayed False) → summed TWICE → 43%.
should_skip_post_experts_all_reduce returns True iff use_reduce_scatter=True
(utils.py:455, doc: "reduce_scatter would double-reduce on top of an all-reduce").

### Fix (Exp 31)
deepseek_v4.py: add `_use_gatherv_pair = _use_tp_moe_gather and
is_dp_gatherv_active() and not max_len`. When set:
  - pass `use_reduce_scatter=True` to self.mlp → MoE SKIPS its internal
    all_reduce (no double reduce).
  - combine uses reduce_scatterv (the single sum+scatter), the symmetric inverse
    of the all_gatherv gather. = ATOM's exact scheme.

### Validation — accuracy RESTORED
gsm8k 5-shot: OFF 0.9477 / 0.9484 vs ON **0.9416 / 0.9424**. Δ ≈ −0.6% (within
±0.65% noise), 0 server errors. ⇒ symmetric all_gatherv + reduce_scatterv pair is
NOW CORRECT for SGLang's non-EP TP-MoE. (Exp 29's "different scheme" conclusion is
RETRACTED — same scheme; it was an implementation double-reduce.)

### Status
gather+scatter symmetric pair correct & env-gated (SGLANG_DP_USE_GATHERV).
Throughput A/B pending (Exp 32).

## Exp 32 — Symmetric pair throughput: +2.8% (real, small win) — 2026-06-12

### Result (8k/1k c256 np1024 ratio0.8 chunk16384/rank delayer ON cuda-graph ON)
| metric | OFF (all_reduce gather + dp_scatter) | ON (all_gatherv + reduce_scatterv) | ON vs OFF |
|---|---|---|---|
| total tok/s | 24,369 | **25,048** | **+2.8%** |
| output tok/s | 2,700 | 2,775 | +2.8% |
| median TTFT ms | 2,291 | 2,193 | −4.3% |
| median TPOT ms | 86.8 | 84.7 | −2.5% |
| bench dur s | 349.1 | 339.6 | −2.7% |
1024/1024 completed both, 0 errors. (Accuracy verified Exp 31: 94.2% ≈ 94.8%.)

### Conclusion — the SYMMETRIC pair (not the gather alone) gives the win
- Exp 27/28 (gather-only swap) = 0% because the heavy collective was the
  post-experts all_reduce, untouched.
- Exp 32 replaces BOTH: the gather (all_gatherv) AND the post-experts reduce
  (all_reduce → reduce_scatterv), skipping the MoE-internal all_reduce. Now the
  heavy per-layer all_reduce IS replaced by reduce_scatter → +2.8% total tput,
  −2.5% TPOT, −4.3% TTFT. Modest but real and consistent across metrics.
- This is exactly ATOM's scheme (all_gatherv + reduce_scatterv, skip internal AR).
  Confirms the per-layer all_reduce→reduce_scatter is a genuine (small) lever.

### Caveat / scale
+2.8% at c256 np1024. The gap vs ATOM (~12% at 8k/1k c256, Exp 18) is only
partially addressed — reduce_scatter moves ~half the bytes of all_reduce but the
collective is not the whole gap (compute remains). Worth: (a) confirm the win
holds across c64/c128/c512; (b) the change is a clean, correct, ATOM-aligned
improvement regardless — candidate to upstream behind a flag or enable for dp8.

### Decision
KEEP the symmetric-pair path. Still env-gated SGLANG_DP_USE_GATHERV (default OFF)
pending broader validation; it is correct (gsm8k) and a real +2.8%. Recommend
sweeping concurrency next, then consider making it default for tp==dp dp-attn.

Artifacts: deepseek_v4.py (_use_gatherv_pair + use_reduce_scatter skip +
reduce_scatterv combine), dp_attention.py (all_gatherv gather),
/workspace/run_gatherv_ab.sh, /workspace/bench_gatherv_ab/, run_gatherv_ab3.master.log.

## Exp 33 — Concurrency sweep: symmetric pair win GROWS with concurrency — 2026-06-12

### Result (8k/1k, ratio0.8, chunk16384/rank, delayer ON, cuda-graph ON; np=conc×8)
| conc | OFF tok/s | ON tok/s | Δtput | OFF TPOT | ON TPOT | ΔTPOT | OFF TTFT | ON TTFT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64  | 12,097 | 12,213 | +1.0% | 44.2 | 44.0 | −0.5% | 1815 | 1768 |
| 128 | 18,293 | 18,500 | +1.1% | 59.4 | 58.7 | −1.1% | 1924 | 1857 |
| 256 | 24,311 | 24,800 | +2.0% | 90.9 | 89.2 | −1.9% | 2062 | 2027 |
| 512 | 30,563 | 31,552 | **+3.2%** | 134.3 | 130.2 | −3.0% | 12999 | 12334 |
0 errors both configs, all completed. (Accuracy verified Exp 31.)

### Findings
- The symmetric all_gatherv + reduce_scatterv pair (replacing all_reduce gather +
  post-experts all_reduce) is a **consistent win at every concurrency**, and the
  win GROWS with concurrency: +1.0% (c64) → +1.1% (c128) → +2.0% (c256) → +3.2%
  (c512). TPOT improves in lockstep (−0.5%→−3.0%); TTFT also slightly lower.
- Makes sense: higher concurrency → larger batches → the per-layer collective
  moves more bytes, so halving its traffic (reduce_scatter vs all_reduce) helps
  more. This is exactly the regime where SGLang lags ATOM most (8k/1k high conc).
- Direction matches the gap shape (gap widens with conc, Exp original problem
  statement). The pair closes a real, growing slice of it — but it's a few %, not
  the full ~12%; compute remains the larger residual.

### Conclusion / recommendation
The symmetric gather+scatter (ATOM-aligned) MoE comm is correct (gsm8k) and a
real, concurrency-scaling throughput win (up to +3.2% at c512, the worst-gap
point). Recommend: enable for tp==dp dp-attention DSV4 (or upstream behind a
flag defaulting on for that config). Remaining SGLang↔ATOM gap is now
attributable mostly to COMPUTE (expert GEMM / fp8 quant / MLA), the last
un-quantified bucket.

Artifacts: /workspace/run_gatherv_sweep.sh, /workspace/bench_gatherv_sweep/
(OFF/ON jsonl c64-512), /workspace/run_gatherv_sweep.master.log.

## Exp 34 — A-fix: gatherv prefill path was silently falling back to all_reduce — 2026-06-15

Re-opened the DP-comm line on the latest upstream (branch
`feat/dp-moe-reduce-scatter`, editable install at `/sgl-workspace/sglang-upstream`)
to chase the residual c256 apple-to-apple gap. Methodology used: module-level
CUDA-event instrumentation (continuous record, single sync at flush — NOT
per-op sync, avoids both the per-op pipeline break and the Exp 19 trace `dur`
inflation) + isolated repeated microbench. ATOM module-event compare was
attempted but is unreliable (model module not imported in ATOM workers; custom
op / compile path) — Exp 19's "ATOM per-kernel inflated" still holds.

### B1 — per-module prefill breakdown (SGLang, graph-off eager, c256)
| module (per layer) | PREFILL (16384 tok/rank) | DECODE (32 tok/rank) |
|---|--:|--:|
| attn | ~14.3 ms | 0.77 ms |
| moe  | ~16.8 ms | 0.93 ms |
| forward total | ~31 ms | ~1.7 ms |
isolated fused_moe @131072 (= dp8 gathered buffer) = 12.2 ms (moe1 4.2 / moe2 4.6
/ top-k reduce 2.4 / quant 0.5), matching the server module's 16.8 ms minus
gate+shared+gather/scatter. KEY: the MoE runs on the *gathered global buffer*
(M = 8x16384 = 131072), i.e. every rank computes all ranks' tokens — but ATOM is
the SAME in tp8dp8 no-EP (engine_core_mgr resets to dp8/tp1, then
flatten_tp_across_dp -> tp=8; both shard expert intermediate 1/8). So the global
buffer is NOT the apple-to-apple gap.

### ROOT CAUSE FOUND (the actual bug) — c256 prefill never used gatherv
Coverage probe on `_dp_gather` over a real c256 run:
| step | gather via gatherv | fallback to all_reduce |
|---|--:|--:|
| PREFILL | **0** | **985 (100%)** |
| DECODE  | 15 | 0 |
dbg: `sizes=[3,3,3,3,3,3,3,3] sum=24  buffer_rows=129325`. `_dp_gatherv_sizes()`
returns `global_num_tokens_for_logprob_cpu` (the LOGPROB token counts, 3/rank),
not the MoE `global_num_tokens_cpu` (~16384/rank). Its sum (24) never equals the
ceil_align'd global buffer (129325), so the `sum(sizes)==buffer_rows` guard fails
and PREFILL always falls back to the heavier all_reduce. Only pure-decode (sizes
happen to match) ever took gatherv. ATOM, by contrast, unconditionally takes the
variable-length all_gatherv whenever any rank has prefill
(`dp_uniform_decode = not any_rank_has_prefill`, model_runner.py:1733).

### A-fix
`_dp_gather` now uses `get_dp_global_num_tokens()` (the buffer-aligned sizes
stored by set_dp_buffer_len, the SAME source the reduce_scatterv combine uses)
as the gatherv sizes; `_dp_gatherv_sizes()` is only the fallback (logits path).
One-line logic change, env-gated only by the existing SGLANG_DP_USE_GATHERV (no
new flag). Committed as `[DP] fix gatherv prefill path: use buffer-aligned sizes`.

### Verification
- Functional: c256 prefill gather now `sizes=[16193,16168,...] sum=129104 ==
  buffer`, ZERO mismatch -> all_gatherv taken on every prefill step.
- gsm8k 5-shot (OFF vs ON, chunk16384/rank, delayer on):
  OFF 0.9484 / ON 0.9431 (strict), Δ -0.53% within noise, 0 server errors.
- Throughput A/B at c256 (np1024, gatherv-ON, only the prefill-fix toggled):
  pre-fix 25,160 -> A-fix 25,885 tok/s (**+2.9%**), median TPOT 84.2 -> 82.0
  (-2.6%), TTFT 2161 -> 2108 (-2.4%).

### Concurrency sweep (gatherv OFF vs ON-with-A-fix, 8k/1k, np=conc*4)
| conc | OFF tok/s | ON tok/s | Δtput | OFF TPOT | ON TPOT |
|---:|---:|---:|---:|---:|---:|
| 64  | 13,157 | 13,387 | +1.7% | 39.9 | 39.3 |
| 128 | 18,579 | 19,421 | +4.5% | 56.4 | 54.2 |
| 256 | 24,937 | 25,754 | +3.3% | 84.8 | 82.3 |
| 512 | 30,255 | 31,637 | +4.6% | 127.5 | 122.0 |
Win grows with concurrency (heavier per-step collective -> larger all_gatherv vs
all_reduce saving). vs ATOM single-stream c256 (26,266): A-fix reaches ~98.5%
(gap ~4% -> ~1.5%).

### C2 — checked the gate/router-gather difference (rejected as a lever)
ATOM computes router locally (M=16384) then cat+gathers hidden+router in one
collective; SGLang gathers hidden only, then computes gate on the global buffer
(M=131072), i.e. an 8x-redundant gate GEMM. Isolated microbench made it look big
(gate GEMM 7168->384: 112us@16384 vs 623us@131072), BUT in-server module-event
measurement shows gate is only **0.068 ms/layer, 7.2% of the MoE module** at
prefill (the small GEMM overlaps with neighbors; isolated absolute time is not
representative). Expected refactor upside <~6%/layer of MoE and likely far less
end-to-end -> LOW ROI, high risk. Not pursued.

### Net
The A-fix is a clean, correct, ATOM-aligned bug fix that makes the existing
gatherv feature actually fire on prefill. c256 apple-to-apple now ~98.5% of ATOM.
Remaining ~1.5% is small; further compute-side levers (attn MLA breakdown, top-k
combine reduce ~19% of MoE) have diminishing ROI, and EP would beat ATOM but
breaks apple-to-apple.

Artifacts: dp_attention.py (`_dp_gather` fix); /workspace/b1_instrument.py,
/workspace/b1_iso_moe.py, /workspace/c1_coverage.py, /workspace/c2_gate_instrument.py
(env/sentinel-gated probes, removed from site-packages after use);
/workspace/run_afix_perf_ab.sh, /workspace/bench_afix/; /workspace/run_c8_sweep.sh,
/workspace/bench_c8_sweep/; /workspace/run_gatherv_gsm8k_ab.sh, /workspace/gsm8k_gatherv/.

### Exp 34 appendix — ATOM-matched sweep (np=conc*8, warm=conc*2) + 3-way table
The Exp 34 throughput A/B and the C8 sweep used np=conc*4. To compare
apple-to-apple with the ATOM Exp 1/13 numbers (which use np=conc*8,
warm=conc*2, ratio0.8), re-ran the SGLang A-fix sweep with the SAME client
settings (gatherv ON incl. A-fix, chunk16384/rank, delayer on, single-stream,
graph on, cons 1.0). Results: /workspace/bench_c9_aligned/, run_c9_aligned_sweep.sh.

SGLang A-fix detailed table (interact = 1000/median_TPOT; tok/s/gpu = total/8):
| workload | TP,DP | conc | total tok/s | tok/s/gpu | out tok/s | Med TTFT ms | Med TPOT ms | Med ITL ms | interact | Med E2E ms |
|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 8k/1k | 8,8 | 64  | 12,498 | 1,562 | 1,384 | 2,020  | 43.0  | 28.2 | 23.2 | 41,125  |
| 8k/1k | 8,8 | 128 | 18,809 | 2,351 | 2,084 | 1,883  | 57.8  | 32.2 | 17.3 | 54,929  |
| 8k/1k | 8,8 | 256 | 25,258 | 3,157 | 2,806 | 2,018  | 87.5  | 37.4 | 11.4 | 82,983  |
| 8k/1k | 8,8 | 512 | 32,322 | 4,040 | 3,591 | 12,062 | 127.2 | 44.4 |  7.9 | 128,644 |

3-way total tok/s comparison (all np=conc*8, ratio0.8, single-stream;
ATOM single/multi from Exp 13):
| conc | SGLang A-fix | ATOM single | ATOM multi | SGL/single | SGL/multi |
|---:|---:|---:|---:|---:|---:|
| 64  | 12,498 | 12,162 | 12,866 | 103% | 97% |
| 128 | 18,809 | 18,651 | 19,436 | 101% | 97% |
| 256 | 25,258 | 27,023 | 27,881 |  93% | 91% |
| 512 | 32,322 | 33,126 | 33,779 |  98% | 96% |

Median TPOT (ms):
| conc | SGLang A-fix | ATOM single | ATOM multi |
|---:|---:|---:|---:|
| 64  | 43.0  | 44.8  | 42.6  |
| 128 | 57.8  | 59.0  | 56.4  |
| 256 | 87.5  | 81.4  | 79.1  |
| 512 | 127.2 | 133.4 | 132.2 |

**Findings**
- vs ATOM single-stream (the true apple-to-apple): SGLang A-fix is 98-103% at
  c64/c128/c512 (matches or slightly beats), and 93% at c256 (the only clearly
  weak point: TPOT 87.5 vs 81.4).
- vs ATOM multi-stream (ATOM's best): 91-97%; the residual is largely ATOM's
  side-stream overlap (single->multi is +2-6%, Exp 13), not a dp-comm issue.
- The np=conc*8 c256 (25,258) is lower than the earlier np1024 A/B (25,885)
  purely due to the longer/heavier run (more prefill-ramp weight) — this is the
  fairer number vs ATOM. So c256 SGLang A-fix ~= 93% of ATOM single (honest),
  not the 98.5% the mixed-np estimate suggested.
- c256 remains the relative weak spot (prefill<->decode interference is worst
  there, consistent with Exp 16/B1). Further upside would need attn (MLA)
  breakdown (C3) or EP (C6, beats ATOM but breaks apple-to-apple).

## Exp 35 — c256 gap localized to prefill per-token COMPUTE (+8%), symmetric measurement — 2026-06-16

Re-opened the c256 residual (~93% of ATOM single-stream) on latest upstream
(branch `feat/dp-moe-reduce-scatter`, editable `/sgl-workspace/sglang-upstream`,
SGLANG_DP_USE_GATHERV=1, chunk131072=16384/rank, delayer on, graph on, cons 1.0).
Artifacts in `/sgl-workspace/c256_analysis/`.

### Methodology guardrail (two corrections made mid-investigation)
This experiment twice produced a WRONG conclusion from single-sided / stale data;
both were caught and retracted. Recorded here so the trap isn't repeated:
1. **"SGLang fragments prefill, ATOM packs 16384"** — WRONG. It cited the Exp 15
   trace (OSL128/ratio1.0/chunk16384), not this config, and compared SGLang
   PER-RANK numbers against ATOM numbers misread as GLOBAL.
2. After collecting ATOM's real per-step data it first looked like ATOM ran one
   global scheduler — also WRONG.
The rule confirmed: **never conclude an SGLang-vs-ATOM gap without measuring BOTH
engines with the SAME method, same config, same basis (per-rank).**

### Verified the gatherv+reduce_scatterv pair is actually live (probe)
Env-gated counter probe on `_dp_gather` / `dp_reduce_scatter_tensor` /
deepseek_v4 combine over a real c256 run: prefill `gatherv_prefill` increments,
`allreduce_fallback_prefill=0`, `allgather_maxlen=0`, `moe_combine_reduce_scatterv`
increments. So A-fix is firing on every prefill step (no fallback). Probe removed
after verification (git clean).

### Fast repro config (3.8 min/run instead of ~13)
c256, ISL8192, OSL1024, ratio0.8, **np=512 warm=256** (vs full np=2048/warm=512).
Holds the prefill<->decode regime (conc/ISL/OSL fixed); only np shrinks. Verified
on BOTH engines it preserves the gap:
| config | SGLang tok/s | ATOM tok/s | SGL/ATOM |
|---|---:|---:|---:|
| fast np512 | 24,761 / 24,984±94 (3x) | 26,742 / 26,705 | 92.6–93.4% |
| full np2048 (handoff) | 25,258 | 27,023 | 93.5% |
OSL must stay 1024 — OSL256 spikes total to 34k (over-weights prefill, leaves the
valley regime; matches Exp 16 warning).

### Phase 1 — prefill vs decode split (per-request bench --output-details)
Identical workload both engines (same input/output token totals):
| metric | SGLang | ATOM | SGL/ATOM |
|---|---:|---:|---:|
| total tok/s | 24,761 | 26,742 | 92.6% |
| decode-only tok/s | 3,870 | 4,026 | 96% (near parity) |
| median TTFT (prefill) | 4,213 | 3,308 | 127% (27% slower) |
| median TPOT (decode step) | 66.4 | 64.0 | 104% |
⇒ The c256 gap is in PREFILL (TTFT), decode is parity. (Confirms Exp 14 with a
clean per-request apple-to-apple split, not the old trace.)

### Prefill chunk composition is NEARLY IDENTICAL (apple-to-apple, per-rank)
Confirmed from source that ATOM `enable_dp_attention=True` with launch
`data_parallel_size=1` is RESET by engine_core_mgr.py:32-40 to dp_size = tp×dp = 8,
tp=1 → **8 independent per-rank EngineCores**, each with prefill budget
`max_num_batched_tokens=16384` (NOT divided by dp; model_runner.py:1114-1115).
SGLang dp8 = chunk131072÷8 = 16384/rank. So **both = 16384 tok/rank/step → the
configs ARE apple-to-apple** (answers the recurring "is this fair?" question).
ATOM `Scheduled prefill batch:` log is PER-RANK (8 EngineCores share one stdout).
Per-rank prefill composition over the fast run:
| | per-rank steps | mean tok/step | full-chunk(>=16k) |
|---|---:|---:|---:|
| ATOM | ~59 | 12,103 | 46% |
| SGLang | ~64 | 11,428 | 45% |
⇒ chunk granularity is NOT the gap (both ~45% full, tok/step within 6%).

### Phase 3 — symmetric per-rank prefill forward-wall timer (THE KEY RESULT)
Added an identical env-gated CUDA-event timer (continuous record, single sync at
flush; rank0 only) around the model forward on BOTH engines:
- SGLang: `model_runner.forward` around `_forward_raw` (SGLANG_FWD_TIMER=1).
- ATOM: `model_runner.run_model` whole body (ATOM_FWD_TIMER=1).
Same fast config, aggregated over the run:
| | prefill steps | prefill tok | **us/tok** | tok/ms | decode ms/step |
|---|---:|---:|---:|---:|---:|
| ATOM | 59 | 729,218 | **168.7** | 5.93 | 41.4 |
| SGLang | 64 | 719,838 | **182.3** | 5.48 | 40.2 |
| SGL/ATOM | | | **+8.1% slower** | | 0.97 (parity) |

**ROOT CAUSE (quantified): SGLang's prefill forward costs ~8% more per token
than ATOM's, at the SAME per-rank token count. Decode forward is parity.**
This is the long-suspected "compute residual" (expert GEMM / fp8 quant / MLA),
now measured directly, not inferred. The 8%/token amplifies to TTFT +27%
(a request spans many prefill forwards + queues behind decode) → total tok/s ~7%.

Caveat on scope match: SGLang timer brackets `_forward_raw` (model only); ATOM
`run_model` includes `compute_logits` (small for prefill: last-token only). Both
prefill paths are eager. The ~8% is robust to this minor scope diff but a
per-module breakdown (Exp 36 plan) should re-confirm.

### Levers already rejected (do NOT re-run)
- `--schedule-conservativeness 2.0` (Exp 5 / handoff C5): no stacking benefit
  after A-fix (-1%). The 8% is compute, not admission throttle.
- prefill-delayer tuning (Exp 5): default is the sweet spot; stronger over-throttles.

### Diagnostic code removed
SGLang `model_runner.py` + `dp_attention.py` + `deepseek_v4.py` and ATOM
`model_engine/model_runner.py` all reverted (git diff clean on the SGLang repo;
ATOM site-packages restored). Scripts kept in `/sgl-workspace/c256_analysis/`:
`bench_one.sh`, `analyze_phase1.py`, `analyze_phase2.py`, `ATOM_baseline.txt`.

### Next: Exp 36 — per-module / per-kernel breakdown of the prefill 8%
KEY HYPOTHESIS TENSION (user): both engines call the SAME aiter kernels for MoE
GEMM / fp8 quant / MLA, so a per-kernel gap "shouldn't" exist. So the 8% must
come from HOW each engine invokes the shared kernels, not the kernels themselves.
Candidate sources to isolate (all testable SGLang-internal + controlled microbench):
1. **GEMM M / gather buffer size**: SGLang MoE runs on the gathered global buffer
   (M = 8×16384 = 131072), ATOM too (flatten tp64) — but verify the EXACT per-rank
   expert-GEMM M and whether SGLang pads more (DpPaddingMode rounding) → bigger M →
   more padding-token compute at the SAME tok/ms denominator (inflates us/REAL-tok).
2. **fp8 quant path**: SGLANG_OPT_FP8_WO_A_GEMM / per-token-group quant shape vs
   ATOM AITER_BF16_FP8_MOE_BOUND=0 path — confirm both hit the same aiter quant
   kernel and same group size; a different quant granularity changes cost.
3. **MLA prefill attention**: indexer / unified_kv_triton (SGLANG_HACK_FLASHMLA_
   BACKEND) vs ATOM's MLA — these may NOT be the same kernel (SGLang uses triton
   indexer + flashmla; ATOM may use a different prefill MLA). Likely the biggest
   suspect since MoE GEMM is genuinely shared aiter.
4. **gate/router redundancy** (C2, Exp 34): SGLang computes gate on global M=131072
   (8× redundant) vs ATOM local — measured ~7% of MoE earlier; re-confirm under
   the per-module timer at prefill.
5. **host-side launch / layout**: extra contiguous()/copy, more kernel launches per
   layer on SGLang's path (non-kernel wall inside the forward).
Method: add a per-MODULE CUDA-event timer (attn / gate / gather / moe1 / moe2 /
quant / topk-combine / scatter) inside one SGLang prefill forward (continuous
record, single sync), get the ms breakdown of the ~182us/tok; then isolated
aiter microbench at the EXACT prefill shapes (M per rank, fp8) to see if the
shared kernel is equal in isolation (→ gap is invocation/shape) or different
(→ gap is a non-shared kernel, e.g. MLA). Start with #3 (MLA) and #1 (M/padding)
as highest-probability; #2/#4/#5 if residual remains.

## Exp 36 — prefill +8% broken down: it is NOT MoE/comm (shared), points to attn(MLA) — 2026-06-16

Executed the Exp 36 plan. Two-stage: (1) SGLang per-module CUDA-event timer to
split the prefill forward; (2) neutral isolated microbench of the shared aiter
kernels + RCCL collectives. ATOM per-module probe attempted and CONFIRMED
infeasible (below). Artifacts `/sgl-workspace/c256_analysis/`.

### SGLang prefill forward per-module breakdown (CUDA-event, rank0, prefill-only)
Timer wrapped attn / hc_norm / gather / moe(mlp) / scatter in the DSV4 decoder
layer (deepseek_v4.py forward). Aggregated over a fast-config c256 run; the 5
blocks sum to 128,982 ms ≈ the Exp 35 forward wall 131,249 ms (≈98% coverage):
| block | share of prefill forward | nature |
|---|---:|---|
| moe (expert GEMM) | 35.6% | shared aiter fused_moe |
| attn (MLA) | 34.2% | SGLang-specific (unified_kv_triton + aiter indexer + compressor) |
| gather (all_gatherv) | 18.2% | DP-MoE comm (RCCL) |
| scatter (reduce_scatterv) | 10.1% | DP-MoE comm (RCCL) |
| hc_norm | 1.9% | fused HC norm |
CORRECTION to the Exp 35 first cut ("other 40%"): the "other" is actually
gather+scatter = **28% DP-MoE communication**, not gate/quant/host glue.

### Stage 2a — isolated aiter MoE GEMM (neutral, engine-independent)
`/workspace/b1_iso_moe.py`, aiter `fused_moe` at the gathered prefill shape:
| M (tokens) | fused_moe | note |
|---|---:|---|
| 16,384 (1 rank real) | 1.72 ms | |
| 131,072 (gathered 8×) | 12.02 ms | both engines run on this M |
8× tokens → 7× time (sub-linear) ⇒ compute-bound, efficient at large M, NO
redundancy penalty from the global buffer. Per-kernel: moe2 37% / moe1 34% /
top-k reduce_kernel 19% / quant 4% / sort 5%. Both engines call THIS SAME aiter
kernel at the SAME M ⇒ MoE GEMM is equal, NOT the gap. (Confirms the user's
"shared aiter kernel shouldn't differ" intuition for MoE.)

### Stage 2b — neutral collective microbench (RCCL floor)
`/workspace/moe_comm_microbench.py`, 8×MI355X, hid7168 bf16, aligned all-prefill
(every rank 16384 → 131072 global):
- all_gatherv = 5.25 ms/call (moves 1879 MB). This is the RCCL hardware floor.
Reconcile vs the in-server module timer: SGLang gather = 6.10 ms/layer-call,
scatter = 3.41 ms/layer-call. gather 6.1 vs floor 5.25 ⇒ SGLang's all_gatherv is
within ~16% of the raw RCCL floor (per-layer reuse overhead). ATOM also uses
all_gatherv+reduce_scatterv (PR #930) over the same bytes on the same RCCL ⇒
**comm is at the hardware floor on both, NOT the gap.**

### ATOM per-module probe — CONFIRMED INFEASIBLE (validates Exp 34 note)
Tried the symmetric per-module timer inside ATOM's DSV4 decoder layer
(model_runner run_model + models/deepseek_v4 layer forward). ATOM crashed at
init: `torch._dynamo.exc.BackendCompilerFailed: cannot extract sympy expressions
from <torch.cuda.Event>`. ATOM's decoder layer is **torch.compile-wrapped
(VllmBackend)**, so inserting CUDA-Event objects breaks Dynamo graph tracing.
⇒ In-model module probes are impossible on ATOM. The neutral microbench (shared
kernel + RCCL floor) is the ONLY way to get an ATOM-comparable baseline for the
shared blocks; the non-shared block (attn) can only be bounded, not directly
ATOM-probed. All ATOM edits reverted (site-packages restored).

### CONCLUSION (by elimination)
The 8% prefill per-token gap (Exp 35: SGLang 182.3 vs ATOM 168.7 us/tok) is:
- NOT MoE GEMM (35.6% of fwd, shared aiter, isolated-equal).
- NOT comm gather+scatter (28% of fwd, at RCCL floor, both engines same primitive).
- ⇒ **most likely in attn / MLA (34.2% of fwd)** — the only large block that is
  NOT a shared kernel: SGLang uses unified_kv_triton + aiter indexer + compressor;
  ATOM has its own prefill MLA. This is exactly consistent with the user's framing:
  shared aiter kernels (MoE) match; the gap lives in the engine-specific MLA path.
(Caveat: this is by elimination, not a direct ATOM attn measurement, which is
impossible via in-model probe. A direct attn comparison would need an isolated
MLA-prefill microbench replicating each engine's attention kernel sequence — the
recommended Exp 37.)

### Next: Exp 37 — isolate the MLA prefill attention cost
Build an isolated microbench of SGLang's MLA prefill kernel sequence (q/kv proj,
aiter indexer, flashmla/unified_kv core attn) at the c256 prefill shape (16384
tok/rank, the DSV4 head/dim config), time it standalone; compare against ATOM's
MLA prefill kernels (the aiter MLA ops ATOM calls, run standalone — these MAY be
shared aiter ops even if the surrounding glue differs). If the isolated MLA
kernels are equal, the residual is SGLang attn glue/launch overhead (host-side);
if different, it's a genuine kernel-path difference. This is the last bucket;
expected end-to-end upside is ~the 8% prefill share of the c256 gap (a few % of
total tok/s), diminishing ROI now that MoE/comm are ruled out.

### Diagnostic code removed
SGLang deepseek_v4.py module timer + ATOM model_runner/deepseek_v4 timers all
reverted (SGLang git diff clean; ATOM site-packages restored). Microbench
scripts reused: /workspace/b1_iso_moe.py, /workspace/moe_comm_microbench.py.

## Exp 37 — split the attn block; out_proj wo_a einsum is NOT slow (correction) — 2026-06-16

Followed up Exp 36 (gap is in attn/MLA, 34% of prefill fwd). Added an env-gated
CUDA-event timer splitting the MLA attn forward (deepseek_v4.py MQALayer.forward)
into qkv_prep / core_attn / out_proj; ran the fast c256 config. Then a neutral
microbench of the wo_a output projection. Diagnostic code removed after (git clean).

### attn sub-block split (CUDA-event, prefill, rank0) — see caveat below
| sub-block | share of attn | spans |
|---|---:|---|
| out_proj | 49.3% | rope_inplace + o.view + wo_a einsum + wo_b |
| qkv_prep | 30.5% | q/kv proj + aiter indexer + compressor |
| core_attn | 20.2% | unified_kv_triton / flashmla core attention |

### Hypothesis (out_proj einsum slow) — TESTED and REJECTED
Our config has `SGLANG_OPT_FP8_WO_A_GEMM=false`, so wo_a runs as
`torch.einsum("tgd,grd->tgr")` in bf16 (not the fp8/deep_gemm path). Hypothesis:
this einsum is the slow bucket. Two findings refuted it:
1. **Can't even enable the fp8 path here**: `SGLANG_OPT_FP8_WO_A_GEMM=true`
   crashes at init — `ModuleNotFoundError: No module named 'deep_gemm'`. This
   ROCm build has NO deep_gemm, so the aligned config MUST use the einsum path.
   (Informative: SGLang on this build is forced onto the bf16 einsum wo_a; ATOM
   presumably has an equivalent optimized o-proj. But we cannot A/B it here.)
2. **The einsum is already fast**: isolated microbench at the real shape
   (T=16384, G=16, R=1024, D=3584) — `torch.einsum` = 1.36 ms = **1411 TFLOP/s**
   (torch.bmm same, 1.44 ms). 0.083 us/tok × 61 layers ≈ 5 us/tok of the ~182 —
   small, near-peak. The wo_a einsum is NOT the gap. (`wo_a_microbench.py`)

### CAVEAT — attn sub-block split is unreliable (single-stream async tail)
Per-block CUDA events on one stream attribute the async tail of the previous
block's still-draining kernels to the next block's measured window. core_attn's
attention kernels likely drain into the out_proj window, inflating out_proj's
49%. So the in-attn sub-attribution (out_proj 49 / qkv 31 / core 20) is NOT a
trustworthy decomposition; only the attn-BLOCK-as-a-whole (34% of fwd, Exp 36)
and the isolated microbenches (MoE equal, comm at floor, wo_a fast) are reliable.

### Net (Exp 37)
Ruled OUT the wo_a einsum as the gap. The reliable picture stands from Exp 35/36:
- gap is prefill, +8.1% us/tok; decode parity.
- MoE GEMM shared/equal; gather+scatter at RCCL floor → not the gap.
- ⇒ the 8% is in the MLA attn block, but it could NOT be cleanly sub-attributed
  (async-tail) and the one concrete sub-suspect (wo_a) is fast. The remaining
  candidates inside attn are core_attn (unified_kv_triton vs ATOM's MLA — a
  genuine non-shared kernel) and qkv_prep (aiter indexer/compressor). A reliable
  next step needs ISOLATED timing of each attn sub-kernel (separate launches with
  their own sync, NOT chained single-stream events) — deferred (Exp 38, low ROI:
  the whole attn gap is ~8% of prefill ≈ a few % of total tok/s).

### Conclusion of the c256 investigation (Exp 34–37)
The c256 gap (~93% of ATOM single-stream) decomposes as: comm fixed/at-floor
(gatherv+reduce_scatterv, shipped), MoE compute equal (shared aiter), decode
parity; the irreducible residual is ~8% per-token PREFILL compute living in the
engine-specific MLA attention path (not the shared MoE/quant kernels, consistent
with the "same aiter kernel shouldn't differ" expectation). Diminishing ROI to
chase further; the shipped gatherv+A-fix is the main, defensible c256 win.

## Exp 38 — re-evaluate the 8x-redundant gate/router GEMM (C2 corrected) — 2026-06-16

User asked to actually quantify the "SGLang does ~8x the gate GEMM" lever that
C2 (Exp 34) had dismissed at "7.2%, low ROI". Re-measured it NEUTRALLY (isolated
microbench of the exact aiter gate kernel), because the C2 number came from a
single in-server module-event that is subject to the same async-tail
mis-attribution found in Exp 37.

### The redundancy (confirmed real, SGLang-specific)
DSV4 dp-attn + TP-MoE (no EP): SGLang gathers hidden → global buffer (M=131072)
then computes the gate on the WHOLE buffer (`MoEGate.forward`, deepseek_v2.py:985
→ `linear_bf16_fp32` → aiter `tgemm.mm`, hidden7168→384 experts, fp32 logits).
So every rank computes the router logits for ALL 8 ranks' tokens and uses only
its own 1/8. ATOM computes the router LOCALLY (M=16384) then gathers hidden+router
together (cat). This IS a genuine SGLang-only redundancy (unlike the expert GEMM,
which is global-buffer on both engines — Exp 36).

### Isolated gate-GEMM microbench (aiter tgemm.mm, hidden7168→384, fp32)
| | M | ms/layer | GFLOP/s |
|---|---:|---:|---:|
| ATOM local | 16,384 | 0.127 | 709 |
| SGLang global | 131,072 | 0.644 | 1120 |
- Redundancy = **5.1x** (not 8x): the small M=16384 GEMM is launch/mem-bound
  (709 GFLOP/s) while M=131072 is more efficient (1120 GFLOP/s), so 8x tokens
  cost only 5.1x time.
- SGLang-only WASTE = 0.517 ms/layer × ~58 MoE layers = **~30 ms/prefill step**.
- vs prefill forward ~2050 ms/step (Exp 35) ⇒ **~1.5% of the prefill forward**
  ⇒ closing it would gain ~1.5% of prefill ≈ **~+1% total tok/s at c256**.
Script: `/sgl-workspace/c256_analysis/gate_microbench.py`.

### CORRECTION to C2 (Exp 34)
C2 reported gate = 0.068 ms/layer = 7.2% of MoE and called it low-ROI. The
isolated kernel is **0.644 ms/layer — ~10x higher** than the in-server event
measured. The C2 in-server number under-counted because the small gate GEMM's
real cost was hidden in neighbor-kernel overlap (single-stream event tail, same
artifact as Exp 37). So C2's "7.2% / negligible" was an under-estimate; the true
redundant cost is ~1.5% of the prefill forward (still modest in absolute terms,
but ~2x what C2 implied, and it is a clean, shared-kernel, low-risk target).

### Is it worth fixing? (assessment)
- Upside: ~+1% total tok/s at c256 (the prefill-weighted point); less at low conc.
- Approach: mirror ATOM — compute the gate on LOCAL hidden (M=16384) BEFORE the
  gather, then carry router_logits through the gather (cat hidden+logits, or a
  second small gather) so the per-rank slice has its routing. This touches the
  DSV4 gather payload + MoE entry, adjacent to the shipped gatherv/reduce_scatterv
  path — needs care to keep the reduce_scatterv combine symmetric and accuracy
  intact.
- Risk: medium (changes the gather contract); ROI ~+1%. Reasonable as a
  follow-up optimization PR, separate from the gatherv+A-fix that already landed.
  NOT a large win, but it is REAL, quantified, and the cleanest remaining lever
  (shared kernel, no MLA/non-shared-kernel surgery).

Net: the "8x gate GEMM" is a real ~1.5%-of-prefill redundancy (C2 under-stated
it). It's the most actionable remaining c256 lever after gatherv+A-fix, but a
modest ~+1% — consistent with the overall finding that the big buckets (MoE,
comm) are already equal/shipped.

### Implementation (gate-local) — built, debugged to CORRECT, but net-zero tput
Implemented env-gated `SGLANG_DP_GATE_LOCAL` (default OFF): `MoEGate.forward`
returns a stashed precomputed-logits tensor (skip gate GEMM);
`DeepseekV4DecoderLayer._dp_gate_local_gather` computes the gate on LOCAL hidden,
gathers hidden + router-logits, fills the global buffer, stashes the global logits.

Two bugs found & fixed (a COLLECTIVE PROBE — ENTER/EXIT per rank+layer+op,
sentinel-gated, flush-each — pinpointed both):
1. **HANG (collective desync).** First version gated gate-local on per-rank
   `forward_batch.forward_mode.is_extend()`. The probe showed: in a MIXED step
   only the prefilling rank (`pf=True`) took gate-local → `all_gatherv(width
   7552)` while the other 7 ranks (`pf=False`) took the normal
   `dp_gather_partial` → `all_gatherv(width 7168)`. Same collective, DIFFERENT
   width across ranks → RCCL hang. FIX: gate-local must be an ALL-RANKS decision
   — key it ONLY on `_use_gatherv_pair` (derived from the synced dp_padding_mode,
   identical on all ranks), NOT per-rank is_extend. After: no hang, "17+25=42" OK.
2. **ACCURACY collapse (bf16 router logits).** gsm8k A/B: OFF 0.9469 vs ON
   **0.5876**. Cause: I cast the fp32 router logits to bf16 to cat+gather them
   with the bf16 hidden — the bf16 round-trip changes the ungrouped top-6 expert
   selection → wrong routing. FIX: gather the logits in **fp32** via a SEPARATE
   small `all_gatherv` (384 fp32 = 1.5KB/tok). After: gsm8k ON **0.9424** ≈ OFF
   0.9469 (Δ−0.45%, within ±0.64% noise). Routing correct.

### Result: gate-local is CORRECT but gives ~0 total throughput (TTFT −11%)
c256 fast-config A/B (gatherv ON both, only SGLANG_DP_GATE_LOCAL toggled):
| | total tok/s | median TTFT | median TPOT |
|---|---:|---:|---:|
| OFF | 24,736 | 4,206 | 76.9 |
| ON  | 24,725 | 3,759 | 76.9 |
| Δ   | **−0.04% (noise)** | **−10.6%** | ~0 |
So the gate redundancy is real and **TTFT improves ~11%** (the gate GEMM saving
shows up in the prefill phase), but **total throughput does NOT move**:
- the second fp32-logits `all_gatherv` (needed for correctness) costs roughly
  what the gate GEMM saved — net ~0 (exactly the "option (a) nets ~0" predicted
  earlier; the fused single bf16 gather that WOULD net-win is incorrect).
- c256 total tput is decode-dominated, so an 11% prefill-TTFT win barely moves it.

### Why a SECOND gather was needed — ATOM uses ONE; the diff is router dtype
Checked ATOM's `dp_gather_hidden_and_router` (model_ops/moe.py:240). ATOM does
the single FUSED gather: cast router_logits to hidden dtype, `cat`, ONE
`all_gatherv`, split. It has NO second gather. The reason it works in bf16:
**ATOM's gate is `ReplicatedLinear` → bf16 logits** (gate output dtype = input
bf16), so the cast in the fused gather is a NO-OP. SGLang's DSV4 gate is
`linear_bf16_fp32` → **fp32 logits** (deliberately, for router numerical
stability), so casting to bf16 for a fused gather LOSES precision.

### Option A tested: make SGLang's gather bf16 too (single fused, like ATOM)
Re-implemented gate-local with a single fused bf16 gather (cast logits→bf16,
exactly ATOM's scheme). gsm8k = **0.5754** — collapses AGAIN (matches the v1 bf16
result 0.5876; fp32 gave 0.9424). So **SGLang's DSV4 router CANNOT tolerate bf16
logits, even though ATOM's can.** Confirmed twice (independent runs). The DSV4
router (sqrtsoftplus + fp32 e_score_correction_bias + ungrouped top-6) is
numerically sensitive enough in SGLang's topk path that bf16 logits flip the
top-6 expert selection → wrong routing. ATOM tolerates bf16 because its
gate/topk/bias path is structured differently (its gate is natively bf16 and it
ships fine at 0.95).

### Conclusion (gate-local, FINAL)
gate-local is CORRECT only with fp32 logits, which forces a SECOND all_gatherv,
which offsets the saved gate GEMM → **net ~0 total tput at c256** (TTFT −11% but
decode-bound total unmoved). The single-fused-bf16 gather that WOULD net-win
(ATOM's scheme) BREAKS SGLang accuracy. Pushed further to find out WHY.

### Single gather IS possible (bitcast); the REAL blocker is per-M GEMM routing
Got a SINGLE gather with fp32 precision via BITCAST: reinterpret the fp32 logits
as 2x bf16 columns (byte-identical), cat with bf16 hidden, ONE bf16 all_gatherv,
split + view back to fp32. Verified byte-exact through RCCL all_gather, even AND
uneven `sizes` (maxerr=0, standalone). So the dtype/2-gather issue is solvable —
**a single gather works.** But it STILL gave gsm8k 0.58. Clean isolation found
the true root cause (NOT precision, NOT gather, NOT memory lifetime — .clone()
didn't help):
- **NOSKIP test** (fused gather but DON'T skip the gate; recompute gate on the
  gathered global buffer) → gsm8k **0.9477**. ⇒ gather + hidden fill are PERFECT;
  the bug is purely in skipping the gate via the stash.
- **GLTOPK probe** (stash logits = gate on LOCAL hidden M≈12–16384/rank, vs gate
  recomputed on GLOBAL M=131072): **8.3% of tokens get a DIFFERENT top-6 expert
  set**, logits_maxdiff≈0.03.
- ROOT CAUSE: aiter's router GEMM (`aiter_dsv3_router_gemm`/`tgemm.mm`) is
  **autotuned per-M** — different kernel/tile for small per-rank M vs global
  M=131072 → different bf16 rounding (~0.03) → flips ~8% of the borderline
  ungrouped-top-6 expert picks → degraded routing → 0.58.

⇒ gate-local is **fundamentally unsafe in SGLang**: computing the gate at local-M
diverges numerically from the model's reference global-M gate (the gate GEMM is
NOT M-invariant on aiter). ATOM avoids this because its whole pipeline gates
locally end-to-end (no global-M reference to diverge from); SGLang's reference IS
the global-M gate.

### CORRECTION — gate-local DOES work; 0.58 was a buffer bug, NOT per-M/local-M
The "per-M GEMM / local-M reference" theory above was WRONG. Verified by
microbench: the SAME tokens gated at M=16384 vs M=131072 give **top-6 flip = 0.0%**
(mean logit diff 3e-05) — `tgemm.mm` is effectively M-invariant for routing. And
ATOM's gate is literally the SAME aiter `tgemm.mm` (linear.py:603), computed at
local-M; ATOM's whole flow is gate@local-M → gather(hidden+logits) → topk@global
— IDENTICAL to gate-local's flow. So local-M routing is NOT the problem.

REAL ROOT CAUSE of the 0.58 (found by NOSKIP isolation + buffer inspection):
`get_global_dp_buffer()` returns a **fresh `torch.empty` every call** (not a
persistent buffer). My gate-local called it TWICE — the helper filled buffer #1
with the gathered hidden, then the call site did `hidden_states =
get_global_dp_buffer()` again → buffer #2 (uninitialized GARBAGE). The MoE then
ran experts on garbage hidden while routing with correct logits → 0.58.
FIX: the helper returns the buffer it filled; the caller uses THAT instance
(never re-calls get_global_dp_buffer). After the fix: **gsm8k 0.9492 ≈ OFF 0.9469**.
(This also explains why NOSKIP=0.9477: NOSKIP recomputes the gate on buffer #2,
but in that variant buffer #2 WAS the one dp_gather filled — different code path
that happened to use one buffer. The bug was specific to the skip-path's double
get_global_dp_buffer.)

### gate-local FINAL result — CORRECT, single gather, but throughput-neutral
With the buffer fix + single fused bitcast gather (fp32 logits losslessly carried
as 2x bf16) + local-M gate skipping the redundant global gate GEMM:
- **Correctness: gsm8k 0.9492** (OFF 0.9469) — fully correct, ATOM-equivalent flow.
- **c256 throughput (fast config np512)**: ON 24,524 vs OFF 24,687–24,736 = −0.7%.
- **c256 throughput (FULL config np2048/warm512, ROCM700A=0, gatherv ON)** —
  the authoritative measurement:
  | | total tok/s | out tok/s | TTFT | TPOT | dur |
  |---|---:|---:|---:|---:|---:|
  | OFF | 25,354 | 2,817 | 2,008 | 87.2 | 670.8s |
  | ON  | 25,306 | 2,811 | 1,983 | 87.2 | 672.1s |
  | Δ   | **−0.19% (neutral)** | −0.2% | **−1.3%** | ~0 | +0.2% |
  Full config confirms & tightens the fast-config result: total tput is NEUTRAL
  (−0.19%, in noise), TTFT slightly better (−1.3%), TPOT parity (decode-bound).
So the ATOM local-M logic IS correctly portable to SGLang (single gather, gsm8k
passes), but at c256 it does NOT improve total throughput: the saved gate GEMM
(~1.5% of prefill) is offset by the wider fused gather (7936 vs 7168) AND c256 is
decode-bound so prefill savings barely move total. TTFT improves slightly.
⇒ gate-local is a correct, ATOM-aligned refactor but NOT a c256 throughput win.
Could help prefill-heavy / TTFT-sensitive workloads. REVERTED (git checkout; PR
pristine). c256 throughput-lever search closed: nothing beyond shipped
gatherv+A-fix moves total tput.
Scripts kept: `/sgl-workspace/c256_analysis/gate_microbench.py`,
`launch_sgl_wofp8.sh` (GATE_LOCAL/WO_FP8 knobs), `gsm8k_gatelocal/`.

---

## Exp 39 — full re-benchmark on updated codebases (2026-06-17)

Fresh container (the prior `/sgl-workspace/sglang-upstream` clone is gone; ATOM
re-installed — `atom/__init__.py` timestamp 2026-06-17 04:05). Goal: re-measure
the tp8+dp8 numbers on the **updated** code bases over a wider grid, and re-check
SGLang↔ATOM with both engines driven by the SAME (ATOM-native) client so client
variance = 0.

### Common config (apple-to-apple)
- 8×MI355X, tp8 + dp-attention (tp8dp8), **multi-stream** (ATOM default; no
  `ATOM_DISABLE_SIDE_STREAMS`), FP8 KV, page/block 256, mem 0.90, max-running 512,
  cuda-graph-max-bs 512, **16384 prefill tokens / rank**, prefill-delayer ON.
- Client: ATOM `atom.benchmarks.benchmark_serving --backend vllm` for BOTH engines.
- Bench params: **ratio=1.0** (fixed lengths; note: differs from the 0.8 used in
  the 06-09 baseline header), request-rate inf, ignore-eos, num_prompts=conc*8,
  warmups=conc*2.
- Grid: ISL∈{1024,8192}, OSL=1024, conc∈{64,128,256,512}.
- Launch scripts: `useful-scripts/benchmarking/dsv4/run_atom_dsv4_aligned.sh`
  (DP_MODE=tp8dp8) and `run_sgl_dsv4_aligned.sh`.

### SGLang-specific config
- `SGLANG_DP_USE_GATHERV=1` (shipped gatherv+reduce_scatterv; PR #28216 is now in
  main — `SGLANG_DP_USE_GATHERV`, `reduce_scatterv`, and the A-fix
  `get_dp_global_num_tokens()` all present in the editable main clone
  `/sgl-workspace/sglang` @ 66ac385f52).
- **`SGLANG_USE_ROCM700A=0`** (per request).
- `SGL_EXTRA_ARGS="--chunked-prefill-size 131072"`: current main auto-divides
  chunked_prefill_size by dp_size when DP attention is on (server_args.py:3537
  `chunked_prefill_size //= dp_size`), so 131072 → **16384/rank** = matches ATOM.
  (The aligned script's default 16384 would give only 2048/rank — NOT aligned.)
- Needed a one-line repo fix to launch: `srt/configs/cohere2_moe.py` `@strict`
  crashes on import under huggingface_hub≥1.x (SKILL §2a). Made `strict` a no-op
  identity (no behavior change; runtime field validation only).

### Accuracy (ATOM, gsm8k 5-shot, lm_eval local-completions, num_concurrent=64)
| Filter | exact_match | stderr |
|---|---:|---:|
| flexible-extract | **0.9500** | ±0.006 |
| strict-match | **0.9492** | ±0.006 |
✓ Correct (~0.95) even WITHOUT `ATOM_USE_TRITON_MOE=1` on this updated build —
the SKILL §1 "silent wrong-MoE → ~0.6" caveat did NOT trigger here.

### ATOM throughput (updated build, multi-stream)
| ISL | OSL | conc | total tok/s | tok/s/gpu | Med TTFT (ms) | Med TPOT (ms) | Med E2E (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1024 | 64  | 4,211  | 526   | 1,307  | 29.2 | 31,132 |
| 1024 | 1024 | 128 | 7,330  | 916   | 1,725  | 33.1 | 35,783 |
| 1024 | 1024 | 256 | 12,418 | 1,552 | 3,564  | 37.1 | 41,480 |
| 1024 | 1024 | 512 | 19,828 | 2,478 | 5,489  | 45.4 | 51,641 |
| 8192 | 1024 | 64  | 14,530 | 1,816 | 5,929  | 33.3 | 40,111 |
| 8192 | 1024 | 128 | 21,609 | 2,701 | 9,545  | 43.0 | 53,988 |
| 8192 | 1024 | 256 | 30,880 | 3,860 | 18,854 | 55.4 | 75,319 |
| 8192 | 1024 | 512 | 39,291 | 4,911 | 38,630 | 80.0 | 118,296 |

vs the 06-09 baseline (SKILL.md, ATOM client, 8192:1024): c128 21,526→21,609
(+0.4%), c256 30,942→30,880 (−0.2%) — flat within noise. (Note 06-09 used
ratio0.8; here ratio1.0, but ATOM fixed-len makes this immaterial.)

### SGLang (gatherv ON, ROCM700A=0) vs ATOM — SAME client, both multi-stream
| ISL | conc | SGL tok/s | ATOM tok/s | SGL/ATOM | SGL TTFT | ATOM TTFT | SGL TPOT | ATOM TPOT |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 64  | 4,642  | 4,211  | **110.2%** | 1,458  | 1,307  | 26.1 | 29.2 |
| 1024 | 128 | 7,870  | 7,330  | **107.4%** | 2,185  | 1,725  | 30.3 | 33.1 |
| 1024 | 256 | 12,974 | 12,418 | **104.5%** | 4,016  | 3,564  | 35.3 | 37.1 |
| 1024 | 512 | 17,185 | 19,828 | 86.7%      | 7,120  | 5,489  | 44.0 | 45.4 |
| 8192 | 64  | 15,295 | 14,530 | **105.3%** | 6,525  | 5,929  | 31.2 | 33.3 |
| 8192 | 128 | 22,102 | 21,609 | **102.3%** | 12,055 | 9,545  | 40.2 | 43.0 |
| 8192 | 256 | 29,621 | 30,880 | 95.9%      | 23,124 | 18,854 | 55.2 | 55.4 |
| 8192 | 512 | 32,254 | 39,291 | 82.1%      | 55,368 | 38,630 | 72.7 | 80.0 |

### Findings
- **Low/mid concurrency (c64–c256): SGLang wins or ties.** 1024:1024 c64–c256 =
  104–110%; 8192:1024 c64/c128 = 102–105%. SGLang TPOT beats ATOM at every point.
- **High concurrency c512: SGLang regresses** (1024:1024 86.7%, 8192:1024 82.1%)
  with much higher TTFT (8192 c512: 55.4s vs 38.6s). Consistent with the standing
  conclusion: high-conc prefill↔decode interference is SGLang's weak point, and
  c512 (newly added here) is more extreme than the usual c256.
- 8192 c256 = 95.9%, slightly better than the historical ~93%.

### Artifacts
- ATOM: `/workspace/bench_results_dsv4_atom_0617/` (8 JSON + summary).
- SGLang: `/workspace/bench_results_dsv4_sgl_0617/` (8 JSON + summary).
- Logs: `/workspace/{atom,sgl}_server.log`, `/workspace/{atom,sgl}_sweep_0617.log`,
  `/workspace/gsm8k_eval.log`.
- Both servers cleaned up (kill process tree incl. DP `multiprocessing-fork`
  children; VRAM back to ~0.3 GB/GPU). NOTE: this container has no `lsof`; find the
  DP EngineCore children via `rocm-smi --showpids` / `ps` and kill the parent tree.

---

## Exp 40 — re-added ATOM_DISABLE_SIDE_STREAMS flag + ATOM single vs multi-stream (2026-06-17)

The updated ATOM build had **dropped** the `ATOM_DISABLE_SIDE_STREAMS` flag (a
knob we had added in a prior session for single-stream A/B). It was not in the new
centralized env registry (`atom/utils/envs.py`) and not read anywhere in the
package — so the earlier single-stream attempt would have been IDENTICAL to
multi-stream (caught before wasting a run). Re-added it as a single master switch.

### Side-stream architecture in the updated ATOM (so the re-add is correct)
Two independent side-stream mechanisms in `atom/models/deepseek_v4.py`, BOTH gated
on `alt_stream is not None`:
1. **Dual-stream MoE** (shared_experts // routed_experts on `alt_stream`):
   `self._use_dual_stream = shared_experts is not None and alt_stream is not None
   and envs.ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD > 0` (deepseek_v4.py:2156). Per-call
   token-count gated → prefill (large batch) skips it; mainly a DECODE optimization.
2. **Async Compressor/indexer overlap** (Main Compressor → `alt_stream`, Indexer
   Compressor → `indexer_stream`): `use_async_compress = self._use_async_compress
   and fc.in_hipgraph` (deepseek_v4.py:1635); `_use_async_compress = alt_stream is
   not None and compressor is not None` (line 1567).
The two `torch.cuda.Stream()` objects are allocated once at model __init__
(deepseek_v4.py:~2668), shared across all blocks. There is NO env toggle in the new
code (compressor overlap is gated only by in_hipgraph).

### The re-added flag (single master switch)
- `atom/utils/envs.py`: registered `ATOM_DISABLE_SIDE_STREAMS` (default "0").
- `atom/models/deepseek_v4.py` (~line 2668): `_enable_side_streams =
  torch.cuda.is_available() and not envs.ATOM_DISABLE_SIDE_STREAMS`; allocate
  `alt_stream`/`indexer_stream` only when true, else None. Leaving them None makes
  every downstream `is not None` guard run inline → disables BOTH mechanisms in one
  switch. Added an info log of the resolved state.
- Usage: `ATOM_DISABLE_SIDE_STREAMS=0` (default) = multi-stream; `=1` = single-stream.
- **Runtime-verified**: all 8 DP ranks log `DSV4 side-streams DISABLED
  (single-stream) (ATOM_DISABLE_SIDE_STREAMS=1): alt_stream=False
  indexer_stream=False`.
- CAVEAT: edited in the installed site-package (`/opt/venv/.../atom/`), NOT a git
  repo — lost on container rebuild. To persist, commit into the ATOM source repo.

### ATOM single-stream (SS) vs multi-stream (MS) — tp8dp8, ATOM client, ratio1.0
Same config/grid as Exp 39 (multi-stream = the Exp 39 ATOM numbers).
| workload | conc | MS tok/s | SS tok/s | SS/MS | MS TPOT | SS TPOT | MS TTFT | SS TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k/1k | 64  | 4,211  | 3,945  | 93.7% | 29.18 | 31.23 | 1,307  | 1,261  |
| 1k/1k | 128 | 7,330  | 6,947  | 94.8% | 33.14 | 34.51 | 1,725  | 2,445  |
| 1k/1k | 256 | 12,418 | 12,078 | 97.3% | 37.12 | 38.90 | 3,564  | 4,197  |
| 1k/1k | 512 | 19,828 | 19,348 | 97.6% | 45.41 | 46.52 | 5,489  | 5,932  |
| 8k/1k | 64  | 14,530 | 13,518 | 93.0% | 33.35 | 36.41 | 5,929  | 4,843  |
| 8k/1k | 128 | 21,609 | 20,982 | 97.1% | 43.02 | 44.69 | 9,545  | 10,044 |
| 8k/1k | 256 | 30,880 | 30,146 | 97.6% | 55.38 | 57.48 | 18,854 | 19,965 |
| 8k/1k | 512 | 39,291 | 38,867 | 98.9% | 79.96 | 80.79 | 38,630 | 38,666 |

### Findings
- **Multi-stream wins everywhere, small margin, shrinking with concurrency**:
  biggest at c64 (SS = 93–94% of MS), nearly even at c512 (97.6–98.9%).
- MS TPOT consistently lower → side-stream overlap mainly helps DECODE (consistent
  with the dual-stream MoE per-call gating that skips large prefill batches).
- ⇒ side-streams are a real but modest optimization (~+1–7% total tok/s), largest
  at low concurrency (decode-heavy).

### Artifacts
- SS: `/workspace/bench_results_dsv4_atom_ss_0617/`; MS: `.../bench_results_dsv4_atom_0617/`.
- Logs: `/workspace/atom_ss_{server,sweep_0617}.log`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 41 — SGLang c512 stability (3 repeats) (2026-06-17)

Q: is SGLang's large c512 deficit vs ATOM (Exp 39: 1k/1k 86.7%, 8k/1k 82.1%) a
stable measurement or run-to-run noise? Re-ran SGLang c512 for both workloads
**3× each**, same config as Exp 39 (gatherv ON, ROCM700A=0, tp8dp8, ATOM client,
ratio1.0, np4096/warm1024, 16384 prefill tok/rank).

| workload | run1 | run2 | run3 | mean | std | range |
|---|---:|---:|---:|---:|---:|---:|
| 1k/1k c512 | 17,195 | 17,157 | 17,348 | **17,233** | 83 (0.48%) | 191 (1.11%) |
| 8k/1k c512 | 32,155 | 32,186 | 32,200 | **32,180** | 19 (0.06%) | 45 (0.14%) |

TPOT/TTFT also tight: 8k/1k TPOT 72.59–72.65, TTFT 55.6–55.7s; 1k/1k TPOT
43.80–43.88, TTFT 7.36–7.70s.

### Conclusion — the deficit is REAL and reproducible, not noise
- 3-run variance is tiny (std 0.06–0.48%). SGLang stably lags ATOM at c512.
- vs Exp 39 multi-stream ATOM: 1k/1k 17,233/19,828 = **86.9%** (matches the single
  run's 86.7%); 8k/1k 32,180/39,291 = **81.9%** (matches 82.1%).
- Consistent with the standing conclusion: high-conc (c512) is SGLang's weak point —
  prefill↔decode interference is worst there. The 8k/1k c512 TTFT ≈ 55.6s (vs ATOM
  ~38.6s) points at heavy prefill queueing as the driver.
- NOTE: this c512 is the worst point; c64–c256 SGLang ties/wins (Exp 39).

### Artifacts
- `/workspace/bench_results_dsv4_sgl_c512_run{1,2,3}/` (2 JSON each).
- Logs: `/workspace/sgl_c512_repeat.log`, `/workspace/sgl_server_c512.log`. Server
  cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 42 — c512 gap root-cause + levers tried (swa-ratio, mixed-chunk) (2026-06-17)

Investigated WHY SGLang lags ATOM so much at c512 (Exp 39/41: 1k/1k 86.9%, 8k/1k
81.9%). All on 1k/1k c512, gatherv ON, ROCM700A=0, tp8dp8, ATOM client, ratio1.0,
np4096/warm1024.

### (a) swa-full-tokens-ratio sweep — NO throughput effect
Overrode the aligned-script default 0.15 via SGL_EXTRA_ARGS.
| metric | 0.15 (mean3) | 0.2 | 0.25 |
|---|---:|---:|---:|
| total tok/s | 17,233 | 17,173 (−0.35%) | 17,128 (−0.61%) |
| Med TTFT ms | 7,500 | 6,966 | 7,224 |
| Med TPOT ms | 43.84 | 43.68 | 43.98 |
Throughput flat (within the 1.1% run-to-run band). TTFT best at 0.2 (~−7%) but not
monotonic; 0.25 regresses. ⇒ swa ratio tunes KV-pool split, NOT prefill/decode
scheduling → no tput lever. Kept default 0.15.

### (b) Gap decomposition (client metrics) — the gap is ALL prefill/TTFT
| | SGL | ATOM |
|---|---:|---:|
| total tok/s | 17,195 | 19,828 (+15.3%) |
| wall duration s | 487.9 | 423.1 |
| Med TPOT ms (decode) | **43.85** | 45.41 (SGL FASTER) |
| Med TTFT ms | 7,355 | 5,489 |
| mean TTFT ms | 13,404 | 5,924 |
| p99 TTFT ms | 53,459 | 9,538 (SGL 5.6×) |
| std TTFT ms | 16,046 | 2,532 (SGL 6.3×) |
- Decode is NOT the problem (SGL TPOT lower). Gap = prefill/TTFT, and the signature
  is VARIANCE: SGL TTFT std 6.3× and p99 5.6× ATOM; SGL mean≫median (right-skew
  tail), ATOM mean≈median (tight). Wall-duration ratio (1.153) == tput gap.

### (c) Scheduler-log root cause (BOTH engines, same case — two-sided)
Per-rank (~64 reqs/rank at c512):
| | SGLang | ATOM |
|---|---|---|
| decode batch occupancy | #running-req median **53/64**, range 2–64 (drains) | output median **64/64** (stable full) |
| prefill granularity | almost always FULL 16384 tok (16 reqs) | MIXED: full 16384 (most common) + many small 1024/2048/3072/4096 (1–4 reqs) |
| prefill-delayer | n/a | delay_rate **2.69%** (well-tuned, mostly allows) |
ROOT CAUSE: **SGLang cannot keep the decode batch full at c512** (median 53/64 ≈
83% occ, dips to single digits) while ATOM holds 64/64. The ~17% decode
under-occupancy ≈ the 15% tput gap. Mechanism: SGLang injects prefill as RIGID full
16384-token chunks and (default) non-mixed steps → each prefill chunk stalls ALL
decode for a step → decode occupancy collapses + TTFT bursts. ATOM uses ADAPTIVE
prefill granularity (small 1–4-req batches when needed) + a well-tuned prefill-delayer
to slip prefill in smoothly, keeping decode full and TTFT low/uniform. ATOM trades
slightly slower decode (TPOT 45.4 vs 43.9) for stable-full occupancy → +15% total.
Why only c512 breaks: more in-flight decode at high conc ⇒ each rigid prefill chunk
disrupts more; c64–c256 has little in-flight decode so SGLang ties/wins (Exp 39).
Logs: `/workspace/{sgl,atom}_server_sched.log` (+ `*_sched_sweep.log`); parsed DP0/
all-rank "Prefill batch"/"Scheduled prefill batch"/decode lines.

### (d) Lever tried: --enable-mixed-chunk — REJECTED (made it WORSE)
gsm8k OK with mixed-chunk (flexible 0.9386 / strict 0.9393, ~0.94 — not a
correctness issue). But:
| metric | SGL base (mean3) | SGL +mixed-chunk | MC vs base |
|---|---:|---:|---:|
| total tok/s | 17,233 | 16,321 | **−5.30%** |
| Med TTFT ms | 7,500 | 8,336 | +11.2% |
| mean TTFT ms | 13,260 | 15,921 | +20.1% |
| std TTFT ms | 15,870 | 18,055 | +13.8% |
| Med TPOT ms | 43.84 | 44.61 | +1.8% |
MC drops 86.9%→**82.3%** of ATOM. Mixing prefill tokens into the decode step
enlarges per-step batch (incl. large prefill chunk) → higher TPOT + WORSE TTFT
variance; with gatherv/MoE padding it nets negative. ⇒ baseline (non-mixed) is the
better SGLang c512 config. ATOM's edge is adaptive-granularity + full-occupancy
scheduling, NOT prefill+decode co-stepping.

### Not yet tried (candidate levers, more on-target than mixed-chunk)
- Raise `schedule_conservativeness` (DP attn auto ×0.3 → 0.3): admit prefill more
  conservatively to protect decode occupancy (directly targets the 53/64 drop).
- Smaller `chunked-prefill-size` (e.g. 8192/rank): shrink per-step prefill shock to
  mimic ATOM's small-batch injection (watch prefill efficiency).

### Artifacts
- swa: `/workspace/bench_results_dsv4_sgl_swa0{2,25}/`; mixed-chunk:
  `/workspace/bench_results_dsv4_sgl_mc/`; scheduler runs:
  `/workspace/bench_results_dsv4_{sgl,atom}_sched/`. gsm8k(MC): `/workspace/gsm8k_mc.log`.
  All servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 43 — client validation: SGLang client vs ATOM client, same server (2026-06-17)

Q: does the c512 number change if driven by SGLang's OWN bench client instead of
ATOM's? Same SGLang server (gatherv ON, ROCM700A=0, tp8dp8, chunk16384/rank,
baseline), same `/v1/completions` endpoint, same params (np4096/warm1024/ratio1.0,
1k/1k c512). Only the client differs:
- ATOM client: `atom.benchmarks.benchmark_serving --backend vllm`.
- SGLang client: `python3 -m sglang.bench_serving --backend sglang-oai` (no shim
  needed vs an SGLang server).

| metric | ATOM client | SGLang client | diff |
|---|---:|---:|---:|
| total tok/s | 17,195 | **15,751** | **−8.4%** |
| wall duration s | 487.9 | 532.6 | +9.2% |
| Med TTFT ms | 7,355 | 8,261 | +12.3% |
| Mean TTFT ms | 13,404 | 15,612 | +16.5% |
| p99 TTFT ms | 53,459 | 58,155 | +8.8% |
| std TTFT ms | 16,046 | 18,370 | +14.5% |
| Med TPOT ms | 43.85 | 46.36 | +5.7% |

### Conclusion — the data is NOT the same; client effect is ~8% at c512
- SGLang's own client reports SYSTEMATICALLY LOWER throughput (and higher
  TTFT/TPOT/duration) than the ATOM client on the IDENTICAL server. The client
  itself accounts for ~8% at c512 — LARGER than the ~3% SKILL §4 saw at c128/c256,
  i.e. **client effect grows with concurrency** (client-side dispatch/burstiness
  becomes part of the bottleneck in high-conc closed loop).
- Implications:
  1. Validates the methodology choice: all SGLang-vs-ATOM engine numbers in this log
     use the ATOM client for BOTH engines, so the client effect cancels → the
     reported gaps (e.g. c512 ~87%) are pure engine differences.
  2. If someone instead used "SGLang client for SGLang, ATOM client for ATOM",
     SGLang would be under-reported ~8% → c512 would look ~79% instead of ~87%.
     ⇒ cross-engine comparison MUST use one client.
- Artifacts: `/workspace/bench_results_dsv4_sglclient/`, log
  `/workspace/sglclient_sweep.log`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 44 — c512 levers: conservativeness (neutral) + chunk-size (helps) + CORRECTED mechanism (2026-06-18)

Tried the two Exp-42 candidate levers on 1k/1k c512 (gatherv ON, ROCM700A=0,
tp8dp8, ATOM client, ratio1.0, np4096/warm1024). Baseline = chunk 16384/rank,
conservativeness eff 0.3.

### Lever 1: schedule_conservativeness eff 0.3 → ~1.0 (`--schedule-conservativeness 3.3`, ×0.3 DP) — NEUTRAL
total 17,233 → 17,009 (−1.3%, in noise), TTFT not improved. Conservativeness
controls admit-caution to avoid RETRACTS; this case never retracts (KV fits), so it
does nothing to the real driver. Rejected.

### Lever 2: chunked-prefill 16384/rank → 8192/rank (`--chunked-prefill-size 65536`) — HELPS +5% (single run)
| metric | base 16384 | 8192 | ATOM |
|---|---:|---:|---:|
| total tok/s | 17,195 | **18,064 (+5.1%)** | 19,828 |
| duration s | 487.9 | 464.4 | 423.1 |
| Med TPOT | 43.85 | **45.62 (WORSE)** | 45.41 |
| Mean TPOT | 43.39 | 45.35 (worse) | 45.84 |
| Med ITL | 40.14 | 41.43 (worse) | 40.67 |
| Mean TTFT | 13,404 | **9,025 (−33%)** | 5,924 |
| std TTFT | 16,046 | 11,543 (−28%) | 2,532 |
| Med TTFT | 7,355 | 6,482 | 5,489 |
| p99 TTFT | 53,459 | 53,985 (~same) | 9,538 |
| Mean E2E | 57,797 | 55,423 (−4.1%) | 52,821 |
8192 closes ~1/3 of the gap: 86.9% → 91.1% of ATOM.

### CORRECTED mechanism (Exp 42's "smaller chunk → smoother decode occupancy" was WRONG)
The smaller-chunk win is a TRADE, and decode actually gets slightly WORSE:
- **Decode is hurt, not helped**: TPOT 43.85→45.62 (+4%), ITL up. More prefill steps
  DO interrupt decode more often (the intuitive objection is correct).
- **The win is on the PREFILL/queue side**: mean TTFT −33%, std −28%. Throughput is
  closed-loop ∝ 1/mean_E2E. Decompose mean E2E change: TTFT −4.4s, decode +1.8s
  (=ΔTPOT 1.77ms × 1024), net −2.6s ≈ measured mean-E2E −2.4s. So +5% tput = (TTFT
  queue win) − (TPOT decode cost); at 1k/c512 the queue win dominates.
- **Why smaller chunk shortens TTFT**: TTFT≈queue-wait (1k prefill compute is tiny).
  What matters is how OFTEN a prefill step fires, not how many reqs it carries.
  Big chunk → scheduler/prefill-delayer batches into infrequent big prefill waves →
  a new req can wait a whole decode interval (high mean/var TTFT). Small chunk →
  cheaper, more frequent prefill steps → reqs admitted in smaller steadier waves →
  lower mean/var TTFT. Cost = more prefill steps eat decode time → TPOT up.
- NOT clean: p99 TTFT unchanged (worst tail same); TPOT is a real tradeoff
  (interactivity-sensitive workloads lose). Single run — repeat ×3 to confirm.

### Reconciles with the old "2048/rank worse than 16384" finding — non-monotonic, two opposing effects
- Effect A (prefill efficiency): smaller chunk → more steps → per-step overhead +
  low-M GEMM MFU → HURTS. Dominates at 8k input (chunk also splits a single 8192-tok
  prefill → efficiency loss amplified) → "bigger is better down to the 16384/rank
  floor" (Exp 16/18).
- Effect B (queue fairness): smaller chunk → more frequent prefill → lower TTFT
  mean/variance → HELPS. Dominates at 1k input (1024 < chunk so a request is never
  intra-chunked; chunk only sets reqs-per-prefill-step granularity).
- ⇒ optimal chunk is workload×concurrency dependent: 8k/c256 wants big (A); 1k/c512
  wants smaller (B). Both prior+current findings are correct in their regime.

### CORRECTION to Exp 42 root-cause framing
Exp 42 attributed the c512 vs-ATOM gap to "SGLang decode occupancy 53/64". But the
client metrics show **SGLang TPOT is BETTER than ATOM at c512 (43.85 < 45.41)** — if
decode were truly under-occupied/inefficient, TPOT would be WORSE. So the vs-ATOM
c512 gap is PURELY TTFT (prefill admission queueing/fairness; SGLang TTFT mean 13.4s
& std 16k vs ATOM 5.9s & std 2.5k), NOT decode occupancy. The "decode 53/64" log
reading was likely a sampling artifact / not the throughput driver.

### Untried / next
- Repeat 8192 ×3 (stability); re-capture scheduler logs at 8192 vs 16384 to confirm
  the "more frequent prefill steps + shorter queue" causal chain (prefill-step
  frequency + queue depth). Try 4096/rank (push effect B; watch effect A). Validate
  best chunk on 8k/c512 (gap is larger there, 82%) and on c128/c256 (don't regress).

### Artifacts
- conservativeness: `/workspace/bench_results_dsv4_sgl_cons/`; chunk8192:
  `/workspace/bench_results_dsv4_sgl_cps/`; logs `/workspace/sgl_{cons,cps}_sweep.log`,
  `/workspace/sgl_server_{cons,cps}.log`. Servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 45 — chunk-size sweep: 8192 stability + 4096 + 8k validation (2026-06-18)

Verified the Exp 44 chunk lever properly: 8192 stability ×3, 4096/rank (push effect
B), and the 8k workload (where chunk < 8192 SPLITS a request → effect A). 1k & 8k,
c512, gatherv ON, ROCM700A=0, ATOM client, ratio1.0, np4096/warm1024.

### 1k/1k c512 (1024<chunk for all → no intra-req split; chunk just sets reqs/step)
| chunk/rank | reqs/step | total tok/s | std | vs ATOM | Med TPOT | Mean TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 16384 (base ×3) | 16 | 17,233 | 83 | 86.9% | 43.84 | 13,260 |
| **8192 (×3)** | 8 | **18,106** | **11** | **91.3%** | 45.46 | 8,874 |
| 4096 (×1) | 4 | 18,099 | — | 91.3% | 46.88 | 7,123 |
| ATOM | — | 19,828 | — | 100% | 45.41 | 5,924 |
- 8192 STABLE (×3 = 18,095/18,121/18,104, std 11 = 0.06%) → the +5% is REAL.
- **8192→4096 plateaus** (18,106≈18,099): TTFT keeps dropping (8,874→7,123) but TPOT
  keeps rising (45.46→46.88) — they cancel. Sweet spot = 8192/rank.

### 8k/1k c512 (chunk<8192 SPLITS the 8192-tok request → triggers effect A)
| chunk/rank | behavior | total tok/s | vs ATOM | Med TPOT | Mean TTFT |
|---|---|---:|---:|---:|---:|
| 16384 (base) | 2 reqs/step | 32,254 | 82.1% | 72.70 | 65,771 |
| **8192** | 1 req/step, no split | **33,475** | **85.2%** | 82.61 | 53,521 |
| 4096 | SPLITS req into 2 | 33,268 | 84.7% | 84.89 | 48,299 |
| ATOM | — | 39,291 | 100% | 79.96 | 38,745 |
- 16384→8192 helps (+3.8%, 1 req/step, no split). **8192→4096 REGRESSES** (85.2→84.7%)
  — crossing into intra-request splitting makes effect A (prefill efficiency) bite,
  exactly as the two-effects model predicts.

### Confirmations
1. The chunk win is REAL & reproducible (8192 ×3 std 0.06%).
2. TPOT rises MONOTONICALLY as chunk shrinks (1k 43.8→45.5→46.9; 8k 72.7→82.6→84.9)
   → smaller chunk DOES interrupt decode more (the intuitive objection is correct);
   throughput is the net of (TTFT gain − TPOT cost), peaking at 8192.
3. The "don't split a request" boundary is real: at 8k, 8192 (=1 req, no split) is
   best; 4096 (splits) regresses. At 1k, 4096 is fine (still >1024, no split) but no
   extra gain. ⇒ **8192/rank is the universal c512 sweet spot** (1k 86.9→91.3%, 8k
   82.1→85.2%).
4. chunk tuning recovers ~1/3 of the gap; ATOM still leads (TTFT 5.9s/38.7s vs
   8.9s/53.5s) → ATOM's scheduling (prefill fairness) is still better; the residual
   is NOT a chunk-size issue.

### Artifacts
- 8192: `/workspace/bench_results_dsv4_A_8192_1k_run{1,2,3}/`, `.../_A_8192_8k/`.
- 4096: `/workspace/bench_results_dsv4_B_4096_1k/`, `.../_B_4096_8k/`.
- Logs: `/workspace/server{A,B}_bench.log`, `/workspace/sgl_server{A,B}.log`.
  Servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 46 — old vs new ATOM at c512: is ATOM's speed a recent change? (2026-06-18)

Q: is ATOM fast vs SGLang because of a recent ATOM scheduler update or faster
prefill kernels? Compared two ATOM commits at c512 (tp8dp8, multi-stream, ATOM
client, ratio1.0, np4096/warm1024).
- **OLD = `914d50323` (2026-06-08)** = `/sgl-workspace/ATOM-previous`.
- **NEW = `bcd38f67` (2026-06-17)** = `/sgl-workspace/ATOM` = the version ALL prior
  ATOM data in this log (Exp 39–45) was measured on.
- (Version note: the installed-pkg metadata `0.1.4.dev80+g<hash>` identifies which
  commit is live; `pip install ./<dir>/` swaps it. OLD has no `ATOM_DISABLE_SIDE_STREAMS`
  flag — that edit was on NEW's site-package, overwritten by installing OLD.)

| | OLD 914d50323 | NEW bcd38f67 | NEW vs OLD |
|---|---:|---:|---:|
| 1k/1k c512 total tok/s | 19,995 | 19,828 | −0.8% |
| 1k Med TTFT / TPOT | 5,766 / 45.33 | 5,489 / 45.41 | −4.8% / +0.2% |
| 8k/1k c512 total tok/s | 39,721 | 39,291 | −1.1% |
| 8k Med TTFT / TPOT | 38,573 / 78.40 | 38,630 / 79.96 | +0.1% / +2.0% |

### Conclusion — the two ATOM versions are IDENTICAL in perf (all metrics ±2%, noise)
- ATOM did NOT change (perf-wise) between 06-08 and 06-17: neither scheduler nor
  prefill-kernel speedup in that window. ATOM was ALREADY this fast at 914d50323.
- ⇒ ATOM's advantage over SGLang is NOT a recent patch; it's inherent to its design
  (adaptive prefill injection + prefill-delayer fairness, per Exp 42/44/45).
  Diffing 914d50323↔bcd38f67 will NOT locate "why ATOM is fast" (no perf delta).
- To find WHEN ATOM became fast, would need a much older ATOM (pre-improvement);
  both of these are already post-improvement.

### Install state after this exp
Installed = OLD 914d50323 (no side-stream flag). Restore to NEW with
`pip install /sgl-workspace/ATOM/` and re-apply the flag if persistent single-stream
A/B is needed (perf is equivalent either way).

### Artifacts
- OLD: `/workspace/bench_results_dsv4_atomOLD/` (2 JSON), log
  `/workspace/atomOLD_sweep.log`, `/workspace/atom_old_server.log`. NEW = the
  existing `/workspace/bench_results_dsv4_atom_0617/`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 47 — pure-prefill (OSL=1) compute: is SGLang's per-step prefill slower? (2026-06-18)

Q: besides the scheduler, is SGLang's per-STEP prefill execution itself slower? To
isolate prefill COMPUTE from the prefill↔decode scheduler interference, ran a
PURE-PREFILL workload (OSL=1, almost no decode) on both engines, same client, same
chunk 16384/rank, conc512, np4096/warm1024, rate inf. Prefill (input) throughput at
saturation = per-step prefill compute rate (per-step time = 131072 tok ÷ system tps).

| ISL | SGL input tok/s | ATOM input tok/s | ATOM/SGL | per-step SGL | per-step ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 57,821 | **+19.7%** | 2,714 ms | 2,267 ms |
| 8192 | 47,013 | 55,999 | **+19.1%** | 2,788 ms | 2,341 ms |

### Conclusion — NOT purely scheduler: SGLang prefill compute is ~20% SLOWER
- With decode interference removed (OSL=1), ATOM still prefills ~19–20% faster →
  **SGLang's per-step prefill execution is genuinely ~20% slower** (real
  compute/kernel gap, NOT scheduling).
- So the c512 gap has TWO components: (1) **prefill compute ~20% slower** (this exp)
  + (2) scheduler/queueing (Exp 42/44/45, partly recoverable via chunk tuning).
  Reconciles the mixed-c512 picture: SGLang decode (TPOT) is fine/better, but TTFT
  is high because prefill is BOTH slower to compute AND queued.
- Direction matches Exp 36 (prefill compute gap lives in the engine-specific MLA
  path; MoE GEMM + comm are shared/equal). Magnitude here (~20% at c512) > the ~8%
  measured per-token at c256 — bigger under c512 saturation / system-throughput view.
- Caveat: input_tps under saturation also reflects prefill BATCHING efficiency, not
  only raw kernel; but both are engine-side (not the prefill↔decode interference).
  Next to pin raw kernel: isolated MLA-prefill kernel microbench at matched shapes.

### Artifacts
- `/workspace/bench_pp_sgl/`, `/workspace/bench_pp_atom/` (isl{1024,8192}_osl1_c512),
  logs `/workspace/pp_{sgl,atom}_sweep.log`, `/workspace/{sgl,atom}_pp_server.log`.
  Servers cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 48 — prefill TRACE: split the ~20% into raw-kernel vs overhead (2026-06-18)

Followed up Exp 47 ("SGLang prefill ~20% slower") with torch-profiler traces to split
it into raw-kernel-compute vs host/launch overhead. BOTH single-stream (SGLang aligned
is already single-stream; ATOM run with ATOM_DISABLE_SIDE_STREAMS=1 so kernels
serialize → clean GPU-busy), pure-prefill load (ISL=8192 OSL=1, conc512), torch
profiler GPU activity, rank0 trace.

### Method note (heeds Exp 19): per-kernel `dur` is UNRELIABLE (esp. ATOM — durs
inflated / overlapping streams gave impossible >20,000 ms/step for some kernels). The
ONLY trusted metric = GPU-active UNION (wall time with ≥1 kernel running). Normalized
per `pa_prefill` count (= attn-layers × steps; SAME aiter kernel + same model both
engines → L-independent unit). SGLang pa=549 (~10 steps), ATOM pa=183 (~3.3 steps).

### Result — GPU-active per attn-layer-step (overlap-robust, the reliable number)
| | GPU-active/step | wall/step (Exp47) | per-step bubble |
|---|---:|---:|---:|
| SGLang | 2.46 s | 2.79 s | **12%** |
| ATOM | 2.28 s | 2.34 s | **3%** |
- GPU-active per unit work: SGLang **+8%** (44.81 vs 41.54 ms/attn-layer-step).
- Per-step wall ratio 2.79/2.34 = 1.19 (= the Exp 47 prefill tput gap) DECOMPOSES as:
  **raw GPU kernel ×1.08 (8%) × host-overhead/bubble ×1.10 (10%) ≈ 1.19 (19%).**

### Conclusion — answer to "are the raw kernels fine?"
NOT identical, but the raw-kernel gap is modest: **~8% is real GPU compute (SGLang
kernels take 8% more GPU-active time), and ~10% is host/launch overhead+bubbles**
(SGLang has 12% per-step idle between kernels vs ATOM's 3%). So the ~20% prefill gap
is roughly HALF kernel, HALF glue/overhead — it's NOT purely scheduler, NOT purely
kernel.
- Per-kernel attribution of the 8% is BLOCKED by ATOM's unreliable per-kernel dur
  (Exp 19). The shared aiter `pa_prefill` (MLA core attn) is the same kernel on both
  (~179 ms/step on SGLang) and should be equal; the 8% likely sits in the MLA
  projection GEMMs / glue (per Exp 36 direction), but needs an ISOLATED kernel
  microbench at matched shapes to pin — trace dur can't do it.
- SGLang's 12% per-step bubble (launch gaps) is a concrete, addressable target
  (kernel launch batching / fewer host syncs / CUDA graph for prefill).

### Artifacts
- SGLang trace `/workspace/sgl_prof/1781761067*TP-0-DP-0.trace.json.gz` (clean 10-step,
  89% busy); ATOM `/workspace/atom_prof/dp0_tp0/*.pt.trace.json.gz`. Analyzer:
  `useful-scripts/benchmarking/dsv4/analyze_trace.py` (+ global-union script inline).
  Both servers cleaned via `rocm-smi --showpids` → kill (VRAM ~0.3 GB/GPU).

---

## Exp 50 — shared-expert-local PoC: +6–7% prefill (2026-06-18)

Per-layer trace (Exp 48 follow-up) showed the MoE "shared-expert + gate" block ~2×
bigger on SGLang. Investigated whether it's a kernel diff or a logic diff.

### Shapes & redundancy check (microbench, avoids unreliable trace dur)
- shared-expert (n,k) are config constants → SAME on both engines. The differing
  factor is M (tokens).
- SGLang (code-confirmed): `disable_shared_experts_fusion` → separate
  `self.shared_experts`; `_shared_expert_use_tp1=False` ⇒ shared expert is
  **TP-sharded**. deepseek_v4.py decoder gathers local→global THEN `self.mlp(global)`
  → shared expert runs on the GLOBAL buffer (M≈131072).
- ATOM: shared expert on LOCAL tokens (M≈16384), before the gather.
- **CORRECTION to the initial "8× redundant" guess:** a TP-sharded shared expert is
  NOT redundant — per-rank FLOPs are identical (TP8-global: M=131072×FFN/8 = 16384×FFN
  ≡ TP1-local: M=16384×FFN). Only the GATE is truly redundant (replicated; Exp 38).
- Same-FLOPs shape microbench (`gemm_a8w8_blockscale_bpreshuffle_ck`):
  | GEMM (same FLOPs) | TP8-global | TP1-local | TP1 faster |
  |---|---:|---:|---:|
  | gate_up (K7168) | M131072,N768: 1.38ms | M16384,N6144: 1.21ms | 13% |
  | down | M131072,N7168,K384: 1.10ms | M16384,N7168,K3072: 0.63ms | 1.74× |
  (the global down GEMM's K=384 is too small → low arithmetic intensity.)
  ALSO ck_xdl (SGLang) is FASTER than ck_tile (ATOM) at these shapes (~1.6×) → ATOM's
  shorter trace time is purely M/shape, NOT a faster kernel; do NOT swap to ck_tile.

### PoC implementation (env-gated SGLANG_DP_SHARED_EXPERT_LOCAL=1, needs SHARED_EXPERT_TP1=1)
Compute the (replicated, TP1) shared expert on LOCAL hidden in the decoder layer
BEFORE the dp gather; skip it inside `self.mlp` (forward_normal); add it back to this
rank's reduce-scattered LOCAL slice. Prefill-only (gated on is_extend). Files:
`models/deepseek_v2.py` (skip_shared_experts param in forward/forward_normal),
`models/deepseek_v4.py` (compute local + skip + add after reduce_scatterv).

### Result — pure-prefill (OSL=1), gsm8k correct (flex 0.9386 / strict 0.9393)
| ISL | SGL base | SGL +SE-local | ATOM | gain | base→new vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 51,171 | 57,821 | **+6.0%** | 84% → 88% |
| 8192 | 47,013 | 50,324 | 55,999 | **+7.0%** | 84% → 90% |
- **+6–7% prefill — BIGGER than the ~1.4% GEMM-shape microbench predicted.** The extra
  gain is because the local path also runs 8× FEWER ROWS through the activation
  fp8-quant + elementwise (global processed M=131072 rows; local M=16384), on top of
  the better GEMM shape. So it's NOT redundant FLOPs but it IS redundant
  per-row quant/overhead on the global buffer. The user's "test it empirically" call
  was right; the microbench under-counted.
- Caveats: (1) requires TP1 shared expert → ~8× shared-expert weight memory/rank
  (~+0.5 GB). (2) pure-prefill only here; **c512 end-to-end (decode-bound) gain
  unverified** — Exp 38 gate-local was neutral at c512, so confirm with a full A/B.
  (3) prefill-path only (decode keeps global shared).

### Next
- FULL c512 1k/1k & 8k/1k end-to-end A/B with SE-local (does the prefill win move
  total tput, or is it decode-bound/neutral like gate-local?).
- Stack with Exp 49 (CK GEMM + batched rope) — are the gains additive?

### Artifacts
- `/workspace/bench_pp_se/` (isl{1024,8192}_osl1), gsm8k `/workspace/gsm8k_se.log`,
  server `/workspace/sgl_se_server.log`. Microbench `/workspace/gemm_micro.py`.
  Edits in `/sgl-workspace/sglang` (deepseek_v2.py, deepseek_v4.py), env-gated default
  OFF. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 51 — ALL 3 prefill levers stacked: 84% → 97–98% of ATOM (2026-06-18)

Stacked the three prefill fixes (all env-gated, default OFF):
`SGLANG_FORCE_CK_W8A8=1 SGLANG_ROPE_BATCHED=1 SGLANG_DP_SHARED_EXPERT_LOCAL=1
SGLANG_SHARED_EXPERT_TP1=1` (+ gatherv ON, ROCM700A=0). gsm8k correct: flex 0.9469 /
strict 0.9477.

### Pure-prefill (OSL=1, ATOM client) — gains are ~ADDITIVE
| ISL | base | +CK+ROPE | +SE-local | ALL3 | ATOM | ALL3 vs base | ALL3 vs ATOM |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 52,506 | 51,171 | **55,944** | 57,821 | +15.8% | **97%** |
| 8192 | 47,013 | 51,205 | 50,324 | **54,645** | 55,999 | +16.2% | **98%** |
⇒ the three independent levers stack to ~+16% and **close the prefill gap to 97–98%**
of ATOM (was 84%).

### Per-layer trace (ISL8192, pa_prefill-windowed) — gap nearly gone
GPU-active/attn-layer-step: base 44.81 → **ALL3 38.53** → ATOM 41.54 ms (SGLang now
BELOW ATOM in raw GPU compute). Per-layer WINDOW (ratio-4): base ~7000us SGLang-slower
→ **ALL3 39.5ms vs ATOM 38.8ms (~600us)**. Op-by-op now aligned: o-proj GEMM both
ck_tile (1904 vs 1802 ≈), shared-expert both local-ish (744+silu+444 vs 752+act+442 ≈),
MLA/MoE/comm/kv-q-proj all shared+equal. Remaining small diffs:
- **RoPE**: SGL `rope_batched` 332us vs ATOM fused `inverse_rope_gptj` 155us (~180us).
- **compressor glue**: SGL `fused_norm_rope`+`fill`×3+`rocprim`×2+elementwise vs ATOM's
  3 tight fused kernels (`hca_compress_forward`+`hca_norm_rope_scatter`+`compressor_update`).
  This is the residual per-step bubble (Exp 48); now the dominant remaining item.

### Reusable per-layer diff METHOD (persisted)
`useful-scripts/benchmarking/dsv4/layer_diff.py` (PERSISTENT, not /workspace):
- `overview <trace>`: list pa_prefill windows (=layers) with window dur + GPU-active union.
- `seq <trace> <ratio:4|128>`: one layer's ordered kernel sequence (grouped).
- `cmp <sglA> <atomB> <ratio>`: side-by-side (the main debug view).
Boundary = `pa_prefill` kernel (alternates dur by compress_ratio 128/4 per config.json,
so match SAME ratio across engines). Trust WINDOW span + GPU-active UNION; per-kernel
`dur` is unreliable (Exp 19). Capture both single-stream + pure-prefill (OSL=1).

### Per-layer op table — SGLang ALL3 vs ATOM (ratio-4 layer, pa→pa window, us)
| stage | op | SGL ALL3 | ATOM | status |
|---|---|---:|---:|---|
| MLA attn | pa_prefill | 4610 | 4680 | = shared |
| out RoPE | rope | batched 332 | inverse_gptj 155 | ⚠ SGL ~2× (~180us) |
| o-proj | cijk + quant | 1437 + 143 | 1526 + 140 | = |
| o-proj | main GEMM | **ck_tile 1904** | **ck_tile 1802** | ✅ aligned (was Triton) |
| mhc | post/pre | 368/483 | 386/522 | = |
| shared-exp | up_gate ck_tile | 744 | 752 | ✅ aligned (local) |
| shared-exp | silu/act | 65 | 48 | = |
| shared-exp | down ck_xdl | 444 | 442 | ✅ aligned (local) |
| gate | router cijk | 572 | 99 | ⚠ SGL gate still GLOBAL (redundant, Exp 38) |
| comm | gather nccl | 4498 | 4694 | = |
| routed MoE | sort+moe1+moe2+reduce | ~12030 | ~12030 | = shared |
| comm | reduce-scatter nccl | 4615 | 4861 | = |
| next pre-attn | kv_a/q_a ck_xdl | 2739 | 2730 | = shared |
| next pre-attn | qk_norm_rope_fused | 729 | 967 | SGL slightly better |
| compressor | glue | fused_norm_rope+fill×3+**rocprim×2**+elementwise (~80us, many launches) | **hca_compress_forward+hca_norm_rope_scatter+compressor_update** (3 fused, ~28us) | ⚠ bubble — top remaining item |
| **TOTAL** | window | **39,472** | **38,846** | gap **~600us** (base was ~7000) |
| | GPU-active/layer | **38.53 ms** | 41.54 ms | SGL now below ATOM |

### Status / next
Prefill essentially matched (97–98%). Remaining prefill items are small: (1)
compressor glue/bubble (SGL fill/rocprim/fused_norm_rope vs ATOM 3 hca_* fused) —
the top remaining; (2) rope (batched 332 vs ATOM fused-inverse 155); (3) gate still
global (Exp 38 gate-local was c512-neutral). Caveat: SE-local needs TP1 shared
(~+0.5GB/rank). Artifacts: `/workspace/bench_pp_all/`, `/workspace/gsm8k_all.log`,
trace `/workspace/sgl_prof_all/*TP-0-DP-0*`. Cleaned (VRAM ~0.3 GB).

---

## Exp 52 — ALL3 c512 END-TO-END A/B (OSL=1024, with decode) (2026-06-18)

Does the matched prefill (Exp 51) move c512 TOTAL throughput, or is it diluted by the
decode-bound regime (gate-local Exp 38 was neutral)? Ran ALL3 (FORCE_CK_W8A8 +
ROPE_BATCHED + SHARED_EXPERT_LOCAL + SHARED_EXPERT_TP1, gatherv ON, ROCM700A=0) at
c512, full OSL=1024, ATOM client, np4096/warm1024.

| workload | SGL base | SGL ALL3 | ATOM | ALL3 vs base | base→ALL3 vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,877 | 19,828 | **+3.7%** | 86.9% → 90% |
| 8k/1k | 32,254 | 34,613 | 39,291 | **+7.3%** | 82.1% → 88% |
ALL3 detail: 1k MedTTFT 6855 TPOT 45.36 E2E 53084; 8k MedTTFT 47984 TPOT 69.56 E2E 103016.

### Conclusion — the prefill win DOES move c512 total tput (unlike gate-local)
- c512 end-to-end gain is REAL (+3.7% 1k, +7.3% 8k), bigger at 8k (more prefill-heavy).
  Closes c512 gap 86.9%→90% (1k), 82.1%→88% (8k) of ATOM.
- Diluted vs pure-prefill (+16%) because c512 is decode-bound — the matched prefill
  only helps the prefill share of the step. But it is NOT neutral (gate-local was),
  because these levers cut a much larger prefill chunk (GEMM + shared-expert rows +
  rope) than the gate alone.
- Remaining c512 gap (~10–12%) is now decode/scheduling + the residual prefill bubble
  (compressor glue) — not the kernels we fixed.

### Net summary of the 2026-06-18 prefill work (Exp 49–52)
3 env-gated levers (default OFF), gsm8k correct (0.9477):
- pure-prefill: 84% → 97–98% of ATOM (+16%); per-layer gap ~7000us → ~600us.
- c512 end-to-end: +3.7% (1k) / +7.3% (8k); 87%/82% → 90%/88% of ATOM.
Levers: `SGLANG_FORCE_CK_W8A8` (MLA proj Triton→CK), `SGLANG_ROPE_BATCHED` (batched
compressor rope), `SGLANG_DP_SHARED_EXPERT_LOCAL` (+`SGLANG_SHARED_EXPERT_TP1`,
shared expert on local hidden). Edits: fp8_utils.py, deepseek_v4_rope.py,
deepseek_v2.py, deepseek_v4.py (all in /sgl-workspace/sglang, default OFF).

### Artifacts
- `/workspace/bench_c512_all3/`, log `/workspace/c512_all3_sweep.log`,
  server `/workspace/sgl_b_server.log`. Baselines: Exp 39/41 + `/workspace/bench_results_dsv4_atom_0617/`.
  Server cleaned (VRAM ~0.3 GB/GPU).

### Follow-up — add chunk 8192/rank on top of ALL3 (Exp 45 lever stacks)
Stacked `--chunked-prefill-size 65536` (=8192/rank, the Exp 45 c512 sweet spot) ON TOP
of ALL3. c512 end-to-end:
| wl | base | ALL3 (16k/r) | **ALL3 + chunk 8k/r** | ATOM | best/ATOM |
|---|---:|---:|---:|---:|---:|
| 1k/1k | 17,233 | 17,877 | **17,921** | 19,828 | **90%** |
| 8k/1k | 32,254 | 34,613 | **36,031** | 39,291 | **92%** |
- 8k/1k: chunk8k adds **+4.1%** on top of ALL3 → vs base **+11.7%**, **82%→92% of ATOM**.
- 1k/1k: +0.2% (saturated; chunk effect already small at 1k c512), 90% of ATOM.
⇒ **best c512 config = ALL3 + chunk 8192/rank**: 1k 90%, 8k 92% of ATOM (8k from 82%).
The chunk lever (effect B, TTFT/queue fairness) is independent of and stacks with the
kernel/locality levers, especially at 8k. Artifacts:
`/workspace/bench_c512_all3_chunk8k/`, `/workspace/c512_all3_chunk8k_sweep.log`.

---

## Exp 53 — A2: output inverse-RoPE full-fuse (contiguous kernel) (2026-06-18)

The remaining rope item (Exp 51): the hot 337us/layer rope is the ATTENTION-OUTPUT
inverse rope `fused_rope_inplace(o[..., -rd:], k=None, ..., inverse=...)` at
deepseek_v4.py:1012. On HIP it fell back to `apply_rotary_emb_triton` (my Exp49
batched, STRIDED 2i/2i+1 interleaved loads = 337us); on CUDA it uses a single fused
kernel. ATOM uses `inverse_rope_gptj` (CONTIGUOUS load + reshape/flip) = 155us.

FIX: new `apply_rotary_emb_contig_kernel` (deepseek_v4_rope.py), mirrors ATOM —
loads the rope slice as a CONTIGUOUS [BLOCK_M, RD] tile (coalesced) and does the
GPT-J pair rotation via tl.reshape + tl.flip; derives cos/sin from the interleaved
freqs_real (cos=fr[2*(d//2)], sin=fr[2*(d//2)+1]). Supports forward+inverse. Wired
in apply_rotary_emb_triton for the 3D case under SGLANG_ROPE_BATCHED (the 2D
compressor rope keeps the prior batched kernel).

Result (trace, ALL3 + contig rope): `apply_rotary_emb_contig_kernel` = **142.7 us/call**
(was strided batched 337; **≤ ATOM's inverse_rope_gptj 155us**). gsm8k correct: flex
0.9439 / strict 0.9447. Per-layer saving ~194us × 61 ≈ 12ms/step (~0.5% prefill) — the
rope item is now fully closed (SGL ≤ ATOM). Remaining per-layer residual is the
compressor glue/bubble (A1) and gate-global (Exp 38).

Artifacts: trace `/workspace/sgl_prof_a2/*TP-0-DP-0*`, gsm8k `/workspace/gsm8k_a2.log`.
Edit in `/sgl-workspace/sglang/python/sglang/srt/layers/deepseek_v4_rope.py`
(under SGLANG_ROPE_BATCHED). Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 54 — make CK-GEMM + batched/contig-RoPE DEFAULT-ON for DSV4 (no env) (2026-06-18)

Converted two levers from env-flag-gated to module toggles (default OFF) that the
DeepseekV4 model flips ON in `__init__` — so DSV4 gets them WITHOUT any env var:
- `fp8_utils.py`: `_FORCE_CK_W8A8=False` + `set_force_ck_w8a8()`; `use_aiter_triton_gemm_w8a8_tuned_gfx950`
  checks `_FORCE_CK_W8A8 or env`. (CK GEMM for MLA proj.)
- `deepseek_v4_rope.py`: `_USE_BATCHED_ROPE=False` + `set_batched_rope()`;
  `apply_rotary_emb_triton` checks `_USE_BATCHED_ROPE or env`. (batched/contig rope.)
- `deepseek_v4.py` `DeepseekV4ForCausalLM.__init__`: imports + `set_force_ck_w8a8(True)`,
  `set_batched_rope(True)`. The env vars `SGLANG_FORCE_CK_W8A8` / `SGLANG_ROPE_BATCHED`
  still work as overrides.
(NOT changed: shared-expert-local stays env-gated — needs TP1 shared, ~+0.5GB/rank;
chunk-prefill is a launch arg.)

Verification — launched DSV4 with NO opt env flags, trace confirms defaults active:
contig rope present=True, strided batched rope=False, Triton a8w8 GEMM=False, ck_tile
QuantGemm=True. gsm8k correct: flex 0.9507 / strict 0.9515. ⇒ DSV4 now uses CK GEMM +
contig rope by default (no env needed). Artifacts: `/workspace/sgl_prof_def/*`,
`/workspace/gsm8k_def.log`. Server cleaned (VRAM ~0.3 GB/GPU).

---

## Exp 49 — FIX the prefill kernel gap: Triton→CK GEMM + batched RoPE (2026-06-18)

From the trace (Exp 48) two prefill kernels differ between engines by IMPLEMENTATION
(same logical op). Identified the call paths and made SGLang match ATOM:

### Kernel 1 — w8a8-block FP8 GEMM (the big one: MLA q/kv/o projections, ~28% of step)
- SGLang `apply_w8a8_block_fp8_linear` (fp8_utils.py) picks the **Triton**
  `gemm_a8w8_blockscale` for the MLA projection shapes because the hardcoded
  `use_aiter_triton_gemm_w8a8_tuned_gfx950(n,k)` list contains them (k=7168 shapes:
  2112/512/4096/4608×7168 etc.). HIP is 7.2.0 so the CK bpreshuffle path is available.
- ATOM `linear.py` per_1x128 path uses the **CK** `gemm_a8w8_blockscale_bpreshuffle`
  (preshuffle) by default (`ATOM_FP8_BLOCKSCALE_WEIGHT_PRESHUFFLE=1`, Triton off) — with
  an explicit comment "Triton FP8 Blockscale GEMM is mostly slower than AITER [CK] GEMM".
- FIX: env-gate `SGLANG_FORCE_CK_W8A8=1` → `use_aiter_triton_gemm_w8a8_tuned_gfx950`
  returns False → SGLang uses CK bpreshuffle (= the trace's `ck_tile…QuantGemmKernel`),
  matching ATOM. (fp8_utils.py:72.)

### Kernel 2 — RoPE (compressor fallback rope; small, ~2% of step)
- SGLang `apply_rotary_emb_triton` (deepseek_v4_rope.py, called in compress_hip.py):
  grid (batch, heads, dim_blocks) = ONE program per token (fine-grained launches).
- ATOM `_inverse_rope_gptj_kernel`: batches BLOCK_S=32 tokens/program.
- FIX: env-gate `SGLANG_ROPE_BATCHED=1` → added `apply_rotary_emb_triton_kernel_batched`
  (BLOCK_M=32 tokens/program, same math), mirroring ATOM. (deepseek_v4_rope.py.)

### Result — pure-prefill (OSL=1, ATOM client) with BOTH flags on
gsm8k stays correct: flexible 0.9462 / strict **0.9469** (≈ baseline 0.94 — both
changes numerically safe).
| ISL | SGL base | SGL +CK+ROPE | ATOM | gain | base→new vs ATOM |
|---|---:|---:|---:|---:|---:|
| 1024 | 48,291 | 52,506 | 57,821 | **+8.7%** | 84% → **91%** |
| 8192 | 47,013 | 51,205 | 55,999 | **+8.9%** | 84% → **91%** |
- **+~8.8% prefill throughput → recovers essentially the entire ~8% raw-kernel gap**
  from Exp 48 (84%→91% of ATOM). The remaining ~9% to ATOM = the host/overhead/bubble
  share (Exp 48), which kernel swaps don't touch.
- Dominant contributor is almost certainly the GEMM (28% of step vs RoPE's ~2%); not
  yet isolated GEMM-only vs RoPE-only (tested combined per request). Can split if needed.

### Status / how to use
- Both changes are env-gated and DEFAULT OFF (`SGLANG_FORCE_CK_W8A8`,
  `SGLANG_ROPE_BATCHED`). Edits in the editable repo `/sgl-workspace/sglang`
  (fp8_utils.py, deepseek_v4_rope.py) — persist in this clone, lost on container rebuild.
- Next: (a) isolate GEMM-only vs RoPE-only; (b) run a FULL c512 1k/1k & 8k/1k A/B
  (not just pure-prefill) to confirm the end-to-end throughput gain; (c) attack the
  remaining ~10% per-step bubble (Exp 48: launch batching / prefill CUDA graph).

### Artifacts
- `/workspace/bench_pp_kern/` (isl{1024,8192}_osl1), gsm8k `/workspace/gsm8k_kern.log`,
  server `/workspace/sgl_kern_server.log`. Baselines: `/workspace/bench_pp_{sgl,atom}/`.
  Server cleaned (VRAM ~0.3 GB/GPU).

### Follow-up — ISL=8192 pure-prefill TRACE of the fixed build (confirms kernel swap)
Re-captured the prefill trace WITH both flags on (ISL8192 OSL1, rank0, 10 steps),
same GPU-active-union-per-attn-layer-step method as Exp 48.
| | GPU-active/step | busy% | vs ATOM |
|---|---:|---:|---:|
| SGL base (Triton GEMM) | 44.81 ms | 89% | +7.9% |
| SGL +CK+ROPE | **42.58 ms** | 92% | **+2.5%** |
| ATOM | 41.54 ms | 82% | — |
- Kernel-level confirmation: Triton `_gemm_a8w8_blockscale` is GONE; CK
  `QuantGemmKernel` (1048 ms ×549, same as ATOM) is now used; old per-token rope GONE,
  batched rope present.
- GPU-active raw-kernel gap to ATOM shrank from +7.9% → **+2.5%** (kernel swap recovered
  most of the raw compute). busy% 89→92% ⇒ CK GEMM also cut launch/bubble slightly,
  which is why pure-prefill THROUGHPUT gain (+8.8%) > GPU-active reduction (−5%).
- Remaining ~2.5% GPU-active to ATOM = minor residual (some projection GEMM / other
  kernel). Trace: `/workspace/sgl_prof_kern/*TP-0-DP-0.trace.json.gz` (92% busy);
  baseline `/workspace/sgl_prof/1781761067*`, ATOM `/workspace/atom_prof/dp0_tp0/*`.

---

## Exp 55 — Port shared-expert-local to sglang-upstream + A/B; debunk the "upstream is slower" / BLOCK_M scare (2026-06-21)

### What was done
1. Ported the **shared-expert-local (SE-local)** PoC from the dev clone (`/sgl-workspace/sglang`)
   onto **`/sgl-workspace/sglang-upstream`** (which already has CK-GEMM + batched/contig
   rope DEFAULT-ON, Exp 54). Edits: `models/deepseek_v2.py` (`skip_shared_experts` param
   in forward/forward_normal), `models/deepseek_v4.py` (`_SHARED_EXPERT_LOCAL` flag +
   compute-local / skip / add-after-reduce-scatterv), `configs/cohere2_moe.py` (the
   `@strict` no-op patch, SKILL §2a, needed to import on upstream too).
2. Also applied the **rope contig-kernel `BLOCK_M=32 → 8` + `num_warps=4`** tweak
   (today's microbench: 1.6–1.8× faster kernel) to both clones.
3. A/B on upstream (user's choice = "plan B"): baseline = **gatherv ONLY**;
   SE-local = **gatherv + TP1 + se-local**. ROCM700A=0, chunk 65536 (=8192/rank,
   server resolves to chunked_prefill_size=8192), ratio 1.0, np=conc*8, ATOM-aligned
   launcher (`run_sgl_dsv4_aligned.sh`).

### Correctness
- gsm8k 5-shot (SE-local ON, upstream): **flex 0.9500 / strict 0.9507** — correct.

### Performance (upstream, plan B baseline = gatherv only)
| workload | baseline (gatherv) | + TP1 + se-local | delta |
|---|---:|---:|---:|
| pure-prefill 1k (OSL=1, c256) | 48,941 | **52,235** | **+6.7%** |
| pure-prefill 8k (OSL=1, c256) | 49,639 | **52,674** | **+6.1%** |
| c512 1k/1k (total tok/s) | 17,738 | 17,136 | **−3.4%** |
| c512 8k/1k (total tok/s) | 33,828 | 34,064 | +0.7% |
- pure-prefill +6–7% (consistent with Exp 50). c512 diluted/negative because **plan B
  bundles TP1**: TP1 replicates the shared expert → in DECODE each rank does the full
  shared-expert GEMM (~8× FLOPs, only saves the all-reduce); at c512 1k/1k (decode-bound)
  that cost outweighs se-local's tiny prefill benefit → −3.4%. 8k/1k (prefill-heavier)
  roughly breaks even (+0.7%, TTFT 48.0s→44.9s).

### The "upstream 34k vs this-morning 36k" investigation — it was an editable-finder BUG
User flagged that c512 8k/1k SE-local hit ~36k earlier (Exp 52 logged 36,031 on the dev
clone, 6/18) but only 34k now on upstream. Root-caused as follows:
- **BUG**: `pip install -e` to "restore" the dev clone did NOT switch the import — TWO
  `__editable__` finders coexisted in site-packages: `…dev380…` → `/sgl-workspace/sglang`
  (dev clone) and `…dev14280…` → `/sgl-workspace/sglang-upstream` (upstream). The
  upstream finder won, so **every run labelled "dev clone" was actually upstream**.
  Fix: `rm` the upstream `.pth` + `_finder.py` + `dist-info`; import then resolves to
  the dev clone. (Lesson: after `pip install -e`, ALWAYS verify `python -c "import
  sglang.…; print(.__file__)"` AND the server log's `Editable project location`.)
- After the fix, re-ran the **TRUE dev clone** 8k/1k c512 SE-local (full ALL3 env:
  FORCE_CK + ROPE_BATCHED + SE-local + TP1 + gatherv):

| config | c512 8k/1k total tok/s | TTFT (ms) | TPOT (ms) |
|---|---:|---:|---:|
| dev clone **BLOCK_M=32** (= Exp 52 cfg) | 35,101 | 38,987 | 88.31 |
| dev clone **BLOCK_M=8** (current default) | 35,086 | 39,062 | 88.34 |
| upstream BLOCK_M=8 | 34,064 / 35,084 | — | — |
| Exp 52 logged (dev clone BM32, 6/18) | 36,031 | — | — |

### Conclusions
1. **rope BLOCK_M=8 vs 32 has ZERO c512 end-to-end effect** (35,086 vs 35,101, −0.04%,
   pure noise). As predicted — rope is a sub-ms prefill op, c512 is decode-bound. The
   BLOCK_M=8 default stays (it's a real prefill kernel win, Exp 53/microbench, and costs
   nothing at c512). User's BLOCK_M hypothesis = ruled out.
2. **upstream ≈ dev clone** today (both 34–35k). The earlier "dev clone +3%" claim was
   the finder bug (both were upstream); there is NO clone/version regression.
3. **36,031 (6/18) vs ~35,100 (today) = ~2.5% cross-day run-to-run variance** (c512
   high-conc single-run variance is 1–3%), not any code change.

### Artifacts
- upstream: `/sgl-workspace/{base_prefill,base_c512,se_prefill,se_c512,se_c512_8k,gsm8k_se_local}.log`
- dev clone: `/sgl-workspace/{D32_c512_8k,D8_c512_8k}.log`, servers `/sgl-workspace/dsv4_{D32,D8}.log`
- Final state: both clones rope = BLOCK_M=8 + num_warps=4; editable → dev clone
  (`/sgl-workspace/sglang`); servers stopped, VRAM ~0.3 GB/GPU.

---

## Exp 56 — C5: extend shared-expert-local to DECODE → fixes the c512 1k/1k regression (2026-06-21)

### Motivation
Exp 55 (plan B) showed SE-local (TP1 + se-local) REGRESSED c512 1k/1k by −3.4% while
helping prefill +6–7%. Root cause analysis: the −3.4% is NOT prefill FLOPs. Per-rank
shared-expert FLOPs are identical between the two schemes:
- normal (TP-sharded, global buffer): `M_global * dim/tp`
- SE-local (TP1, local tokens):        `M_local * dim = M_global/tp * dim`
The penalty came from SE-local being PREFILL-ONLY (`is_extend()` gate): in DECODE the
normal path ran with the **replicated (TP1) weights on the gathered global batch at full
dim = ~dp_size x** the sharded cost. c512 1k/1k is decode-bound, so that decode penalty
dominated.

### Change (one-liner: broaden the gate)
`deepseek_v4.py` `_do_shared_local`: drop `is_extend()`, switch `_use_gatherv_pair` →
`_use_tp_moe_gather`, so SE-local applies to BOTH prefill (gatherv/reduce_scatterv) and
decode (dp_scatter). The shared expert is a per-token MLP → computing on this rank's
local tokens ≡ computing on the global buffer then taking the local slice (gsm8k-verified).
The existing add-back (after the if/else) already covers both reduce_scatterv and
dp_scatter. Stable for CUDA graph (gate no longer depends on padding mode). Amended into
the SE-local commit `30fa179536`.

### Correctness
gsm8k 5-shot (C5, gatherv+TP1+se-local): **flex 0.9507 / strict 0.9515** — correct.

### c512 results (ratio 1.0, conc 512, same client, dev clone, ROCM700A=0, chunk 8192/rank)
| config | 1k/1k | vs base | 8k/1k | vs base |
|---|---:|---:|---:|---:|
| baseline (gatherv only) | 17,738 | — | 33,828 | — |
| SE-local prefill-only (Exp 55) | 17,136 | **−3.4%** | 34,064 | +0.7% |
| **SE-local C5 (prefill+decode)** | **17,954** | **+1.2%** | **35,403** | **+4.7%** |

Decode TPOT confirms the ~dp_size x removal:
- 1k/1k TPOT 48.08 → **46.54** ms; 8k/1k TPOT 79.10 → **73.69** ms.
vs ATOM (same-day): 1k/1k 91.2%, 8k/1k 90.7% of ATOM (was ~87%).

### Net
SE-local is now a positive lever at c512 for BOTH workloads (no 1k/1k regression). Still
env-gated (`SGLANG_DP_SHARED_EXPERT_LOCAL` + `SGLANG_SHARED_EXPERT_TP1`); TP1 weight-memory
cost (~+0.5 GB/rank) unchanged — C5 only removes the decode COMPUTE penalty, not the
replication memory.

### Artifacts
- `/sgl-workspace/{gsm8k_c5,c5_c512}.log`, server `/sgl-workspace/dsv4_c5.log`.
- Code: dev clone `deepseek_v4.py` (amended into commit `30fa179536`); server stopped,
  VRAM ~0.3 GB/GPU.
