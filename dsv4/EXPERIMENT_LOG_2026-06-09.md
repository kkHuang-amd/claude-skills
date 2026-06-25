# DeepSeek-V4-Pro serving perf — experiment log (2026-06-09)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

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

