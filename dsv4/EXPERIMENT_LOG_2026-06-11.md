# DeepSeek-V4-Pro serving perf — experiment log (2026-06-11)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

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

