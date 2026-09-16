# MI355X decode trace, AgentX c128, pdi=24 (mi355x, 2026-09-16)

> ## ⚠ SUPERSEDES commit 564a590 — every step number there was wrong
>
> **Speculative decoding emits TWO `step[TARGET_VERIFY]` annotations per
> `scheduler.run_batch`** — a small DSPARK draft forward and the full-model
> verify — and **they carry the same `bs`**, so `trace_summary.py`'s (type, bs)
> grouping averaged them. Cross-check on one rank: 32 `step[TARGET_VERIFY]`
> against 17 `scheduler.run_batch`.
>
> | | 564a590 said | actually |
> |---|---:|---|
> | verify step wall p50 | 39.3 ms | **4.0 ms draft + 74.0 ms full** |
> | `compute` p50, bs=10 | 14.85 ms | **28.36 ms** (full step) |
> | MoE calls/step | 32 | **3 (draft) / 61 (full)** |
>
> 39.3 ms was the median of a bimodal set and described neither cluster. The
> conclusion it supported — "`compute` matches B200 within 1.4 %" — **is
> withdrawn**; see §3.
>
> Two consequences worth reading even if you skip the rest:
> - **B200's published numbers are almost certainly affected too.** Same tool,
>   same grouping, and B200's own `n=38` verify steps at one bs look like the
>   same 2-per-batch pattern. Re-derive before comparing.
> - **The "bimodal distribution, quote p50 not mean" advice in both docs was
>   misdiagnosing this bug.** Split by class and each cluster is tight: `moe`
>   min/p50/max 33.87 / 34.37 / 38.85 within the full step.
>
> Fixed in `analysis/trace_common.py:verify_classes()`, which splits by MoE call
> count (never by annotation order, so a wrong ordering assumption cannot swap
> the classes) and is wired into `trace_summary.py`.
>
> Credit: caught by the MI355X operator, not by me. The tell was that my
> "32 MoE calls/step" is exactly (3+61)/2.

Raw traces are node-local at
`mi355x:/shared_nfs/kk/pr35619/trace_c128_pdi24_steady/` (steady state — use
this) and `.../trace_c128_pdi24/` (mid-ramp, kept only for §2).

## Capture provenance

| | |
|---|---|
| arm | c128, dp8 + ep8 + MegaMoE + EPLB, DSPARK, hicache 1.5 |
| pdi | **24** — verified in `$RESULT_DIR/sglang_command.txt`, not script intent |
| settings | `{"activities":["CPU","GPU"],"num_steps":40,"profile_by_stage":true,"record_shapes":true,"with_stack":false}` + explicit `output_dir` |
| steady state | per-request `#full token` = 167,538, pool usage plateaued at 0.16 |
| sglang | `sglang-MegaMoE @ ff7f522abd`; aiter `ffa945f93` + 2 uncommitted fixes |

`output_dir` was added because `SGLANG_TORCH_PROFILER_DIR` — exported before the
server started — appeared in none of the 9 schedulers' `/proc/*/environ`. That is
not conclusive (exec-time environ, not Python's runtime `os.environ`), and
`profile_utils.py:117` only falls back to the env var when the field is absent,
so passing it removes the risk of landing in `/tmp`. No perturbation-relevant
field changed.

## 1. Step walls, split by class

**Full-model verify: 74.0 ms. DSPARK draft: 4.0 ms. ~78 ms per `run_batch`.**

The full step wall is near-identical on every decoding rank, as group-synchronous
steps require, while `compute` and `barrier` trade off against each rank's own
batch:

| rank | bs | full-step wall p50 | compute p50 | barrier p50 | draft wall p50 |
|---|---:|---:|---:|---:|---:|
| 3 | 9 | 74.01 | 24.94 | 44.27 | 4.06 |
| 7 | 10 | 74.02 | 28.36 | 40.71 | 4.02 |
| 1 | 14 | 74.01 | **38.91** | 30.73 | 4.02 |
| 6 | 16 | 73.93 | 35.25 | 33.95 | 3.98 |
| 0 | 17-20 | 74.09 | 28.87 | 40.62 | 3.93 |

`compute` spans 24.94-38.91 ms across ranks while the wall stays at 74 ms — the
textbook signature that `barrier` is absorbing a wait, not doing work.

**No ratio against B200 is quoted.** B200's 15.9 ms is retracted as mid-ramp,
and by the argument above it is probably also class-mixed. Both need re-deriving.

## 2. The fixed 120 s settle is unsafe — quantified on this node

My first capture used the prompted rule (first `done=` + 120 s) and fired at
per-request KV **67,759**, under half of steady state:

```
minute  perreq_KV   usage  batch
02:33      45,824   0.010    2.0
02:36      67,759   0.040    7.0   <-- first capture
02:39     159,334   0.170   14.0   <-- plateau starts
03:00     167,538   0.160   13.0   <-- second capture
```

Same run, mid-ramp vs steady, class-mixed medians (the only form in which I have
the mid-ramp numbers): step wall 28.3 -> 39.3 ms, `attn` p50 7.95 -> 15.62 ms.
Both nodes should replace the constant with a predicate: wait for per-request
`#full token` / `#running-req` to plateau (>= ~130k on these arms). A fixed sleep
is racing a context ramp whose length depends on trace content and page cache.

## 3. Per-role, full-model verify, bs=10 (rank 7, n=16)

| role | p50 ms/step | mean | min | max |
|---|---:|---:|---:|---:|
| moe | **34.37** | 35.49 | 33.87 | 38.85 |
| attn | **15.67** | 15.67 | 15.62 | 15.76 |
| gemm | **9.37** | 9.36 | 9.34 | 9.38 |
| comm | 6.48 | 6.43 | 5.85 | 6.83 |
| copy | 3.97 | 3.98 | 3.95 | 4.03 |
| quant | 2.21 | 2.21 | 2.21 | 2.21 |
| other | 0.98 | 0.99 | 0.98 | 1.00 |
| norm_rope | 0.72 | 0.72 | 0.72 | 0.72 |
| sample | 0.40 | 0.40 | 0.40 | 0.40 |
| summed kernel | 75.24 | | | |
| step wall | **74.0** | 75.2 | 73.4 | 79.0 |

Spreads are tight once the classes are separated — the earlier 34x role spreads
were the two classes, not step-to-step variance.

`gemm` is still not directly comparable to B200's: `deep_gemm::..._mega_moe_impl`
contains `gemm` while ROCm's equivalent is `megamoe_stage1/2_compact`, and ROLES
tests `moe` first. **Compare `gemm + moe` as one bucket**: MI355X 43.74 ms.

## 4. MoE calls per step — resolved, the platforms agree

**61 calls in the full-model verify step**, 3 in the draft. B200 reports 61-64,
one per layer. The earlier "32" was the average of the two classes, and the
open question about whether to normalise by 2x is void — **both platforms issue
one MoE call per layer and no renormalisation is needed.**

## 5. Overlap and streams

| | streams | summed kernel | step wall | overlap |
|---|---:|---:|---:|---:|
| B200 multi-stream | 132 | 24.55 ms | 15.9 ms | 1.54x |
| B200 single-stream | 4 | ~24.5 ms | — | — |
| **MI355X, full verify** | **2-3** | 75.24 ms | 74.0 ms | **1.00x** |

Per step: min 1, p50 2, max 3, identical on ranks 0, 5 and 7 and in both
captures. MI355X's critical path is its whole kernel sum.

**This is probably not the cause.** B200's own 132->4 A/B costs only 4-5 % of
step time, and MI355X already sits at that setting — stream count cannot carry a
gap of this size. Recorded as a measured difference, not an explanation.

## 6. The 74 ms wall is already free of concurrent EXTEND

The previous reading — "all eight ranks have `EXTEND` in the file, so 74 ms
includes waiting for prefills" — is **file-level coexistence, not time overlap**.
GPU clocks align. Every rank's `EXTEND` is either before the verify burst
(ranks 0/1/3/6/7, ~820-1080 ms, ending just as verify starts) or ~10 s after it
(ranks 2/4/5; 0 kernels on those ranks during rank 0's 1286 ms verify window).

`trace_ranks.py` now filters `TARGET_VERIFY full` to steps with no interval
overlap against any of the 8 `EXTEND`s. **n_hit=0** on all five decoding
ranks; p50 73.9-74.1 ms (1.002x). The number does not drop.

What remains inside the 74 ms is still ~half `barrier`, but it is the wait
*among decoding ranks*: wall locked at ~75 ms mean while compute ranges
24.95 (rank 3, bs=9) to 38.97 (rank 1, bs=14) and barrier runs the other way.
`compute` at bs=10 (rank 7) is still **28.37 ms**. That is the barrier-free
decode-compute figure; the 74 ms wall is the group-synchronous decode step
with no concurrent prefill in this window.

Rank 6 also has a 5321 ms `TARGET_VERIFY` — classed **draft** (3 MoE calls),
not full. Ignore it for full-verify stats.

## 7. Scheduler-log numbers (complete run, not the trace run)

`mi355x:/workspace/results/megamoe-eplb-c128-b200aligned/server.log`:

```
running-req/rank   p50=12.00  mean=12.30  p90=17.00
accept len         p50= 3.78          cuda graph replay 100.0% of steps
gen tput/rank      p50=389.36 tok/s   implied step ms p50=121.76
implied ITL        p50= 32.28 ms      #full token p50=1,763,584 (~151k/req)
pool capacity      ~12,074,667 tokens ISL mean 108,371   OSL mean 967
```

Worth noting against §1: the log-implied step is 121.76 ms while the traced
`run_batch` verify work is ~78 ms (74 + 4), so ~64 % of the scheduler's step is
inside these two annotations and the rest is prefill and gaps.

`step_ms` by batch (p50, per-token on the shared 3.77 divisor):

```
batch= 4  109.47  29.04     batch=13  122.74  32.56
batch= 9  112.04  29.72     batch=16  130.40  34.59
batch=12  119.90  31.80     batch=21  136.49  36.20
```

ISL mean 108,371 vs B200's 114,401 at pdi=24: B200 carries 5.6 % longer
sequences, so its advantage is understated.

## 8. Tooling changes pushed with this

`analysis/trace_common.py`:

- **`verify_classes()`** — splits draft vs full-model verify by MoE call count;
  `trace_summary.py` now groups by `(type, class, bs)`. This is the fix for the
  bug at the top of this file.
- ROLES gained three ROCm patterns, dropping unclassified time from 1.90 to
  0.72 ms/step: `Cijk_` -> `gemm` (rocBLAS/Tensile), `rotary` -> `norm_rope`,
  `fillbuffer|fill_padded_rows|fill_compress_tail` -> `copy`. `rotary` is generic
  and may reclassify B200 kernels currently in `other`.

Left in `other` deliberately, because bucketing them is a guess I should not make
unilaterally: `sglang::write_c4_prefill` / `write_cN_prefill` (DSv4 compressed-KV
writes — `attn` or `copy`?) and residual `at::native::*elementwise*`.

`trace_ranks.py` now uses the same `(type, class, bs)` grouping and prints a
final "no concurrent EXTEND" table. Older cross-rank tables (commit 564a590
and the mixed numbers in 2c7b0c9) still mix the two classes — ignore them.

## 9. Operational note

The first capture attempt died in an **intermittent EPLB rebalance deadlock**:
`returned=` frozen with `errors=0`, `/metrics` still 200, schedulers alive, VRAM
still full. Last server activity was `Resetting ExpertDistributionRecorder...`
from all 8 ranks; every rank then hung in a collective and the NCCL watchdog took
the process down 600 s later with `c10::DistBackendError`. Nothing reached
launcher stdout. A straight retry worked.

## 10. MI355X per-kernel listing at bs=10 — answering B200's §2b request

> **⚠ The `delta` column in this section is superseded.** It subtracts B200's
> summed kernel time from MI355X's elapsed time — see §13 for why that is
> invalid. B200 has since published `credited` elapsed time, and the correct
> cross-platform table is in FINDINGS ("B200 `credited` elapsed time"):
> **moe +27.67 (61.5 %), attn +9.44 (21.0 %), gemm −4.35**, total +45.02 on a
> 30.20 → 75.22 ms GPU-busy basis. The *ordering* in this section (moe > attn)
> was right and B200's "`attn` is the largest" is withdrawn; only the magnitudes
> here are wrong. The MI355X column needs no correction — this node is serial.
> The per-kernel pairings below are unaffected, being per-call at matched call
> counts.

*(mi355x, 2026-09-16, second session. Source: rank 7 `TARGET_VERIFY full`
bs=10, n=16, steady capture. Reproduce with `/tmp/mi355x_for_b200.py`, kept at
`analysis/kernel_dump.py`.)*

Summed kernel **74.03 ms p50 / 75.24 ms mean** per step. B200 published 74.17
for this cell from my earlier table; the difference is p50-vs-mean, not data.
All MI355X numbers below are **means**, to match B200's dump.

Both sides are reclassified the same way before comparing: B200's `nvjet_*`
→ `gemm` and `silu_mul_clamp` → `moe` (as B200 did), plus B200's
`flash_fwd_mla_combine` → `attn` and `mega_moe_pre_dispatch` → `moe`, which
empties B200's `comm` bucket; and MI355X's `_fused_clamp_silu_mul_kernel`
(0.262) out of `comm` → `moe`. After that, `comm` means *only* a real
cross-rank collective on both sides.

| bucket | B200 (TP-4, n=15) | MI355X (TP-7, n=16) | delta | share of gap |
|---|---:|---:|---:|---:|
| moe | 15.84 | **35.75** | **+19.91** | 81.6 % |
| attn | 7.62 | **15.67** | **+8.05** | 33.0 % |
| gemm | 24.03 | 9.36 | **−14.67** | −60.1 % |
| comm | 0.00 | 6.17 | +6.17 | 25.3 % |
| copy | 0.06 | 3.98 | +3.92 | 16.1 % |
| quant | 1.52 | 2.21 | +0.69 | 2.8 % |
| sample | 0.14 | 0.40 | +0.26 | 1.1 % |
| norm_rope | 0.56 | 0.72 | +0.16 | 0.7 % |
| other | 1.09 | 0.99 | −0.10 | −0.4 % |
| **summed kernel** | **50.86** | **75.25** | **+24.39** | |

`gemm` and `moe` still must be read as one bucket, for B200's reason (its
`deep_gemm` absorbs wait) — **39.87 → 45.11, +5.24 (1.13x)**, close to B200's
+4.11. The new information is *inside* that bucket: the two platforms split it
completely differently, 60/40 gemm:moe on B200 against 21/79 on MI355X. So the
−14.67 `gemm` row is **not** MI355X being faster at GEMM; most of it is B200's
absorbed wait sitting in `deep_gemm` (which B200 measured at 56.8 → 48.9
µs/call as work grows). Do not quote it as a GEMM efficiency result.

### `attn` 15.67 is one kernel — 93 % of the attention gap

Matched by role, per step, with calls/step in parentheses:

| what | B200 | MI355X | ratio |
|---|---|---|---:|
| **MLA decode core** | `flash_fwd_splitkv_mla_fp8` **2.454** (61) | `_paged_decode_split_kernel` **9.974** (61) | **4.06x** |
| split-K reduce | `flash_fwd_mla_combine` 0.428 | `_paged_decode_reduce_kernel` 0.831 (61) | 1.94x |
| MLA post fusion | `mhc_post_tilelang` 0.737 (125) | `mhc_fused_post_pre_gemm_sqrsum` 1.250 (121) | 1.70x |
| MLA pre + norm | `mhc_pre_big_fuse_with_norm` 0.948 (122) | `mhc_pre_big_fuse_rmsnorm` 0.894 (121) | 0.94x |
| indexer logits | `smN_paged_mqa_logits` 0.901 (30) | `pa_mqa_logits_fp4_prefill` 1.092 (30) | 1.21x |
| indexer topk | `topk_main` + `topk_persistent_cluster` 0.710 (30+30) | `sglang::topk_main_kernel` 0.830 (30) | 1.17x |
| compressed-KV read c4 | `flash_c4_prefill` 0.298 (60) | `flash_c4_prefill` 0.307 (60) | 1.03x |
| compressed-KV read cN | `flash_cN_prefill` 0.193 (31) | `flash_cN_prefill` 0.151 (31) | 0.78x |
| fused norm/rope for MLA | 0.182 + 0.138 (61) | `fused_norm_rope_flashmla` 0.271 (61) | 0.85x |

**The MLA decode kernel alone is +7.52 ms of the +8.05 ms `attn` gap (93 %)**, at
an identical 61 calls/step — one per layer — so it is 40.2 µs/call against
163.5 µs/call. Every other attention-family kernel is within 0.78-1.94x, and the
five DSv4-specific ones (mhc pre/post, indexer, c4/cN) are within 1.7x. This is
the narrowest target either node has produced so far.

`attn` is *not* pace-pinned on MI355X (see below), and the c4/cN pair matching
within 3 % is a good sign the two traces are measuring the same work.

### The compressed-KV kernels land identically — no reclassification needed

B200 asked where they go on ROCm. Same place: `flash_c4_prefill` /
`flash_cN_prefill` → `attn`, `write_c4_prefill` / `write_cN_prefill` → `other`
(0.259 / 0.131 here). Both nodes split them the same way, so that asymmetry does
not exist.

### `comm` 6.17 is a genuine collective — and the buckets are structurally different

B200 asked. Answer: **yes.** It is one kernel, `ep_combine_intranode_0`, 6.166
ms over 61 calls/step, an aiter EP-combine all-to-all. B200's `comm` bucket by
contrast contains **no** collective at all — its all-to-all is fused inside
`mega_moe_impl`. So the two `comm` rows are not comparable in either direction,
and MI355X's is 6.17 ms of *exposed* collective that on B200 is hidden inside a
`moe` kernel. Adding it to the joint bucket gives **gemm+moe+comm 39.87 →
51.28, +11.41 (1.29x)**, which is the most defensible single roll-up.

### MoE is three kernels on MI355X, and the prepare stage has no B200 analogue

| | B200 | MI355X |
|---|---|---|
| | `mega_moe_impl` 14.85 (61) | `megamoe_prepare_compact` **16.244** (61) |
| | `mega_moe_pre_dispatch` 0.242 | `megamoe_stage1_compact` 13.799 (61) |
| | `silu_mul_clamp` 0.740 (61) | `megamoe_stage2_compact` 5.444 (61) |
| | `moe_hash_topk_fused` 0.02 | `_fused_clamp_silu_mul` 0.262 (61) |
| total | **15.85** | **35.75** (2.26x) |

`megamoe_prepare_compact` is the **largest single kernel in the whole step** and
on its own exceeds B200's entire MoE time. B200's `pre_dispatch` equivalent is
0.242 ms. Whether that is real work, the fused all-to-all wait, or both cannot
be told from this trace — but it is where 66 % of the `moe` gap sits.

## 11. Answering §3 — bs never repeats across ranks, so I deconfounded it the other way

B200 asked for `compute` on two ranks at the same `bs`. **Not available on this
node, in either capture** — every rank sits at its own `bs` and no value
repeats (checked both the steady and the mid-ramp directory). But **rank 0
spans four `bs` values on its own**, which fixes the rank and varies the batch,
and that answers the same question from the other side.

Rank 0 only, steady capture, `TARGET_VERIFY full` p50:

| bs | n | wall | compute | attn | gemm | moe | comm | quant |
|---:|--:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 4 | 73.55 | 28.76 | 14.17 | 10.90 | 35.34 | 5.31 | 2.32 |
| 18 | 7 | 74.04 | 28.78 | 14.26 | 10.89 | 35.50 | 5.48 | 2.32 |
| 19 | 3 | 77.86 | 34.45 | 18.44 | 12.22 | 33.71 | 5.42 | 2.37 |
| 20 | 2 | 78.75 | 35.06 | 19.02 | 12.25 | 33.99 | 5.39 | 2.37 |

**MI355X's `compute` is not pace-pinned, and only `moe` absorbs wait.** Applying
B200's own §2c test at fixed rank, bs 17 → 20: `attn` +34 %, `gemm` +12 %,
`quant` +2 %, `comm` flat, and **`moe` −3.8 %** — the only bucket that shrinks
as work grows. On B200 *both* `gemm` (−3.04) and `moe` (−3.10) shrink. So the
withdrawal in FINDINGS §2 applies **asymmetrically**: B200's `compute` is an
upper bound contaminated by slack, MI355X's `compute` is a measurement. That
does not resurrect the ratio (B200's side is still unusable), but it does mean
MI355X's 28.36 ms at bs=10 needs no caveat.

### Cross-rank spread: `bs` explains the mid-ramp capture entirely and the steady one only partly

All ranks, `TARGET_VERIFY full` p50 `compute`:

| capture | bs → compute |
|---|---|
| mid-ramp | 4 → 17.15, 5 → 18.69, 7 → 19.78, 10 → 23.24, 11 → 24.25, 12 → 24.40, 13 → 25.14 |
| steady | 9 → 24.94, 10 → 28.36, 14 → **38.91**, 16 → 35.25, 17 → 28.76, 18 → 28.78, 19 → 34.45, 20 → 35.06 |

Mid-ramp is **monotone in `bs`**, so B200's objection is exactly right there:
that capture shows no rank imbalance once `bs` is controlled. Steady is **not
monotone** — rank 1 at bs=14 does 38.91 ms of compute while rank 0 at bs=18 does
28.78 — so at steady state `bs` does *not* explain the spread and there is
residual per-rank variation of about 1.35x. The driver is `attn` (25.20 vs
14.26), i.e. **KV tokens, not request count**: `bs` × context is not monotone in
`bs`, so `bs` is a poor proxy for attention work at steady state. Reporting
`compute` against per-rank KV tokens is what this actually needs, and no
annotation in the trace carries it.

### The step is paced, and `barrier` is pure slack — `compute` + `barrier` is constant

Across the five decoding ranks, for every step whose wall is ~74.0 ms:

| rank | bs | compute | barrier | sum |
|---:|---:|---:|---:|---:|
| 3 | 9 | 24.94 | 44.27 | 69.21 |
| 7 | 10 | 28.36 | 40.71 | 69.07 |
| 1 | 14 | 38.91 | 30.73 | 69.64 |
| 6 | 16 | 35.25 | 33.95 | 69.20 |
| 0 | 17 | 28.76 | 40.62 | 69.38 |
| 0 | 18 | 28.78 | 40.88 | 69.66 |

`compute` spans **1.56x** (24.94-38.91) while the sum holds within **0.9 %**
(69.07-69.66). That is as clean a demonstration as this data can give that
`barrier` is slack and the group step is paced by its straggler — rank 1, at
38.91 ms of compute. Even the straggler still spends 30.73 ms in `moe`+`comm`,
so that time is not all wait.

## 12. B200's proposed estimator does not work on MI355X

FINDINGS §5 suggests achieved FLOPs/bandwidth per kernel "which `record_shapes`
already supports". **It does not, on this node's traces.** Of 177,036 events in
the rank-7 file, 34,486 carry `Input Dims`, and **every one of them is an
`aten::*` `cpu_op`** (`as_strided`, `empty`, `view`, `slice`, `copy_`, ...).
Zero attention or MoE ops carry dims — the aiter/sglang kernels that hold the
time are launched through custom ops that never register shapes. So the
bandwidth estimator needs either shapes plumbed into those custom ops or the
dimensions taken from the model config plus the per-rank KV token count, not
from `record_shapes`.

The other candidate estimator — a **TP-only / single-DP-rank run** — is still
viable here and is now the cheapest way to get an uncontaminated compute number
on either node.

## 13. ⚠ The per-role tables in §10 subtract kernel-seconds from elapsed seconds

*(mi355x, 2026-09-16. Raised by the MI355X operator; it invalidates the delta
column in §10 and in B200's §2b, though not the per-kernel pairings.)*

Every per-role number either node has published is a **sum of kernel
durations**. On a platform with stream concurrency that sum double-counts. B200
reports 50.86 ms of summed kernel inside a **30.0 ms wall**, so the union of its
kernel intervals — the time the GPU was actually busy — is **at most 30.0 ms**,
and B200's `gemm` 24.03 or `moe` 15.84 cannot each be occupying that much of a
30 ms step. **So yes: B200's elapsed decode work is a ~30 ms quantity, not a
50.86 ms one, and §10's delta column compares B200's kernel-seconds against
MI355X's elapsed seconds.**

MI355X is the degenerate case where the distinction vanishes. New tool
`analysis/busy_ms.py` sweeps the intervals; on rank 7, `TARGET_VERIFY full`
bs=10, n=16:

| | ms/step (mean) |
|---|---:|
| step wall | 75.23 |
| summed kernel | 75.24 |
| **union (GPU busy)** | **75.22** |
| sum / busy | **1.000x** |
| busy / wall | **1.000** |
| idle inside the step | **0.01** |

Per role, `sum` = `union` = `exclusive` to two decimals on every bucket. So the
MI355X decode step is **perfectly serial and has no idle**: no two kernels ever
overlap, and there is no gap between them. Every MI355X number in §10 is already
elapsed time and needs no correction. The same holds on all five decoding
ranks (sum/busy 1.000-1.009x, idle 0.01 ms).

### What this does to the comparison

- **In elapsed terms the gap is bigger, not smaller.** ≤30.0 ms against 74.0 ms,
  i.e. **≥2.47x** — the wall ratio. The "+24.39 ms more kernel time" in §10 and
  the "1.50x kernel x 1.67x overlap" decomposition are arithmetically fine as a
  decomposition *of the wall*, but they must not be read as elapsed
  contributions.
- **Every bucket delta in §10 is understated, and `gemm` may flip sign.** B200's
  roles have to shrink to fit 30 ms. Uniform 1.67x scaling — which is *not* what
  actually happens — would put B200 at gemm ~14.4, moe ~9.5, attn ~4.6, turning
  the deltas into roughly moe +26, attn +11, gemm −5. The real split is whatever
  the sweep says, which is why B200 has to run it rather than accept a scaling.
- **The headline per-kernel findings survive.** They are per-call durations of a
  single kernel at a matched call count, not bucket sums. And for the MLA decode
  kernel specifically, B200's own §2c classified `flash_fwd_splitkv_mla_fp8` as
  **pure** (1.14 → 2.90 ms as bs goes 1 → 12, scaling with work), so its 2.454 ms
  is real work. If anything the **4.06x is a floor**: a kernel sharing the device
  with concurrent work is *slowed* by contention, so B200's 2.454 ms under
  1.67x concurrency would only get shorter if it ran alone.
- **The `moe` comparison stays the weak one.** B200's `mega_moe_impl` absorbs
  wait (its §2c) *and* its sum is concurrency-inflated, so both corrections hit
  the same bucket. `megamoe_prepare_compact` 16.24 ms still has no counterpart,
  but its ratio against B200 is not quotable until B200 publishes credited time.

### Request to B200: run `analysis/busy_ms.py` on your steady capture

```bash
python3 analysis/busy_ms.py <your trace dir> 10
```

It prints, per role, `sum` / `union` / `exclusive` / `credited`, plus total busy
and idle-in-step. **`credited`** (exclusive time, plus a 1/k share of every
segment where k roles overlap) is the column to publish: the credits sum exactly
to total GPU busy time, so a per-role table built from `credited` is an
attribution of elapsed time and its deltas mean what they look like. `union` per
role does *not* sum to total busy when roles overlap each other, which is why
per-role unions alone are not enough.

Two numbers nobody has for B200 and which this produces for free: **how much of
the 30 ms wall the GPU was idle**, and **which roles B200 is actually overlapping
with which** (from `union` minus `exclusive`). If B200's idle is large, part of
the 2.47x is launch/sync gaps rather than kernel work — a different fix again.

## Appendix — full MI355X role composition, rank 7, `TARGET_VERIFY full` bs=10

n=16 steps, summed kernel 75.24 ms/step. Since the step is perfectly serial
(§13), **`ms/step` here is elapsed time and the roles sum to the step wall.**
Names are demangled; `[mangled]` marks ones `c++filt` could not decode (aiter
writes bf16 as the non-standard `DF16b`), where the identifier was recovered
from the Itanium length prefixes. Reproduce with
`python3 analysis/kernel_dump.py <trace dir>`.

Cut at 0.02 ms/step. Note two naming corrections to §10: what earlier tables
printed as `flash_cN_prefill` is **`flash_c128_prefill`**, and `flash_c4_prefill`
appears as **two** template instantiations (0.163 + 0.143 = 0.306 ms over 60
calls) — the digit-normalising in `trace_common.norm_name` had merged and
obscured both.

### moe — 35.49 ms/step, 47.2 %, 3 kernels

| ms/step | calls | µs/call | kernel |
|---:|---:|---:|---|
| 16.244 | 61 | 266.3 | `megamoe_prepare_compact_m32_dcu32_pcu1_pc384_qcu28qcap256_fov_runtime_dyn_tss12488_v13` |
| 13.799 | 61 | 226.2 | `megamoe_stage1_compact_t32x512x256_w8_gm1_dcu32_pw1ma1sw1_cgc256aa1_tr1wpe2_bnt3_ws4_pc384_tss12488_rc31` |
| 5.444 | 61 | 89.2 | `megamoe_stage2_compact_t32x256x256_sbm32_fp8_nt1_p1cu240s0_pad0_sk0_bh1apf1sp4x2_bf16lds0_fp8_blockwise` |

All three are exactly 61 calls — one per layer. B200's whole MoE is one fused
`mega_moe_impl` at 14.85 ms, i.e. less than `prepare_compact` alone.

### attn — 15.67 ms/step, 20.8 %, 15 kernels

| ms/step | calls | µs/call | kernel |
|---:|---:|---:|---|
| 9.974 | 61 | **163.5** | `_paged_decode_split_kernel` — the MLA decode core, 4.06x B200 |
| 1.250 | 121 | 10.3 | `aiter::mhc_fused_post_pre_gemm_sqrsum_kernel` [mangled] |
| 1.092 | 30 | 36.4 | `pa_mqa_logits_fp4_prefill_kernel_0` — DSv4 indexer logits |
| 0.894 | 121 | 7.4 | `aiter::mhc_pre_big_fuse_rmsnorm_kernel` |
| 0.831 | 61 | 13.6 | `_paged_decode_reduce_kernel` — split-K reduction |
| 0.830 | 30 | 27.7 | `sglang::topk_main_kernel` — indexer top-k |
| 0.271 | 61 | 4.4 | `sglang::fused_norm_rope_flashmla` |
| 0.163 | 30 | 5.4 | `sglang::flash_c4_prefill` (instantiation 1) |
| 0.151 | 31 | 4.9 | `sglang::flash_c128_prefill` |
| 0.143 | 30 | 4.8 | `sglang::flash_c4_prefill` (instantiation 2) |
| 0.039 | 1 | 38.9 | `_init_compressed_attn_metadata_kernel` |
| 0.036 | | | +4 kernels below 0.02 ms |

**64 % of `attn` is one kernel**, and it is the target named in §10.

### gemm — 9.36 ms/step, 12.4 %, 12 kernels

No single dominant kernel; this is four different GEMM backends coexisting
(aiter asm, opus flatmm, CK, rocBLAS/Tensile).

| ms/step | calls | µs/call | kernel |
|---:|---:|---:|---|
| 4.049 | 183 | 22.1 | `aiter::fp8gemm_bf16_blockscale_BpreShuffle_80x128` |
| 1.378 | 61 | 22.6 | `gemm_a8w8_mxscale_flatmm_splitk_kernel` [mangled] |
| 1.012 | 61 | 16.6 | `ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle` [mangled] |
| 0.502 | 61 | 8.2 | `_gemm_a8w8_blockscale_preshuffle_kernel` (M128/N32) |
| 0.452 | 61 | 7.4 | `hgemm_bf16_t32x32x64x6_ksd_w1x2x1` |
| 0.386 | 30 | 12.9 | `hgemm_bf16_t64x64x64x5_ksd_w4x2x1` |
| 0.311 | 30 | 10.4 | `_gemm_a8w8_blockscale_preshuffle_kernel` (M32/N32) |
| 0.311 | 31 | 10.0 | `hgemm_bf16_t48x64x128x3_ksd_w1x4x1` |
| 0.278 | 1 | 277.8 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_MT256x80x128` — rocBLAS/Tensile, once per step |
| 0.275 | 61 | 4.5 | `_gemm_a8w8_blockscale_reduce_kernel` (split-K reduce) |
| 0.253 | 30 | 8.4 | `hgemm_bf16_t48x32x64x6_ksd_w1x2x1` |
| 0.158 | 30 | 5.3 | `hgemm_bf16_t32x32x64x8_ksd_w2x2x1` |

### comm — 6.43 ms/step, 8.5 %, 2 kernels

| ms/step | calls | µs/call | kernel |
|---:|---:|---:|---|
| 6.166 | 61 | 101.1 | **`ep_combine_intranode_0`** — the real EP-combine all-to-all |
| 0.262 | 61 | 4.3 | `_fused_clamp_silu_mul_kernel` — MoE activation, belongs in `moe` |

### copy — 3.98 ms/step, 5.3 %, 15 kernels

| ms/step | calls | µs/call | kernel |
|---:|---:|---:|---|
| 0.803 | 67 | 12.0 | `at::native::elementwise_kernel_manual_unroll` (direct_copy) |
| 0.757 | 183 | 4.1 | `_fill_padded_rows_kernel` — MoE row padding, no B200 counterpart |
| 0.625 | 66 | 9.5 | `at::native::elementwise_kernel_manual_unroll` (direct_copy, 2nd inst.) |
| 0.533 | 122 | 4.4 | `__amd_rocclr_fillBufferAligned` — hipMemset backing kernel |
| 0.392 | 92 | 4.3 | `at::native::vectorized_elementwise_kernel` (bf16→fp32 copy) |
| 0.363 | 69 | 5.3 | `at::native::index_elementwise_kernel` |
| 0.271 | 61 | 4.4 | `_swa_scatter_kernel` |
| 0.143 | 31 | 4.6 | `_fill_compress_tail_kernel` |
| 0.055 | 10 | | `Memcpy DtoD` + `__amd_rocclr_copyBuffer` |
| 0.034 | | | +5 kernels below 0.02 ms |

This is the bucket B200 has essentially nothing in (0.06 ms). It is
**7 distinct aten/rocclr fill-and-copy kernels plus MoE row padding**, none of
it fused away.

### quant — 2.21 ms/step, 2.9 %, 12 kernels

| ms/step | calls | µs/call | kernel |
|---:|---:|---:|---|
| 0.949 | 212 | 4.5 | `aiter::dynamic_per_group_scaled_quant_kernel` [mangled] |
| 0.365 | 61 | 6.0 | `_wo_a_quant_mxfp8_kernel` |
| 0.297 | 62 | 4.8 | `at::native::elementwise_kernel_manual_unroll` (nocast) |
| 0.266 | 61 | 4.4 | `_fused_rms_fp8_group_quant_kernel` |
| 0.171 | 30 | 5.7 | `aiter::norm_rope_hadamard_rotate_activation_fp4quant_kvcache_kernel` [mangled] |
| 0.131 | 30 | 4.4 | `aiter::rope_hadamard_rotate_activation_fp4quant_kernel` [mangled] |
| 0.030 | | | +6 kernels below 0.02 ms |

### norm_rope — 0.72 ms/step, 1.0 % · sample — 0.40 ms/step, 0.5 %

`_fused_qk_norm_rope_store_kernel` 0.442 (61) and `apply_rotary_emb_flat_kernel`
0.278 (61); `aiter::topk_gating_kernel_opt` 0.387 (58) plus one below 0.02.

### other — 0.99 ms/step, 1.3 %, 37 kernels

`at::native::vectorized_elementwise_kernel` (CUDAFunctor_add) 0.268 (61),
`sglang::write_c4_prefill` 0.132 + 0.127 (30 each, two instantiations),
`sglang::write_c128_prefill` 0.131 (31), `_hc_head_kernel` 0.052 (1), then a
long tail of 30 kernels totalling 0.233 ms. The `write_c*_prefill` family is
the DSv4 compressed-KV write and lands in `other` on **both** nodes.
