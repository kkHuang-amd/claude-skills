# FINDINGS — joint source of truth (B200 + MI355X)

Append only. Sign every block with node + date so a superseded number stays
identifiable. Detail and provenance live in the per-node `<node>-<topic>.md`
files; this file carries conclusions.

---

## ⚠ WITHDRAWN — the block below mixed DSPARK draft and full-model verify steps
*(mi355x, 2026-09-16, superseding its own entry from earlier the same day)*

Speculative decoding emits **two** `step[TARGET_VERIFY]` annotations per
`scheduler.run_batch`, at the **same `bs`**, so grouping by (type, bs) averaged a
4.0 ms draft step with a 74.0 ms full-model step. Corrected numbers, MI355X c128
pdi=24 steady state:

| | withdrawn | correct |
|---|---:|---|
| verify step wall p50 | 39.3 ms | **74.0 ms** full, 4.0 ms draft |
| `compute` p50 at bs=10 | 14.85 ms | **28.36 ms** |
| MoE calls/step | 32 | **61** full, 3 draft |

**"`compute` matches B200 within 1.4 %" is withdrawn.** MI355X's full-step
`compute` is 28.36 ms at bs=10 (24.94-38.91 across ranks bs 9-20).

**B200's published numbers are very likely affected the same way** — same tool,
same grouping, and B200's `n=38` verify steps at a single bs fit the same
2-per-batch pattern. Re-derive before either side quotes a ratio.

Resolved on the way: MoE calls/step **match** at 61 (one per layer); the earlier
"32 vs 61-64" discrepancy was the mixing artefact, so no renormalisation is
needed. Also, the "bimodal, quote p50 not mean" advice in both docs was
misdiagnosing this bug — split by class and each cluster is tight.

Fixed in `analysis/trace_common.py:verify_classes()` (splits by MoE call count,
not annotation order) and wired into `trace_summary.py`. `trace_ranks.py` is
**not** yet fixed and still mixes the classes.

What survives: the steady-state-vs-mid-ramp finding below, the stream-count
measurement (2-3 vs 132), the group-wide prefill barrier bound, and the
scheduler-log numbers, none of which depend on the split.

<details>
<summary>Withdrawn text, kept for traceability</summary>

## MI355X steady-state decode step: 39.3 ms — and `compute` nearly matches B200
*(mi355x, 2026-09-16 — detail in `mi355x-decode-trace.md`)*

At matched pdi=24, accept len (3.78 vs 3.770), per-request KV working set
(~151k vs ~152k) and 100 % cuda-graph replay, MI355X's steady-state
`TARGET_VERIFY` step wall is **39.3 ms p50 at bs=10**, captured at 167,538
tokens/request with pool usage plateaued. Flat across bs 9-20 (38.7-41.7 ms).

**No ratio is quoted against B200's 15.9 ms** — that number is retracted as
mid-ramp and B200 is re-capturing. 39.3 ms is simply the MI355X side.

**The most important number is not the wall, it is `compute`:** MI355X 14.85 ms
vs B200 15.06 ms at bs=10 — **within 1.4 %**. `compute` is barrier-free by
construction, so on present evidence the two platforms do comparable decode
compute per step, and the wall-clock gap lives in `barrier` (MI355X 21.8 ms,
~55 % of its step). That is the opposite of a kernel-efficiency story and should
be the next thing both nodes confirm, since the two captures are still not
like-for-like.

**Stream count differs sharply but is probably not the cause.** MI355X runs 2-3
streams against B200's 132, and its kernel sum equals its wall (1.00x overlap vs
B200's 1.54x). But B200's own 132->4 A/B cost only 4-5 % of step time, and
MI355X sits at essentially B200's single-stream setting — so stream count cannot
carry a gap of this size. Recorded as a measured difference, not a cause.

Open, and blocking a clean answer:

1. **Both captures are contaminated by the group-wide prefill barrier.** In the
   MI355X steady capture, ranks 0/1/3/6/7 have `TARGET_VERIFY` while **all eight**
   have `EXTEND`. Rank 6 is the extreme: barrier 174.69 ms of a 195.97 ms step.
   The two nodes need to report the same thing — either both filtered to steps
   with no concurrent `EXTEND` in the group, or both quoting `compute`.
2. **`gemm` is not comparable as bucketed.** B200 19.39 vs MI355X 9.34 ms/step
   p50 is probably naming: `deep_gemm::..._mega_moe_impl` matches `gemm` while
   `megamoe_stage1/2_compact` matches `moe` first. Use `gemm + moe` as one
   bucket until settled — B200 31.61 vs MI355X 43.21.

</details>

---

## The fixed 120 s settle is unsafe on both nodes — use a steady-state predicate
*(mi355x, 2026-09-16, confirming B200's retraction from its own data)*

MI355X's first capture used the prompted rule (first `done=` + 120 s) and landed
at 67,759 tokens/request, less than half steady state. Measured cost on one run:

| | mid-ramp | steady | delta |
|---|---:|---:|---:|
| per-request KV | 67,759 | 167,538 | 2.47x |
| step wall p50 | 28.3 ms | 39.3 ms | **+39 %** |
| `attn` p50 | 7.95 ms | 15.62 ms | +96 % |
| `moe` p50 | 24.23 ms | 33.87 ms | +40 % |

`attn` scaling with context is expected. **`moe` growing 40 % is not** — MoE work
is context-independent, so that is barrier absorbed into the fused all-to-all,
i.e. more waiting at steady state.

Replace the constant with a predicate on `server.log`: wait for per-request
`#full token` / `#running-req` to plateau (>= ~130k on these arms). A fixed sleep
is racing a ramp whose length depends on trace content and page-cache warmth.

---

## prefill_decode_interval was a launcher difference, and it is now matched
*(mi355x, 2026-09-16)*

`dsv4_fp4_mi355x_sglang_mtp.sh` now emits B200's values: pdi 24 (20 plus
`--balance-abs-threshold 32` at `CONC>=160`), `--load-balance-method
total_requests`, router `--policy cache_aware`. Every cross-platform ITL number
taken on MI355X before 2026-09-15 used pdi=10 and is not comparable.

Measured on MI355X at c128, the knob alone is a clean ITL<->TTFT trade, same
direction as B200's A/B: ITL p90 50.6 -> 33.9 ms (-33 %), TTFT p50 2.52 -> 4.01 s
(+59 %), and the same trade reproduces at c256 (ITL -30 %, TTFT p50 +187 %).

**The throughput half of that is not established.** tok/s/chip rose 6.6 % at
c128 and 4.7 % at c256, both inside the ~5 % replicate spread, and both are
partly explained by ISL mean drifting up (+2.6 % / +1.6 %) — the metric is ~99 %
input tokens, so longer prompts inflate it without the engine doing better.
Normalising to requests/s leaves +3.9 % / +3.0 %. What *is* solid is output
throughput, +8.3 % / +9.1 %, consistent with the ITL win.

---

## KV pool usage must not be matched across platforms
*(mi355x, 2026-09-16 — corrects a misreadable line in `analysis/METHOD.md`)*

`full token usage` is a fraction of each node's own pool capacity, and the pools
differ 5.4x (B200 2,217,472 tokens, MI355X ~12,074,667). Matching the fractions
would force genuinely different working sets. Match the **absolute** figure,
`#full token` / batch: B200 151,908 vs MI355X ~151,000 per request, i.e. aligned
within 3 %, while the fractions read 0.62 vs 0.15. METHOD.md's
"align kv pool usage" applies to partial-vs-complete runs on one node.

---

## MI355X 74 ms full-verify wall is already EXTEND-free
*(mi355x, 2026-09-16 — local, no B200 needed)*

Filter: `TARGET_VERIFY full` whose GPU-annotation interval overlaps **no**
`EXTEND` on any of the 8 ranks. Clocks align across ranks (shared ~6.56e12 µs
origin). Result on the steady c128 pdi=24 capture:

```
rank n_full n_free n_hit  p50   mean  min   max
   0     16     16     0  74.1  75.2  73.5  78.9
   1     16     16     0  74.0  75.2  73.4  79.0
   3     16     16     0  74.0  75.2  73.4  78.9
   6     17     17     0  73.9  75.0  72.7  78.9
   7     16     16     0  74.0  75.2  73.4  79.0
p50 spread 73.9-74.1 ms (1.002x)   n_hit=0
```

**74.0 ms does not drop.** It was already an upper bound only in the weak
sense that every rank's file also contains an `EXTEND`; in time, those
`EXTEND`s finish *before* the 16-step verify burst (ranks 0/1/3/6/7) or start
~10 s *after* it (ranks 2/4/5, 0 overlapping kernels). The 31-45 ms `barrier`
inside the 74 ms step is therefore **not prefill-EXTEND wait**. It still
anti-correlates with `compute` across the five decoding ranks (rank 1: 38.97
compute / 31.85 barrier; rank 3: 24.95 / 45.48; wall locked at ~75 ms mean),
i.e. it is the DP/EP wait among the ranks that are actually decoding.

`trace_ranks.py` now splits draft/full and prints this section. Rank 6's 5321 ms
annotation is a **draft** (3 MoE calls), not a full step — exclude it from
full-verify stats; it does not overlap any `EXTEND` either.

B200 still must re-derive (steady state + class split + this filter) before a
ratio. Detail: `mi355x-decode-trace.md` §6.

---

## Intermittent EPLB rebalance deadlock on MI355X
*(mi355x, 2026-09-16)*

Roughly 1 run in 3 on an EPLB arm hangs with a signature that defeats every
liveness check: `returned=` frozen, `errors=0`, `/metrics` answering 200,
schedulers alive, VRAM still allocated. Last server activity is
`Resetting ExpertDistributionRecorder...` from all 8 ranks; all 8 then hang in a
collective and the NCCL watchdog kills the process 600 s later with
`c10::DistBackendError`. Nothing reaches launcher stdout. Judge liveness only by
`returned=`/`done=` moving. A straight retry cleared it.

---

## RESOLVED: the gap is inside the decode step, and it is not compute
*(b200, 2026-09-16, against mi355x's steady-state capture the same day)*

Both nodes now have a steady-state, class-split capture at pdi=24 with matched
KV working sets (**B200 165,220 tok/req vs MI355X 167,538, 1.4 % apart**). This
is the first genuinely controlled comparison; every earlier one mismatched the
ramp position, the step class, or both.

| per full-model verify step | B200 (bs=12) | MI355X (bs=10) | ratio |
|---|---:|---:|---:|
| **step wall p50** | **30.0 ms** | **74.0 ms** | **2.47x** |
| draft step wall | 2.0 | 4.0 | 2.0x |
| summed kernel | 50.18 | 75.24 | 1.50x |
| `compute` (attn+gemm+quant+norm_rope+sample) | 31.52 | 28.37 | ~~0.90x~~ **not comparable, see §2** |
| `barrier` (moe+comm) | 14.42 | 40.85 | ~~2.77x~~ **not comparable, see §2** |
| summed kernel / wall (overlap) | **1.67x** | **1.00x** | |

### 1. The discriminator is answered — and it inverts the earlier reading

| | B200 | MI355X | ratio |
|---|---:|---:|---:|
| log-implied step (scheduler) | 79.2 ms | 121.76 ms | 1.54x |
| decode step (full + draft) | 32 ms | 78 ms | **2.44x** |
| remainder (prefill, gaps) | 47.2 ms | 43.8 ms | **0.93x** |

The decode step accounts for **all** of the 42.6 ms log-implied gap — 46 ms of
it — while the portion outside the decode step is slightly *better* on MI355X.
**The prefill barrier is not where the gap lives.** The old "only 22 % of B200's
wall is inside decode steps" pointed the other way and was an artefact of the
retracted 15.9 ms.

### 2. WITHDRAWN — "compute is equal, so it is not the kernels"

*(withdrawn by b200, 2026-09-16, one revision after it was published)*

This block said `compute` was 28.37 ms on MI355X against 31.47 on B200, so the
kernels were not the problem and the whole gap was `barrier`. **That comparison
is invalid: B200's `compute` is pace-pinned and contains absorbed wait, so it is
an upper bound on B200's real compute rather than a measurement of it.**

The evidence is in `b200-decode-trace.md` §"`compute` is pace-pinned": across
bs 1 → 12 on B200, `attn` gains 3.42 ms while `gemm` *loses* 3.04 ms and `moe`
3.10 ms, leaving `compute` flat at 31.0-31.7 and the wall at 30.0. Real GEMM
work cannot shrink as batch grows — under DP attention the GEMM segment runs on
a padded token count and should be constant. The slack is being spent inside
deep_gemm kernels, which carry a device-side cross-rank grid sync.

MI355X's `compute` does *not* look pinned — it spans 24.9-38.9 across ranks —
so the two nodes' `compute` figures are not measuring the same thing and must
not be divided.

**What survives:** §1 is untouched. The decode step is **2.47x** slower on
MI355X (wall 30.0 vs 74.0 ms) and accounts for the entire log-implied gap, while
the portion outside it is 0.93x. Wall time is elapsed time and needs no
barrier-free assumption.

**What reopens:** whether that 2.47x is kernel work or intra-step wait. Both
nodes attribute wait into kernel durations, so no role-level split currently
answers it. Needed: an estimator of real compute that is immune to absorbed
wait — e.g. a single-rank / TP-only run with no DP group to sync with, or
per-kernel achieved FLOPs/bandwidth against peak, which `record_shapes` already
makes possible and which nobody has computed on either node.

### 2b. Where the 44 ms actually goes, matched at bs=10

*(b200, 2026-09-16. B200 = mean of TP-4 n=15 and TP-7 n=10 at bs=10; MI355X =
rank 7 bs=10. All p50.)*

The wall gap splits in two before any role is named:

| | ms |
|---|---:|
| wall, B200 → MI355X | 30.0 → 74.0 (**+43.95**) |
| of which: more kernel time | **+23.65** |
| of which: overlap B200 gets and MI355X does not | **+20.30** |

B200 fits 50.52 ms of kernels into a 30.0 ms wall (hides 20.48); MI355X fits
74.17 into 74.0 (hides 0.17).

The +23.65 ms of kernel time attributes as follows. **B200's roles are
reclassified first** — a full per-kernel dump (below) showed `other` holding
2.04 ms of nvjet/cuBLAS GEMM and 0.74 ms of `silu_mul_clamp` MoE activation, so
those are moved into `gemm` and `moe`:

| bucket | B200 | MI355X | delta | % of gap | comparable? |
|---|---:|---:|---:|---:|---|
| attn | 7.19 | 15.67 | **+8.48** | **36.4 %** | composition unverified on MI355X |
| comm | 0.67 | 6.48 | +5.81 | 24.9 % | **no — see below** |
| gemm+moe | 39.63 | 43.74 | **+4.11** | **17.6 %** | contains absorbed wait both sides |
| copy | 0.06 | 3.97 | +3.91 | 16.8 % | likely real |
| quant | 1.52 | 2.21 | +0.69 | 3.0 % | yes |
| sample+norm_rope+other | 1.79 | 2.10 | +0.31 | 1.3 % | yes |

**⚠ WITHDRAWN: "`attn` is the largest difference, not MoE".** This whole table
is built on summed kernel time, which double-counts B200's concurrency. On
elapsed (`credited`) time `moe` is +27.67 and 61.5 % of the gap while `attn` is
+9.44 — see the `busy_ms.py` block at the end of this file. The table is kept
only so the withdrawal is traceable; do not quote its deltas.

**The `comm` row is not a comparison and must not be quoted.** B200's `comm`
bucket contains **no cross-rank collective at all**: it is
`flash_fwd_mla_combine_kernel` (0.428 ms, attention split-K reduction) and
`mega_moe_pre_dispatch_kernel` (0.242 ms, MoE dispatch prep). B200's actual
all-to-all is fused inside `mega_moe_impl`. Whether MI355X's 6.48 ms is a real
collective is unknown from its published table.

**`copy` is probably real, and an earlier revision was wrong to explain it away
as a classification artefact.** That argument assumed B200's `other` held
fill-type kernels; the dump shows it holds GEMM and activation instead. MI355X's
`fillbuffer` / `fill_padded_rows` / `fill_compress_tail` have no B200
counterpart.

#### B200 per-kernel composition, TARGET_VERIFY full bs=10, TP-4, n=15

Summed kernel 50.87 ms/step. Only the B200 side can be listed here: MI355X's
raw traces (`mi355x:/shared_nfs/kk/pr35619/trace_c128_pdi24_steady/`) are not
reachable from B200 — this node's `/shared_nfs` is
`10.238.19.129:/AI55QU/models`, six model directories and no `kk/`, a different
export that merely shares the mount point. Every MI355X number in this document
comes from its published tables in `mi355x-decode-trace.md`, not from
re-analysis.

- **gemm 21.99** (3 kernels): `deep_gemm::smN_fp8_fp4_gemm_1d1d_impl` 20.17
  (396 calls), `cublasLt::splitKreduce_kernel` 1.25 (182),
  `deep_gemm::smN_tfN_hc_prenorm_gemm_impl` 0.57 (122)
- **moe 14.86** (2): `deep_gemm::smN_fp8_fp4_mega_moe_impl` 14.85 (61),
  `moe_hash_topk_fused` 0.02
- **attn 7.19** (16) — the real MLA decode kernel is only a third of it:
  `flash_fwd_splitkv_mla_fp8` 2.454 (61),
  `mhc_pre_big_fuse_with_norm_tilelang` 0.948 (122),
  `deep_gemm::smN_paged_mqa_logits` 0.901 (30),
  `mhc_post_tilelang` 0.737 (125), `topk_persistent_cluster` 0.357 (30),
  `topk_main` 0.353 (30), `flash_c4_prefill` 0.298 (60),
  `fused_norm_rope_indexer_fp4` 0.214 (30), flashinfer `RMSNormKernel` 0.211
  (62), `flash_cN_prefill` 0.193 (31), `fused_k_norm_rope_flashmla` 0.182 (61),
  `fused_norm_rope_flashmla` 0.138 (61),
  `fused_q_indexer_rope_hadamard_fp4_quant` 0.116 (30), + 3 below 0.05
- **other 3.87** (33) — **2.78 ms of this is mis-bucketed**:
  `silu_mul_clamp_kernel` 0.740 (61, MoE activation),
  `nvjet_smN_tss_*` / `nvjet_smN_tst_*` 2.044 total (GEMM),
  `_router_triton_kernel` 0.379 (58), `write_c4_prefill` 0.153 (60),
  `at::native::vectorized_elementwise` 0.275, `write_cN_prefill` 0.079 (31),
  + 22 kernels totalling ~0.19
- **comm 0.67** (2): listed above, neither is a collective
- **quant 1.52**: `per_token_group_quant_flat_kernel` 1.191 (335),
  `fp8_wo_a_group_major_quant_ue8m0` 0.326 (61)
- **norm_rope 0.56**: `fused_q_norm_rope` 0.305 (61), `deepseek_rope_kernel`
  0.258 (61)
- **sample 0.14**: `mask_topk_ids_padded_region` 0.141 (61)
- **copy 0.06**: one `Memcpy DtoH`; 12 kernels, all ≤0.03

**Request to MI355X: the same per-role kernel listing at bs=10.** Three things
cannot be settled without it — how much of `attn` 15.67 is attention math
versus indexer/topk/norm helpers, whether `comm` 6.48 is a genuine collective,
and where the DSv4 compressed-KV kernels (`flash_c4_prefill` / `write_c4_prefill`
family) land on ROCm. B200 splits them across `attn` and `other`.

### 2c. Which B200 kernels absorb wait — measured, not assumed

Same kernel, same call count, *less* work, *more* time:

| kernel | bs=1 | bs=12 | calls/step | verdict |
|---|---:|---:|---:|---|
| `deep_gemm::smN_fp8_fp4_gemm_1d1d_impl` | 22.48 | 19.36 | 396 both | **absorbs** (56.8 → 48.9 µs/call) |
| `deep_gemm::smN_fp8_fp4_mega_moe_impl` | 17.22 | 13.84 | 61 both | **absorbs** |
| `flash_fwd_splitkv_mla_fp8_*` (attn) | 1.14 | 2.90 | 61 both | pure — scales with work |

So **two** deep_gemm kernels absorb the group wait on B200, not just the MoE one;
`attn`, `quant`, `norm_rope`, `comm` and `copy` track their own work. This is
what makes the `attn` and `comm` rows above usable and the `gemm+moe` row not.

### 3. Candidate A — DP load imbalance: NOT ESTABLISHED, and the earlier
### framing of it was invalid

An earlier version of this block compared a B200 `compute` spread of 0.6 ms
against an MI355X spread of 14 ms (24.9-38.9) and called the difference
imbalance. **That comparison does not hold: each rank sits at its own `bs`, so
rank and batch size move together and neither table separates them.**

What *is* established, because B200's capture happens to repeat several `bs`
values across different ranks — **at fixed `bs`, B200's cross-rank spread is
0.0-0.3 ms**:

| bs | ranks | `compute` p50 | spread |
|---:|---|---|---:|
| 9 | TP-4 / TP-6 / TP-7 | 31.48 / 31.16 / 31.36 | 0.32 |
| 10 | TP-4 / TP-7 | 31.45 / 31.60 | 0.15 |
| 11 | TP-0 / TP-4 | 31.38 / 31.29 | 0.09 |
| 12 | TP-0 / TP-4 | 31.47 / 31.48 | 0.01 |

MI355X's published table has one `bs` per rank and no repeats, so the matching
statement cannot be made there. **Until it is, there is no measured claim that
MI355X's imbalance exceeds B200's.**

To settle it, MI355X should report `compute` for two or more ranks *at the same
`bs`* within one capture, or bucket `compute` by bs the way `decode_stats.py`
buckets `step_ms`.

### 3b. The overlapping batch sizes — controlled on bs, but still not comparable

`bs` 9 and 10 appear on both nodes, so these rows are matched on batch:

| bs | B200 `compute` p50 | MI355X `compute` | ratio |
|---:|---|---:|---:|
| 9 | 31.14-31.45 | 24.94 | 0.79-0.80x |
| 10 | 31.47-31.67 | 28.36 | 0.90x |

**Matching on bs does not rescue this comparison.** Per §2, B200's `compute` is
pace-pinned, so these ratios measure B200's *slack* as much as either node's
work. Recorded only so the next person does not re-derive them and draw the
conclusion we just withdrew.

The one asymmetry worth keeping from these rows: B200's `compute` barely moves
between bs 9 and 10 (31.14-31.67) while MI355X's moves 24.94 → 28.36 (+13.7 %).
Consistent with B200 being pinned and MI355X tracking real work — which is the
§2 argument again, not independent evidence for it.

### 4. Candidate B — the "stream overlap is worth 4-5 %" bound does NOT apply here

**This retracts how both docs used that A/B.** B200 fits 50.18 ms of kernels
into a 30.0 ms wall (**1.67x**); MI355X fits 75.24 into 74.0 (**1.00x**). The
B200 flag A/B that measured 4-5 % moved the ratio only **1.49x -> 1.33x** (132
streams -> 4) — it never reached 1.00x, so it bounds the cost of *fewer*
streams, not of *no overlap at all*. MI355X sits outside the range that
experiment explored, and citing 4-5 % to dismiss overlap is unsupported.

The wall gap decomposes cleanly into the two candidates:
**1.50x (more kernel time) x 1.67x (less overlap) = 2.50x, against 2.47x
measured.**

### 5. What each side should do next

- **MI355X:** publish `compute` for two or more ranks **at the same `bs`**, so
  rank and batch stop moving together — that is what §3 needs and it needs no
  new run, only a regroup of the existing capture. Then a `compute`-vs-bs curve
  for §3b. And establish whether any kernel overlap is reachable at all on ROCm,
  since 1.00x is the single largest multiplier in the table.
- **B200:** batch ranges barely overlap (B200 bs 1-12, MI355X 9-20, common
  ground only at 9 and 10). Re-capture at a larger batch to extend the
  `compute`-vs-bs curve into MI355X's operating range.
- **Both:** the open question is now an estimator of real decode compute that
  absorbed wait cannot contaminate. Two candidates, neither attempted:
  a **TP-only / single-DP-rank run** of the same shapes (no group to sync with,
  so nothing to absorb), or **achieved FLOPs/bandwidth against peak** per
  kernel, which `record_shapes` already supports. Until one exists, quote step
  wall — it is elapsed time and needs no barrier-free assumption.
- **Both:** always publish step wall, its `bs`, *and* the capture window's KV
  working set together. Two of the three were missing from every capture before
  today, and that is what made three separate numbers wrong.
- **Both:** `trace_summary.py` prints `ms/step` (a mean) and `p50` as separate
  columns. B200 published the mean under a `p50` header for one revision. State
  which you are quoting.

---

## The decode gap is one attention kernel plus the MoE prepare stage
*(mi355x, 2026-09-16 second session, answering b200's §2b / §3 / §5 requests.
Detail and the full per-kernel listing in `mi355x-decode-trace.md` §10-12.)*

MI355X's per-kernel dump at bs=10 is now published, reclassified the same way
B200 reclassified its own (plus `flash_fwd_mla_combine` → `attn` and
`mega_moe_pre_dispatch` → `moe` on B200, and `_fused_clamp_silu_mul` → `moe`
on MI355X, so that `comm` means only a real collective on both sides).

| bucket, ms/step mean | B200 TP-4 n=15 | MI355X TP-7 n=16 | delta |
|---|---:|---:|---:|
| moe | 15.84 | **35.75** | **+19.91** |
| attn | 7.62 | **15.67** | **+8.05** |
| gemm | 24.03 | 9.36 | −14.67 |
| comm (real collective) | 0.00 | 6.17 | +6.17 |
| copy | 0.06 | 3.98 | +3.92 |
| quant + sample + norm_rope + other | 3.31 | 4.32 | +1.01 |
| **summed kernel** | **50.86** | **75.25** | **+24.39** |

Read `gemm+moe` as one bucket per B200's §2c: **39.87 → 45.11, +5.24 (1.13x)**,
consistent with B200's +4.11. Rolling the exposed collective in as well gives
**gemm+moe+comm 39.87 → 51.28, +11.41 (1.29x)**.

**Two named targets, and they are narrow:**

1. **The MLA decode kernel is 4.06x.** `flash_fwd_splitkv_mla_fp8` 2.454 ms
   (61 calls) on B200 against `_paged_decode_split_kernel` 9.974 ms (61 calls)
   on MI355X — 40.2 vs 163.5 µs/call at one call per layer on both. That single
   kernel is **+7.52 of the +8.05 ms `attn` gap (93 %)**. Every other
   attention-family kernel matches within 0.78-1.94x, including the DSv4
   compressed-KV pair `flash_c4_prefill` (0.298 vs 0.307, within 3 %).
2. **`megamoe_prepare_compact` 16.24 ms has no B200 counterpart.** MI355X's MoE
   is three kernels (prepare 16.244 / stage1 13.799 / stage2 5.444, 61 calls
   each) where B200 has one fused `mega_moe_impl` 14.85. The prepare stage alone
   exceeds B200's entire MoE time and holds 66 % of the `moe` gap. B200's
   nearest-named kernel, `mega_moe_pre_dispatch`, is 0.242 ms.

**Answers to the three open asks:**

- **`comm` 6.48 is a genuine collective** — one kernel, `ep_combine_intranode_0`
  6.166 ms / 61 calls, an aiter EP-combine all-to-all. B200's `comm` has no
  collective at all (its a2a is fused in `mega_moe_impl`), so the two `comm`
  rows are not comparable in either direction; MI355X's is *exposed* time that
  B200 hides inside a `moe` kernel.
- **Compressed-KV kernels land identically on both nodes** — `flash_c4/cN_prefill`
  → `attn`, `write_c4/cN_prefill` → `other`. That asymmetry does not exist.
- **`bs` never repeats across ranks on MI355X**, in either capture, so §3's
  request cannot be met as posed. **Rank 0 spans bs 17-20 on its own**, which
  fixes the rank instead, and gives the same test: `attn` +34 %, `gemm` +12 %,
  `quant` +2 %, `comm` flat, `moe` **−3.8 %**. **Only `moe` absorbs wait on
  MI355X**, where on B200 both `gemm` and `moe` do. So §2's withdrawal is
  **asymmetric**: B200's `compute` is an upper bound, MI355X's is a measurement.
  The ratio stays withdrawn — one usable side is not a comparison.

**New: `bs` is a bad control at steady state, and the mid-ramp capture misled.**
Mid-ramp `compute` is monotone in `bs` (4 → 17.15 … 13 → 25.14), so there B200's
confound objection is exactly right. Steady state is **not** monotone: rank 1 at
bs=14 does 38.91 ms of compute against rank 0 at bs=18 doing 28.78. The driver
is `attn` (25.20 vs 14.26), i.e. **KV tokens, not request count** — `bs` ×
context is not monotone in `bs`. So residual per-rank variation of ~1.35x
survives controlling for `bs`, and what this needs is `compute` against per-rank
KV tokens, which no trace annotation carries.

**And the step is paced, with `barrier` as pure slack.** On the five decoding
ranks, `compute` spans 1.56x (24.94-38.91) while `compute + barrier` holds
within 0.9 % (69.07-69.66) at a ~74.0 ms wall. The straggler is rank 1 at
38.91 ms; even it spends 30.73 ms in `moe`+`comm`.

**⚠ Correction to §5: `record_shapes` cannot give the bandwidth estimator.** Of
177,036 events in the rank-7 file, 34,486 carry `Input Dims` and **all of them
are `aten::*` `cpu_op`s** — zero attention or MoE ops. The aiter/sglang kernels
holding the time are launched through custom ops that never register shapes. The
estimator needs shapes plumbed into those ops, or dims from the model config
plus per-rank KV counts. The **TP-only / single-DP-rank run** is still viable and
is now the cheapest uncontaminated compute number available to either node.

Also, and separately from all of the above: **the 74.0 ms wall is already free
of concurrent prefill** — see the EXTEND-overlap block above. `n_hit=0` on all
five decoding ranks.

**Reproduce:** `python3 analysis/kernel_dump.py <trace dir>` prints the
per-(rank, bs) table, the repeated-`bs` check, and the per-role kernel listing.

---

## ⚠ Every per-role table on both sides subtracts kernel-seconds from elapsed seconds
*(mi355x, 2026-09-16, correcting its own block above and b200's §2b. Detail in
`mi355x-decode-trace.md` §13.)*

All per-role numbers published so far are **sums of kernel durations**, which
double-count concurrency. B200 reports 50.86 ms of summed kernel inside a
**30.0 ms wall**, so the union of its kernel intervals is **≤30.0 ms**: B200's
elapsed decode work is a ~30 ms quantity, and its `gemm` 24.03 / `moe` 15.84
cannot each occupy that much of a 30 ms step. **MI355X's sums are elapsed
time; B200's are not. The two were being subtracted from each other.**

New tool `analysis/busy_ms.py` sweeps the intervals and reports per role
`sum` / `union` / `exclusive` / `credited`, plus total busy and idle-in-step.
MI355X, rank 7, `TARGET_VERIFY full` bs=10, n=16:

| | ms/step |
|---|---:|
| step wall | 75.23 |
| summed kernel | 75.24 |
| **union (GPU busy)** | **75.22** |
| sum / busy | **1.000x** |
| idle inside the step | **0.01** |

Per role, `sum` = `union` = `exclusive` on every bucket, on all five decoding
ranks (sum/busy 1.000-1.009x). **The MI355X decode step is perfectly serial with
no idle** — no two kernels ever overlap and there is no gap between them. So
MI355X's side needs no correction; B200's does.

**Consequences:**

- **The elapsed gap is ≥2.47x, the wall ratio** — ≤30.0 vs 74.0 ms. The
  "+24.39 ms more kernel time" and the "1.50x kernel × 1.67x overlap"
  decomposition remain valid *as a decomposition of the wall*, but are not
  elapsed contributions.
- **Every bucket delta is understated and `gemm` may flip sign.** Uniform 1.67x
  scaling of B200 (not what actually happens) would give moe +26, attn +11,
  gemm −5 instead of +19.91 / +8.05 / −14.67. The real split is whatever the
  sweep says.
- **The two headline per-kernel findings survive**, being per-call durations at
  matched call counts. And B200's own §2c classified
  `flash_fwd_splitkv_mla_fp8` as **pure** (1.14 → 2.90 ms across bs 1 → 12), so
  its 2.454 ms is real work — making **4.06x a floor**, since contention from
  1.67x concurrency can only have inflated it.
- **The `moe` comparison is the weak one**: B200's `mega_moe_impl` absorbs wait
  *and* is concurrency-inflated. `megamoe_prepare_compact` 16.24 ms still has no
  counterpart, but no ratio until B200 publishes credited time.

**Request to B200:** `python3 analysis/busy_ms.py <your trace dir> 10` and
publish the **`credited`** column — exclusive time plus a 1/k share of every
k-way overlap segment, so the credits sum exactly to total GPU busy and the
per-role deltas become an attribution of elapsed time. Per-role `union` alone
does not sum to total busy when roles overlap each other. It also gives two
numbers nobody has for B200: **idle inside the 30 ms step**, and which roles
overlap which (`union` − `exclusive`). A large idle would mean part of the
2.47x is launch/sync gaps, which is a different fix again.

---

## B200 `credited` elapsed time — answering the busy_ms.py request, and it
## makes `moe` the gap, not `attn`
*(b200, 2026-09-16. `busy_ms.py` on
`b200:/workspace/agentx/traces/b200-tp8-ep8-dpatrue-c128-pdi24trace_steady/`,
`TARGET_VERIFY full` bs=10. Reclassified to the agreed scheme —
`nvjet_*`→gemm, `silu_mul_clamp`→moe, `mega_moe_pre_dispatch`→moe,
`flash_fwd_mla_combine`→attn — so `comm` means a real collective on both sides.)*

MI355X was right and the correction is large. Two ranks, and they agree:

| rank 4, n=15 | sum | union | excl | **credited** |
|---|---:|---:|---:|---:|
| gemm | 24.03 | 21.32 | 6.23 | **13.71** |
| moe | 15.84 | 15.07 | 1.12 | **8.08** |
| attn | 7.62 | 6.80 | 5.75 | **6.23** |
| quant | 1.57 | 1.53 | 0.60 | 1.05 |
| other | 1.03 | 1.03 | 0.30 | 0.64 |
| norm_rope | 0.56 | 0.56 | 0.27 | 0.38 |
| sample | 0.14 | 0.14 | 0.00 | 0.07 |
| copy | 0.06 | 0.06 | 0.01 | 0.03 |
| comm | 0.00 | — | — | **0.00** |
| **total** | **50.87** | | | **30.20** |

| | wall | summed | sum/busy | **GPU busy** | **idle in step** |
|---|---:|---:|---:|---:|---:|
| rank 4, n=15 | 30.79 (p50 29.95) | 50.87 | **1.684x** | **30.20** | **0.59** |
| rank 7, n=10 | 32.03 (p50 30.15) | 51.92 | 1.681x | 30.88 | 1.15 |

Rank 7 credited: gemm 13.52, moe 8.50, attn 6.43 — within 0.4 ms of rank 4 on
every bucket.

### The idle number neither node had

**B200 is 98.1 % busy inside the step** (0.59 ms idle of 30.79), MI355X 99.99 %
(0.01 of 75.23). **Neither node has idle to reclaim.** The whole gap is kernel
work or wait absorbed inside kernels; there is no launch-gap or scheduling
slack to recover on either side.

### What overlaps what on B200 (the request's second half)

Concurrent elapsed time per role pair, ms/step, rank 4:

| pair | ms/step |
|---|---:|
| **gemm x moe** | **13.39** |
| attn x gemm | 0.83 |
| gemm x other | 0.63 |
| moe x quant | 0.57 |
| gemm x quant | 0.27 |
| attn x norm_rope | 0.22 |
| everything else | <0.2 each |

Exclusive: gemm 6.23, attn 5.75, moe 1.12, quant 0.60, rest <0.31.

**B200's concurrency is essentially one thing: the MoE kernel running underneath
the dense GEMM.** `moe`'s union is 15.07 ms of which 13.39 is concurrent with
`gemm` and only 1.12 is exclusive. MI355X runs the same two stages serially.
That is the structural difference, stated as elapsed time rather than as a
1.67x ratio.

### The corrected cross-platform table — `credited` vs `credited`

Both columns are now elapsed attributions of GPU busy time.

| bucket | B200 credited | MI355X credited | delta | % of gap |
|---|---:|---:|---:|---:|
| **moe** | 8.08 | **35.75** | **+27.67** | **61.5 %** |
| **attn** | 6.23 | **15.67** | **+9.44** | **21.0 %** |
| comm (real collective) | 0.00 | 6.17 | +6.17 | 13.7 % |
| copy | 0.03 | 3.98 | +3.95 | 8.8 % |
| quant+other+norm_rope+sample | 2.14 | 4.32 | +2.18 | 4.8 % |
| **gemm** | 13.71 | 9.36 | **−4.35** | **−9.7 %** |
| **GPU busy** | **30.20** | **75.22** | **+45.02** | 100 % |

`gemm+moe` as one bucket: **21.79 → 45.11, +23.32 (2.07x)**.

### Consequences, including one of B200's own claims withdrawn

- **"`attn` is the largest difference, not MoE" is WITHDRAWN.** That was
  §2b, built on summed kernel time. On elapsed time `moe` is +27.67 and 61.5 %
  of the gap, `attn` +9.44 and 21 %. MI355X's sum-based ordering (moe +19.91 >
  attn +8.05) was the right one; `credited` widens moe further because B200's
  MoE is the bucket that was most concurrency-inflated (sum 15.84 → credited
  8.08, a factor of 1.96).
- **`gemm` flips sign, as MI355X predicted.** B200 spends 13.71 ms of elapsed
  time in dense GEMM against MI355X's 9.36 — B200 is 1.46x *slower* here. Note
  B200's `gemm` absorbs wait (§2c), so its real GEMM work is below 13.71 and
  the true flip may be smaller.
- **`megamoe_prepare_compact` (16.24 ms, 61 calls, no B200 counterpart) is now
  the single biggest named target**, since it alone exceeds B200's entire
  credited `moe` of 8.08.
- **The MLA decode kernel 4.06x stands as a floor**, unchanged: per-call at
  matched call counts, and B200's side is pure work that concurrency can only
  have inflated.
- **What is still not separable:** both nodes' `moe` absorbs group wait, so
  +27.67 mixes real MoE work with wait on both sides and its sign is safe but
  its magnitude is not. The TP-only / single-DP-rank run remains the way out,
  and MI355X has shown `record_shapes` cannot substitute for it.

**Reproduce:** `python3 analysis/busy_ms.py <trace dir> 10`. The reclassified
run and the pairwise matrix need the four FIX patterns above applied to
`classify`; folding them into `trace_common.ROLES` is the obvious next tooling
step so both nodes stop patching locally.

### DONE — the four patterns are now in `trace_common.ROLES`
*(mi355x, 2026-09-16)*

No more local patching: `classify()` produces the agreed buckets natively.
Two override rules are tested **before** `comm`, because the `comm` patterns are
substring matches on words that also occur in non-collective kernels —
`mla_combine|mha_combine|splitkv.*combine` → `attn`, and
`pre_dispatch|silu_mul_clamp|clamp_silu_mul|silu_mul` → `moe`. `nvjet` was added
to `gemm`, and `ep_combine|ep_dispatch` to `comm` so MI355X's real collective
keeps matching explicitly rather than incidentally.

Verified on 11 kernel names from both platforms, and the MI355X bs=10 totals
move exactly onto the hand-reclassified values: **`moe` 35.49 → 35.75, `comm`
6.43 → 6.17** (`_fused_clamp_silu_mul` 0.262 crossing over), everything else
unchanged. Any table produced before this commit that did *not* apply the
manual FIX list is in the old buckets; both nodes' published credited tables
already used the new ones, so no published number changes.
---

## CU contention measured: ~5 %, real but not the explanation. Both tables are
## valid, for different questions
*(b200, 2026-09-16. Raised by the b200 operator: is B200's large `gemm` and its
1.67x sum/wall ratio caused by CU contention stretching concurrent kernels,
rather than by absorbed wait or by genuine concurrency?)*

**Direct test, and it settles the mechanism.** `gemm_1d1d_impl` invocations were
bucketed by what fraction of their own duration overlapped a `moe` kernel. Same
kernel, same step, same rank, matched bs=9, multi-stream vs single-stream
capture:

| `gemm_1d1d_impl`, per call | multi-stream | single-stream | delta |
|---|---:|---:|---:|
| the overlapped population (75-99 %) | **211.3 µs** mean, 210.8 p50 | **201.4 µs**, 199.9 p50 | **+4.9 %** |
| the non-overlapped population (0 %) | 20.2 µs | 20.3 µs | 0 % |

**CU contention is real and is ~5 %**, confined to the kernel that actually
overlaps; kernels that do not overlap are unaffected to within 0.5 %.

Three things follow.

- **The 1.67x sum/wall ratio is genuine concurrency, not contention
  inflation.** A 5 % stretch cannot produce a 67 % ratio, and the overlapped
  GEMM is still 201 µs with streams off — it is a big kernel, not a stretched
  small one.
- **The "5 % ITL vs 30-versus-50 ms" objection resolves: single-stream is not
  serial.** Measured `sum/busy` is **1.55x** with the flag off against 1.70x
  with it on (rank 0-2, `pdi10trace40` vs `1stream_trace40`). The flag removes
  roughly 8 % of the overlap, not all of it. So the ~5 % ITL cost of the flag
  and the ~5 % contention measured above are the *same* 5 %, and they are
  mutually consistent — neither bounds the cost of having no overlap at all,
  which is MI355X's situation at 1.00x.
- **B200's larger `gemm` is not contention either.** It survives with streams
  off. The likely cause is that the two platforms partition the same work
  differently: B200 runs a per-layer dense GEMM (61 calls/step, ~211 µs,
  12.9 of its 24.03 ms) deliberately underneath the MoE all-to-all, where
  MI355X appears to fold that work into `megamoe_stage1/2`. This is the
  concrete reason `gemm+moe` must be read as one bucket.

### Both tables stand, for different questions

The summed-kernel table and the `credited` table were treated as rivals. They
are not — and per-call comparison is now backed by measurement rather than
assumption:

| question | use | B200 | MI355X | ratio |
|---|---|---:|---:|---:|
| which kernel is slower **per call** | sum / per-call, valid to ~5 % | 50.86 | 75.25 | **1.48x** |
| where the **elapsed** step time goes | `credited` / `union` | 30.20 | 75.22 | **2.49x** |

`gemm+moe` on summed kernel: **39.87 vs 45.11, only 1.13x.**

**And the clean decomposition: 2.49x = 1.48x (more kernel work) x 1.68x (B200's
concurrency).** MI355X issues 48 % more kernel-time and cannot overlap any of
it, which is what turns 48 % into 149 %.

So the ordering question — "is it `moe` or `attn`" — has different answers by
construction, and both are right: on elapsed time `moe` dominates (+27.67), on
per-call kernel work the buckets are close (gemm+moe 1.13x) and the outliers are
the two named kernels. **Neither table alone characterises the gap.**

### Caveat retained on the earlier "gemm absorbs wait" claim

§2c inferred that `gemm_1d1d_impl` absorbs group wait from its being 22.48 µs*
at bs=1 and 19.36 at bs=12 on an unchanged 396 calls. That inference is **not
safe**: at bs=1 `moe` is also larger (16.80 vs 13.70), so the same data is
equally explained by more overlap with a longer MoE kernel, i.e. by the ~5 %
contention measured here. The `mega_moe_impl` absorption claim is unaffected.
(*ms, not µs — 22.48 ms summed over 396 calls.)

---

## Identifying B200's 217 us per-layer GEMM: not routed MoE, most likely the
## alt-stream shared expert waiting on the a2a
*(b200, 2026-09-16. Withdraws an unverified claim of its own.)*

An earlier note called this kernel "a per-layer dense GEMM deliberately
overlapped with the MoE all-to-all, probably the shared expert". **That was
speculation.** Here is what is now measured.

### Method: attribute in EXTEND, because decode cannot be attributed at all

Decode steps are cuda-graph replays — `graph id` 32/92 on every kernel, and
**0 of 15,732** `gemm_1d1d_impl` calls resolve to a CPU op, because the launches
happened at capture time outside the window. **EXTEND steps are eager**
(`graph id` 0) and **670 of 792** resolve. Use the GPU kernel event's own
`External id` against `cpu_op` — the `cuda_runtime` correlation path only
intersects 76 of 948 ids and is the wrong route.

This works on the existing captures; no re-profiling was needed.

### What the dense GEMM family actually is

All of it is `sglang::deep_gemm_fp8_fp8_bf16_nt`. Five dim sets, from the two
EXTEND steps at 6144 tokens (`[A, A_scale, B, B_scale, out]`, so N and K are
readable):

| dims | projection | calls |
|---|---|---:|
| `[6144,7168] … [2048,7168] → [6144,2048]` | fused qkv_a (q_lora 1536 + kv_lora 512) | 122 |
| `[6144,1536] … [65536,1536] → [6144,65536]` | q_b_proj | 122 |
| `[6144,16384] … [7168,16384] → [6144,7168]` | o_proj | 122 |
| `[6144,7168] … [6144,7168] → [6144,6144]` | **shared-expert gate_up** (7168→2×3072) | 122 |
| `[6144,3072] … [7168,3072] → [6144,7168]` | **shared-expert down** (3072→7168) | 122 |

So **`num_fused_shared_experts == 0` on this arm** and the alt-stream
shared-expert path at `deepseek_v2.py:1282-1287` is live — the shared expert is
its own pair of dense GEMMs, not folded into the routed experts.

### The answer, and the residual uncertainty

- **It is NOT the routed-MoE GEMM.** Routed MoE is the `GemmType=4` grouped
  instantiation (template carries `1024u, 4096u`, 16 groups). In decode that is
  61 calls/step at **18.5 us** with **0 % overlap** with the `moe` bucket. It is
  also the one thing EXTEND cannot resolve, because the MoE custom ops register
  no shapes — the same gap MI355X reported for `record_shapes`.
- **The 217 us kernel is `GemmType=0`, i.e. dense**, and the shared expert is
  one of the five dense candidates above.
- **The exact projection cannot be pinned**, because the tile instantiation is
  chosen from M, and M is 6144 in EXTEND against ~40 in decode. Instantiations
  do not carry across the two windows.

**Quantitative argument that it is not doing its own work:** at fp4 (0.5 B per
param) and TP8, the largest per-layer projection is o_proj at 117 M params, i.e.
58.7 MB dense and ~7 MB per rank — single-digit microseconds at B200's HBM rate.
The smallest observed decode group is 15 us, already several times that bound,
which is normal small-M GEMM inefficiency. **217 us is ~30x the bound for any of
these projections.** No projection's own work explains it.

Combined with the 99 % overlap against the `moe` bucket, and with the fact that
the shared expert is exactly what the code places on `alt_stream` to overlap the
MoE call, the consistent reading is: **the 217 us is the alt-stream
shared-expert GEMM, mostly waiting on the MoE all-to-all rather than computing.**
That also explains why the *same* instantiation runs 91 calls/step at 24.8 us
with 0 % overlap.

**To close it, one clean experiment:** re-capture decode with `with_stack=true`,
or compare against the breakable-cuda-graph path, where
`deepseek_v2.py:1289`'s own comment says the shared experts "overlap nothing".

**Standing lesson for both nodes:** kernel identity cannot be established inside
a graph-replayed window. Attribute in EXTEND, then carry the *symbol family*
(dense vs grouped) rather than the tile instantiation across to decode.

---

## ⚠ The 217 us GEMM is SM-STARVED, not waiting — and this invalidates the
## 4.9 % contention figure and part of the summed-kernel comparison
*(b200, 2026-09-16. Raised by the b200 operator asking why a shared-expert GEMM
would ever wait on the routed a2a. It does not.)*

Measured, TP-4, `TARGET_VERIFY full` bs=10, n=15:

| | grid | streams | calls/step | duration |
|---|---|---|---:|---:|
| `deep_gemm::..._mega_moe_impl` | **[146,1,1]** | 1 (default) | 61 | 243.4 us |
| the big `gemm_1d1d_impl` | **[148,1,1]** | 61 distinct | 61 | 217.4 us |

**B200 has 148 SMs and the MoE kernel asks for 146 blocks** — it is a persistent
kernel that owns essentially the whole GPU. The GEMM asks for 148 and gets ~2.

Timing relative to its overlapping MoE call: **starts +25.0 us after it, ends
+1.9 us after it, 99.1 % covered, never on the same stream.** A semaphore wait
would start early and block; this starts late, crawls, and finishes only when
the MoE releases the machine. **That is SM starvation.** The same instantiation
costs **24.8 us** on its 91 non-overlapped calls/step — that is its real cost.

### Three earlier conclusions are affected

1. **The 4.9 % contention measurement is invalid as a bound.** It compared the
   multi- and single-stream captures, but `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`
   **does not disable this overlap**: `moe_alt_stream` at
   `deepseek_v4.py:1798` is gated on `_is_cuda`, not on that flag. So the
   overlapped population is still ~201 us with the flag off, and the A/B never
   varied the thing being tested (`perf-bottleneck-attribution` #3 — enabling
   condition vs outcome). **The real starvation effect is ~10x on this kernel,
   not 5 %.**
2. **It explains the "5 % ITL vs 1.67x sum/wall" puzzle completely.** The 1.67x
   is genuine concurrency but buys almost no throughput, because most of the
   co-resident time is one kernel crawling on ~2 SMs. That is why removing part
   of the overlap costs only ~5 %.
3. **B200's summed `gemm` overstates GEMM work by ~11.7 ms/step.** Of its
   24.03 ms, the starved population is 13.26 ms whose unimpeded cost is
   61 x 24.8 us = 1.5 ms. So on *work*, B200's step is ~39 ms rather than
   50.86, against MI355X's 75.25 — **~1.9x, not 1.48x**. `credited` does not fix
   this either: splitting shared time 1/k over-credits a starved kernel.

### What this does not change

The elapsed comparison stands: GPU busy **30.20 vs 75.22 ms**, and the wall
**30.0 vs 74.0**. Starvation is already inside those numbers. The per-kernel
findings that rest on matched call counts also stand — the MLA decode kernel at
4.06x, and `megamoe_prepare_compact` 16.24 ms with no B200 counterpart.

### For MI355X

Worth checking the equivalent on ROCm: **what grid does the MegaMoE kernel
request against the CU count**, and does any dense kernel co-reside with it?

The reason this matters is that B200's 1.68x invites the inference "MI355X is
missing kernel overlap, so its 75 ms step should compress toward ~45 ms." That
inference does not hold. Of B200's **15.92 ms** of co-resident time (busy 30.20
minus 14.28 of summed per-role exclusive), **13.39 ms — 84 % — is the single
`gemm` x `moe` pair**, and in it the GEMM holds ~2 of 148 SMs and contributes
only ~1.5 ms of work (61 x 24.8 us, its unimpeded cost). So the time actually
*saved* by overlap is single-digit ms in a 30 ms step, **order 5-13 %**, not
40 %. Chasing overlap cannot close a 2.47x gap; the two named kernels can.

**Two honest caveats on that estimate.** (a) It has roughly ±5 ms of slop: the
"real work ~39 ms in a 30.2 ms window" route implies ~8.9 ms hidden, while the
pairwise route implies ~4 ms, and the gap is in how much the *non*-starved
`gemm` calls overlap and in the 1.5 ms estimate. The order of magnitude is
sound; the number is not. (b) **This is a statement about B200, not about
ROCm.** B200 gains little because `mega_moe_impl` claims 146 of 148 SMs and
leaves nothing to overlap into. If aiter's MegaMoE leaves a real fraction of CUs
idle, MI355X could overlap *better* than B200 does, and then it would be worth
doing. That is exactly what the grid-vs-CU number would settle — this is a
request for a measurement, not advice to skip the work.

---

## MegaMoE grid vs CU count: MI355X is the opposite case to B200 — 30 of 256 CUs
*(mi355x, 2026-09-16, answering b200's request. Detail in
`mi355x-decode-trace.md` §14.)*

Hardware: gfx950, **SPX, 256 CUs**. The grid is **not in the trace** — ROCm's
PyTorch profiler emits only `{device, stream, correlation, kind}` on kernel
events, no grid/block/registers. (`bytes` and `memory bandwidth (GB/s)` exist
but only on the 479 `gpu_memcpy` events, never on the 59,033 compute kernels,
so that is not the bandwidth estimator either.) It is recoverable because
**aiter encodes the launch config in the kernel name** and the generators are
local:

| kernel | grid | of 256 CUs | µs/call | ms/step |
|---|---:|---:|---:|---:|
| `megamoe_prepare_compact` | **30** | **≤11.7 %** | 266.3 | **16.244** |
| `megamoe_stage1_compact` | 256 | 100 % | 226.2 | 13.799 |
| `megamoe_stage2_compact` | 240 | 93.8 % | 89.2 | 5.444 |

`launch_grid = prepare_blocks + quant_blocks + 1`, and the name carries
`pcu1` + `qcu28` (`mega_moe_prepare.py:61,74,262`); stage1 is
`gm1 × num_cu` (`:144,218`); stage2 is `p1cu240` (`:369,670`).

**MI355X's largest kernel — 16.24 ms/step, 21.6 % of the decode step — cannot
occupy more than 30 of 256 CUs.** And **nothing co-resides with it**: inside
`TARGET_VERIFY full` bs=10, stream 7 holds 3,096 events/step and 75.220 ms while
stream 6 holds 2.2 events/step and **0.024 ms**. Total co-resident time is
**24 µs/step** against B200's 15.92 ms.

**This inverts B200's overlap conclusion, and B200 was right to demand the
number before generalising.**

| | B200 | MI355X |
|---|---|---|
| big MoE kernel grid | 146 of **148** SMs | prepare **30** of **256** CUs |
| room to overlap into | ~2 SMs — none | ~226 CUs unclaimed for 16.24 ms/step |
| co-resident time | 15.92 ms, 84 % one starved GEMM | **0.024 ms** |

B200 gains little because `mega_moe_impl` claims the machine. That does not
transfer: MI355X leaves most of the device unclaimed for a fifth of the step,
which is the condition B200 itself named as making the work worthwhile.

**What is NOT established** (`perf-bottleneck-attribution`): this is an
occupancy *ceiling*, not a utilisation measurement. It does not show the prepare
stage wastes the device — 30 workgroups bounds how many CUs can host work, not
how busy they are, and the kernel is a producer/consumer ticket protocol whose
266 µs may be partly irreducible serialisation. It also does not predict a
recovery of ~16 ms. The falsification is cheap and local: **co-schedule real
work against the prepare stage and see whether the step wall moves.** Also worth
one config probe first — whether `pcu1`/`qcu28` are simply mistuned for a
256-CU part, which is a parameter in `mega_moe_prepare.py`, not a rewrite.
## ✅ MEASURED: a truly serial B200 arm. Overlap is worth 8 %, the gap is `moe`,
## and `gemm` is equal on the two platforms
*(b200, 2026-09-16. Proposed by the b200 operator: stop inferring the cost of
no-overlap and measure it. This supersedes every summed-kernel and `credited`
bucket comparison in this file.)*

### Why no previous arm was serial — a real flag bug

`deepseek_v4.py:2736`: `use_stream_pool = _is_cuda or (_is_hip and ...)`. **On
CUDA the stream pool is created unconditionally**, and `moe_alt_stream`
(`:1798`) only tests `_is_cuda`. `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP` gates
*only* the three attention streams (`:853`). So the flag never disabled the one
overlap that matters, and the "single-stream" arm still ran at `sum/busy` 1.55x.

One-line fix, default-preserving because the flag defaults True:

```python
use_stream_pool = (
    (_is_cuda and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get())   # was: _is_cuda
    or (_is_hip and (...)) or (_is_npu and ...)
)
```

With it, `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` gives **`sum/busy`
1.034-1.052x on all six ranks** — verified as an outcome, not assumed from the
flag.

### The result, at matched steady state (169,932 tok/req vs 165,220 and MI355X's 167,538)

| | multi-stream | **serial** | delta |
|---|---:|---:|---:|
| step wall p50 | 29.95 ms | **32.68 ms** | **+8.0 %** |
| summed kernel | 50.87 | **34.16** | **−33 %** |
| GPU busy | 30.20 | 32.61 | +8.0 % |
| `sum/busy` | 1.684x | **1.043x** | |
| the 217 us GEMM population | 61/step @ 217 us, 99 % moe-overlap, 13.26 ms | **gone** — all 152 calls/step @ 18.2 us, 0 % overlap, 2.77 ms | |

**Predictions were stated in advance and one is falsified.** Starvation
predicted +5-13 % on the wall; "overlap hides real work" predicted +50-67 %.
Measured **+8.0 %**. The 16.7 ms that vanished from the summed kernel was
phantom — an earlier estimate put the phantom at ~11.8 ms, so it was
*understated*.

**So overlap on B200 is worth 8 % of the step**, and of the 2.47x wall gap
against MI355X, **2.26x is kernel work and only 1.09x is overlap.**

### The clean per-role attribution — both sides serial, both sums are elapsed

B200 rank 0, bs=9, n=17 (`sum ≈ union ≈ exclusive`) against MI355X rank 7,
bs=10:

| bucket | B200 serial | MI355X | delta | % of gap |
|---|---:|---:|---:|---:|
| **moe** | 13.42 | **35.75** | **+22.33** | **54 %** |
| **attn** | 8.20 | **15.67** | **+7.47** | **18 %** |
| comm (real collective) | 0.00 | 6.17 | +6.17 | 15 % |
| copy | 0.13 | 3.98 | +3.85 | 9 % |
| quant | 1.14 | 2.21 | +1.07 | 3 % |
| other+norm_rope+sample | 1.47 | 2.11 | +0.64 | 2 % |
| **gemm** | **9.81** | **9.36** | **−0.45** | −1 % |
| **total** | **34.17** | **75.24** | **+41.07** | **2.20x** |

**`gemm` is equal between the platforms** (B200 5 % higher). Both earlier
readings — summed `+14.67` and credited `−4.35` — were starvation artefacts in
opposite directions. There is no dense-GEMM gap.

**MI355X's ordering was right all along**: `moe` is the gap, at +22.33 ms and
54 % of it, with `attn` second at +7.47. `megamoe_prepare_compact` alone
(16.24 ms, no B200 counterpart) exceeds B200's *entire* `moe` of 13.42.

### One reverse-direction contention effect, for the record

`mega_moe_impl` is **220 us/call serial against 243 us multi-stream** — it was
being slowed ~9 % by the starved GEMM stealing 2 of its SMs. So contention is
real, and it ran *against* the overlap, not for it.

### Consequences for MI355X

- The overlap upside on ROCm is now bounded by measurement, not inference:
  **~8 %**, and only if aiter's MegaMoE leaves CUs free (B200's takes 146 of
  148). The grid-vs-CU number is still the thing to check, but the prize is
  small either way — it cannot touch a 2.20x kernel-work gap.
- **Everything now points at `moe`.** +22.33 ms of 41.07, and the named
  suspect is `megamoe_prepare_compact`. That is where the work is.

### Retractions this supersedes

The decomposition "2.49x = 1.48x work x 1.68x overlap" is wrong; it is
**2.26x work x 1.09x overlap**. Every summed-kernel bucket delta in §2b and
every `credited` delta in the `busy_ms.py` block is superseded by the serial
table above — on B200 both were measuring starvation, in opposite directions.

---

## JOINT: the `moe` gap is `megamoe_prepare_compact`, and it runs on 11.7 % of
## the GPU. B200's overlap conclusion does not transfer — MI355X was right
*(b200, 2026-09-16, combining its serial arm with mi355x's grid-vs-CU answer.)*

Both sides are now serial (B200 `sum/busy` 1.043x by patch, MI355X 1.000x by
construction), so the two per-role tables are elapsed and subtract cleanly. Put
B200's fused MoE against MI355X's three stages:

| | B200 | MI355X | grid / device |
|---|---:|---:|---|
| `mega_moe_impl` (fused) | **13.42** | — | 146 of 148 SMs |
| `megamoe_prepare_compact` | — | **16.24** | **30 of 256 CUs (11.7 %)** |
| `megamoe_stage1_compact` | — | 13.80 | 256 of 256 |
| `megamoe_stage2_compact` | — | 5.44 | 240 of 256 |
| `moe` bucket total | **13.42** | **35.75** | |

**stage1 + stage2 = 19.24 ms against B200's entire fused 13.42 (1.43x)** — a
normal cross-platform kernel difference. **The whole excess is `prepare`:
16.24 ms with no B200 counterpart, i.e. 40 % of the total 41.07 ms decode-step
gap, on a kernel that cannot use more than 11.7 % of the GPU.**

**Correction to this document's own advice.** An earlier block said the overlap
prize was "small either way" and merely asked for the grid number. The number
inverts the conclusion: B200 gains 8 % from overlap because `mega_moe_impl`
claims 146 of 148 SMs and leaves nothing to overlap into; MI355X leaves **226 of
256 CUs unclaimed for 16.24 ms per step** and currently co-resides **24 µs**.
The B200 measurement bounds B200, not ROCm.

### The two targets, in order

1. **`megamoe_prepare_compact`'s parallelism** — 16.24 ms at grid 30. This is
   the single biggest item in the gap and it is a launch-config/algorithm
   question, not a cross-platform hardware one. Raising its CU usage attacks the
   gap directly; overlapping it only hides it.
2. **The MLA decode kernel, 4.06x per call** — `flash_fwd_splitkv_mla_fp8`
   40.2 us vs `_paged_decode_split_kernel` 163.5 us, one call per layer on both.
   This is 93 % of the `attn` gap (+7.47 ms, 18 % of the total).

Together these two kernels are ~58 % of the 41.07 ms. `gemm` is equal, `comm`
is exposed rather than slower, and overlap is worth 8 % on the node that has no
room for it.

---

## ✅ CONFIRMED: `megamoe_prepare_compact` is a cross-rank WAIT — r = −0.942,
## and 10 of its 16.24 ms is idle, not work
*(mi355x, 2026-09-16, measured with `analysis/prepare_wait.py`. This resolves
the hypothesis block below and **revises B200's target #1**. Detail in
`mi355x-decode-trace.md` §15-16.)*

Per-call µs against each rank's own `compute`, steady capture, five decoding
ranks:

| kernel | r | verdict |
|---|---:|---|
| **`megamoe_prepare_compact`** | **−0.942** | **WAIT** |
| `megamoe_stage1_compact` | +0.672 | real work |
| `megamoe_stage2_compact` | +0.943 | real work |
| MLA decode (`_paged_decode_fused`) | +0.961 | real work |
| `ep_combine_intranode` | −0.665 | **inconclusive** — flat within one rank |

**The within-rank control settles it**: rank 0 alone spans bs 17→20, so rank is
fixed, and as its `compute` rises 28.76 → 35.06 `prepare` **falls 251.2 → 216.3
µs/call** while `stage1`, `stage2` and MLA decode all rise. `prepare` is the
only kernel that shortens as its own rank gets busier.

**`compute + prepare` is constant at 44.1-45.2 ms (2.5 % spread) while
`compute` spans 1.56x.** `prepare` is the single synchronisation point of the
step — `stage1`/`stage2`/`ep_combine` are near-equal across ranks once it has
levelled everyone.

**Therefore the 16.24 ms is mostly idle.** The straggler is by definition not
waiting for anyone, so its **102.9 µs/call = 6.28 ms/step is the protocol's own
cost**; rank 7's 16.24 ms is that floor plus **9.96 ms of idle**, and the
lightest rank's 19.83 ms is the floor plus 13.55. **B200's "prepare is 40 % of
the 41.07 ms gap" overstates the real cost by ~10 ms — it is 6.28 ms, 8.5 % of
the step.**

### Both proposed fixes are wrong, for the same reason

- **✗ Raising CU usage / retuning `pcu1`+`qcu28`.** A `wait_i32_until_equals`
  spin does not parallelise. §14's "226 of 256 CUs idle" is real but the CUs are
  idle **because the rank is waiting**, not because the kernel is
  under-parallelised. That inverts §14's own implication.
- **✗ Co-scheduling work against it.** On the seven waiting ranks the capacity
  is already free and they still cannot finish before the straggler; on the
  straggler there is only 6.28 ms to hide.

### What would work

1. **Balance on KV tokens, not request count.** The wall is set by the
   straggler's `compute`, driven by the MLA decode kernel at **165.3-330.7
   µs/call (2.0x)**, which tracks **KV tokens, not `bs`** — rank 1 has bs=14 at
   330.7 µs, rank 0 has bs=18 at 166.3. The launcher runs
   `--load-balance-method total_requests`, i.e. balances the wrong quantity.
   SGLang already has `LoadBalanceMethod.TOTAL_TOKENS`
   (`data_parallel_controller.py:92,125-130`) fed by
   `LoadSnapshot.num_total_tokens` (`load_snapshot.py:201`), a per-rank KV
   occupancy metric. Upper bound on the prize: straggler `compute` 38.91 → ~31.2
   mean, so **wall 74.0 → ~66 ms, order 10 %**. **Not free** — it fights
   `--policy cache_aware`, which creates the skew on purpose for prefix reuse,
   so it is an ITL↔TTFT A/B. **EPLB cannot help**: this imbalance is in
   attention/KV, not expert routing.
2. **Pipeline the 6.28 ms floor, do not widen it.** The floor is **61 cross-rank
   gate crossings per step**, one per layer. Overlapping `prepare(n+1)` with
   `stage1/2(n)` hides the crossing behind MoE GEMM **on the straggler too**,
   which is the only place hiding shortens the step. B200 gets this free: its
   fused `mega_moe_impl` has no per-layer gate. Note this is the *opposite* of
   "co-schedule unrelated work" — the win is pipelining the dependency chain.

**The measurement that decides how far (2) goes: the TP-only / single-DP-rank
run.** At `npes = 1` there is nobody to wait for. If the 102.9 µs floor
collapses it is still synchronisation and pipelining is the fix; if it holds,
that is real dispatch-plan emission and then `pcu1` — one CTA emitting the plan
— *is* worth raising after all.

**Net effect on the joint target list:** the **MLA decode kernel is target #1**
on its own numbers (7.17-20.18 ms/step, 4.06x per call at matched bs=10), and
`prepare` splits into a load-balancing problem (~10 ms idle, ~7.7 ms of wall)
plus a 6.28 ms pipelining problem.

### Also fixed: a self-inflicted artefact worth knowing about

aiter selects **two** MLA decode variants by batch size —
`_paged_decode_split_kernel` at bs 9-10, `_paged_decode_fused_kernel` at
bs 14-20. A first version of the script tracked only the split variant, read
0.0 on three ranks, and reported a spurious "mla_decode is a WAIT". Tracked
separately it is the clearest *work* signal in the step. The published 4.06x is
the **split** variant at the matched bs=10 cell; it is not a claim about the
fused one.

---

## ⚠ superseded by the block above — HYPOTHESIS, unmeasured:
## `megamoe_prepare_compact` is a cross-rank barrier
*(mi355x, 2026-09-16, from the generator source. Detail in
`mi355x-decode-trace.md` §15. **Mechanism established, magnitude NOT** —
the falsification is scripted and unrun; see the shell note below.)*

B200's target #1 is "`megamoe_prepare_compact`'s parallelism — 16.24 ms at grid
30 … raising its CU usage attacks the gap directly". Reading
`aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py` suggests **the grid is a
consequence, not the cause**, and that the kernel is an inter-rank protocol
rather than compute:

- `:43,57,62` — parameterised by `npes` (rank count): `total_experts = npes*epr`,
  `assert dispatch_blocks % npes == 0`
- `:111` — `comm_ops.atomic_add_agent` ticket draw
- **`:188` — `comm_ops.wait_i32_until_equals(epoch_gate, gate_epoch)`, an
  explicit spin-wait**
- `:177-197` — `s_waitcnt(0)`, `store/load_i32_system`, `fence_system_acquire`
  — **system scope**, i.e. cross-device
- `:201-235` — `emit_dispatch_plan` / `emit_dispatch_group` with `fz_npes`,
  `fz_rank`

The grid is small **by design for a small batch**, not by oversight:
`mega_moe_config.py:273` scales quant CTAs with tokens (64 → 192 from 8K to 32K)
and `mega_moe_v2.py:144-151` states "small batches continue to launch only their
useful subset and do not pay for the capacity". At decode (~40 tokens) that
subset is `qcu28`, and `pcu1` means the prepare role is **one** CTA.

**If the time is in the wait, both proposed fixes miss.** More CUs do not speed
up `wait_i32_until_equals`, and co-scheduling work against it hides a wait that
is already free capacity. It would also mean MI355X **exposes both halves of the
all-to-all** — dispatch inside `prepare` (16.24 ms, bucketed `moe`) plus
`ep_combine_intranode` (6.17 ms, bucketed `comm`) = **22.41 ms/step of exposed
EP traffic** — where B200 fuses its a2a inside `mega_moe_impl` (13.42 ms
serial, total). That reframes the +22.33 ms `moe` gap as substantially
**communication exposure**, not slower MoE math, and it would make the MLA
decode kernel target #1 by default.

**Not proven.** Per `perf-bottleneck-attribution`, source shows the wait exists,
not that the time is in it. Consistent but non-probative: `moe` was the only
bucket that *shrank* as work grew on rank 0 (−3.8 %, bs 17→20), and
`compute + barrier` holds within 0.9 % while `compute` spans 1.56x.

**Falsification, scripted and ready:**
`python3 analysis/prepare_wait.py <trace dir>` compares each MoE kernel's
µs/call against the rank's own `compute`; a barrier anti-correlates (busy rank
arrives late, waits less). If `prepare` is anti-correlated (r < −0.5) while
`stage1`/`stage2`/`mla_decode` track own work, reorder the targets. The
independent check is the **TP-only / single-rank run**: at `npes = 1` there is
nobody to wait for, so a barrier-dominated `prepare` collapses and real work
does not.

**Why it is unrun:** the MI355X session's shells wedged (see that node's
CONTINUE_HERE) before the script could execute. It needs one command in a fresh
terminal.

---

## The KV skew is real and measured from the scheduler log — and B200's evidence
## that it is balanced is invalid (it came from the pinned arm)
*(mi355x, 2026-09-16. New `analysis/kv_skew.py`, needs only a `server.log`.
Detail in `mi355x-decode-trace.md` §17.)*

### MI355X: the balancer works, on the wrong quantity

Per-DP-rank medians, steady-state half of the trace run's `server.log`:

| quantity | spread | max/min |
|---|---|---:|
| `running-req` — what `total_requests` balances | 8 - 11 | **1.38x** |
| `#full token` — what attention costs | 1.04M - 2.36M | **2.27x** |
| tok/req — request length per rank | 110,789 - 213,393 | **1.93x** |

Request count is level to 1.38x, but per-request length varies 1.93x, so KV ends
up 2.27x skewed. That is `--policy cache_aware` working as designed — it routes
a conversation to the rank holding its prefix, so long conversations
concentrate. Rank 0: 9 requests of 110k. Rank 6: 11 requests of 213k.

Independent of §15-16: the skew is visible in the scheduler's own log, with no
trace and no kernel inference.

### Why B200 does not see it — and the first reason is a measurement error

**1. B200's balance evidence is from the pace-pinned arm.** FINDINGS §3 reports
cross-rank `compute` spread 0.0-0.3 ms at fixed `bs` (31.16-31.60 over bs
9/10/11/12). Read it *along* `bs` instead of across ranks: B200's `compute`
moves **+0.5 % from bs 9 to 12** where MI355X moves **+13.7 % from bs 9 to 10**.
A `compute` that does not respond to batch size is the pinning signature B200
itself later established and withdrew. **That table shows a saturated
measurement, not a level load**, and no cross-rank spread has been published
from the serial arm.

**2. B200 has no dedicated barrier kernel, so the wait has nowhere visible to
land.** MI355X parks it in `prepare`, a separate 16 ms line item. B200's is
absorbed inside fused `mega_moe_impl` — and B200's own §2c data gives that
kernel the same conviction signature used against `prepare`: **17.22 ms at bs=1
vs 13.84 ms at bs=12 on an unchanged 61 calls**, shorter when the rank is
busier. By this document's own test, that is a wait.

**3. The same skew costs B200 ~7x less, because its MLA kernel is cheap.** Cost
= skew x per-unit attention cost, and the second factor differs 4-8x:

| | B200 | MI355X |
|---|---|---|
| MLA decode µs/call | 18.7 → 47.5 (bs 1→12) | 117.5 → 330.7 (bs 9→14) |
| ms/step across its own rank spread | 1.14 → 2.90, **Δ1.76** | 7.17 → 20.18, **Δ13.0** |
| fraction of own step wall | **5.9 %** | **17.6 %** |

A B200 exactly as imbalanced as MI355X would pay ~1.8 ms and rightly ignore it.

**Consequence: targets #1 and #2 are not independent.** The imbalance penalty is
downstream of the MLA decode kernel's per-call cost — speed that kernel by k and
the imbalance cost falls by roughly k. Do the kernel first, then re-measure
balance, rather than running both A/Bs at once.

### Two requests to B200, one command each, on existing captures

```bash
python3 analysis/kv_skew.py      <your server.log>
python3 analysis/prepare_wait.py <your SERIAL trace dir>
```

`prepare_wait.py` now carries CUDA patterns (`mega_moe_impl`, `gemm_1d1d_impl`,
`flash_fwd_splitkv_mla`) and runs unchanged on B200. **Use the serial arm** —
under pinning every `compute` reads the same and the correlation is meaningless,
which is exactly how reason 1 above happened.

**Predictions, stated in advance:** `kv_skew.py` shows B200's `#full token` skew
well above its `running-req` skew (same workload, same router policy, same
balance method); and `prepare_wait.py` shows **`b200_moe` anti-correlated** with
per-rank `compute`, i.e. `mega_moe_impl` is where B200's imbalance has been
sitting. If either fails, the question becomes why B200 *is* balanced on the same
workload — and the first check would be whether both launchers really pass the
same `--load-balance-method` and router `--policy`, which has only ever been
verified from script intent on the B200 side, not from its `sglang_command.txt`.
## End-to-end cost of TRUE full serialization on B200: 3 %, not 8 % — and the
## dominant trace overlap is worth ~nothing
*(b200, 2026-09-16. Full arm, no profiler, `DURATION` 3600 so it matches the
pdi=24 reference's 3,628 s. `RESULT_DIR=...-c128-pdi24-serial`. Launcher gate
passed: aiperf error rate 1/11,428 = 0.009 %.)*

`SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0` **with** the `use_stream_pool` patch, so
this is the first arm on this node with no alt streams at all:

| metric | multi-stream (ref) | **serial** | delta |
|---|---:|---:|---:|
| output tok/s/GPU | 403.6 | **390.4** | **−3.26 %** |
| total tok/s/GPU | 46,170 | **44,610** | **−3.39 %** |
| **ITL p90** | 20.7 ms | **21.32 ms** | **+3.00 %** |
| interactivity p90 | 48.31 | 46.91 | −2.89 % |
| TTFT p50 | 4.59 s | 5.031 s | **+9.62 %** |
| cache hit | 0.9635 | 0.9624 | −0.11 % |
| ISL / OSL mean | 114.4k / 1,009 | 112.3k / 992 | −1.8 % / −1.7 % |
| avg GPU power | 5,723 W | 5,606 W | −2.05 % |
| duration | 3,628 s | 3,629 s | matched |

### It reconciles with the trace, via the pdi dilution

The trace measured the **decode step wall** at +8.0 % (29.95 → 32.68 ms). End to
end it is +3.0 % on ITL, because at pdi=24 only ~40 % of the scheduler's step is
inside decode steps (32 of 79.2 ms log-implied). **8.0 % x 0.40 = 3.2 %**,
against 3.00 % measured. The two numbers are the same effect at two scopes;
quote the step wall for kernel work and the 3 % for user-visible latency.

### The overlap that dominates the trace is worth approximately zero

Two A/Bs bracket it:

| removed | out tok/s/GPU | ITL p90 |
|---|---:|---:|
| attention streams only (flag alone, pdi=10) | −3.07 % | +5.88 % |
| **everything, incl. MoE/shared-expert (patch + flag, pdi=24)** | **−3.26 %** | **+3.00 %** |

The `pdi` differs, so this brackets rather than proves. But removing *all*
overlap costs about what removing only the attention streams costs — so the
`gemm` x `moe` co-residency, which is **84 % of B200's co-resident time and the
entire reason `sum/busy` reads 1.68x**, buys essentially nothing end to end.
That is what a starved kernel on 2 of 148 SMs should be worth, and it closes
the loop with the 217 us population vanishing under serialization.

**TTFT is the one real casualty** (+9.6 %): prefill loses its attention-stream
overlap too, and prefill is where those streams actually pay.

### Bottom line for the cross-platform work

Overlap is worth **3 % end to end** on B200. Against MI355X's 2.20x kernel-work
gap it is noise. The targets remain `megamoe_prepare_compact` (16.24 ms/step at
30 of 256 CUs, 40 % of the gap) and the MLA decode kernel (4.06x per call, 18 %).

**The patch is not upstreamed and has been reverted on this node.** Re-apply it
to reproduce:

```python
# deepseek_v4.py:2736
use_stream_pool = (
    (_is_cuda and envs.SGLANG_OPT_USE_MULTI_STREAM_OVERLAP.get())   # was: _is_cuda
    or (_is_hip and (...)) or (_is_npu and ...)
)
```

It is arguably a bug fix worth sending upstream: without it the flag silently
does not control the largest overlap on CUDA.

---

## B200 answers both requests: KV skew 2.45x (worse than MI355X) and its `moe`
## IS the wait (r = −0.717). The `moe` gap is half work, half imbalance
*(b200, 2026-09-16, running mi355x's `kv_skew.py` and `prepare_wait.py`
unchanged. **Both of its predictions hold, and B200's §3 balance claim is
withdrawn.**)*

### Prediction 1 confirmed — B200 is *more* KV-skewed than MI355X

`kv_skew.py` on the serial full arm's `server.log` (9,871 decode lines, steady
half):

| quantity | B200 | MI355X |
|---|---:|---:|
| `running-req` (what `total_requests` balances) | 7-9, **1.29x** | 8-11, 1.38x |
| `#full token` (what attention costs) | 752k-1,842k, **2.45x** | 1.04M-2.36M, 2.27x |
| tok/req | 125.6k-265.6k, **2.11x** | 110.8k-213.4k, 1.93x |

Also verified from the artefact rather than script intent, as asked:
`sglang_command.txt` carries `--load-balance-method total_requests` and
`--prefill-decode-interval 24`.

**So the skew is not a ROCm phenomenon — it is the balance method, identically
on both nodes.** MI355X's diagnosis of the cause (`cache_aware` routing
concentrating long conversations) applies unchanged.

### `compute` spread of 0.0-0.3 ms at fixed `bs` — WITHDRAWN

§3 used that spread as evidence B200's ranks were level. **MI355X is right that
it is invalid**: it came from the multi-stream capture, whose `compute` this
document itself later showed to be pace-pinned. In the **serial** arm the per-rank
spread is real — `compute` 16.75-19.79 ms and MLA decode **24.3-48.0 µs/call
(1.98x)** across ranks. B200's ranks are *not* level, and never were; the
measurement was saturated.

### Prediction 2 confirmed — the wait lands in `mega_moe_impl`

`prepare_wait.py`, serial trace, 7 rank/bs cells:

| kernel | r | spread µs/call | verdict |
|---|---:|---|---|
| **`mega_moe_impl`** | **−0.717** | 214.5-261.2 | **WAIT** |
| `flash_fwd_splitkv_mla` | +0.917 | 24.3-48.0 | real work |
| `gemm_1d1d_impl` | +0.358 | 17.0-17.4 | flat — inconclusive, and confirms the starvation is gone |

Busiest rank shortest, lightest rank longest — the same signature MI355X used on
`prepare`. **B200 hides its barrier inside the fused MoE kernel, exactly as
MI355X predicted.**

### Both waits removed, the `moe` gap splits in half

Take each node's busiest rank as its protocol floor (the straggler waits for
nobody):

| | B200 | MI355X |
|---|---:|---:|
| floor, µs/call | 214.5 (fused, all stages) | `prepare` 102.9 + `stage1` 226.2 + `stage2` 89.2 |
| floor, ms/step (x61) | **13.08** | **25.52** |
| idle on the lightest rank | **2.85** | **9.96** |
| `moe` bucket as measured | 13.42 | 35.75 |

Reconciles: 25.52 + 9.96 = 35.5 against 35.75 measured.

**So the +22.33 ms `moe` gap is ~12.4 ms of real MoE work (1.95x: a fused
one-kernel pipeline against three stages) plus ~9.96 ms of cross-rank idle.**
And the same 2.45x skew costs B200 only ~2.85 ms, because its MLA kernel is
4.06x cheaper per call — which is MI355X's "cost = skew x per-unit attention
cost" argument, now with B200's own number in it.

### Revised joint decomposition of the 41.07 ms

| component | ms | share | nature |
|---|---:|---:|---|
| MoE pipeline work (3-stage vs fused) | +12.4 | 30 % | kernel work |
| MI355X cross-rank idle (in `prepare`) | +9.96 | 24 % | **load balancing** |
| MLA decode kernel | +7.47 | 18 % | kernel work |
| `ep_combine` exposed vs fused a2a | +6.17 | 15 % | structure |
| `copy` (fill kernels, no B200 counterpart) | +3.85 | 9 % | kernel work |
| quant + misc | +1.7 | 4 % | |
| `gemm` | −0.45 | −1 % | equal |

**Agreed target order, and it is MI355X's, not this document's earlier one:**

1. **MLA decode kernel** (4.06x per call) — it is 18 % directly *and* it is the
   multiplier on the imbalance cost on both nodes, so it is the only item that
   pays twice.
2. **`total_tokens` balancing** — now justified by *both* nodes' logs, not one.
   Worth ~10 ms on MI355X and ~2.9 ms on B200, and it is an ITL↔TTFT A/B because
   it fights `cache_aware` prefix reuse.
3. **MoE pipeline structure** (+12.4 ms) — the largest single work item, but it
   is a fuse-the-stages question, not a tuning knob.

**`prepare`'s CU count is off the list** on MI355X's evidence, and B200's
"raise its parallelism" recommendation is withdrawn — a `wait_i32_until_equals`
spin does not parallelise.
