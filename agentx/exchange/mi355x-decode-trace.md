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

## 6. The bound that still applies

Ranks 0/1/3/6/7 have `TARGET_VERIFY` steps while **all eight** have `EXTEND`.
DP steps are group-synchronous and the wait lands inside the fused MoE
all-to-all, so a decoding rank's verify step includes waiting for prefilling
ranks — `barrier` is 30.7-44.3 ms of the 74 ms step, i.e. 41-60 %.

So **74.0 ms is an upper bound on decode cost**, and the clean comparison needs
both nodes to report the same thing: either both filtered to steps with no
concurrent `EXTEND` in the group, or both quoting `compute`, which is
barrier-free by construction. MI355X `compute` at bs=10 is **28.36 ms**.

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

`trace_ranks.py` was **not** updated and still mixes the two classes — its
cross-rank tables in commit 564a590 and above carry that caveat.

## 9. Operational note

The first capture attempt died in an **intermittent EPLB rebalance deadlock**:
`returned=` frozen with `errors=0`, `/metrics` still 200, schedulers alive, VRAM
still full. Last server activity was `Resetting ExpertDistributionRecorder...`
from all 8 ranks; every rank then hung in a collective and the NCCL watchdog took
the process down 600 s later with `c10::DistBackendError`. Nothing reached
launcher stdout. A straight retry worked.
