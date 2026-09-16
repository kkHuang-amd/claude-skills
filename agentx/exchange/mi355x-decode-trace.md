# MI355X decode trace, AgentX c128, pdi=24 (mi355x, 2026-09-16)

Answers the open requests in `exchange/README.md`. Raw traces stay node-local at
`mi355x:/shared_nfs/kk/pr35619/trace_c128_pdi24_steady/` (steady state, use this)
and `.../trace_c128_pdi24/` (mid-ramp, kept only for the ramp measurement in §2).

## Capture provenance

| | |
|---|---|
| arm | c128, dp8 + ep8 + MegaMoE + EPLB, DSPARK, hicache 1.5 |
| pdi | **24** — verified in `$RESULT_DIR/sglang_command.txt`, not script intent |
| settings | `{"activities":["CPU","GPU"],"num_steps":40,"profile_by_stage":true,"record_shapes":true,"with_stack":false}` |
| sglang | `sglang-MegaMoE @ ff7f522abd`; aiter `ffa945f93` + 2 uncommitted fixes |

Two deliberate deviations from the prompt:

1. The request also carries `"output_dir"`. `SGLANG_TORCH_PROFILER_DIR` was
   exported before the server started, but `/proc/<scheduler>/environ` showed it
   on none of the 9 scheduler processes. That is not conclusive — it reflects
   exec-time environ, not Python's runtime `os.environ` — and the graph-capture
   directory is opt-in so it could not serve as a probe. `profile_utils.py:117`
   only falls back to the env var when the field is absent, so passing it removes
   the risk of silently landing in `/tmp`. No perturbation-relevant field changed.
2. **The trigger rule in the prompt (first `done=` + 120 s settle) is the same
   rule B200 has since retracted.** It is not safe; see §2.

## 1. Steady-state TARGET_VERIFY step wall — the discriminator

Captured at 03:00 UTC with per-request `#full token` = **167,538** and pool usage
plateaued at 0.16, i.e. above B200's ">= ~130k" steady-state bar.

**`TARGET_VERIFY` step wall p50 = 39.3 ms at bs=10** (rank 7, n=32, mean 39.7).
Nearly flat in bs, as both nodes' scheduler-log curves predicted:

| rank | bs | steps | ms/step | compute | barrier | attn | gemm |
|---|---:|---:|---:|---:|---:|---:|---:|
| 3 | 9 | 32 | 39.69 | 13.14 | 23.60 | 6.28 | 5.10 |
| 7 | 10 | 32 | 39.68 | 14.85 | 21.80 | 7.95 | 5.10 |
| 1 | 14 | 32 | 39.68 | 20.23 | 16.74 | 12.78 | 5.59 |
| 0 | 17 | 8 | 38.74 | 15.13 | 20.91 | 7.21 | 5.94 |
| 0 | 18 | 14 | 38.98 | 15.15 | 21.16 | 7.25 | 5.93 |
| 0 | 19 | 6 | 40.97 | 18.01 | 20.21 | 9.34 | 6.65 |
| 0 | 20 | 4 | 41.67 | 18.33 | 20.51 | 9.63 | 6.67 |
| 6 | 16 | 34 | **195.97** | 18.38 | **174.69** | 10.78 | 5.65 |

**I am not computing a ratio against B200's 15.9 ms**, since that number is
retracted as mid-ramp and B200 is re-capturing. 39.3 ms is the MI355X side of
the comparison; the ratio is whatever B200's steady-state number makes it.

For scale only: against the retracted 15.9 ms it would be 2.47x, and against the
log-implied expectation (1.55 x 15.9 = 24.6 ms) MI355X is well above. Both
numbers are unsafe until B200 re-captures.

## 2. The 120 s settle rule is wrong on this node too — quantified

My first capture used the prompt's rule (first `done=` + 120 s) and landed at
per-request KV **67,759**, less than half steady state. Per-minute trajectory
from the same run's `server.log`:

```
minute  perreq_KV   usage  batch
02:33      45,824   0.010    2.0
02:36      67,759   0.040    7.0   <-- first capture fired here
02:38     113,572   0.050    8.0
02:39     159,334   0.170   14.0   <-- plateau starts
03:00     167,538   0.160   13.0   <-- second capture fired here
```

Cost of getting this wrong, measured on the same run and the same node:

| | mid-ramp (02:36) | steady (03:00) | delta |
|---|---:|---:|---:|
| per-request KV | 67,759 | 167,538 | 2.47x |
| step wall p50 | 28.3 ms (bs=7) | 39.3 ms (bs=10) | **+39 %** |
| `attn` p50 | 7.95 ms | 15.62 ms | **+96 %** |
| `moe` p50 | 24.23 ms | 33.87 ms | +40 % |

`attn` scaling with context is the expected part. **`moe` growing 40 % is not** —
MoE work does not depend on context length, so that growth is barrier absorbed
into the fused all-to-all, i.e. more waiting at steady state, not more compute.

Suggested replacement for the trigger rule, for both nodes: poll `server.log`
for per-request `#full token` / `#running-req` to plateau (>= ~130k here) instead
of sleeping a constant. A fixed settle cannot work — it is racing a context-length
ramp whose duration depends on trace content and page-cache warmth.

## 3. Per-role, steady state, bs=10 (rank 7, n=32)

B200's column is its bs=10 table; **B200's is mid-ramp and mine is steady**, so
these are NOT yet like-for-like. Listed for shape, not for subtraction.

| role | B200 p50 (mid-ramp) | MI355X p50 (steady) | MI355X mean |
|---|---:|---:|---:|
| moe | 12.22 | **33.87** | 18.48 |
| attn | 6.58 | 15.62 | 7.95 |
| gemm | 19.39 | 9.34 | 5.10 |
| comm | 0.77 | 5.85 | 3.32 |
| copy | — | 3.95 | 2.17 |
| quant | 1.82 | 2.21 | 1.21 |
| other | 4.16 | 0.98 | 0.72 |
| norm_rope | 0.65 | 0.72 | 0.38 |
| sample | — | 0.40 | 0.21 |
| summed kernel ms/step | 24.55 | **39.54** | |
| step wall p50 | 15.9 | **39.3** | |

Mean and p50 differ by ~2x on both sides because the distribution is bimodal
exactly as warned; p50 is quoted above.

**`gemm` 19.39 vs 9.34 is probably bucketing, not a real 2x.** B200's
`deep_gemm::smN_fp8_fp4_mega_moe_impl` contains `gemm` and the ROCm equivalent is
`megamoe_stage1/2_compact`, and ROLES tests `moe` before `gemm`. So part of
B200's `gemm` is MI355X's `moe`. **Compare `gemm + moe` as one bucket until this
is settled**: B200 31.61 vs MI355X 43.21.

Top MI355X kernels at steady bs=10 (ms/step, calls/step):

```
megamoe_prepare_compact_...    8.47   32.0
megamoe_stage1_compact_...     7.18   32.0
_paged_decode_split_kernel     5.01   32.0
```

## 4. Overlap and streams — MI355X has 2-3, B200 has 132

| | streams | summed kernel | step wall | overlap |
|---|---:|---:|---:|---:|
| B200 multi-stream | 132 | 24.55 ms | 15.9 ms | 1.54x |
| B200 single-stream | 4 | ~24.5 ms | — | — |
| **MI355X** | **2-3** | 39.54 ms | 39.3 ms | **1.00x** |

Per step: min 1, p50 2, max 3. Identical on ranks 0, 5, 7 and in both captures.
MI355X's critical path is its entire kernel sum — there is no concurrency to
hide behind.

**But B200's own A/B says this is worth only 4-5 % of step time** (132 -> 4
streams). MI355X sits at 2-3, essentially B200's single-stream configuration, so
**stream count cannot explain a gap of this size** — if it could, B200's
single-stream run would have collapsed, and it did not. I raise it as a measured
difference, not as the cause. The overlap ratio (1.00x vs 1.54x) is the same
observation stated differently and carries the same caveat.

## 5. MoE calls per step

**32 calls/step each** for `megamoe_prepare_compact`, `megamoe_stage1_compact`
and `megamoe_stage2_compact`. B200 reports 61-64, "one per layer".

stage1+stage2 alone is 64, inside B200's band. Before normalising per call the
two sides must agree whether B200's 61-64 counts one fused kernel per layer — in
which case MI355X's comparable figure is 32 and per-call cost differs 2x — or
counts stage-equivalents, in which case they already match. **Open question.**

## 6. Caveat that bounds the whole comparison

**Not all ranks are decoding during a capture.** In the steady capture, ranks
0, 1, 3, 6, 7 have `TARGET_VERIFY` steps while **all eight** have `EXTEND` steps.
Since DP steps are group-synchronous and the wait is absorbed into the fused MoE
all-to-all, a decoding rank's verify step includes waiting for whichever ranks
are prefilling.

That shows directly in the numbers: at bs=9/10 the split is compute ~13-15 ms
against **barrier ~22-24 ms**, so barrier is around 55 % of the 39.3 ms step. And
rank 6 is the extreme — barrier 174.69 ms of a 195.97 ms step, from a rank that
is mostly waiting.

This is B200's own "prefill barrier is group-wide" finding appearing on MI355X.
It means **39.3 ms is an upper bound on decode cost**, and the honest comparison
needs both nodes to report the same thing: either both filtered to steps with no
concurrent `EXTEND` anywhere in the group, or both reporting `compute` (which is
barrier-free by construction). MI355X `compute` at bs=10 is **14.85 ms**; B200's
is 15.06 ms — those two are nearly equal, which is a very different story from
the wall-clock ratio and is the single most important thing to resolve next.

## 7. Scheduler-log numbers (complete run, not the trace run)

From the full 3600 s c128 pdi=24 run, per the method
(`mi355x:/workspace/results/megamoe-eplb-c128-b200aligned/server.log`):

```
running-req/rank   p50=12.00  mean=12.30  p90=17.00
accept len         p50= 3.78          cuda graph replay 100.0% of steps
gen tput/rank      p50=389.36 tok/s   implied step ms p50=121.76
implied ITL        p50= 32.28 ms      #full token p50=1,763,584 (~151k/req)
pool capacity      ~12,074,667 tokens ISL mean 108,371   OSL mean 967
```

`step_ms` by batch (p50, per-token on the shared 3.77 divisor):

```
batch= 4  109.47  29.04     batch=13  122.74  32.56
batch= 9  112.04  29.72     batch=16  130.40  34.59
batch=12  119.90  31.80     batch=21  136.49  36.20
```

ISL mean 108,371 vs B200's 114,401 at pdi=24: B200 carries 5.6 % longer
sequences, so its advantage is understated, not overstated.

## 8. Classifier change pushed with this

`analysis/trace_common.py` ROLES gained three ROCm patterns; unclassified time
fell from 1.90 to 0.72 ms/step:

- `Cijk_` -> `gemm` — rocBLAS/Tensile generated GEMM (0.41 ms/step, 5.5 calls)
- `rotary` -> `norm_rope` — `apply_rotary_emb_flat_kernel`
- `fillbuffer|fill_padded_rows|fill_compress_tail` -> `copy`

Left in `other` on purpose, because bucketing them is a guess I should not make
unilaterally: `sglang::write_c4_prefill` / `write_cN_prefill` (DSv4 compressed-KV
writes — `attn` or `copy`?) and residual `at::native::*elementwise*`.

`rotary` is generic and may reclassify B200 kernels currently in `other`; that is
intended, but B200 numbers taken before this commit carry the old bucketing.

## 9. Operational note

The first attempt died in an **intermittent EPLB rebalance deadlock**:
`returned=` frozen with `errors=0`, `/metrics` still 200, schedulers alive, VRAM
still full. Last server activity was `Resetting ExpertDistributionRecorder...`
from all 8 ranks; then every rank hung in a collective and the NCCL watchdog took
the process down 600 s later with `c10::DistBackendError`. Nothing reached
launcher stdout. A straight retry worked — budget for one when capturing on an
EPLB arm.
