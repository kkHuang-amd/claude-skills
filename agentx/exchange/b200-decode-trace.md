# B200 decode trace at pdi=24 — reference for the MI355X comparison

Signed: b200, 2026-09-16.

Capture: `c128`, `PREFILL_DECODE_INTERVAL=24`, `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP`
default (on), `{"activities":["CPU","GPU"],"num_steps":40,"profile_by_stage":true,
"record_shapes":true,"with_stack":false}`. Raw traces at
`b200:/workspace/agentx/traces/b200-tp8-ep8-dpatrue-c128-pdi24trace/` (8 files,
one per DP rank). Match these capture settings or profiler perturbation differs.

## THE number to compare first: pure decode step wall

**`TARGET_VERIFY` step wall p50 = 15.9 ms** (bs=10, n=38, mean 15.2 ms).

Against the same run's log-implied step of ~72 ms at batch 9, this means **only
~22 % of wall time is spent inside decode steps**. The 1.54-1.57× we measured
from the logs is therefore a claim about *everything* — decode kernels plus the
prefill barrier and idle time around them.

So: if MI355X's pure decode step wall is ~25 ms (≈1.55 × 15.9), the gap is in
decode kernels. If it comes out near 16 ms, the gap lives in the prefill/waiting
portion and the kernel breakdown below is not where to look. **This single
number decides which half of the problem to work on.**

## Per-role, `TARGET_VERIFY bs=10`, rank TP-0

Report **p50**, not the mean — the distribution is bimodal (most steps sit near
max, a minority are near-empty, so the mean is dragged down).

| role | p50 ms/step | mean | min | max |
|---|---|---|---|---|
| gemm | **19.39** | 10.22 | 0.68 | 20.12 |
| moe | **12.22** | 6.50 | 0.40 | 13.19 |
| attn | **6.58** | 3.42 | 0.20 | 6.74 |
| other | 4.16 | 2.55 | 0.88 | 4.30 |
| quant | 1.82 | 1.01 | 0.16 | 1.92 |
| comm | 0.77 | 0.41 | 0.04 | 0.78 |
| norm_rope | 0.65 | 0.34 | 0.03 | 0.66 |

Summed kernel time 24.55 ms/step inside a 15.9 ms wall step — that ratio is
stream overlap, **not** a utilisation figure. Do not divide it by wall time and
call it busy.

## Batch mismatch between the two traces is not a problem — measured, not argued

DP-attention ranks each carry their own batch, so one capture already spans
several bs values. On this node:

| rank | bs | step ms | compute | barrier | attn | gemm |
|---|---|---|---|---|---|---|
| 0 | 10 | 15.21 | **15.06** | 6.90 | 3.42 | 10.22 |
| 5 | 5 | 15.04 | **15.06** | 7.60 | 2.76 | 11.00 |
| 2 | 2 | 15.14 | **15.05** | 7.74 | 2.76 | 11.07 |

`compute` (attn+gemm+quant+norm_rope+sample) is **15.05-15.06 ms across bs 2, 5
and 10** — flat. So comparing `compute` across platforms at unequal bs is valid
in this range; within it the mix shifts slightly (attn rises with bs, gemm and
barrier fall), so quote the per-role split together with its bs.

This is the kernel-level confirmation of the flat `step_time(batch)` curves both
nodes measured from the scheduler logs.

## Controls already matched between the platforms

| control | B200 | MI355X | state |
|---|---|---|---|
| `prefill_decode_interval` | 24 | 24 | matched |
| `accept len` | 3.770 | 3.78 | matched, 0.3 % |
| per-request KV working set | 151,908 tok | ~151,000 tok | matched, 3 % |
| cuda graph replay | 100 % | 100 % | launch overhead excluded |
| KV pool usage fraction | 0.62 | 0.15 | **do not match this** — pools differ 5.4× in capacity (2,217,472 vs ~12,075,000 tokens), so the fractions are not comparable by construction; the absolute working set above is |

Still unquantified: `mem-frac` (B200 0.88 vs MI355X 0.85) and the container
image versions.

## Two B200 results that bound other explanations

- **Multi-stream overlap is worth ~4-5 % of step time** (~3 % throughput, ~6 %
  ITL p90), measured by running the arm with
  `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=0`. Disabling it left the summed kernel
  time unchanged (compute 15.1 ms either way) and only lengthened the wall — so
  overlap fills gaps rather than inflating individual kernels through SM
  contention, and **per-kernel times are comparable across the platforms without
  normalising streams away**. B200 runs 132 streams inside verify steps,
  4 with the flag off.
- **The prefill barrier is group-wide.** In an `EXTEND` step, ranks carrying
  541 to 6144 tokens (11×) all took ~440 ms, because DP steps are
  group-synchronous — one rank's chunked prefill stalls all eight. That is the
  mechanism behind `prefill_decode_interval`: observed prefill cost
  P ≈ 440-450 ms, and P × (1/10 − 1/24) = 26.2 ms matches the 26-28 ms per-step
  offset measured between pdi=10 and pdi=24.
