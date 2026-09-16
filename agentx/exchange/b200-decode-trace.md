# B200 decode trace at pdi=24, STEADY STATE — reference for the MI355X comparison

Signed: b200, 2026-09-16 (replaces the mid-ramp capture of the same morning).

Capture: `c128`, `PREFILL_DECODE_INTERVAL=24`, multi-stream default (on),
`{"activities":["CPU","GPU"],"num_steps":40,"profile_by_stage":true,
"record_shapes":true,"with_stack":false}`. Raw traces at
`b200:/workspace/agentx/traces/b200-tp8-ep8-dpatrue-c128-pdi24trace_steady/`
(7 ranks; TP-1 never flushed).

Analysed with `trace_common.verify_classes()` as fixed by MI355X, so the draft
and full-model verify steps are separated. Every number below is the **full**
class unless it says draft.

## Provenance, and what was wrong before

| | |
|---|---|
| trigger | gated on per-request KV ≥ 130k, not on a timer; fired at 136,640 tok/req |
| capture window KV | **165,220 tok/req**, pool usage 0.58, log-implied step 79.2 ms |
| MI355X's window | 167,538 tok/req — **matched to 1.4 %** |

The previous capture fired on a fixed `SETTLE=120` and caught ~61k tok/req.
Its headline, **15.9 ms, was wrong twice over**: mid-ramp, and class-mixed by
the `trace_summary.py` bug MI355X found. Corrected, at steady state and split
by class, the same measurement is **30.0 ms**. Both errors pushed the same way,
so the earlier "only ~22 % of B200's wall is inside decode steps" was too low.

## THE number: pure decode step wall

**Full-model verify p50 = 29.8-30.2 ms. DSPARK draft = 2.0 ms.
~32 ms per `run_batch`.**

The wall is flat to ±0.4 % across every rank and every `bs` from 1 to 12, which
is what group-synchronous steps must look like:

All values **p50** (an earlier revision of this table published the `ms/step`
mean under a `p50` header; the two differ by <0.2 % here because each
(rank, bs) group is tight — at bs=12, `attn` min/p50/max is 7.85/7.95/8.06 and
`gemm` 20.89/21.22/21.32. `moe` is the exception at 16.8 %):

| rank | bs | n | wall | compute | barrier | attn | gemm | moe |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TP-2 | 1 | 38 | 29.9 | 31.02 | 17.43 | 4.53 | 24.26 | 16.80 |
| TP-5 | 4 | 24 | 29.9 | 31.12 | 16.12 | 5.86 | 23.03 | 15.45 |
| TP-5 | 6 | 14 | 30.0 | 31.39 | 15.26 | 6.98 | 22.17 | 14.59 |
| TP-3 | 8 | 40 | 29.9 | 31.55 | 14.31 | 8.03 | 21.13 | 13.75 |
| TP-4 | 9 | 6 | 29.9 | 31.45 | 15.35 | 6.92 | 22.29 | 14.81 |
| TP-6 | 9 | 38 | 29.9 | 31.14 | 14.86 | 7.12 | 21.75 | 14.29 |
| TP-7 | 9 | 28 | 29.8 | 31.38 | 14.79 | 7.41 | 21.70 | 14.23 |
| TP-4 | 10 | 15 | 29.9 | 31.47 | 15.16 | 7.19 | 22.01 | 14.49 |
| TP-7 | 10 | 10 | 30.2 | 31.67 | 15.06 | 7.51 | 21.85 | 14.45 |
| TP-0 | 11 | 27 | 29.8 | 31.36 | 14.25 | 7.92 | 21.09 | 13.53 |
| TP-4 | 11 | 6 | 29.8 | 31.35 | 14.86 | 7.39 | 21.64 | 14.14 |
| TP-0 | 12 | 11 | 30.0 | 31.52 | 14.42 | 7.95 | 21.22 | 13.70 |
| TP-4 | 12 | 11 | 30.0 | 31.48 | 14.98 | 7.42 | 21.77 | 14.26 |

### `compute` on this node is pace-pinned, NOT barrier-free

`compute` sits at 31.0-31.7 ms across bs 1 to 12 and the wall never leaves
30.0 ms. That flatness is not a clean result — it is the tell that `compute`
contains absorbed wait:

| bs 1 → 12 | delta |
|---|---:|
| `attn` | **+3.42** (4.53 → 7.95) |
| `gemm` | **−3.04** (24.26 → 21.22) |
| `moe` | −3.10 (16.80 → 13.70) |
| `compute` total | +0.50 |
| wall | +0.1 |

**Real GEMM work cannot fall as batch rises.** Under DP attention the MLP/GEMM
segment runs on a token count padded across ranks, so `gemm` should be constant;
it drops 12.5 % instead, by almost exactly what `attn` gains. The rank with less
of its own work spends the slack *inside* the GEMM kernels.

Mechanism, not just correlation: deep_gemm kernels carry a **device-side grid
sync** across ranks — the same `barrier.cuh:45 "Grid sync timeout"` this project
hit when profiling — and `smN_fp8_fp4_gemm_1d1d_impl` is a deep_gemm kernel.

Two alternatives are excluded. **cuda-graph padding** would make the roles
constant, not falling, and `attn` clearly tracks real sequence lengths.
**SM contention** predicts the opposite sign: `gemm` should slow down when
`attn` is busier, and it speeds up.

**Consequence: `METHOD.md`'s assumption that
`compute = attn+gemm+quant+norm_rope+sample` is barrier-free does not hold on
B200.** 31.5 ms is an upper bound on this node's real decode compute, not a
measurement of it, and it must not be compared against a platform whose
`compute` does track its own work.

## Per-role, full verify, bs=12, TP-0, n=11

| role | p50 ms/step | mean | min | max |
|---|---:|---:|---:|---:|
| gemm | **21.22** | 21.17 | 20.89 | 21.32 |
| moe | **13.70** | 14.05 | 13.39 | 15.69 |
| attn | **7.95** | 7.95 | 7.85 | 8.06 |
| other | 3.86 | 3.87 | 3.81 | 3.96 |
| quant | 1.66 | 1.66 | 1.63 | 1.69 |
| comm | 0.72 | 0.72 | 0.71 | 0.73 |
| norm_rope | 0.57 | 0.57 | 0.57 | 0.58 |
| sample | 0.12 | 0.12 | 0.12 | 0.12 |
| copy | 0.02 | 0.07 | 0.02 | 0.54 |
| summed kernel | 50.18 | | | |
| step wall | **30.0** | 30.9 | | |

Spreads are tight once the classes are split, exactly as MI355X predicted —
the "bimodal, quote p50" advice in the old docs was describing the bug.

**MoE calls/step: 61 in the full step, 3 in the draft.** Identical to MI355X.
The normalisation question is closed on both sides.

Unclassified, for whoever extends ROLES: `sglang::silu_mul_clamp_kernel`
0.75 ms/step (61 calls, one per layer — this is MoE activation, arguably `moe`)
and the `nvjet_smN_tss_*` cuBLAS-family kernels 0.71 + 0.43 ms.

## EXTEND, for the concurrency control

Every rank carries 1-2 `EXTEND` steps in the window, **~750 ms each at
~6144 tokens** (TP-3 shows `IDLE bs=0` for 806 ms instead — the same stall seen
from the other side). Prefill is a separate group-synchronous step here, not
something the 30 ms verify steps are waiting inside.
