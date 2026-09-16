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

| rank | bs | n | wall p50 | compute p50 | barrier p50 | attn | gemm | moe | comm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| TP-0 | 11 | 27 | 29.8 | 31.38 | 14.58 | 7.92 | 21.11 | 13.86 | 0.72 |
| TP-0 | 12 | 11 | 30.0 | 31.47 | 14.77 | 7.95 | 21.17 | 14.05 | 0.72 |
| TP-2 | 1 | 38 | 29.9 | 31.03 | 17.86 | 4.54 | 24.26 | 17.23 | 0.63 |
| TP-3 | 8 | 40 | 29.9 | 31.57 | 14.79 | 8.03 | 21.16 | 14.23 | 0.56 |
| TP-4 | 10 | 15 | 29.9 | 31.45 | 15.53 | 7.19 | 21.99 | 14.86 | 0.67 |
| TP-5 | 4 | 24 | 29.9 | 31.09 | 16.46 | 5.86 | 23.00 | 15.79 | 0.67 |
| TP-6 | 9 | 38 | 29.9 | 31.16 | 14.96 | 7.13 | 21.76 | 14.40 | 0.56 |
| TP-7 | 9 | 28 | 29.8 | 31.36 | 14.85 | 7.41 | 21.68 | 14.29 | 0.56 |

`compute` is **31.0-31.6 ms across bs 1 to 12** — flat, confirming the earlier
finding at a valid window. Within it `attn` rises with bs (4.54 → 7.95) and
`gemm` falls (24.26 → 21.17), so quote the split with its bs.

`compute + barrier` is ~46-49 ms and near-constant per rank while `compute`
alone varies — the group-sync signature MI355X describes, on this node too.

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
