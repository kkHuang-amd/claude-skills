# OPEN QUESTION — root cause of the R1 gfx1250 gap: TP1-vs-TP2 confound vs hardware

Status: **OPEN / not first priority** (FIX A already recovers production accuracy end-to-end;
this note is to preserve the unresolved *mechanistic* root cause so it isn't re-derived wrong).
node context: H21-18 gfx1250 (TP1) vs gfx950 baseline (TP2). Date: 2026-07-09.

## What is SOLVED (engineering)
- **FIX A** (keep softmax `p` in fp32 for the `P·V` dot: `tl.dot(p, v.to(tl.float32),
  out_dtype=tl.float32)` in `triton_ops/decode_attention.py` + `extend_attention.py`) recovers
  R1 on gfx1250 to **0.925 (40Q) / 0.980 (200Q) / 0.950 (1319Q)** at full speed, production recipe
  (cuda-graph ON, real a8w4 MoE). So removing the `p->bf16` downcast is a *sufficient* cure.

## What is NOT solved (mechanism / true root cause)
Central question: **why does the SAME `p->bf16` downcast cost ~11% on gfx1250 but ~0 on gfx950?**

Key facts that reshape the question:
1. **Same kernel on both.** gfx950 and gfx1250 use the *same launch script and the same Triton
   MLA kernels*; the only difference is **TP: gfx1250 = TP1, gfx950 = TP2**. So this is NOT a
   "different attention backend / kernel path" story — both run the identical `p->bf16` code.
2. **gfx950 matmul microbench REFUTED the hardware-precision hypothesis (E41 -> E42).** gfx950
   bf16 matmul error vs fp64 = ~1.65e-3, *identical* to gfx1250. That number is the **bf16 output
   downcast** (rel precision 2^-8 ~ 3.9e-3), NOT accumulation; hipBLAS accumulates fp32 on both,
   so it is architecture-independent and cannot explain the gap. The proxy microbench measured the
   wrong thing (a single torch/hipBLAS matmul, not the kernel's internal accumulation nor the
   full-model reduction structure).

=> The two runs differ in only two, currently **confounded**, variables:
   - **hardware** (gfx1250 vs gfx950), and
   - **TP degree** (1 vs 2).

## Leading hypothesis: TP degree, not hardware
TP degree changes the **whole-model floating-point reduction order/grouping**. TP2 splits
contraction dims across 2 ranks, computes partial sums, and combines them via (typically fp32)
all-reduce. The same per-element `p->bf16` error can accumulate/cancel very differently under
TP2's "split into smaller partials + fp32 cross-rank combine" vs TP1's "one long accumulation".
This plausibly explains "same bf16 downcast, TP1 drops 11%, TP2 does not". It also explains why the
single-matmul microbench was blind to it (the effect lives in the model-level reduction structure,
not a lone matmul).

Caveat: MLA `P·V` is per-head and TP splits heads across ranks, so `P·V` per head is bit-identical
regardless of TP; the TP-sensitive part is more likely the downstream projections / residual /
all-reduce chain that the bf16-p error feeds into. Mechanism is a hypothesis, not proven.

## DECISIVE experiment to deconfound (pick one; neither needs FIX A)
- **A (preferred, single node in hand): gfx950 @ TP1**, everything else unchanged.
  - gfx950-TP1 also drops to ~0.81  => differentiator is **TP** (bf16-p is a real universal bug,
    TP2's reduction structure masks it). Root cause = bf16-p + TP-dependent masking.
  - gfx950-TP1 stays ~0.93          => differentiator is **hardware** (gfx950 tolerates the same
    bf16-p triton kernel); dig into arch-specific in-kernel accumulation / wave width.
- **B: gfx1250 @ TP2** (needs 2 cards), no FIX A.
  - recovers to ~0.93 => proves **TP**, hardware-independent.

## Follow-up if TP is confirmed the cause
- The "true" microbench should exercise the **actual attention kernel** (force fp32 vs bf16
  output/accumulation of `P·V`), and ideally under TP1 vs TP2, not the torch/hipBLAS matmul proxy.
- Reconcile with E39's earlier "MoE + attention two-kernel mutual masking" theory vs E37's
  "FIX A alone suffices".

## Cross-refs
- EXPERIMENT_LOG.md: E36–E39 (FIX A validation, SOLVED), E41 (refuted hw hypothesis), E42 (gfx950
  matmul microbench = identical -> E41 refuted).
- results_gfx950.md (E42 raw data), STATUS.md (microbench DONE E42), CHANGES.md (FIX A).
- scripts/matmul_prec.py (the matmul proxy microbench — note: insufficient for this question).
