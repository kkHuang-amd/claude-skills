# Handover: run the bf16-vs-fp32 matmul microbench on gfx950 (confirm gfx950 bf16 is accurate)

## Why
E41 (node H21-18 gfx1250) showed gfx1250's **bf16 matmul is lossy** (~0.16-0.28% vs fp64;
fp32 ~1e-6), which is the root cause of the R1 accuracy gap (the attention P·V dot downcast the
softmax weights to bf16). Hypothesis: **gfx950's bf16 matmul (CDNA4 MFMA) is accurate (~fp32)**,
which is why gfx950 never needed the fix. This handover runs the SAME test on gfx950 to confirm.

## What to run on the gfx950 box
Script (portable, torch-only, no arch-specific code): `scripts/matmul_prec.py` (this dir).
```bash
cp <skill>/scripts/matmul_prec.py /tmp/matmul_prec.py
HIP_VISIBLE_DEVICES=<a gfx950 gpu> python3 /tmp/matmul_prec.py
```
It builds attention-PV-like matrices (A = softmax rows / probabilities, B = gaussian), feeds the
SAME values as bf16 and fp32, and compares each to an fp64 recompute. Prints, for K in {256,512,2048}:
- `bf16 matmul vs fp64` (total, incl. input rounding)
- `fp32 matmul vs fp64`
- `HARDWARE accum (same vals vs fp64)`: bf16 vs fp32 (isolates accumulation precision)
- `ratio bf16/fp32`

## gfx1250 reference numbers (E41, for comparison)
| K | bf16 vs fp64 | fp32 vs fp64 | ratio | same-vals bf16-accum vs fp64 |
|---|---|---|---|---|
| 256  | 2.87e-3 | 2.8e-7 | ~10000x | 1.65e-3 |
| 512  | 2.83e-3 | 4.0e-7 | ~7000x  | 1.66e-3 |
| 2048 | 2.89e-3 | 8.2e-7 | ~3500x  | 1.67e-3 |
Key gfx1250 signal: even with the SAME bf16 values, `A_bf16 @ B_bf16` differs from an fp64
recompute by **1.65e-3** — if it accumulated in true fp32 this would be ~1e-6, so gfx1250's bf16
matmul uses a LOSSY (bf16/reduced-precision) accumulation.

## How to read the gfx950 result
- **CONFIRMS the hypothesis** if gfx950's `bf16 matmul vs fp64` and especially the "same-vals
  bf16-accum vs fp64" are MUCH smaller than gfx1250's (e.g. ~1e-6 for the accum, i.e. gfx950 bf16
  accumulates in true fp32). Then: gfx950 bf16 is accurate -> it never needed FIX A; the R1 gap
  was gfx1250-specific bf16 lossiness.
- If gfx950 shows the SAME ~1.6e-3 bf16 accum error, then bf16 matmul is lossy on BOTH and the
  gfx950-vs-gfx1250 difference must be elsewhere (unexpected given gfx950=0.93 with the bf16-PV
  code) — re-open the question.

## Report back
Append an entry to EXPERIMENT_LOG.md tagged `[node: <gfx950-host> gfx950]` with the gfx950
numbers next to the gfx1250 reference table, and update the E41 conclusion (confirmed / not).

## Notes
- Pure torch/hipblas (not the exact triton WMMA) — same as the gfx1250 run, so it's apples-to-apples.
- If the box OOMs or the shape is awkward, the K values / M,N in `matmul_prec.py` can be tuned; keep
  the same shapes on both nodes for a fair comparison.
