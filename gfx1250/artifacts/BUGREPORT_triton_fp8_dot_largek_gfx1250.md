# triton `tl.dot(fp8, fp8)` returns garbage for contraction dim K >= 128 on AMD gfx1250

## Summary
On AMD **gfx1250**, a triton `tl.dot` with **both operands `float8_e4m3fn`** silently returns
garbage (~1e34–1e36) once the **contraction dimension `K >= 128`**. `K <= 64` is correct, and the
same kernel with the operands upcast to `bfloat16` (`bf16 x bf16`) is correct at all `K`. The dot
itself does not error or produce NaN; the corrupt (~1e36) values only become NaN downstream (e.g.
after a `softmax`/`exp`).

This makes any fp8 KV-cache attention that performs an `fp8 x fp8` QK or PV dot with `K >= 128`
produce garbage — e.g. DeepSeek-R1 MLA (nope contraction dim `K = 512`) with `--kv-cache-dtype
fp8_e4m3` degenerates to empty/whitespace output.

## Environment
- GPU: AMD gfx1250 (`torch.cuda.get_device_properties(0).gcnArchName == "gfx1250"`)
- triton: 3.7.1 (custom ROCm build)
- torch: ROCm build; dtype `torch.float8_e4m3fn`
- Docker image: `henryx/xsgl:v0.5.14-gfx1250-rocm-nightlies-20260709-trial-3`

## Reproduction
`triton_fp8_dot_largek_gfx1250_repro.py` (single program, single tile, no K-loop, no masking;
small in-range values so there is no fp8 overflow). Compares `tl.dot` output to an fp64 recompute
of the **same** fp8-rounded operands.

```
arch: gfx1250
triton: 3.7.1

fp8_e4m3fn x fp8_e4m3fn, M=16 N=16, small in-range values (|a|<=~2):
     K     fp8 rel_l2     fp8 |c|max    bf16 rel_l2
    64      0.000e+00      4.860e+00      0.000e+00
   128      3.216e+34      1.062e+36      0.000e+00  <-- GARBAGE
   256      4.912e+36      2.182e+38      0.000e+00  <-- GARBAGE
   512      3.123e+36      2.215e+38      0.000e+00  <-- GARBAGE
```

Expected: `rel_l2 ~1e-7..1e-5` for all K (as seen for K=64 and for the bf16 control).
Observed: `rel_l2 ~1e34+` and `|c|max ~1e36` for K >= 128.

## Scope / notes
- **Threshold**: correct at `K = 64`, broken at `K in {128, 256, 512}`. (Only powers of two tested
  due to the minimal kernel's `tl.arange`; the break is at/below 128.)
- **dtype-specific**: `bf16 x bf16` with identical shapes/values is correct at all K (rel_l2 0).
  Only `fp8 x fp8` is affected. Upcasting either operand to bf16 before `tl.dot` avoids it.
- **kernel-args independent**: reproduces with default launch args and with the AMD tuning args
  `waves_per_eu=1, matrix_instr_nonkdim=16, kpack=2, num_stages=1` — no difference.
- **arch-specific**: the identical fp8 x fp8 K=512 dot is **correct on gfx950** (a real DeepSeek-R1
  MLA decode with the same triton code + fp8 KV scores GSM8K 0.941 on gfx950, vs 0.000 on gfx1250).
  So this is a gfx1250 fp8-MFMA / codegen issue, not a generic triton bug.
- Silent: no compile error, no runtime error, no NaN at the dot — corruption is only detectable by
  comparing against a reference or by a downstream NaN.

## Suspected cause
gfx1250-specific lowering/codegen of the fp8 (e4m3) matrix-multiply for K >= 128 (e.g. wrong
K-tiling / accumulation / MFMA instruction selection for the fp8 path). bf16 lowering is correct.

## Workaround
Upcast fp8 operands to bf16 before `tl.dot` when K >= 128 on gfx1250 (K-dim bf16 accumulation is
correct), or tile the contraction dim into <= 64 chunks.
