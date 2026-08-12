# FlyDSL performance patterns learned from Kimi-K3 KDA

## Scope

This note captures reusable techniques from the gfx950 Kimi-K3 fused
`f_b + convolution + recurrent KDA + gated RMSNorm` comparison.

Measured identical-contract CUDA graph latency:

```text
Batch   AITER FlyDSL   SGLang Triton   FlyDSL advantage
1       11.69 us       29.50 us        2.52x
8       13.34 us       30.06 us        2.25x
16      14.52 us       31.54 us        2.17x
32      19.78 us       38.55 us        1.95x
```

The Triton kernel was numerically correct, graph-safe, used gfx950 MFMA,
reported zero scratch/VGPR memory spills, and still lost badly. The gap is
therefore mostly schedule/lowering quality rather than an algorithm mismatch.

Source anchors:

```text
FlyDSL:
/sgl-workspace/aiter-kda-4495/aiter/ops/flydsl/kernels/kimi_k3_kda_decode_fb.py

Triton experiment:
/root/.cursor/worktrees/k3-kda-7c4e91af/sglang-ff23c246a4b4/
python/sglang/kernels/ops/attention/kda_fused_decode_triton_hip.py
```

## Ranked root causes

### 1. Explicit subgroup reductions versus compiler-generated tensor reductions

FlyDSL assigns K to eight-lane subgroups. Each lane owns four adjacent FP32
values and reduces with three XOR shuffles (`1, 2, 4`). These are wave-local
operations and need no workgroup barrier.

The Triton implementation expresses reductions as `tl.sum` over `[32,128]`
tensors. Its lowering introduces layout conversions, DPP/DS reduction trees,
and repeated CTA synchronization. The inspected artifact contained roughly
56 `s_barrier` instructions, versus three intentional workgroup barriers in
the FlyDSL kernel.

Reusable rule:

> For small fixed reductions on AMD, explicitly map a subgroup and use
> wave-local shuffles. Do not assume a high-level tensor reduction will remain
> wave-local.

### 2. Do the actual GEMV, not a fake GEMM

The Triton kernel forces the `128x128 @ 128x1` projection through `tl.dot` by
replicating the single output column 16 times. This causes:

- a `128x128 @ 128x16` MFMA operation;
- 32 KiB LDS staging;
- eight MFMA instructions;
- a redundant max/reduction to collapse identical columns;
- extra layout conversions and barriers.

FlyDSL gives each of threads 0–127 one output row, performs vector4 FP32 FMAs,
reduces four register values, and writes one BF16 value.

Reusable rule:

> MFMA is not automatically faster. For fixed skinny GEMV boundaries, compare
> direct vector FMA against padded/replicated MFMA and include LDS/reduction
> cost, not only FLOP throughput.

### 3. Guarantee vector VMEM transactions

FlyDSL represents recurrent state fragments as explicit FP32x4 vectors and
uses `vec_load`/`vec_store`. One lane issues one aligned 16-byte state
transaction.

The Triton artifact used mostly scalar `buffer_load_dword` and
`buffer_store_dword` instructions for state traffic. Similar byte volume
therefore required many more instructions and address calculations.

Reusable rule:

> Make vector width part of the kernel's logical ownership model. Validate the
> generated ISA contains the expected `buffer_load/store_dwordx4`; source-level
> contiguous tensors are not sufficient evidence.

### 4. Reuse each loaded state fragment for both dot products

FlyDSL loads and decays one state vector, then accumulates both:

```text
h_decay dot k
h_decay dot q
```

from the same register values. It computes `k dot q` once and forms:

```text
recurrent = (h_decay dot q) + v_new * (k dot q)
```

The Triton lowering constructs a large `h_new` tensor and performs a later
reduction, extending live ranges and increasing synchronization/layout work.

Reusable rule:

> When two reductions consume the same bandwidth-dominant tensor, accumulate
> them together before writing the updated tensor.

### 5. Keep BF16 staging in LDS, not global output

The numerical contract requires recurrent output to round to BF16 before
RMSNorm. FlyDSL preserves this boundary in LDS and writes final output once.

The Triton experiment writes BF16 recurrent values to the global output,
barriers, reloads them, and overwrites the same output after RMSNorm.

Reusable rule:

> A required rounding boundary does not imply a global-memory boundary. Stage
> rounded intermediates in LDS when they are consumed in the same workgroup.

### 6. Control register lifetime, not only spill count

Triton metadata:

```text
211 VGPR
106 SGPR
occupancy 2
32 SGPR values spilled into VGPR lanes
zero scratch-memory spills
```

The unrolled 32-row state tiles and tensor reductions keep many values live.
Zero scratch spills is necessary but not sufficient; high register pressure
still limits scheduling freedom and occupancy.

FlyDSL explicitly chooses register vectors and uses
`amdgpu-expert-scheduling-mode`.

Reusable rule:

> Track VGPR/SGPR count, occupancy, lane spills, code size, and live ranges.
> “Zero scratch spills” alone is not a performance verdict.

## Portable optimization checklist

Before accepting a fixed-shape AMD kernel:

1. Count generated barriers; explain every one.
2. Count vector versus scalar VMEM operations for the dominant state tensor.
3. Verify reduction scope is lane/wave/workgroup as intended.
4. Check whether skinny projections are doing duplicated MFMA work.
5. Reuse loaded fragments across all dependent reductions.
6. Keep intermediate rounding in registers/LDS where possible.
7. Record VGPR, SGPR, lane spills, scratch, occupancy, and code size.
8. Benchmark graph replay with rotating weights/data to avoid cache-only wins.
9. Compare complete boundaries, including state mutation and output epilogue.
10. Validate output, state, and auxiliary cache independently.

## Triton experiments in priority order

These are the most promising ways to narrow the gap:

1. Implement eight-lane K subgroups with explicit XOR shuffle support or AMD
   inline assembly; target fewer than ten total barriers.
2. Reshape state ownership around FP32x4 and verify vector4 VMEM in ISA.
3. Replace fake-MFMA `f_b` projection with one-output-per-thread vector GEMV.
4. Fuse `h dot k` and `h dot q` in one state pass.
5. Stage rounded recurrent BF16 values in LDS.
6. Sweep state rows per wave only after the above; optimize generated VGPR and
   barrier count rather than source `BLOCK_V`.

A realistic stock-Triton target is approximately 20–25 us. Matching FlyDSL's
12–20 us likely requires low-level AMD shuffle/layout control that ordinary
tensor-level Triton does not currently expose reliably.

## Production lesson

The active FlyDSL integration reduced the production KDA graph bucket from
approximately 1.88 ms to 1.26 ms for 69 launches and improved total serving
throughput by 2.9%–5.6% across concurrency 2–32.

The important pattern is not “always choose FlyDSL.” It is:

> Use the DSL that lets the implementation encode the hardware ownership,
> vector width, reduction scope, and synchronization contract explicitly—and
> verify those choices in generated ISA and endpoint measurements.
