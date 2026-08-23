# Kimi-K3 dedicated dense MXFP4 kernel tuning — 2026-08-20

## Decision

The spill-free FlyDSL streamed-B, Route A, and dense/tail GEMM2 experiments
were rolled back from both worktrees on 2026-08-21 at the user's request.
Production remains on the gfx950 ASM path. The measurements and design notes
below are retained as historical evidence.

The work invalidates the earlier conclusion that FlyDSL cannot approach the
gfx950 ASM kernel: removing B/scale live ranges eliminates all scratch and
brings the plain GEMM to parity or better at several isolated shapes. However,
the improvement does not pass the matched endpoint gate, and fused
shared/prefix epilogues remain slower than the existing high-parallelism
`_add3` kernel.

The removed SGLang opt-in interface was
`SGLANG_K3_MOE_LATENT_UP_FLYDSL`; it is no longer registered or documented.

## Profiler attribution

Matched M2048 kernel traces:

```text
                                      ASM 256x256    generic FlyDSL BM128
median/average kernel duration         33.25 us       91.15 us
VGPR                                   256            256
SGPR                                   112            112
LDS                                    163,840 B      49,152 B
scratch/private                        0              1,220 B/workitem
scratch load/store instructions        0              383
workgroups                             224            448
```

The generic GEMM2 kernel materialized all 14 K tiles of B and both scale
streams before compute. This caused compiler spill despite using the same VGPR
count as ASM.

## Streamed-B implementation

AITER changes:

```text
aiter/ops/flydsl/kernels/mxfp4_gemm2.py
aiter/ops/flydsl/mxfp4_gemm2_kernels.py
```

The new compile-time `stream_b` variant loads one B/scale tile at a time,
reducing scratch from `1,220 B/workitem` to zero. Additional experimental
parameters cover A stages, BM256, dense expert indexing, waves-per-EU and
post-mischeduler selection. All defaults preserve existing callers.

SGLang changes:

```text
python/sglang/kernels/ops/kimi_k3/latent_mxfp4_aiter_hip.py
python/sglang/srt/environ.py
docs/docs/references/environment_variables.mdx
test/registered/kernels/ops/kimi_k3/test_latent_mxfp4_aiter_hip.py
```

The adapter uses fixed Kimi-K3 schedules only for isolated shapes that passed
microbench and otherwise falls back to the existing ASM path.

## Plain GEMM tuning

Best spill-free schedules:

```text
M       schedule                         FlyDSL us
2048    BM256, A3, XCD4                  36.63
4096    BM256, A3, XCD4                  62.64
8192    BM128, A3, XCD8                 109.89
16384   BM128, A3, WPE2, XCD4           230.29
```

The kernel is bit-identical to the current ASM output for retained schedules.
BM256/A2 was rejected because some runs produced nonzero relative error.

The SGLang fail-closed adapter retains only M2048 and M16384. M4096/M8192
chain-level tests did not show stable gains after the existing `_add3`.

## Fused epilogue result

Direct and LDS-cshuffle epilogues were implemented and tested with exact staged
rounding:

```text
projected   = bf16(accumulator)
with_shared = bf16(projected + shared)
result      = bf16(with_shared + optional_prefix)
```

Both were rejected and removed. Even without scratch, each GEMM workgroup must
serially process many shared/prefix values, while the separate `_add3` launch
uses much higher output parallelism. Representative M8192:

```text
ASM GEMM + add3:                   about 195-216 us
FlyDSL cshuffle fused epilogue:    about 278-508 us
```

The existing ASM `beta*C` hook is fast but does not satisfy the contract: its
C operand/layout interpretation produced cosine only `0.77-0.84` versus the
current tail.

## Endpoint gates

8K input screening uses M8192, which is fail-closed to ASM in the retained
adapter. Results are effectively neutral/noisy and do not justify promotion.

Matched 16K input, 32 warmups, four measured requests per concurrency:

```text
case       C2 tok/s   C2 TTFT   C16 tok/s   C16 TTFT   C16 TPOT
ASM         1975.69   1401.12     7389.98     7640.48      29.33
FlyDSL      1942.70   1748.39     7385.55     7691.15      29.34

FlyDSL delta:
  C2  throughput -1.67%, TTFT +24.79%
  C16 throughput -0.06%, TTFT +0.66%, TPOT +0.03%
```

Capacity is unchanged at `1,376,952` tokens. The quick gate required at least
`-0.5%` TTFT improvement at C16/C64 and therefore failed. Full C2-C64 and
GSM8K were intentionally skipped.

## Correctness

```text
Kimi-K3 focused tests: 69 passed
additional subtests:   6 passed
AITER FlyDSL stage2:   1 passed
pre-commit:             passed
IDE diagnostics:       no errors
CUDA Graph replay:     passed
```

## Artifacts

```text
/workspace/kimi-k3-runs/mxfp4-dense-kernel-2026-08-20/
  profile/
  isa/
  tuning/
  endpoint/
  aiter.patch
  sglang.patch
```

Patch SHA-256:

```text
AITER  46073213e59ab3b834a32811364009a37de8e7644cf13dc4273e871972a89a6d
SGLang 14935dda4da93a84231ee86fa31ecde18acb814c5edd0cdf647c5f5bedb34055
```

## Next

Before further model integration, profile the candidate inside a real 16K
prefill to reconcile isolated M16384 kernel timing with endpoint TTFT. If the
kernel call itself is faster in situ, investigate synchronization/JIT/cache
effects around the callsite. If it is not faster in situ, move the dense core
to CK Tile or develop a new gfx950 ASM shader; do not spend more time on the
current FlyDSL epilogue mapping.
