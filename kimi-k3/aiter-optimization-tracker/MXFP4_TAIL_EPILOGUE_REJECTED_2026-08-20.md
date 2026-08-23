# Kimi-K3 MXFP4 latent-up fused epilogue assessment — 2026-08-20

Follow-up profiler/tuning work eliminated the generic FlyDSL GEMM scratch and
reached plain-GEMM parity, but confirmed that the fused epilogue mapping still
regresses endpoint performance. See
[`MXFP4_DENSE_KERNEL_TUNING_2026-08-20.md`](MXFP4_DENSE_KERNEL_TUNING_2026-08-20.md).

## Decision

Reject both evaluated implementation routes before model integration:

```text
Route A: existing AITER Triton preshuffle MXFP4 fused add
Route C: existing AITER FlyDSL MXFP4 GEMM2 adapted as a dense one-expert GEMM
```

Neither route satisfies the micro performance gate against the current gfx950
ASM `gemm_a4w4 + _add3` chain. No epilogue changes remain in AITER or SGLang,
and no endpoint/GSM8K runs were needed.

## Baseline operation

The retained up-only profile executes:

```text
per-1x32 activation quant
gfx950 ASM gemm_a4w4
_add3(gemm_output, shared_output, optional prefix_sum)
```

Target shape:

```text
M >= 2048
K = 3584
N = 7168
BF16 output
```

## Route A: existing Triton fused GEMM

The first attempt exposed two compatibility boundaries:

1. Triton 3.7 cannot bind AITER's `float4_e2m1fn_x2` pointer directly; the
   official Triton contract uses uint8 packed values.
2. The official preshuffle layout differs from the gfx950 ASM weight/scale
   layout used by the current adapter.

After converting to the official Triton uint8 weight/scale layout, the kernel
ran but was slower and numerically different from the production ASM path:

```text
M      prefix   baseline us   Triton us   candidate / baseline   cosine
2048   no            60.113       67.375          1.121x          0.9363
2048   yes          126.656      142.319          1.124x          0.9526
4096   no            90.427      110.555          1.223x          0.9356
4096   yes          100.970      143.253          1.419x          0.9522
8192   no           193.741      240.237          1.240x          0.9357
8192   yes          203.756      267.809          1.314x          0.9522
```

The path fails the requirement that every tested M/prefix case improve and at
least one improve by 3%.

## Route C: existing FlyDSL MXFP4 GEMM2

AITER's `mxfp4_gemm2` was exercised as a dense NE=1 GEMM with fixed Kimi-K3
dimensions. It is bit-identical to the current ASM GEMM for the tested inputs,
but the GEMM body is too slow at the large-M shapes that matter:

```text
M      prefix   baseline chain us   estimated fused us   estimated speedup
2048   no                   67.181               88.173          0.762x
2048   yes                 219.307               79.589          2.755x*
4096   no                  192.897              189.600          1.017x
4096   yes                 192.423              213.842          0.900x
8192   no                  184.735              414.018          0.446x
8192   yes                 205.351              418.555          0.491x
```

`*` The isolated M2048 prefix baseline was an outlier and does not override the
consistent M4096/M8192 failure. At M8192, even subtracting the entire measured
add3 cost leaves FlyDSL about 2x slower than the current chain.

Extending only the FlyDSL epilogue cannot recover the GEMM-body gap. A
competitive Route C would require a new gfx950 ASM shader or a substantially
new dense FlyDSL/CK implementation and tuning campaign, not a small fusion.

## Artifacts

```text
/workspace/kimi-k3-runs/mxfp4-tail-epilogue-2026-08-20/
  micro/run_route_a_gate.py
  micro/route-a-gate-v4.json
  micro/route-a-gate-v4.log
  micro/run_route_c_gate.py
  micro/route-c-gate.json
  micro/route-c-gate.log
```

## Next

Keep the up-only profile unchanged. The next higher-value direction is to
extend the existing all-reduce + RMSNorm epilogue to emit MXFP4 activation
values and E8M0 scales directly for latent-up. That preserves the fast gfx950
ASM GEMM while removing the standalone activation-quant launch and BF16 latent
materialization.
