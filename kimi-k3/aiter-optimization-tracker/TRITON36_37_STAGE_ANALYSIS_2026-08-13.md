# Triton 3.6 versus 3.7 prefill/decode analysis — 2026-08-13

## Method

SGLang `profile_by_stage` captured prefill and decode separately within one
server launch and one request workload for each runtime:

```text
8192 input / 128 output
C32 / 32 requests
5 prefill steps + 5 decode steps
TP0-TP7
CUDA Graph enabled
```

Only Triton and `triton-kernels` changed. Torch, SGLang, AITER, FlyDSL, model,
flags and hardware were fixed.

## Coverage correction

The stage-separated prefill traces are not complete GPU-kernel records.
TP0 contains 552 `aiter::fused_moe_` CPU calls but only 92 MoE stage1/stage2
GPU events in both runtimes: 16.7% GPU visibility. The global prefill+decode
capture confirmed that Triton 3.7 has a much broader ROCTracer visibility
failure:

```text
kernel                         Triton 3.6   Triton 3.7
MoE stage1                          11868            92
_agg_kernel                         27156           186
fused KDA                            8901             0
MLA merge                            3096             0
cross-device reduce                 23936           187
```

CPU `aiter::fused_moe_` counts are identical at 1564, proving the work still
executed. Aggregate GPU time and family totals below are retained as visible
samples only; they are not complete stage attribution.

## Workload-level result under profiling

```text
                         Triton 3.6   Triton 3.7   Delta
duration                    103.01 s      106.59 s   +3.48%
total token throughput     2584.57       2497.76     -3.36%
median TTFT               89136.15 ms   91533.14 ms +2.69%
median TPOT                 108.98 ms     118.29 ms +8.54%
```

Profiling overhead is large, so these absolute values are not production
performance numbers. The direction agrees with the unprofiled endpoint A/B:
Triton 3.7 is slower, with a larger decode-side effect.

## Prefill

All eight prefill traces had superficially comparable launch counts. Rank
medians:

```text
kernel launches: 119256 -> 119348
GPU kernel sum:  6916.61 -> 6701.12 ms  (-3.12%)
GPU busy union:  6916.61 -> 6701.11 ms  (-3.12%)
stage span:     67070.42 -> 68498.73 ms  (+2.13%)
```

The stage became 2.13% slower while the *visible* GPU kernel time fell 3.12%.
Because only 16.7% of MoE GPU executions were visible, this does not establish
that complete GPU kernel time fell.

### Triton kernel changes

The clearest slower Triton kernel was:

```text
chunk_kda_fwd_kernel_intra_token_parallel
p50: 260.96 -> 271.84 us  (+4.17%)
```

Other major Triton kernels improved:

```text
_recompute_w_u_fwd_kernel:               108.96 -> 107.92 us  (-0.95%)
chunk_kda_fwd_kernel_inter_solve_fused:   57.72 ->  56.00 us  (-2.98%)
kda_gate_chunk_cumsum_vector_kernel:       4.56 ->   4.24 us  (-7.02%)
chunk_gated_delta_rule_fwd_kernel_h:      22.68 ->  18.84 us (-16.94%)
_agg_kernel:                              17.28 ->  11.96 us (-30.79%)
```

The slower intra-token KDA sample is real for the visible launches, but
coverage is insufficient to use net Triton kernel time for stage attribution.

### Synchronization/CPU wait

The strongest wall-time signal was distributed synchronization:

```text
record_param_comms p50: 6.812 -> 7.193 s (+5.59%)
c10d::broadcast_ aggregate across ranks: +3.13 s
```

Nested `aten::_to_copy`, `aten::copy_`, and `aten::index_put_` scopes each
showed about +3.02 s aggregate across ranks. These are overlapping/nested
measurements, not additive costs, and indicate waiting at the same
communication/materialization boundary.

Prefill candidate signal: Triton 3.7 changes pacing around KDA and increases
visible synchronization/communication wait. Incomplete GPU coverage prevents
claiming that this is the primary cause.

## Decode

ROCtracer exposed CUDA Graph replay internals inconsistently by rank. Some
ranks recorded about 14.5k kernel launches while others recorded only 2.5k;
the visible-rank set also differed between Triton 3.6 and 3.7. Aggregate launch
counts and total durations are therefore not directly comparable.

Per-kernel p50 values remain useful. No visible major decode kernel regressed
enough to explain the 8.54% TPOT loss:

```text
AITER cross-device reduce:        12.44 -> 12.52 us  (+0.64%)
Opus MoE stage1:                  84.72 -> 84.64 us  (-0.09%)
Opus MoE stage2:                  44.52 -> 44.44 us  (-0.18%)
SGLang Triton _agg_kernel:         8.76 ->  8.04 us  (-8.22%)
AITER MLA stage1:                 71.40 -> 71.18 us  (-0.31%)
SGLang FlyDSL fused KDA:          18.52 -> 18.48 us  (-0.22%)
AITER Triton _fwd_kernel_stage2: no material regression
```

Decode attribution: the visible kernels are flat or faster. The remaining gap
is most consistent with CUDA Graph replay/runtime behavior, hidden graph
internals, or later decode scheduling that the five-step trace does not expose.

## Compact Rank0 trace: prefill root cause

The validated compact method captured comparable late-prefill plus short-decode
windows with graph replay kernels visible in both runtimes. Rank0 separates
prefill from decode at the first `kda_packed_decode_kernel` dispatch.

Prefill:

```text
metric                    Triton 3.6   Triton 3.7   delta
stage span                   4153.94      4375.87 ms  +5.34%
visible kernel sum           1018.02      1195.41 ms +17.43%
_fwd_kernel count                 24           24
_fwd_kernel p50              5990.80     13780.42 us +130.03%
_fwd_kernel total             143.73       330.90 ms +130.22%
```

The `_fwd_kernel` total increase is 187.17 ms, explaining 84.34% of the
221.93 ms prefill span increase.

The compiled source identifies this kernel as:

```text
python/sglang/kernels/ops/attention/extend_attention.py:290
```

The launch configuration is unchanged (`num_warps=4`, `waves_per_eu=1`,
`num_stages=1`, shared memory 65536), but AMDGCN resource use changed:

```text
                             Triton 3.6   Triton 3.7
VGPR count                           483          512
private segment bytes                  0          472
scratch load/store instructions        0          186
```

Triton 3.7 pushes the extend-attention kernel to 512 VGPRs and introduces
extensive scratch spills. This directly explains the measured 2.30x kernel
slowdown and most of the Rank0 prefill gap.

Other prefill kernels are much smaller contributors. The next visible
regression is `layer_norm_gated_fwd_kernel` at +4.38% p50; MoE stage1 is
+1.38%. KDA intra-token and recompute kernels are slightly faster.

Decode windows contain a different number of replay steps, so total durations
are not comparable. Per-launch p50 shows:

```text
cross-device reduce:              +4.51%
AITER Triton GEMM K4224:          +3.21%
Opus split-K GEMM:                +2.66%
_fwd_kernel_stage2_asm MLA merge: -0.62%
major MoE/sort kernels:           approximately flat
```

The compact trace therefore localizes the dominant prefill regression, while
decode still has several smaller changes rather than one comparable 2x
culprit.

## Fix direction

The problematic specialization is:

```text
Lq=576
Lv=512
BLOCK_M=64
BLOCK_N=64
num_warps=4
num_stages=1
waves_per_eu=1
matrix_instr_nonkdim=16
kpack resolves to 1 on gfx950
```

It comes from the generic HIP fallback in
`_get_block_sizes_for_extend_attention`; gfx950 only has a special case for
`128 < Lq <= 256`, so Kimi's `Lq=576` uses the large `BLOCK_M=64` fallback.
That tile is brittle under Triton 3.7 register allocation.

Test gfx950/Lq=576 candidates in this order:

```text
BLOCK_M=32, BLOCK_N=64, num_warps=4
BLOCK_M=32, BLOCK_N=64, num_warps=8
BLOCK_M=16, BLOCK_N=64, num_warps=4 or 8
BLOCK_M=32, BLOCK_N=32, num_warps=4
```

Reducing `BLOCK_M` directly shrinks the `[BLOCK_M, BLOCK_DV]` fp32 accumulator;
increasing `num_warps` spreads the tile across more lanes. Both target the
512-VGPR pressure. Reducing `BLOCK_N` is a fallback because it increases loop
iterations.

`waves_per_eu`, `matrix_instr_nonkdim`, `kpack`, and `num_stages` are identical
between the 3.6/3.7 compiled metadata, so they are not the initiating change.
Raising `waves_per_eu` may force a lower register budget and create more spills;
do not use it as the first fix.

Acceptance gates:

```text
private segment = 0
no scratch_load/scratch_store instructions
correctness against the Triton 3.6 output
_fwd_kernel p50 <= 6.5 ms on the matched Rank0 shape
no TTFT/throughput regression on C2-C32
```

## Conclusion

```text
Valid:
  Triton 3.7 makes the profiled workload slower.
  CPU/PyTorch API flow confirms identical MoE model work.
  Triton 3.7 collapses ROCTracer CUDA Graph kernel visibility by >99%.
  Compact Rank0 traces localize most prefill loss to extend_attention _fwd_kernel.
  Triton 3.7 introduces 472-byte scratch and 186 spill instructions.

Not valid:
  Comparing total GPU kernel time between 3.6 and 3.7.
  Claiming that prefill synchronization is the primary cause.
  Claiming that MoE work is absent in Triton 3.7.
```

The package-level endpoint A/B remains causal evidence against Triton 3.7.
The compact trace provides a defensible prefill localization; decode
localization remains partial.

## Artifacts

```text
/workspace/kimi-k3-runs/stage2-runs/
  2026-08-13-triton36-vs37-stage-traces/
```

Retained:

```text
stage-comparison-summary.json
stage-comparison.log
analyze_stage_pair.py
runtime, client and server logs
```

The stage-separated raw traces were deleted at the user's request. The valid
combined prefill+decode TP0-TP7 traces remain in the global-traces directory
below.

The retained compact prefill+decode traces are under:

```text
/workspace/kimi-k3-runs/stage2-runs/
  2026-08-13-triton36-vs37-compact-global-traces/
```
