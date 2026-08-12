# Kimi-K3 production trace and bottleneck analysis — 2026-08-11

## Runtime and method

```text
8× AMD Instinct MI355X / gfx950
Torch 2.9.1+rocm7.2.0
Triton 3.6.0
A8W4 production profile
Triton prefill / AITER MLA decode
AITER fused KDA + f_b
MLA gate and KDA group64 enabled
```

Three CPU/GPU Chrome traces were captured after warmup:

```text
prefill C8:  8192 input / 1 output
decode C2:   512 input / 256 output
decode C32:  512 input / 128 output
```

Only TP0 was analyzed. All eight raw rank traces are retained. SGLang's
`profile_prefix + merge_profiles` combination did not match its own prefixed
filenames, so raw unprefixed traces were used instead.

Artifacts:

```text
stage2-runs/2026-08-11-production-trace/
```

The reusable analyzer writes per-trace and combined JSON summaries:

```text
analyze_traces.py
combined-summary.json
torch-profile/*/summary.json
```

## Trace overview

### Prefill C8

```text
GPU kernel span:       5540.8 ms
GPU idle within span:    20.2%
>=10 us idle gaps:     1120.1 ms
short kernels <10 us:    28.3 ms
```

Largest kernel families:

```text
quickreduce all-reduce       22.0%
Triton prefill attention     11.6%
BF16 GEMM                     8.6%+
attention-residual aggregate  7.9%
MoE stage1                    7.5%
MoE stage2                    6.5%
KDA prefill kernels           7%+
```

The trace also confirms repeated prefill materialization at
`[M,1536]`, `[M,512]`, and `[M,12,128]`. These are KDA/prefill boundaries,
not the removed routed-MoE `[M,3584]` copies.

### Decode C2

```text
GPU kernel span:       7844.9 ms
GPU idle within span:     2.0%
short kernels <10 us:  4075.7 ms
```

Largest individual costs:

```text
AITER 1-stage all-reduce       9.3%
attention-residual aggregate   8.4%
small BF16 front/tail GEMMs    20%+
MoE stage1                     5.9%
grouped top-k                  5.7%
MoE stage2                     3.8%
fused KDA + f_b                3.9%
route quant/sort preparation   8%+
```

Low-concurrency decode is launch-fragmented: more than half of summed kernel
time is in kernels shorter than 10 us.

### Decode C32

```text
GPU kernel span:       10120.2 ms
GPU idle within span:      3.0%
short kernels <10 us:   1899.8 ms
```

Largest costs:

```text
MoE stage1                    19.3%
MoE stage2                    10.3%
AITER 2-stage all-reduce       8.1%
attention-residual aggregate   5.8%
BF16 GEMMs                     9%+
fused KDA + f_b                3.2%
grouped top-k                  3.0%
```

High-concurrency decode shifts from launch overhead to MoE compute and
communication.

## Ranked bottlenecks

### 1. TP8 communication and collective scheduling

Evidence:

- 8–9% of decode kernel time;
- 22% of prefill kernel time;
- present at both low and high concurrency.

The current 8-GPU custom all-reduce changes from one-stage to two-stage at
80 KiB. A follow-up rebuilt the dispatcher with logged, exact stage controls.
It found that the preliminary operator comparison had reused a stale JIT module
and had not actually switched B32 to one-stage.

```text
B32 paired operator:
  one-stage 21.467 us
  two-stage 12.443 us

Contemporaneous C32:
  baseline 6199.77 tok/s
  global   6085.06 tok/s (-1.85%)
  exact    6096.66 tok/s (-1.66%)
```

Production traces confirmed one-stage p50 `18.16 us` versus baseline two-stage
`11.40 us` at C32. Rank skew improved rather than regressed; the forced kernel
itself was slower. Decision: reject both the global threshold and exact-B32
override.

The full shape sweep is non-monotonic: B6 and B16 favor one-stage, while B12
and B24+ favor two-stage. Any future work must use an exact shape table rather
than a byte threshold.

Details:
[`ALLREDUCE_SHAPE_INVESTIGATION_2026-08-11.md`](ALLREDUCE_SHAPE_INVESTIGATION_2026-08-11.md).

### 2. MoE route preparation and stage kernels

Evidence:

- grouped top-k plus route quant/sort is about 10–14% at C2;
- stage1 + stage2 is about 30% at C32.

The highest-value new implementation target is an AITER/Opus-compatible fused
route → expert sort → MX quant handoff. SGLang already has a fused route+quant
kernel for the TRT-LLM runner, but its packed output is not the sorted layout
consumed by AITER/Opus. This requires a new contract rather than enabling an
existing flag.

For C32, independently retuning the exact Opus stage1/stage2 rows is lower
risk, but the theoretical endpoint ceiling is smaller than removing route
preparation launches at C2.

### 3. Attention-residual aggregation

Evidence:

- 6–8% of decode kernel time;
- 8% of prefill kernel time.

The existing ROCm kernel already keeps the bank tile in registers. #4572 was
tested and rejected at the endpoint. Further work is only justified if it
combines collective completion with aggregation or removes a launch boundary;
another standalone aggregate kernel is unlikely to help.

## Existing optimization evidence

The traces confirm that the selected stack remains active:

- `kimi_k3_kda_decode_fb_bf16_gfx950`;
- Opus A8W4 MoE stage1/stage2;
- AITER custom all-reduce;
- AITER MLA decode;
- no routed-output or routed-input `[M,3584]` copy regression.

## Decision and next implementation target

No trace experiment is retained in production. The all-reduce threshold knob
was removed after its C32 regression.

The next code-level target should be the AITER/Opus route-sort-quant boundary,
starting with a standalone contract and B1/B2/B32 operator benchmark before
any SGLang wiring.
