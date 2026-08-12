# Kimi-K3 B300 versus MI355X trace comparison — 2026-08-11

## Scope

```text
TP8, no DCP
random 8192 input / 1024 output
C2 endpoint and stage-separated CPU/GPU traces
five profiled EXTEND/DECODE scheduler steps
```

B300 artifacts:

```text
/sgl-workspace/b300-kimi-k3-traces-c2-c32/
```

Matched MI355X artifacts:

```text
stage2-runs/2026-08-11-b300-mi355x-comparison/
```

## Endpoint gap

```text
C2 total throughput:  B300 1840.25 vs MI355X 969.04 tok/s  (52.7%)
C2 output throughput: B300  204.47 vs MI355X 107.67 tok/s  (52.7%)
C2 TTFT:              B300  669.81 vs MI355X 941.41 ms     (1.41×)
C2 TPOT:              B300    8.98 vs MI355X  17.66 ms     (1.97×)
C2 ITL:               B300    8.97 vs MI355X  17.67 ms     (1.97×)
```

The gap is primarily decode, not prefill.

## Radix-cache mismatch

B300 leaves radix cache enabled. Its first profiled EXTEND request is:

```text
64 new tokens + 8128 cached tokens
```

MI355X production explicitly uses `--disable-radix-cache` and computes all
8192 tokens. B300 TTFT, E2E, input/total throughput, and EXTEND trace are
therefore favorably biased and are not apples-to-apples.

TPOT/ITL and the DECODE traces remain the strongest comparison.

### Follow-up no-cache controls

Matched B300 no-cache traces now exist at:

```text
/sgl-workspace/b300-kimi-k3-traces-c2-c32-two-cases/
```

Both cases set `disable_radix_cache=true`. The second additionally forces
single-stream execution and disables PDL. Relative to normal no-cache B300,
single-stream/no-PDL changes:

```text
C2 output throughput: 186.46 -> 167.24 tok/s
C2 median TPOT:          9.33 ->  10.59 ms
C32 output throughput: 913.20 -> 867.88 tok/s
C32 median TPOT:        27.47 ->  29.21 ms
```

This confirms the B300 overlap advantage survives a matched no-cache setup.
It does not measure cache on/off because both cases disable radix cache.
Continuation details are recorded in
[`PAUSE_TRACK_2026-08-11.md`](PAUSE_TRACK_2026-08-11.md).

## Matched C2 decode

### Timeline

```text
                         B300       MI355X
step span                9.66 ms    18.21 ms
summed kernel duration  15.09 ms    18.85 ms
busy union               9.07 ms    18.85 ms
overlap saved            6.02 ms     0.00 ms
GPU streams                 5          effectively 1
kernel launches/step      1937        2370
```

ROCm profiler exported child kernels for one graph replay inside the five-step
window; MI355X component totals are normalized to that captured replay.

B300's 6.02 ms of overlap explains about 69% of the measured 8.69 ms TPOT
gap, or 62% of the trace busy-time gap.

### Serial component cost

```text
ms per decode step          B300    MI355X
other GEMMs                 6.618    6.874
collectives                 2.573    1.835
MoE compute                 2.248    1.115
route / sort / quant        0.901    2.615
attention residual          1.180    1.580
KDA decode                  0.464    0.745
copies                      0.024    0.525
unclassified fixed ops      0.548    3.035
```

MI355X is not losing on every kernel: its summed C2 MoE compute and collective
times are lower. Its major serial deficits are route preparation, fixed
small-kernel chains, copies, attention residual, and KDA.

## What B300 overlaps

Visible cross-stream overlap per C2 step includes:

```text
GEMM + GEMM                  1.39 ms
collective + MoE compute     0.84 ms
GEMM + route preparation     0.49 ms
collective + routing         0.36 ms
```

The remaining overlap comes from multiway concurrency and same-timeline PDL.
Representative pairs include:

- MoE stage1 with `all_reduce_pull_res`;
- wide CUTLASS projection with BFA tiny GEMMs;
- wide projection with route/quant;
- routing with collective completion.

MI355X's Kimi-K3 model explicitly disables `alt_streams` on HIP, and the trace
shows effectively no overlap.

## Software-path differences

```text
B300:
  trtllm_mla
  flashinfer_mxfp4 / TRT-LLM MoE
  route_quant_fused
  TMA attention residual
  CUDA alt streams + PDL

MI355X:
  Triton prefill / AITER MLA decode
  AITER Opus A8W4 MoE
  grouped top-k + separate expert sort/MX quant
  ROCm register-tile attention residual
  HIP alt streams disabled
```

## Priority

1. Add an opt-in HIP multi-stream experiment:
   - overlap BFA tiny GEMMs with the wide KDA projection;
   - overlap shared all-reduce with routed MoE compute.
2. Build the AITER/Opus route → expert sort → MX quant fusion.
3. Remove remaining B1/B2 copies and fixed tiny-kernel chains.
4. Retune standalone MLA/KDA/attention-residual kernels only after the missing
   concurrency is addressed.

The first experiment must measure real timeline overlap and endpoint TPOT; a
faster isolated kernel is insufficient.
