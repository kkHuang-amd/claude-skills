# Kimi-K3 reclone endpoint reproduction — 2026-08-13

## Outcome

The fresh `/sgl-workspace/sglang` and `/sgl-workspace/aiter` installation did
not reproduce the 2026-08-12 vendored production throughput matrix within the
0.5% acceptance band. C2 passed; C4, C8, C16 and C32 regressed progressively.

All benchmark requests succeeded. Capacity remained exactly 933,883 tokens and
the server log contained no runtime OOM or request failure.

## Runtime

```text
SGLang 89df50593f75deb882b489323c157b0225869eeb
AITER  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
Torch  2.9.1+rocm7.2.0.lw.git7e1940d4
Triton 3.7.0+amd.rocm7.2.0.git89002410
FlyDSL 0.3.0
sglang-kernel 0.4.6.post1, built locally for gfx950
TP8 / MI355X
```

The handover recorded SGLang `f9dd3a0` and Triton 3.6.0. The current SGLang
branch adds `89df505 perf(kimi-k3): add opt-in Radix-4 router`; the reproduction
explicitly set `SGLANG_K3_RADIX4_TOPK=0`.

## Workload and flags

```text
random input/output: 8192 / 1024
requests:             8 × concurrency
warmup requests:      64 per point
request rate:         infinite
radix cache:          disabled
Radix-4:              disabled
B2 fusions:           disabled
```

The remaining production flags match `start_prompt.md`, including SGLang-owned
FlyDSL, AITER M16384, stage1 scratch reuse, MLA gate and KDA group64.

## Results

| Concurrency | Handover tok/s | Reclone tok/s | Delta | Gate |
|---|---:|---:|---:|---|
| C2 | 968.57 | 970.53 | +0.20% | pass |
| C4 | 1,741.98 | 1,718.97 | -1.32% | fail |
| C8 | 2,881.25 | 2,839.40 | -1.45% | fail |
| C16 | 4,432.25 | 4,309.25 | -2.78% | fail |
| C32 | 6,191.41 | 5,881.93 | -5.00% | fail |

Median TPOT was 17.55, 19.11, 22.57, 28.59 and 39.54 ms from C2 through C32.

## Decision

Do not treat the recloned environment as performance-equivalent to the
handover stack. The widening concurrency-dependent loss should be isolated
before using this environment as a new baseline.

The first controlled comparisons should be:

1. Run the same matrix at the handover SGLang commit `f9dd3a0`.
2. If the loss remains, compare the preserved Triton 3.7 runtime against the
   handover's Triton 3.6 runtime only after explicit approval to change Triton.
3. Verify dispatch and per-kernel timing only after the commit comparison.

## Artifacts

```text
stage2-runs/2026-08-13-reclone-repro/run_repro.sh
stage2-runs/2026-08-13-reclone-repro/server.log
stage2-runs/2026-08-13-reclone-repro/c{2,4,8,16,32}.log
stage2-runs/2026-08-13-reclone-repro/c{2,4,8,16,32}.jsonl
```
