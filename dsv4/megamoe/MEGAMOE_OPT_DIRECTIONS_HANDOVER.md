# MegaMoE optimization directions — HANDOVER (condensed 2026-07-24)

This file covers MegaMoE only.

For the A2A EP umbrella, mori-EP, FlyDSL-EP dynamic recv, correctness, and TBO work,
read:

```text
/dockerx/home/wunhuang/tmp/claude-skills/dsv4/A2A_EP_HANDOVER.md
```

Taxonomy:

```text
A2A EP
├─ mori-ep
├─ flydsl-ep
└─ mega-moe     fused A2A/GEMM implementation
```

MegaMoE is one A2A EP implementation; it is not the parent category.

## 1. Scope and workspace

- FlyDSL: `/sgl-workspace/FlyDSL`, branch/workspace `mega_moe_v1`
- SGLang: `/sgl-workspace/sglang`, dirty `feat/mega-moe`
- hardware: 8x MI355X gfx950
- model: DeepSeek-V4-Pro A8W4
- topology: TP8/DP8/EP8

Do not mix MegaMoE changes into:

```text
/sgl-workspace/sglang-flydsl-a2a
```

That branch owns the separate FlyDSL-EP backend.

## 2. Historical validated shipping state

- compact-only MegaMoE: 29,482 tok/s
- historical DP reference: 30,501 tok/s
- MegaMoE / DP: 96.7%
- full GSM8K: 0.934

Workload:

- input/output 8192/1024
- concurrency 256

The compact-only path remained the recommended MegaMoE baseline. Experimental
features were default-off unless explicitly stated.

## 3. MegaMoE architecture

MegaMoE fuses:

```text
dispatch -> GEMM1 -> quant/activation -> GEMM2 -> combine
```

The fused kernel is persistent because its cross-PE barrier requires all workgroups
to remain co-resident. This removes intermediate HBM traffic and launch overhead,
but imposes a structural occupancy limit.

Important distinction:

- FlyDSL-EP uses separate dispatch/combine kernels around aiter fused MoE.
- MegaMoE fuses communication and expert compute into persistent kernels.

Consequently, conventional TBO applies naturally to FlyDSL-EP but conflicts with
MegaMoE's persistent CU ownership.

## 4. Final bottleneck attribution

### What was measured

- achieved HBM read bandwidth: ~1.1–1.2 TB/s
- MI355X HBM peak: ~8 TB/s
- achieved fraction: ~15%
- fused grid: roughly one workgroup per CU

### Correct conclusion

The kernel was not at the HBM bandwidth ceiling. The persistent fused design forces
approximately one wave/SIMD and exposes on-chip dependency, LDS-read, and synchronization
latency that a second wave cannot hide.

Earlier labels such as “HBM bandwidth-bound” were incorrect because they were inferred
from stall sites and successful knobs rather than measured utilization.

### B200 comparison

B200 hides latency with asynchronous MMA/TMA-style pipelines while retaining fusion.
CDNA4 synchronous MFMA benefits more from another wave, but the persistent barrier
prevents launching a second workgroup/CU.

This is an architectural trade-off, not a simple tuning knob.

## 5. Completed experiments

### Dispatch redesign

- recv/2B-i and 2B-ii were correct but slower than compact.
- Direct GEMM gather from staging starved compute.
- Compact dispatch remained the winner.

### Occupancy / tile changes

- tile_n=128: slower
- tile_k=128: not viable
- waves-per-EU hints: no meaningful gain
- “2 waves had no benefit” was later corrected: the persistent grid never actually
  launched a second workgroup/CU, so higher occupancy was not truly tested

### Arithmetic changes

- a4w4 cheaper MFMA: no useful serving win
- MFMA arithmetic throughput was not the dominant limit

### Weight cache hints

- `b_nt=2` improved decode microbench by roughly 5%
- serving regressed 1.5–1.7%:
  - 28,978 / 29,049 tok/s vs 29,482 baseline
- reason: 8k/1k total throughput is prefill-token dominated; prefill regression
  outweighed decode improvement
- final recommendation: keep `b_nt=0`

### Deeper prefetch

- B-upfront probe changed both issue timing and interleave, and regressed prefill ~7%
- it did not cleanly disprove a true extra-buffer pipeline
- expected upside was bounded by the residual VMEM-wait fraction
- full rewrite was judged poor ROI

### EP4

Rejected. Per-expert token count is invariant to EP degree; EP4 gives each rank more
experts and weight work, not more tokens per expert.

### TBO

Conventional TBO and fused persistent MegaMoE compete:

- TBO needs communication kernels to release CUs while another ubatch computes.
- MegaMoE's persistent dispatch/GEMM kernel retains those CUs across the barrier.

Using TBO would require un-fusing dispatch or abandoning persistence, both of which
remove the core MegaMoE benefit.

## 6. Important corrected misjudgments

1. Profiler stall site was treated as the bottleneck.
   - MFMA stalls can mean operands are late, not that MFMA throughput is saturated.
2. “Allowed occupancy” was treated as achieved occupancy.
   - Reducing LDS allowed a second workgroup but the grid never launched it.
3. Low achieved bandwidth was used to dismiss prefetch.
   - Prefetch hides latency; it does not require bandwidth saturation.
4. A decode micro win was assumed to transfer to serving.
   - Input/output token mix made prefill dominate the final metric.
5. Lower EP degree was assumed to increase per-expert M.
   - A2A routes every token to its owner regardless of EP degree.

Rules:

- measure achieved utilization
- verify the intended outcome actually happened
- change one variable per experiment
- benchmark the full serving token mix
- distinguish architectural constraints from local kernel knobs

## 7. Remaining directions

### Architectural fork: un-fused non-persistent GEMM

Potential benefit:

- allow 2+ workgroups/CU
- hide on-chip latency

Costs:

- restore intermediate activation HBM traffic
- add kernel launches
- lose direct in-kernel dispatch consumption
- prior un-fused attempts lost to compact

Only pursue after a clean decomposition estimates:

```text
separate dispatch + activation buffer + non-persistent GEMM
```

against compact fused end-to-end.

### GEMM2 sensitivity

Stage2 is roughly 25–36% of MegaMoE. Cache-hint sensitivity was not explored as
thoroughly as GEMM1, but expected ROI is low.

## 8. Detailed records

- `KERNEL_OWNER_DECODE_PLAN.md`
- `GEMM_CODESIGN_WIP.md`
- `COMPACT_SINGLE_ROUND_DESIGN.md`
- `DUAL_MODE_DISPATCH_DIAGNOSIS.md`
- `MEGAMOE_HANDOFF.md`
- `EXPERIMENT_LOG.md`

## 9. Guardrails

- Keep experimental paths default-off.
- Run oracle relL2 after every kernel edit.
- Use bs64 as a tight correctness/perf detector.
- Use bs2048 median-of-many; historical noise was about ±1.2%.
- Kill GPU processes by numeric PID.
- Confirm all GPUs return to ~0.3 GB used memory.
- Do not claim a bottleneck without direct utilization evidence.
