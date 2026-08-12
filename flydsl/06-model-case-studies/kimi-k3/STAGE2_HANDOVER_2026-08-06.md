# Kimi-K3 Stage 2 performance optimization handover

Date: 2026-08-06

## CONTINUE HERE

Stage 2 is complete for the fixed TP8 8192/1024 concurrency-32 serving
workload. The retained stack is:

```text
SGLang: /sgl-workspace/sglang
branch: k3_dcp
commit: d197e0d2c3cdb012ba61aab9c38fce79c211fb00

AITER: /sgl-workspace/aiter
branch: kimi-k3-stage2-opus
commit: 6dc26b7a817bba2ae92ff22a7a63c0c024f4c7e5
upstream: ROCm/aiter PR #4534

ROCm/HIP: 7.2.26015-fc0010cf6a
GPU: 8x gfx950
FlyDSL: 0.3.0
```

No server or benchmark is active at handover time.

The A4W4 correctness bug has been fixed in the SGLang working tree, but A4W4
is slower than the retained A8W4 profile for the target workload. Keep A8W4 as
the performance default. Do not revisit the tested non-split-K dense GEMM
config. Re-profile before starting a new kernel family.

## Retained Stage 2 change

AITER commit `6dc26b7a8`, upstream PR
[`#4534`](https://github.com/ROCm/aiter/pull/4534), replaces the K3 A8W4
SiTUv2 MoE stage-2 dispatch with the tuned gfx950 Opus implementation while
retaining FlyDSL for the relevant stage-1 decode shapes.

The primary AITER checkout was moved from detached commit `d9e5ef7ce` to local
branch `kimi-k3-stage2-opus` at `6dc26b7a8`. Old ignored JIT binaries and
`aiter_meta` were moved to `/tmp/aiter-pre-stage2-generated` before the clean
rebuild. Do not restore those old binaries into the accepted checkout.

Final runtime logs confirm imports from `/sgl-workspace/aiter` and dispatch to:

```text
opus_moe2_afp8_wfp4_atomic_t32x128x256_sbm32_occ1_cache_b3_ws2
```

## Authoritative serving comparison

Methodology for both sides:

```text
TP8, DCP off, DSPARK off, Triton attention
Radix Cache disabled
fixed 8192 input / 1024 output
concurrency 32
64 warmups / 256 measured requests
seed 42
three runs per side
```

PR first-parent control, AITER `7a56d7462`:

```text
throughput: 524.63, 524.37, 524.54 tok/s
median:     524.54 tok/s
TPOT:       46.95, 47.01, 46.96 ms
ITL:        37.88, 37.91, 37.88 ms
```

Optimized, AITER `6dc26b7a8`:

```text
throughput: 538.06, 537.97, 538.25 tok/s
median:     538.06 tok/s
TPOT:       45.88, 45.86, 45.84 ms
ITL:        37.13, 37.15, 37.13 ms
```

Median delta:

```text
output throughput: +2.58%
median TPOT:       -2.34%
median ITL:        -1.98%
```

After integration into the primary AITER checkout, a final matched run produced
538.61 output tok/s, 45.87 ms TPOT, and 37.14 ms ITL.

## Trace evidence

The production graph A/B found:

```text
rank-median total: 38.62 -> 38.09 ms
MoE stage 2:        4.10 -> 3.79 ms (-7.44%)
MoE route/sort:     2.98 -> 2.94 ms (-1.15%)
dense GEMM:         8.49 -> 8.33 ms (-1.97%)
MoE stage 1:        7.01 -> 7.06 ms (+0.79%)
```

The stage-2 kernel change is therefore visible in the production graph and the
serving gain is above the 1% retention threshold.

## Correctness

Accepted stack:

- Stage 1 SGLang focused tests: 7 passed.
- Clean AITER K3 Opus CSV oracle: 25 cases recorded, zero skipped/failures.
- Smoke test passed.
- Optimized PR experiment GSM8K 200: 0.985.
- Final primary-checkout GSM8K 200: 0.980.

The eager CPU/GPU stack-profile export was attempted twice. Both attempts
stalled while exporting all eight rank traces and triggered the 300-second
scheduler watchdog. Production GPU graph traces succeeded. Use the previous
eager stack attribution in `TP8_ROCM_TRACE_ANALYSIS.md` unless the profiler
export path is fixed or rank-local profiling is added.

## Rejected experiments

### A4W4 SiTUv2: correctness fixed, performance rejected

The original A4W4 score of 0.005 was an SGLang integration bug, not inherent
model sensitivity. SGLang used `shuffle_weight_a16w4`'s GU-interleaved layout
for every K3 SiTU activation mode, while AITER A4W4 consumes the generic
separated layout. A real-checkpoint layer-1 comparison before the fix had
cosine similarity 0.0057 and relative L2 error 1.148 between A4W4 and A8W4.

The uncommitted SGLang fix makes the layout follow AITER's activation-mode
precedence:

```text
A8W4 or A16W4 -> GU-interleaved shuffle
A4W4          -> generic separated shuffle
both A8/A4    -> A8W4 precedence
```

Validation after the fix:

```text
layout/Stage-1 tests: 11 passed
GSM8K 50:             1.000
GSM8K 200:            0.980

A4W4 throughput runs: 531.09, 530.42, 530.84 tok/s
A4W4 median:          530.84 tok/s
A4W4 median TPOT:      46.81 ms
A4W4 median ITL:       38.13 ms
```

Post-fix A8W4 regression:

```text
throughput runs: 538.07, 537.24, 537.33 tok/s
median:          537.33 tok/s
median TPOT:      45.93 ms
median ITL:       37.21 ms
```

Relative to the pre-fix retained A8W4 median of 538.06 tok/s, the post-fix
A8W4 delta is -0.14%, within serving noise. A4W4 is correct but 1.21% slower
than post-fix A8W4, with worse TPOT/ITL, so keep:

```text
AITER_SITUV2_A8W4=1
AITER_SITUV2_A4W4=0
```

### Non-split-K shared dense GEMM

Branch `tune/kimi-k3-shared-gemm-config`, commit `a86897463`, switches selected
small-M K3 BF16 GEMMs to graph-safe non-split-K Triton configs to reduce
dual-stream resource contention.

Relative to accepted Opus A8W4:

```text
throughput: 538.06 -> 535.97 tok/s (-0.39%)
TPOT:        45.86 -> 46.07 ms (+0.46%)
ITL:         37.13 -> 37.35 ms (+0.59%)
dense GEMM:   8.33 -> 8.54 ms (+2.60%)
```

This branch was rejected. The K3 PTPC/MXFP8 config from AITER PR `#4435` was
not adopted because the new production trace shows BF16 dense kernels, not a
matching A8W8 preshuffle path.

## Persistent artifacts

All Stage 2 logs, benchmark JSONL, traces, and summaries are under:

```text
/dockerx/var/amdsgl/kk/workspace/claude-skills/kimi-k3/stage2-runs/2026-08-06-baseline
```

Most useful machine-readable files:

```text
environment.txt
graph_trace_summary.json
opus_ab_summary.json
dense_a868_ab_summary.json
final-opus-csv-oracle-clean.log
final-sglang-focused-tests.log
final-integrated/
a4w4-fixed-serving/
a8w4-post-fix-serving/
```

## Final launch recipe

The existing launcher defaults already enable the accepted A8W4 path:

```bash
DCP_SIZE=1 ATTENTION_BACKEND=triton \
MAX_RUNNING_REQUESTS=32 CUDA_GRAPH_MAX_BS_DECODE=32 \
RADIX_CACHE=0 ENABLE_INT8_MAMBA_CHECKPOINT=1 \
ENABLE_CACHE_REPORT=1 \
bash /dockerx/var/amdsgl/kk/workspace/useful-scripts/benchmarking/kimi-k3/launch_server.sh
```

Before serving, verify:

```bash
git -C /sgl-workspace/aiter rev-parse HEAD
# 6dc26b7a817bba2ae92ff22a7a63c0c024f4c7e5

git -C /sgl-workspace/sglang rev-parse HEAD
# d197e0d2c3cdb012ba61aab9c38fce79c211fb00
```

Do not mix Python/config files from another AITER revision with the compiled
modules in `/sgl-workspace/aiter/aiter/jit`.
