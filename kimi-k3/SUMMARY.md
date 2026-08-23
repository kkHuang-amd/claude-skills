# Kimi-K3 optimization summary

Updated: 2026-08-23

Fresh-chat entry point:
[`HANDOFF_2026-08-21.md`](HANDOFF_2026-08-21.md).

## Current integration

```text
SGLang remote:
  https://github.com/HaiShaw/sglang
  branch perf/k3_opts_0812
  HEAD 455b744aa77b2078de7577619dc12d2775fc1091

AITER remote:
  https://github.com/kkHuang-amd/aiter
  branch integration/k3-core-only
  HEAD b56d27beee1ce0132ca868da24d1a25e88e6ef42
```

Current capacity-aware non-EP MoE-front profile:

```text
AITER #4834:
  27 gfx950 BF16 tuned-GEMM rows
  non-EP N6016 enabled only for M48-M192

SGLang #35406 adaptation:
  preferred: latent-up-only MXFP4, M>=2048
  EP/N4480 unchanged

Up-only endpoint deltas vs same-code baseline:
  throughput C2/C4/C8/C16/C32/C64 =
  noisy/+0.23/+0.53/+0.82/+1.28/+2.21%

  TTFT =
  -3.15/-2.44/-3.45/-3.08/-3.62/-3.13%

GSM8K 50:   1.000, invalid 0.000
GSM8K 1319: 0.950, invalid 0.001
capacity:   1,519,705 -> 1,376,952 tokens (-9.39%)
```

Details:
[`aiter-optimization-tracker/MOE_LATENT_SPLIT_2026-08-20.md`](aiter-optimization-tracker/MOE_LATENT_SPLIT_2026-08-20.md).

SGLang PR #35770's gfx950 Triton prefill feature commit was manually ported
and evaluated behind `SGLANG_K3_TRITON_PREFILL_PROFILE=off|bf16|fp8`.
Micro gates passed (`1.61x` fresh BF16 8K, `2.14x` absorbed BF16 8K+8K), but
Triton FP8 reached only `0.988x` AITER's 8K geomean throughput and `0.729x`
at 68K; paired GSM8K-1319 was `0.947` versus `0.958`, and 32 paired 68K
prompts matched first tokens only `81.25%`. The port was therefore removed
from the current worktree on 2026-08-21 while its patch snapshots and
validation artifacts were retained. Production remains on AITER prefill.
Details:
[`aiter-optimization-tracker/TRITON_PREFILL_PR35770_2026-08-21.md`](aiter-optimization-tracker/TRITON_PREFILL_PR35770_2026-08-21.md).

Dedicated dense MXFP4 follow-up found the generic FlyDSL gap was caused by
`1,220 B/workitem` scratch and 383 spill instructions. Streamed-B removes all
scratch and reaches isolated plain-GEMM parity, but neither fused epilogue nor
matched 16K endpoint passes the promotion gate. The uncommitted AITER Route A
and dense/tail GEMM2 experiments were rolled back on 2026-08-21; production
stays on gfx950 ASM.
Details:
[`aiter-optimization-tracker/MXFP4_DENSE_KERNEL_TUNING_2026-08-20.md`](aiter-optimization-tracker/MXFP4_DENSE_KERNEL_TUNING_2026-08-20.md).

AITER standalone latent RMSNorm + MXFP4 quant was then integrated default-off.
It improves the isolated norm+quant chain by `1.7-2.3x` and passes GSM8K-1319
at `0.955`, but the full C2-C64 endpoint sweep is effectively neutral
(`-0.02%` to `+0.15%` throughput for C4-C64; TTFT changes within `±0.54%`).
Capacity remains `1,376,952`; production stays on the decomposed path. Details:
[`aiter-optimization-tracker/RMSNORM_MXFP4_FUSION_2026-08-20.md`](aiter-optimization-tracker/RMSNORM_MXFP4_FUSION_2026-08-20.md).

CUDA Graph decode crossover testing then selected a default-off M32-M256
latent-up MXFP4 band. Relative to up-only BF16 decode it improves C32/C64
throughput by `+1.08/+0.82%` and TPOT by `-1.20/-0.90%`; lower concurrencies
remain neutral. GSM8K-1319 rerun is `0.954`, invalid `0.001`, and capacity is
unchanged. BF16 up weights must remain for M1-M31, M257-M2047 and fail-closed
fallback. Details:
[`aiter-optimization-tracker/DECODE_LATENT_MXFP4_2026-08-20.md`](aiter-optimization-tracker/DECODE_LATENT_MXFP4_2026-08-20.md).

Two Draft PRs were prepared from clean bases:

```text
perf/k3-gfx950-independent-fusions-clean
  /sgl-workspace/sglang-pr-independent-clean
  base origin/main 0077f84d3
  selected SGLang-vendored FlyDSL/gate/group64/M2-M4/M16384 integrations

perf/k3_fused_kda_decode
  /sgl-workspace/sglang-pr34198
  base origin/perf/k3_fused_kda_decode 8950e2e0d
  KDA-only FlyDSL vendoring + C2 winner
```

```text
Independent integrations: https://github.com/sgl-project/sglang/pull/35287
KDA decode update:         https://github.com/sgl-project/sglang/pull/34198
```

The independent branch has no #4617/#4647 hard dependency. MLA Q/cache was
removed from it after AITER pin `d9e5ef7` failed batch64 correctness; that
feature requires a newer pin containing AITER #4342 semantics. Details:
[`aiter-optimization-tracker/PR_EXTRACTION_AND_AITER_GAPS_2026-08-18.md`](aiter-optimization-tracker/PR_EXTRACTION_AND_AITER_GAPS_2026-08-18.md).

Branch-specific full accuracy validation:

```text
independent clean branch GSM8K 1319: 0.953, invalid 0.001
PR #34198 local update GSM8K 1319:    0.950, invalid 0.001
```

MLA Q/cache was prepared separately for review:

```text
branch: perf/k3-mla-q-cache-fusion
commit: 9f9e2e2be
GSM8K 1319: 0.955, invalid 0.001
Draft PR: https://github.com/sgl-project/sglang/pull/35308
dependency: AITER #4342 / 770790cd semantics or newer
```

Kimi-specific gfx950 FlyDSL kernels are maintained in SGLang. AITER retains
only the core dependencies:

```text
#4617 caller-provided fused_moe output
#4647 stage1 scratch reuse
shared FlyDSL helpers/toolchain
```

## Fresh environment reproduction

The stack was rebuilt and rerun on 2026-08-12 with the recorded aligned
production command. Torch and Triton remained unchanged.

```text
Machine:                crsuse2-m2m-002.crusoe.amd.com
Focused vendored tests: 46 passed
GSM8K 50:              1.000
GSM8K 200:             0.985 (one deterministic invalid; recorded result 0.990)
Capacity:              934463 (recorded 933883, +0.06%)

C2:    980.24 tok/s
C4:   1764.84 tok/s
C8:   2921.82 tok/s
C16:  4492.89 tok/s
C32:  6295.63 tok/s

B2 C2: 1073.02 tok/s, 15.89 ms median TPOT
B2 C4: 1770.31 tok/s
```

All endpoint requests succeeded. Full methodology, caveats and artifact paths:
[`aiter-optimization-tracker/ENVIRONMENT_REPRODUCTION_2026-08-12.md`](aiter-optimization-tracker/ENVIRONMENT_REPRODUCTION_2026-08-12.md).

## ATOM recipe comparison

ATOM `f782218a5` was benchmarked on
`crsuse2-m2m-002.crusoe.amd.com` using its Kimi-K3 TP8 recipe and the same
fixed 8192/1024, 64-warmup endpoint methodology.

```text
C2:    799.99 tok/s  (-18.39% vs reproduced SGLang production)
C4:   1483.11 tok/s  (-15.96%)
C8:   2521.14 tok/s  (-13.71%)
C16:  4290.71 tok/s   (-4.50%)
C32:  6117.82 tok/s   (-2.82%)
C64:  8380.39 tok/s   (+4.57%)
```

All C2-C32 requests succeeded (496/496), as did C64 (512/512). At C64,
SGLang produced `8014.22 tok/s`; ATOM's median TPOT was `52.66 ms` versus
SGLang's `55.58 ms`. ATOM used the recipe's FP8 KV cache and PTPC-FP8 online
quantization, so this is a framework-recipe comparison rather than identical
precision. Details:
[`aiter-optimization-tracker/ATOM_KIMI_K3_PERFORMANCE_2026-08-12.md`](aiter-optimization-tracker/ATOM_KIMI_K3_PERFORMANCE_2026-08-12.md).

On 2026-08-13, the current ATOM `5479c5af3` / AITER `e8b4507e5` stack was
validated separately on the same machine. The fixed 8192/1024 C64 run
completed 512/512 requests at `8742.34 tok/s` with `49.84 ms` median TPOT:
`+4.32%` throughput and `-5.36%` TPOT versus the recorded ATOM C64 result.
Because Torch/ROCm and the repository SHAs changed, this confirms that the
current stack exceeds the recorded level; it is not an exact old-stack
reproduction.

On 2026-08-21, current ATOM `27f8639b` and isolated AITER main `dc4bdf1c`
were rerun at fixed 8192/1024 C2 using the current Kimi-K3 recipe. Across five
runs per side, multi-stream beat single-stream in every run. The conservative
five-run median is `780.43` versus `767.18 tok/s` (`+1.73%`), with median TPOT
`22.17` versus `22.64 ms` (`-2.08%`). The multi-stream side warmed upward, so
the five-run mean gain of `+3.99%` is not the production claim. This confirms
the direction for an opt-in SGLang prototype but requires SGLang-specific
graph, collective-order, endpoint, and correctness gates. Details:
[`aiter-optimization-tracker/ATOM_C2_STREAM_AB_2026-08-21.md`](aiter-optimization-tracker/ATOM_C2_STREAM_AB_2026-08-21.md).

The matched current-main C64 follow-up reverses the historical stream result:
multi-stream reaches `8680.09 tok/s` versus single-stream `9878.91`
(`-12.14%`), with TTFT `+9.76%` and TPOT `+14.70%`. Graph inspection confirms
multi contains `maybe_dual_stream_forward` and single does not. The current
single path is about `17.45%` faster than the old single baseline, so the
historical `+4.47%` C64 multi-stream claim no longer applies. Any SGLang
prototype must begin at exact C2 and disable before C64. Details:
[`aiter-optimization-tracker/ATOM_C64_STREAM_AB_2026-08-21.md`](aiter-optimization-tracker/ATOM_C64_STREAM_AB_2026-08-21.md).

A controlled cross-engine rerun then used one common streaming OpenAI client,
one exact prompt manifest, and current AITER `dc4bdf1c` for both frameworks.
The apparent native-client C64 gap disappeared: SGLang reached `9800.69 tok/s`
and ATOM `9846.50` (`+0.47%`, within noise). ATOM halves median TTFT
(`14689` vs `27580 ms`) while SGLang has much lower TPOT (`31.74` vs
`44.70 ms`); median E2E is effectively equal near 60 seconds. The former
`9041` versus `9879 tok/s` comparison primarily measured client scheduling
differences. All future SGLang/ATOM claims must use the common client and
persisted manifest. Details:
[`aiter-optimization-tracker/COMMON_CLIENT_SGLANG_ATOM_C64_2026-08-21.md`](aiter-optimization-tracker/COMMON_CLIENT_SGLANG_ATOM_C64_2026-08-21.md).

The same controlled comparison at C2 shows a real low-concurrency engine gap:
five-run medians are `1138.02 tok/s` for SGLang and `768.04` for ATOM
(`-32.51%`). ATOM has 9.37% lower TTFT, but its TPOT is 51.77% higher and
median E2E 48.20% worse over 1024 output tokens. All 80 requests per engine
succeeded with low round-to-round spread. Common-client C2/C64 together show
that SGLang's decode path dominates at low concurrency while aggregate
throughput converges under C64 saturation. Details:
[`aiter-optimization-tracker/COMMON_CLIENT_SGLANG_ATOM_C2_2026-08-21.md`](aiter-optimization-tracker/COMMON_CLIENT_SGLANG_ATOM_C2_2026-08-21.md).
The matched four-case C2/C64 trace, armed route, and dense crossover campaign
is complete. One exact complete decode replay per rank shows ATOM's graph is
`+7.989 ms` longer at C2, led by routed-MoE stage2 (`+1.665 ms`),
collectives (`+1.206 ms`), and merged MoE front (`+1.173 ms`). At C64 ATOM's
graph is `-2.377 ms` shorter even though endpoint TPOT is `+12.956 ms`; the
graph-external discrepancy is deferred and remains unresolved.

Armed C64 routes contain mean active experts `133.87` for SGLang versus
`18.03` for ATOM and BM32 blocks `145.21` versus `33.91`. Matched current-AITER
replay makes A8W4 faster for both full route chains (`3.43%` on SGLang routes,
`7.44%` on ATOM routes), so the trace inversion is route workload, not A16W4
kernel superiority.

The corrected dense full-chain matrix passed 125 supported rows with seven
explicit skips. The combined fastest prepared policy would cost exactly
`3,273,054,208 B` (`3.048269 GiB/GPU`) after layer weighting and is not a
promotion candidate. The low-cost MLA QKV-A PTPC M64 candidate was evaluated
first and rejected at production TP8 retrace. KDA input-projection MXFP4 at
M32+ is the next separate default-off candidate; latent-up remains covered by
existing endpoint evidence. Production remains unchanged. Graph-external work
remains deferred. Canonical report:
[`aiter-optimization-tracker/SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md`](aiter-optimization-tracker/SGLANG_ATOM_C2_C64_TRACE_ATTRIBUTION_2026-08-22.md).

The first separate candidate, gfx950 MLA QKV-A PTPC FP8 for exact decode M64,
was rejected and removed after its complete production TP8 chain measured
`+6.007 us` slower than BF16. The next candidate, KDA input-projection MXFP4
for exact decode M32/M64, is now implemented in SGLang default-off and
fail-closed. Focused production-faithful graph replay measures `29.409 ->
28.823 us` (`1.020x`) at M32 and `42.709 -> 29.917 us` (`1.428x`) at M64;
numerical, storage, and changed-input graph gates pass. Exact incremental
storage is `23,969,792 B/layer`, or `1,653,915,648 B`
(`1.540329 GiB/GPU`) across 69 KDA layers. Production remains unchanged:
real capacity, common-client C2/C64, and complete prefill/decode retrace gates
remain before endpoint or accuracy work. Details:
[`aiter-optimization-tracker/KDA_INPROJ_MXFP4_M32_M64_2026-08-23.md`](aiter-optimization-tracker/KDA_INPROJ_MXFP4_M32_M64_2026-08-23.md).

The QKV-A-only rejection did not cover ATOM's intended shared-input topology.
A production-faithful follow-up quantized normalized hidden once and reused the
same PTPC FP8 activation/scale for MLA QKV-A and g_proj while retaining
SGLang's fused sigmoid/multiply output gate. Exact M64 improves
`34.320 -> 29.920 us` (`1.1471x`) with non-overlapping p90s; M32 regresses and
is excluded. The default-off gfx950 exact-decode-M64 SGLang path passes focused
dispatch/fallback/numerical/storage/changed-input graph tests (`6 passed`).
Prepared QKV-A+gate storage is `627,922,944 B` (`0.584799 GiB/GPU`) across 24
MLA layers. Production remains unchanged; capacity and matched TP8
trace/endpoint gates remain. Details:
[`aiter-optimization-tracker/MLA_SHARED_PTPC_M64_2026-08-23.md`](aiter-optimization-tracker/MLA_SHARED_PTPC_M64_2026-08-23.md).

The same current AITER main was adapted for the SGLang K3 profile by restoring
caller-owned MoE output, default-off stage1 scratch reuse, and the 27 N4480/
N6016 BF16 front tuning rows. Focused validation passed, including 50 vendored
FlyDSL tests. A corrected five-round C2 A/B measured `1136.17 tok/s` versus
`1146.00` on the existing AITER integration (`-0.86%`); median TTFT was
`871.71` versus `844.71 ms` and TPOT `14.99` versus `14.86 ms`. Keep current
AITER experimental because it misses the 0.5% promotion gate. Details:
[`aiter-optimization-tracker/CURRENT_AITER_SGLANG_C2_AB_2026-08-21.md`](aiter-optimization-tracker/CURRENT_AITER_SGLANG_C2_AB_2026-08-21.md).

The old ATOM native, RTL, and converted Perfetto traces were removed on
2026-08-13. Native profiling was then repeated in explicit production
`--cudagraph-mode FULL` with `--mark-trace`. On this Torch 2.13/ROCm 7.14
build, both the run and per-batch capture traces still contained host scopes
but zero GPU activities, so `tools/parse_trace.py` could not find its required
`gpu_user_annotation`.

A complete same-run graph trace was therefore captured with rank-0 PyTorch CPU
profiling for host/operator scopes and `rocm-trace-lite --mode full` for all
eight GPUs, then merged on the BS64 capture and first-decode transitions:

```text
/workspace/kimi-k3-runs/c64-graph-host-kernel-2026-08-13/
  atom-c64-full-graph-host-kernel.perfetto.json.gz  (138 MiB)

5,759,978 events:
  4,855,000 KernelExecution
    865,324 cpu_op
     39,590 user_annotation
```

The merged file includes the BS64 graph-capture host hierarchy, production
FULL-CUDAGraph replay host scopes, and measured GPU kernel durations. It opens
directly in Perfetto. The raw native run/capture pair and RTL database remain
beside it for reproducibility.

The matched single-stream production graph trace was captured with the same
method and `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`:

```text
/workspace/kimi-k3-runs/c64-single-stream-graph-host-kernel-2026-08-13/
  atom-c64-single-stream-full-graph-host-kernel.perfetto.json.gz  (143 MiB)

6,464,900 events:
  5,572,000 KernelExecution
    874,630 cpu_op
     18,206 user_annotation
```

Its raw native run/capture pair and RTL database are retained beside the
merged trace. The trace-mode 8192/64 C64 request wave completed 64/64 requests;
its throughput is profiler-perturbed and is not a benchmark result.

The matched current-stack single-stream C64 run disabled dual-stream MoE with
`ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD=0`:

```text
multi-stream:   8742.34 tok/s, 49.84 ms median TPOT
single-stream:  8399.50 tok/s, 53.33 ms median TPOT
single delta:     -3.92% throughput, +7.01% TPOT
```

Both unprofiled points completed 512/512 fixed 8192/1024 requests. Their
benchmark JSON/log files remain under
`/workspace/kimi-k3-runs/c64-stream-ab-2026-08-13/`; the superseded raw
multi/single RTL and Perfetto files were deleted. Historical details:
[`aiter-optimization-tracker/C64_ATOM_STREAM_AB_2026-08-13.md`](aiter-optimization-tracker/C64_ATOM_STREAM_AB_2026-08-13.md).

On 2026-08-14, a full C2-C64 fixed 8192/1024 stream A/B used
`num_prompts=concurrency*8`, `warmups=concurrency*2`, and
`random_range_ratio=1.0`. Multi-stream exceeded single-stream at every point:

```text
C2:  756.23 vs  671.05 tok/s (+12.69%)
C4: 1391.64 vs 1240.28 tok/s (+12.20%)
C8: 2470.31 vs 2243.62 tok/s (+10.10%)
C16: 4107.02 vs 3797.69 tok/s (+8.15%)
C32: 6377.19 vs 5958.70 tok/s (+7.02%)
C64: 8787.26 vs 8411.14 tok/s (+4.47%)
```

Multi-stream also had lower median TPOT at every concurrency. Artifacts:
`/workspace/kimi-k3-runs/c2-c64-8k1k-stream-ab-2026-08-14/`. Details:
[`aiter-optimization-tracker/C2_C64_ATOM_STREAM_AB_2026-08-14.md`](aiter-optimization-tracker/C2_C64_ATOM_STREAM_AB_2026-08-14.md).

Matched trace/code analysis finds:

```text
ATOM single-stream vs SGLang:       +385.28 tok/s (+4.81%)
ATOM single-stream TPOT delta:                    -4.04%
ATOM dual-stream increment:       +342.84 tok/s (+4.08%)
SGLang decode secondary stream:     0.011 s vs 15.446 s main-stream work
SGLang MLA CatArray launches:       1536 (24 layers x 64 decode steps)
ATOM fused Q/concat/cache launches: ~1534
```

SGLang is effectively single-stream in this MI355 production configuration.
The existing AITER fused Q/identity-RoPE/concat/cache operator was wired and
evaluated. It removed all 24-per-step target CatArray launches and passed
correctness. BF16 gains are small:

```text
BF16 C32: +0.63% throughput, -0.86% TPOT
BF16 C64: +0.53% throughput, -0.65% TPOT
```

With FP8 KV the same fusion passes the performance gate:

```text
FP8 C32: +3.26% throughput, -4.03% TPOT
FP8 C64: +3.57% throughput, -4.23% TPOT
FP8 capacity: 1868927 tokens
GSM8K 50: 1.000
```

The implementation is retained default-off. It is accepted for the Kimi-K3
FP8-KV profile but is not a BF16 production default. Details:
[`aiter-optimization-tracker/MLA_Q_CACHE_FUSION_RESULTS_2026-08-13.md`](aiter-optimization-tracker/MLA_Q_CACHE_FUSION_RESULTS_2026-08-13.md).

The full A8W4 + fused-Q/KV + FP8-KV endpoint sweep on
`perf/k3_opts_0812@e5f2bd991` completed 1,008/1,008 measured requests:

```text
C2:  980.24 -> 1013.37 tok/s (+3.38%), TPOT 17.47 -> 16.88 ms
C4: 1764.84 -> 1811.92 tok/s (+2.67%), TPOT 18.92 -> 18.30 ms
C8: 2921.82 -> 2980.01 tok/s (+1.99%), TPOT 22.15 -> 21.58 ms
C16: 4492.89 -> 4563.04 tok/s (+1.56%), TPOT 27.71 -> 27.09 ms
C32: 6295.63 -> 6356.34 tok/s (+0.96%), TPOT 37.59 -> 36.90 ms
C64: 8014.22 -> 8072.59 tok/s (+0.73%), TPOT 55.58 -> 54.58 ms
```

All points improve, although C32/C64 are below 1% and this is a historical
same-machine comparison rather than an interleaved paired A/B.

Enabling the optional B2 profile on the same fused FP8-KV configuration raises
C2 from `1013.37` to `1109.82 tok/s` (`+9.52%`) and lowers median TPOT from
`16.88` to `15.34 ms` (`-9.12%`). All 16 measured requests succeeded.

`perf/k3_opts_0812` was synced with `HaiShaw/main@65d62109d` in merge commit
`dc80c57a8`. Three K3 conflicts were resolved, 40 focused/integration tests
passed, and matched performance remained within the 0.5% gate:

```text
C2:  1013.37 -> 1011.53 tok/s (-0.18%), TPOT unchanged at 16.88 ms
C64: 8072.59 -> 8053.26 tok/s (-0.24%), TPOT unchanged at 54.58 ms
```

PR #34837 AITER BF16 prefill and PR #34580 tuned Triton decode were then
enabled together with a capacity-safe mixed backend:

```text
--attention-backend triton
--prefill-attention-backend aiter
--decode-attention-backend triton
SGLANG_AITER_FP8_PREFILL_ATTN=0
SGLANG_MLA_DECODE_TUNE=1
```

The full C2-C64 sweep completed 1,008/1,008 requests. Relative to the prior
A8W4 fused FP8-KV profile:

```text
C2  -3.07% throughput, TTFT -2.60%
C4  -0.95% throughput, TTFT -13.29%
C8  +0.28% throughput, TTFT -9.29%
C16 +2.43% throughput, TTFT -11.10%
C32 +3.81% throughput, TTFT -10.11%
C64 +6.51% throughput, TTFT -16.05%
```

Capacity remains 1,861,342 tokens. Using top-level AITER instead applies a
memory multiplier and drops capacity below the C64 workload, so the mixed
backend is the retained launch shape.

Enabling the optional B2 flags raises this profile's C2 from `982.24` to
`1071.03 tok/s` (`+9.04%`) and reduces median TPOT from `17.46` to `15.94 ms`
(`-8.71%`). It recovers most, but not all, of the tuned-decode C2 regression.

A compact C64 trace captured eight valid 18-19 MiB rank files. TP0 confirms
both paths:

```text
AITER/Opus prefill gqa_d192_v128: 408 calls, 506.44 us median
Triton decode stage1:              24 calls, 103.30 us median
Triton decode stage2:              24 calls,   8.66 us median
```

Details:
[`aiter-optimization-tracker/K3_AITER_PREFILL_TRITON_DECODE_2026-08-17.md`](aiter-optimization-tracker/K3_AITER_PREFILL_TRITON_DECODE_2026-08-17.md).

The K3 fused Q/cache boundary was extended to `triton_mla` decode with mixed
output/cache dtypes. Accepted BF16-Q result:

```text
GSM8K 1319: 0.951, invalid 0.001
target Q CatArray: 0 calls
C64: 8597.81 -> 8656.21 tok/s (+0.68%)
TPOT: 52.86 -> 52.32 ms (-1.02%)
```

An FP8 Q/Q-PE experiment failed the full accuracy gate (`0.929`) and regressed
C64 by 1.19% versus BF16 Q, so the experiment flag was removed. Details:
[`aiter-optimization-tracker/TRITON_MLA_Q_CACHE_FUSION_2026-08-17.md`](aiter-optimization-tracker/TRITON_MLA_Q_CACHE_FUSION_2026-08-17.md).

The 2026-08-17/18 KDA/B2/M4 follow-up produced three retained opt-in results:

```text
BF16-Q + FP8-KV sweep:
  C2/C4/C8/C16/C32/C64 =
  998.71 / 1830.68 / 3041.08 / 4729.40 / 6654.66 / 8656.21 tok/s

KDA decode winner:
  69-layer graph 9.20 -> 8.38 us/layer (-8.9%)
  matched C2 endpoint +0.19% throughput / -0.20% TPOT
  decision: retain opt-in, below 0.5% endpoint gate

Historical KDA winner + old M2 projection fusions:
  C2 1002.26 -> 1096.04 tok/s (+9.36%)
  TPOT 17.099 -> 15.561 ms (-8.99%)

Historical raw M4 mixed preroute:
  kernel 22.80 -> 20.19 us (+11.5%)
  C4 1834.22 -> 1867.15 tok/s (+1.80%)
  TPOT 18.230 -> 17.883 ms (-1.90%)
  decision: superseded by cooperative producer; raw dispatch removed
```

Full GSM8K policy gates on 2026-08-18:

```text
KDA winner + B2: GSM8K 50=1.000; GSM8K 1319=0.947, rerun=0.946
historical raw M4: GSM8K 50=1.000; GSM8K 1319=0.954
Invalid rate: 0.001 for every full run
BF16-Q reference: 0.951
```

The user-defined correctness threshold is accuracy greater than 0.94. Both M4
and combined KDA/B2 pass and are correctness-approved as opt-in profiles.

The current all-winner 8K/1K profile was rerun on `perf/k3_opts_0812` with
FP8 KV, AITER prefill, tuned Triton decode, MLA Q/cache, Radix-4, automatic C2
KDA, KDA B2, unified M2/M4 cooperative preroute and M16384:

```text
C2    1142.09 tok/s   TPOT 14.90 ms
C4    1967.69 tok/s   TPOT 16.88 ms
C8    3091.64 tok/s   TPOT 21.01 ms
C16   4797.55 tok/s   TPOT 26.10 ms
C32   6707.67 tok/s   TPOT 35.62 ms
C64   8710.86 tok/s   TPOT 51.96 ms
```

All 1,008 requests succeeded. Relative to the prior BF16-Q + FP8-KV canonical
sweep, throughput changes are `+14.36/+7.48/+1.66/+1.44/+0.80/+0.63%`.
Details:
`/workspace/kimi-k3-runs/best-profile-c2-c64-2026-08-18/RESULTS.md`.

Rejected redesigns:

```text
KDA three-stage split-V: 15.93-16.45 us, slower than 12.79 us baseline
KDA async state-to-LDS: 12.99 us, slower than direct-register path
A8W8 MFMA preroute hybrid: 32.10-36.20 us, slower than BF16
M4 fused shared-down: isolated micro win but C4 endpoint regressed
M4 SiTU-on-load MFMA: 37.14 us vs 14.24 us production chain
M4 single-wave preactivation: 33.83 us vs 29.19 us current chain
```

The M4 shared-down regression is not an overlap issue. SGLang K3 explicitly
uses `alt_streams=None` on HIP. The synthetic PyTorch microbenchmark overstated
the BF16 fallback cost; production uses fused SiTU plus a tuned BF16 GEMM.
The later production-faithful MFMA prototype was numerically correct and
graph-safe, but every N program recomputed SiTU and the best of 24 schedules
was 2.61x slower. Production `situ_and_mul` plus BF16 GEMM remains selected.

A FlyDSL producer-side follow-up then paired gate/up projection rows and emitted
`[4,768]` preactivated BF16 rows exactly once. Correctness and graph replay
passed, but the best of 28 schedules regressed the complete chain by 15.91%.
The dual accumulator/reduction and OCML cost exceeded the removed SiTU launch.
The experiment was removed before endpoint validation.

The follow-up two-wave cooperative FlyDSL design is retained default-off.
Adjacent waves compute gate/up separately, rendezvous through a small BF16 LDS
handoff, and one wave emits the preactivated row using fast exp2 SiTU. M2 and
M4 now share this design. With
interleaved shared weights and CU256/WPB8/WPE3/WCM3:

```text
micro complete chain: 29.34 -> 23.61 us (-19.5%)
C4 five-round mean: 1862.80 -> 1920.22 tok/s (+3.08%)
C4 TPOT: 17.622 -> 17.036 ms (-3.33%)
M2 MoE-only C2: 1065.26 -> 1076.75 tok/s (+1.08%)
M2 MoE-only TPOT: 16.036 -> 15.858 ms (-1.11%)
M2 cooperative + KDA: 1111.97 tok/s, 15.33 ms TPOT
unified GSM8K 1319: 0.951, invalid 0.001
```

Enable with `SGLANG_K3_AITER_MOE_PREROUTE_FP8=1` and
`SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1`. It covers exact M2/M4 and is
SGLang-source only.

An M8 extension was rejected at the micro gate. token_tile=8 measured
`147.24 us` due to 114 KiB LDS and long accumulator live ranges; the best
token_tile=4 schedule was still `36.97 us` versus the `32.91 us` production
chain (`+12.35%`). No C8 endpoint wiring was added.

Rejected experiment source was subsequently removed from the working tree:
KDA split-V, M4 fused FP8 shared-down, M8/M16 build allowances and the A8W8
MFMA hybrid benchmark path. The superseded raw M4 mixed-preroute API/dispatch
was also removed, leaving the cooperative producer as the single M4 path.
The older M2 tri/shared-down specialization was removed as well, leaving one
cooperative MoE design for M2/M4; `B2_FUSIONS` now controls KDA B2 only.
Retained KDA/M4 suites pass 23 tests.

Full handoff, flags, traces, source paths and stop decisions:
[`aiter-optimization-tracker/KDA_B2_M4_OPTIMIZATION_2026-08-18.md`](aiter-optimization-tracker/KDA_B2_M4_OPTIMIZATION_2026-08-18.md).

A direct SGLang A8W4→A16W4 flag switch was also tested with FP8 KV and MLA
Q/cache fusion. The server reached ready state, but GSM8K 50 was only `0.040`
with `0.020` invalid, so the trace stop gate fired. AITER contains the kernel
solution, but SGLang's caller/layout contract is not compatible through flags
alone. Details:
[`aiter-optimization-tracker/A16W4_FP8_MLA_Q_CACHE_SMOKE_2026-08-13.md`](aiter-optimization-tracker/A16W4_FP8_MLA_Q_CACHE_SMOKE_2026-08-13.md).

The next single-stream isolation experiment is ATOM with PTPC-FP8 online
quantization disabled. Dual-stream remains a separate future SGLang feature.
Absolute cross-framework kernel durations are not comparable because ATOM RTL
used Torch 2.13/ROCm 7.14 while SGLang PyTorch profiling used Torch 2.9/ROCm
7.2. Full trace analysis:
[`aiter-optimization-tracker/C64_ATOM_SGLANG_TRACE_COMPARISON_2026-08-13.md`](aiter-optimization-tracker/C64_ATOM_SGLANG_TRACE_COMPARISON_2026-08-13.md).

## Validated production result

```text
Focused vendored tests: 46 passed
GSM8K 50:              1.000
GSM8K 200:             0.990
Max token capacity:    933883

C2:   968.57 tok/s
C4:  1741.98 tok/s
C8:  2881.25 tok/s
C16: 4432.25 tok/s
C32: 6191.41 tok/s
```

All endpoint points are within 0.25% of the selected golden stack.

## 2026-08-13 Triton runtime attribution

The initial reclone used Triton 3.7 and regressed progressively at high
concurrency. Replacing only Triton with the handover commit
`3.6.0+git42270451` recovered C8-C32:

```text
C2:   966.08 tok/s  (-0.26%)  pass
C4:  1743.73 tok/s  (+0.10%)  pass on focused repeat
C8:  2887.31 tok/s  (+0.21%)  pass
C16: 4437.74 tok/s  (+0.12%)  pass
C32: 6176.87 tok/s  (-0.23%)  pass
```

Triton 3.6 improved C8/C16/C32 over the tested 3.7 build by
1.69%/2.98%/5.01%. Treat Triton 3.7 as the main high-concurrency regression
source and retain Triton 3.6 for the handover-equivalent runtime. Capacity
remained 933883 and all requests succeeded.

Details:
`aiter-optimization-tracker/TRITON36_AB_2026-08-13.md`.

Paired C32 traces confirmed that PyTorch/ROCtracer does not expand production
HIP CUDA Graph replays: only one replay was visible during a 15-second active
decode window. The endpoint A/B causally attributes the regression to the
Triton 3.7 runtime/codegen stack, but available traces do not localize it to a
specific Triton kernel. Details:
`aiter-optimization-tracker/TRITON36_37_C32_TRACE_2026-08-13.md`.

Stage-separated all-rank traces refined the attribution:

```text
prefill stage span:       +2.13%
prefill GPU kernel time:  -3.12%
record_param_comms p50:   +5.59%
profiled decode TPOT:     +8.54%
```

Coverage validation later showed that the stage-separated prefill traces expose
only 92 of 552 MoE GPU executions per TP0 (16.7%). A combined prefill+decode
capture records identical CPU MoE counts for 3.6/3.7, but Triton 3.7 hides over
99% of CUDA Graph replay kernels from ROCTracer, including MoE, fused KDA and
MLA merge. Therefore total GPU time and the earlier synchronization attribution
are not valid cross-version comparisons.

The later compact Rank0 trace restores comparable graph visibility and
localizes the prefill loss to
`python/sglang/kernels/ops/attention/extend_attention.py::_fwd_kernel`:

```text
p50:             5990.80 -> 13780.42 us (+130.03%)
total:            143.73 ->   330.90 ms (+187.17 ms)
prefill span:    4153.94 ->  4375.87 ms (+221.93 ms)
explained span:  84.34%
```

Triton 3.7 changes this kernel from 483 VGPR / no private segment to 512 VGPR /
472-byte private segment with 186 scratch spill instructions. Decode has
several smaller 2.7-4.5% regressions but no comparable single-kernel culprit.
Details:
`aiter-optimization-tracker/TRITON36_37_STAGE_ANALYSIS_2026-08-13.md`.

Future SGLang traces must use the validated compact method in
`aiter-optimization-tracker/SGLANG_TRACE_CAPTURE_METHOD_2026-08-13.md`:
one unmerged file per rank, late prefill plus short decode in one manual
profiler session, no stacks/shapes, and a 500 MiB compressed limit per rank.

## Optional Triton 3.7 extend-attention fix

Enable:

```bash
SGLANG_TRITON_37_EXTEND_LQ576_N32=1
```

The gfx950 Lq576/Lv512 extend-attention tile changes from N64 to N32 only on
Triton >=3.7. This removes 472-byte scratch spilling and restores the isolated
kernel from 12.57 to 5.24 ms.

Endpoint result:

```text
C2:   973.94 tok/s
C4:  1748.71 tok/s
C8:  2901.37 tok/s
C16: 4460.33 tok/s
C32: 6198.56 tok/s
```

C32 improves 5.38% over the Triton 3.7 baseline and is 0.12% above the
handover target. Capacity remains 933883. Keep default-off until GSM8K
validation is recorded.

Details:
`aiter-optimization-tracker/TRITON37_EXTEND_N32_RESULTS_2026-08-13.md`.

## Optional B2 profile

```text
C2:      968.57 -> 1054.19 tok/s
C2 TPOT:  17.67 ->   16.16 ms
C4:     1741.98 -> 1743.45 tok/s
```

Enable:

```bash
SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
SGLANG_K3_AITER_B2_FUSIONS=1
```

M>=4 fails closed to the production path.

## Validated Radix-4 router profile

Enable:

```bash
SGLANG_K3_RADIX4_TOPK=1
```

Validation:

```text
Focused tests: 45 passed
C2 paired median: 970.38 -> 993.11 tok/s (+2.34%)
C2 TPOT:          17.63 -> 17.21 ms (-2.38%)
C4:             1741.98 -> 1777.66 tok/s (+2.05%)
C8:             2881.25 -> 2930.39 tok/s (+1.71%)
C16:            4432.25 -> 4491.62 tok/s (+1.34%)
C32:            6191.41 -> 6233.41 tok/s (+0.68%)
GSM8K 200:       0.985
capacity:        933883
```

The isolated integration adds deterministic AITER-compatible exact-tie
selection, NaN exclusion and a gfx942/gfx950 guard beyond upstream #34490.
Keep it default-off until those correctness fixes are reconciled upstream.

## Source selection

```bash
SGLANG_K3_FLYDSL_SOURCE=auto
SGLANG_K3_FLYDSL_SOURCE=sglang
SGLANG_K3_FLYDSL_SOURCE=aiter
```

The SGLang-owned M16384 profile is enabled with:

```bash
SGLANG_K3_AITER_M16384_PROFILE=1
```

## A16W4 caller-contract result

The direct A16W4 flag switch originally produced GSM8K-50 accuracy 0.040
because SGLang shuffled SiTU w13 weights and scales as A8W4 GUGU while the
AITER runner consumed `GateMode.SEPARATED`. The caller now selects:

```text
A16W4: special A16W4 lane shuffle, gate_up=False
A8W4:  special A16W4 lane shuffle, gate_up=True
A4W4:  generic separated shuffle
```

Fresh TP8 FP8-KV validation:

```text
focused layout tests: 6 passed
runtime: gemm1_a16w4_port_* and gemm2_a16w4_port_*
Opus/A8W4 stage GEMMs: absent
MLA FP8 fused cache write: present
GSM8K 50:  1.000, invalid 0.000
GSM8K 200: 0.980, invalid 0.005
GSM8K 1319: 0.948, invalid 0.001 (one response)
capacity: 1,868,927 tokens (unchanged)
```

The complete set leaves one deterministic invalid response, which the user
accepted. Keep the caller-contract fix, but do not select A16W4 for production.

The corrected C64 compact trace completed 64/64 requests and captured eight
valid rank files (approximately 149 MiB compressed). TP0 contains 204,494 GPU
kernels over a 23.350-second profiler span:

```text
MoE GEMM 27.69% · dense GEMM 22.43% · MLA attention 18.10%
collectives 16.07% · attention residual 7.52% · KDA 3.48%
A16W4 stage variants: 4 · A8W4/Opus stage GEMMs: 0
FP8 fused MLA cache write: present
```

The matched 8,192/1,024 endpoint sweep completed 1,008/1,008 requests:

```text
C2:  1013.37 ->  981.61 tok/s (-3.13%), TPOT 16.88 -> 17.32 ms
C4:  1811.92 -> 1698.47 tok/s (-6.26%), TPOT 18.30 -> 19.50 ms
C8:  2980.01 -> 2812.07 tok/s (-5.64%), TPOT 21.58 -> 22.75 ms
C16: 4563.04 -> 4352.35 tok/s (-4.62%), TPOT 27.09 -> 28.05 ms
C32: 6356.34 -> 5729.64 tok/s (-9.86%), TPOT 36.90 -> 40.79 ms
C64: 8072.59 -> 7354.88 tok/s (-8.89%), TPOT 54.58 -> 59.40 ms
```

A16W4 loses throughput and regresses TPOT at every point despite matched
correctness and capacity. Retain A8W4 as production MoE mode.

The apparent same-kernel ATOM/SGLang timing gap was investigated separately.
Two findings:

```text
1. SGLang passed stride-[6016,1] input to a stride-less A16W4 port.
   Isolated correctness failed on 22.4% of elements.
   Fix: materialize contiguous A16W4 input before fused_moe.
   GSM8K 50 remains 1.000; C64 performance is unchanged.

2. The random clients produced different route distributions.
   SGLang: 582 unique experts; ATOM: 454 unique experts.
   Isolated 582/454 replay ratio: stage1 1.283x, stage2 1.247x.
   Observed trace ratio:          gemm1 1.324x, gemm2 1.362x.
```

Thus the large kernel gap is primarily active-expert/BM32 padding work, not a
different AITER kernel implementation.

Details:
`aiter-optimization-tracker/A16W4_FP8_MLA_Q_CACHE_SMOKE_2026-08-13.md`.

## Decisions already made

Keep:

```text
fused KDA + f_b
MoE zero-copy output
stage1 scratch reuse
MLA gate
KDA group64
SGLang-vendored Kimi FlyDSL kernels
B2-only optional profile
Radix-4 K3 TopK router
```

Optional only:

```text
FP8 preroute/shared-down
FP8 latent tail
A4W4 C16 profile
```

Do not retry without an architecture change:

```text
V3/V3-R role-grid + P23
V4 multi-CU persistent route prep
generic TILE_M M4/M8/M16
forced all-reduce global/exact-B32
standalone #4572/#4577 endpoint paths
```

## Remaining optimization work

Completed integration:

```text
SGLang #34490 Radix-4 Kimi-K3 TopK router
measured MI355X saving: 4.2-5.2 us/layer for M1-M64
status: all gates passed; retained default-off
```

This exact Radix-4 nibble-histogram/DPP design was not tested in the prior
V3/V3-R/V4 work. The earlier reports anticipated a broader radix direction,
but V3 used 16 repeated argmax rounds and failed for different architectural
reasons.

Highest-value unresolved areas:

```text
fixed tiny-kernel chains
copies/materialization
attention residual and KDA launch boundaries
route / sort / quant handoff
```

Credible route directions:

```text
one-CTA LDS E896 sorter
stage1 ABI consuming route metadata and token-major scale directly
```

Next, finish analysis of the retained B300 normal versus
single-stream/no-PDL summaries.

## Detailed sources

```text
aiter-optimization-tracker/SGLANG_VENDOR_FLYDSL_2026-08-12.md
aiter-optimization-tracker/FRESH_INTEGRATION_2026-08-12.md
aiter-optimization-tracker/AITER_DEPENDENCY_MATRIX_2026-08-12.md
aiter-optimization-tracker/B2_FUSION_SOLIDIFICATION_2026-08-11.md
aiter-optimization-tracker/B300_MI355X_TRACE_COMPARISON_2026-08-11.md
aiter-optimization-tracker/PAUSE_TRACK_2026-08-11.md
aiter-optimization-tracker/PR34490_RADIX4_ASSESSMENT_2026-08-12.md
aiter-optimization-tracker/PR34490_RADIX4_RESULTS_2026-08-12.md
```
