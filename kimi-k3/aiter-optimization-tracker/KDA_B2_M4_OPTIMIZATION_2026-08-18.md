# Kimi-K3 KDA, B2 and M4 optimization handoff — 2026-08-18

## Scope

This report consolidates the 2026-08-17/18 work on:

- BF16-Q fused MLA cache output with FP8 KV;
- C2-C64 canonical endpoint sweep;
- B300 versus MI355X C2 decode traces;
- gfx950 KDA decode optimization;
- optional B2 MoE/KDA projection fusions;
- M4 single-launch mixed preroute;
- rejected split-V, async-state, A8W8 hybrid and M4 shared-down designs.

Current repositories:

```text
SGLang /sgl-workspace/sglang
  branch perf/k3_opts_0812
  HEAD c8e8c5080d4c2d24b9777caf4b11654ad76f2840
  KDA/M4 work is uncommitted

AITER /sgl-workspace/aiter
  branch integration/k3-core-only
  HEAD 284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
  pre-existing modified op_tests/test_moe_2stage.py remains
```

Do not include `cuda_graph_runner_memory_usage.pickle` in a commit.

## Selected launch shape

Common retained profile:

```bash
SGLANG_K3_AITER_MLA_Q_CACHE_FUSION=1
SGLANG_K3_TRITON_FP8_Q=0
SGLANG_AITER_FP8_PREFILL_ATTN=0
SGLANG_MLA_DECODE_TUNE=1
SGLANG_K3_FLYDSL_SOURCE=sglang
SGLANG_K3_AITER_M16384_PROFILE=1
SGLANG_USE_AITER=1
SGLANG_AITER_K3_OPT=1
AITER_FLYDSL_FORCE=1
AITER_SITUV2_A8W4=1
AITER_SITUV2_A4W4=0
AITER_FLYDSL_STAGE1_SCRATCH_REUSE=1
SGLANG_K3_FLYDSL_AR_NORM=1
SGLANG_K3_KDA_FUSED_BACKEND=aiter
SGLANG_K3_AITER_MLA_GATE=1
SGLANG_K3_AITER_KDA_GROUP64=1
SGLANG_K3_AITER_LATENT_TAIL_FP8=0
```

The KDA C2 winner is selected automatically whenever
`SGLANG_K3_KDA_FUSED_BACKEND=aiter` reaches an exact-C2 batch. Other batches
retain the original build options; no per-knob C2 flags are required.

Unified M2/M4 cooperative MoE preroute:

```bash
SGLANG_K3_AITER_MOE_PREROUTE_FP8=1
SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1
SGLANG_K3_FLYDSL_SOURCE=sglang
```

Optional KDA B2 group64 specialization:

```bash
SGLANG_K3_AITER_B2_FUSIONS=1
```

M2 and M4 use the cooperative producer. M8/M16 fail closed to merged BF16.

## BF16-Q + FP8-KV canonical sweep

Fixed 8192/1024, TP8, 64 warmups, 8×concurrency measured requests:

```text
C2    998.71 tok/s   output 110.97   TPOT 17.16 ms
C4   1830.68 tok/s   output 203.41   TPOT 18.25 ms
C8   3041.08 tok/s   output 337.90   TPOT 21.41 ms
C16  4729.40 tok/s   output 525.49   TPOT 26.49 ms
C32  6654.66 tok/s   output 739.41   TPOT 35.96 ms
C64  8656.21 tok/s   output 961.80   TPOT 52.32 ms
```

All 1,008 measured requests succeeded. CSV:

```text
/workspace/kimi-k3-runs/triton-q-cache-fusion-2026-08-17/
  bf16-q/sweep-8k1k/performance-table.csv
```

## C2 B300 versus MI355X decode attribution

Matched no-radix C2 traces, all eight ranks, normalized per decode step:

```text
B300 step        9.31 ms
MI355X step     18.08 ms
gap              8.77 ms

lost stream overlap equivalent   5.47 ms
extra summed kernel work         3.30 ms
```

This is a stack comparison, not hardware-only:

```text
B300:  TRT-LLM MLA + FlashInfer MXFP4
MI355X: Triton MLA + AITER A8W4 SiTU
```

The MI355X Kimi code explicitly disables `alt_streams` on HIP. ROCm supports
streams generally, but this SGLang K3 execution is effectively single-stream.

Major summed-work gaps:

```text
Router/top-k/quant/sort   +1.78 ms and +184 calls
Dense/shared path        +1.69 ms and +116 calls
Norm/elementwise         +1.05 ms and +201 calls
MLA/KDA                   -1.05 ms and -231 calls
```

Artifacts and parsers:

```text
/workspace/kimi-k3-runs/triton-q-cache-fusion-2026-08-17/
  bf16-q/c2-trace-2026-08-17/
    DECODE_COMPARISON_B300.md
    decode-comparison.json
    decode-kernel-table.csv
    model-op-flow-summary.csv
    model-op-flow-kernel-map.csv
    layer-kernel-path-comparison.csv
```

## KDA decode single-kernel optimization

Baseline kernel:

```text
kimi_k3_kda_decode_fb_bf16_gfx950
```

ATT showed:

```text
total stalls                       64.3%
projection/convolution barrier     28.3%
LDS/SMEM wait                      28.2%
VMEM wait + load                   28.9%
output barriers                     5.6%
late output-gate load               3.6%
```

Accepted opt-in variant:

```text
kimi_k3_kda_decode_fb_bf16_gfx950_wpe3_cfa1_pf1_fnr1_fd21
```

Changes:

- cooperative `f_a` LDS load;
- projection waves 0-1 / convolution waves 2-3;
- BF16 `fdot2` projection;
- recurrent BF16 square accumulation folded into RMSNorm reduction;
- one output barrier removed;
- `waves_per_eu=3`.

Resource change:

```text
VGPR    88 -> 76
LDS   1536 -> 2048 bytes
scratch 0
```

Single-node graph reports about 12 us because graph replay overhead dominates.
Production-like 69-layer graph:

```text
baseline   9.20 us/layer
winner     8.38 us/layer
gain       8.9%
```

Matched endpoint:

```text
baseline  1002.26 tok/s   TPOT 17.0986 ms
winner    1004.19 tok/s   TPOT 17.0646 ms
delta       +0.19%             -0.20%
```

This misses the 0.5% endpoint gate, so KDA winner stays opt-in.

Full details:

```text
/workspace/kimi-k3-runs/triton-q-cache-fusion-2026-08-17/
  bf16-q/KDA_C2_OPTIMIZATION_RESULTS.md
```

## Combined KDA winner + B2 fusions

Both B2 flags enabled with the KDA winner:

```text
run 1   1096.01 tok/s   TPOT 15.562 ms
run 2   1096.07 tok/s   TPOT 15.559 ms
mean    1096.04 tok/s   TPOT 15.561 ms
```

Versus matched baseline:

```text
throughput   +9.36%
TPOT         -8.99%
ITL          -8.97%
TTFT         -0.42%
```

Trace-confirmed kernels:

```text
kimi_k3_b2_tri_projection_bf16_fp8_gfx950...
kimi_k3_b2_situ_shared_down_bf16_fp8_gfx950...
kimi_k3_kda_input_m2_n6288_stored6284_k7168_e4m3g64_gfx950...
kimi_k3_kda_decode_fb_bf16_gfx950_wpe3_cfa1_pf1_fnr1_fd21
```

Eight valid TP-rank traces, 161 MiB compressed:

```text
/workspace/kimi-k3-runs/triton-q-cache-fusion-2026-08-17/
  bf16-q/kda-winner-b2-fusions-c2-2026-08-17/
```

The trace was summarized and then deleted with explicit user approval. See the
three `RAW_TRACE_DELETION_2026-08-18*.tsv` manifests in the Kimi workspace.

## M4 single fused mixed preroute

The rejected A8W8 MFMA hybrid required separate activation quantization and a
separate BF16 router launch:

```text
activation quant   9.15 us
FP8 MFMA          12.82 us
router BF16       14.28 us
```

It lost to the 22-23 us merged BF16 front at M4/M8/M16.

A new single-launch A16W8 mixed kernel was built instead:

```text
kimi_k3_m4_mixed_tri_bf16_fp8_gfx950_tt4_cu256_wpb4_wpe0_wcm2
```

It stages four BF16 rows in 57 KB LDS, reuses each weight load across four
token accumulators, handles routed/shared FP8 rows and BF16 router rows in one
grid, and preserves BF16-round-to-FP32 router logits.

```text
M4 mixed   20.19 us vs BF16 22.80 us  (+11.5%)
M8 mixed   34.01 us vs BF16 23.47 us  reject
M16 mixed  62.65 us vs BF16 22.43 us  reject
```

Matched C4 endpoint:

```text
baseline    1834.22 tok/s   TPOT 18.230 ms
candidate   1867.15 tok/s   TPOT 17.883 ms
delta          +1.80%             -1.90%
```

This raw-gate/up M4 result was later superseded by the cooperative
preactivated producer. Its independent API/dispatch was removed; M8/M16 remain
BF16.

### Rejected M4 shared-down fusion

The M4 fused SiTU + FP8 shared-down kernel measured 16.16 us against a
synthetic PyTorch reference of 31.63 us, but that reference was not the
production baseline. Production uses a fused SiTU kernel plus tuned BF16 GEMM
on the same HIP stream.

Endpoint:

```text
front-only M4          1867.15 tok/s
front + shared-down    1811.27 tok/s
matched BF16 baseline  1834.22 tok/s
```

The earlier multi-stream explanation was wrong. K3 HIP is single-stream. The
actual failure is that each persistent output block recomputes the complete
SiTU activation, while the production chain computes activation once and uses
a tuned tiled GEMM. Runtime `shared_down_covered()` therefore remains
fail-closed for M>2.

Full result:

```text
/workspace/kimi-k3-runs/triton-q-cache-fusion-2026-08-17/
  bf16-q/preroute-mixed-m4/RESULTS.md
```

### Rejected M4 SiTU-on-load MFMA follow-up

A production-faithful BF16 follow-up first measured the actual chain:

```text
SiTU GPU kernel                 3.239 us
Tensile BF16 M4/N7168/K768     5.319 us
GPU kernel sum                 8.558 us
CUDA Graph chain median       14.241 us
```

This confirms the prior approximately 8.72 us/layer production baseline. A
SGLang-owned exact-M4 Triton MFMA prototype accepted raw gate/up input, applied
SiTU, rounded to BF16, and performed down projection in one kernel.

```text
relative RMSE                  0.001662
cosine                         0.9999987
graph replay                   bit-exact
best of 24 schedules          37.140 us
prototype / production          2.608x slower
```

The prototype failed the 5% micro gate before model wiring. Independent N
programs must recompute the same SiTU input, so changing the dot-product core
from fdot2 to MFMA does not fix the architecture. Endpoint, GSM8K and trace
gates were skipped by design. The prototype source was removed.

Decision: production `situ_and_mul` plus tuned BF16 GEMM remains the selected
M4 shared-down path. Do not retry input-activation fusion inside an
N-partitioned GEMM. A future attempt must move SiTU into the shared-projection
producer so the activated rows are emitted exactly once.

Full result:

```text
/workspace/kimi-k3-runs/situ-down-m4-2026-08-18/RESULTS.md
```

### Rejected M4 producer-side preactivation

The next architecture moved SiTU to the FlyDSL shared-projection producer
instead of the N-partitioned down GEMM. Exact-M4 shared work changed from
1,536 independent gate/up rows to 768 paired rows. Each pair reused hidden LDS,
accumulated both FP8 projections with native fdot2, preserved the independent
BF16 rounding boundaries, applied OCML SiTU, and emitted `[4,768]` BF16
preactivated rows.

Correctness passed:

```text
routed relative RMSE             0.001658
preactivated shared RMSE          0.0000092
router relative RMSE              0.0000998
shared cosine                     1.0000001
alternate beta 3/17 RMSE          0.00000054
CUDA Graph replay                 bit-exact
M2                                fail-closed
```

Micro performance did not pass:

```text
current tri producer             20.173 us
current full chain               29.186 us
initial preactivated chain       34.689 us
best preactivated chain          33.829 us
best delta                       +15.91% regression
```

The schedule campaign covered 28 CU/WPB/WPE/cache combinations. Pairing removed
768 row jobs and the standalone SiTU launch, but the producer's second
accumulator set, second wave reduction and OCML work cost more. The 5% micro
gate failed, so model wiring, endpoint, GSM8K and trace gates were skipped.
Experimental source/API changes were removed; the original multitoken suite
still passes (4 tests).

FlyDSL remains a better technical fit than Triton for this producer because the
validated path relies on fdot2, DPP router reduction and persistent row
scheduling. The FlyDSL result nevertheless closes producer-side preactivation
for this pair-row architecture. Do not attempt a combined Triton rewrite
without a new arithmetic or pipeline design.

Full result:

```text
/workspace/kimi-k3-runs/preactivated-producer-m4-2026-08-18/RESULTS.md
```

### Accepted M4 cooperative preactivated producer

A final FlyDSL redesign split each shared gate/up pair across adjacent waves
instead of retaining both accumulator sets in one wave. Lane 0 of each wave
writes its independently BF16-rounded projection into a small LDS handoff;
after one block barrier, the gate wave evaluates SiTU and emits `[4,768]`
preactivated BF16 rows. The production BF16 down GEMM remains unchanged.

Selected exact-M4 schedule:

```text
CU256 / WPB8 / WPE3 / WCM3
interleaved gate/up weight rows
fast exp2 SiTU
```

Fast SiTU rounded bit-identically to the precise OCML producer in focused
tests. Correctness and micro results:

```text
focused suite                  11 passed
preactivated shared RMSE       0.00000014
graph replay                   bit-exact
current complete chain         29.34 us
selected complete chain        23.61 us
micro latency                 -19.5%
```

Five-round matched C4:

```text
total throughput    1862.80 -> 1920.22 tok/s  (+3.08%)
median TPOT           17.622 ->   17.036 ms    (-3.33%)
median ITL            17.590 ->   17.006 ms    (-3.32%)
median TTFT         1736.288 -> 1736.302 ms    (+0.00%)
throughput CV          0.029% / 0.072%
```

All 640 measured requests succeeded. C2 fail-closed guard was `1000.10 tok/s`
and `17.14 ms` TPOT. Accuracy passed:

```text
GSM8K 50     1.000, invalid 0.000
GSM8K 1319   0.955, invalid 0.001
```

An exact-M4 compact trace confirmed the new
`cooppreact768_fast1_interleaved1` kernel, production BF16 down GEMM, and no
old M4 FP8 shared-down. Raw traces were summarized and deleted.

Decision: retain default-off behind
`SGLANG_K3_PREROUTE_PREACTIVATED_SHARED=1` for exact M2/M4 and SGLang source.
M1 retains its B1 path; AITER-source mode fails closed.

Full result:

```text
/workspace/kimi-k3-runs/coop-preactivated-m4-2026-08-18/RESULTS.md
```

#### M8 cooperative follow-up

The retained M4 architecture was tested at M8 before extending model wiring.
A direct token_tile=8 version was numerically correct but measured
`147.24 us` versus the `32.91 us` production chain because eight-token live
ranges and approximately 114 KiB LDS created severe resource pressure.

A focused token-tile/CU/WPB sweep found token_tile=4, CU256, WPB8 as the best
alternative:

```text
M8 production chain                32.906 us
M8 cooperative token_tile=4        36.969 us
delta                              +12.35% regression
```

token_tile=4 scans the front weights twice, while token_tile=8 loses to
LDS/register pressure. The micro gate failed, so C8 endpoint and GSM8K were
skipped. The temporary M8 allowance was removed; cooperative preactivation
covers only exact M2/M4.

```text
/workspace/kimi-k3-runs/coop-preactivated-m8-2026-08-18/RESULTS.md
```

#### M2 design unification

The prior M2 MoE-only path (`FP8 tri → fused SiTU + FP8 shared-down`) was
compared directly against the cooperative producer plus production BF16 down.
KDA group64 and KDA winner flags were disabled during the design A/B.

```text
micro complete boundary       22.843 -> 21.027 us  (-7.95%)
C2 five-round throughput     1065.26 -> 1076.75    (+1.08%)
C2 median TPOT                 16.036 -> 15.858 ms (-1.11%)
C2 median ITL                  16.030 -> 15.850 ms (-1.12%)
```

Both stable clusters had throughput CV below `0.006%`. The cooperative design
also composed successfully with KDA B2:

```text
C2 total throughput          1111.97 tok/s
median TPOT                    15.33 ms
GSM8K 1319                     0.951, invalid 0.001
```

Decision: use cooperative preactivation for both M2 and M4. Remove the older
M2 tri/shared-down specialization. `SGLANG_K3_AITER_B2_FUSIONS` now controls
only KDA B2 group64.

```text
/workspace/kimi-k3-runs/m2-preroute-design-ab-2026-08-18/RESULTS.md
```

## Rejected architectural experiments

Do not repeat without changing the architecture:

```text
Three-stage split-V KDA:
  V2/V4/V8 = 16.45/16.20/15.93 us vs 12.79 us baseline.
  Prepare/workspace/finalize launches cost more than added parallelism saves.

Single-kernel async state-to-LDS:
  32-row 16 KB double buffer = 12.99 us; direct register path is faster.

A8W8 MFMA preroute hybrid:
  M4/M8/M16 = 35.77/36.20/32.10 us vs 22-23 us BF16.

M8/M16 single mixed preroute:
  LDS/register pressure or M4 subtiles reread weights 2x/4x.

M4 fused shared-down:
  isolated synthetic baseline looked favorable, endpoint regressed.

M4 SiTU-on-load MFMA:
  37.14 us vs 14.24 us production; N programs repeat SiTU.

M4 single-wave dual-accumulator preactivation:
  best chain 33.83 us vs 29.19 us; dual accumulators/reductions dominate.
```

Rejected-source cleanup:

```text
removed KDA split-V wrapper, kernels, tests and benchmark mode
removed M4 fused FP8 shared-down kernel and launcher
removed M8/M16 multitoken build/test allowance
removed rejected A8W8 MFMA hybrid benchmark path
removed superseded raw M4 mixed-preroute API/dispatch
removed superseded M2 tri/shared-down MoE specialization
```

The retained KDA C2 winner and unified exact-M2/M4 cooperative preactivation
remain.
Focused retained suites pass 23 tests.

## GSM8K policy gates — 2026-08-18

Both policies used the retained BF16-Q + FP8-KV mixed backend. Each ran a
50-question smoke followed by the complete 1,319-question set at temperature
zero.

```text
Policy                  GSM8K 50   GSM8K 1319   Invalid
KDA winner + B2 flags      1.000         0.947     0.001
KDA/B2 complete rerun          -         0.946     0.001
M4 mixed preroute          1.000         0.954     0.001
BF16-Q reference           1.000         0.951     0.001
```

Decision:

- M4 passes the full correctness gate and remains a valid default-off opt-in.
- The user-defined acceptance threshold is accuracy greater than 0.94.
  Combined KDA/B2 passes on both complete runs (0.947 and 0.946) and is
  correctness-approved as an opt-in profile.
- Both servers completed without initialization/runtime errors; all eight GPUs
  returned to idle VRAM after testing.

Artifacts:

```text
/workspace/kimi-k3-runs/gsm8k-policy-gates-2026-08-18/
  kda-b2/gsm8k-50.log
  kda-b2/gsm8k-1319.log
  kda-b2/gsm8k-1319-rerun.log
  m4/gsm8k-50.log
  m4/gsm8k-1319.log
```

## Tests

Latest focused results:

```text
KDA decode/split-V suite       20 passed
MoE preroute/mixed suite       10 passed
Canvas TypeScript checks       clean for generated working canvases
```

## Next work

Highest-value next steps:

1. Decide whether to retain the M4 mixed-front opt-in in the next local commit.
2. Decide whether the correctness-approved combined KDA/B2 profile should
   remain opt-in or become a default.
3. Analyze route/sort/quant handoff; it remains the largest serial module gap.
4. Revisit attention-residual `_agg_kernel` only with an ATT-backed design.
5. Keep future raw profiler captures temporary and append a new deletion
   manifest after their analysis is durable.

