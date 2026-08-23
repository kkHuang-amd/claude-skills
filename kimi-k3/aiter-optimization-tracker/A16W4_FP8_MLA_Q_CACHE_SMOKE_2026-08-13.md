# Kimi-K3 SGLang A16W4 caller contract — 2026-08-13

## Decision

Keep the SGLang A16W4 caller-contract correction because it restores
correctness, but reject A16W4 as the production MoE mode. It recovered
GSM8K-50 from 0.040 to 1.000 and produced 0.948 accuracy with one invalid
response over the complete 1,319-question set. The matched C2-C64 endpoint
sweep then showed 3.13-9.86% lower throughput than A8W4 at every concurrency.

## Configuration

```text
Machine: crsuse2-m2m-002.crusoe.amd.com
SGLang: d1aed97a6 (local MLA Q/cache fusion commit)
AITER:  284a1eb401bb15f6368a68b34eb0cd693ee1fcd3
TP: 8
KV cache: fp8_e4m3
MLA Q/cache fusion: enabled

AITER_SITUV2_A8W4=0
AITER_SITUV2_A4W4=0
AITER_FLYDSL_FORCE=1
```

The AITER commit contains the same A16W4 FlyDSL gemm1/gemm2 solution family as
the ATOM image's ancestor commit. That source equivalence is insufficient to
guarantee caller compatibility.

## Initial flag-only result

```text
server ready: yes
max_total_num_tokens: 1868927
GSM8K 50 accuracy: 0.040
invalid: 0.020
```

Stop gate triggered before trace capture.

## Root cause and implementation

SGLang treated default A16W4 like A8W4 and passed gate/up-interleaved weights
and scales while the runner used `GateMode.SEPARATED`. The corrected contract
uses the special A16W4 lane shuffle in both modes but passes
`gate_up=False` for A16W4 and `gate_up=True` only for A8W4. A4W4 remains on
the generic separated path.

Focused validation:

```text
test_mxfp4_situ_weight_layout.py: 6 passed
unset / explicit 0/0: A16W4 special shuffle, separated gate/up
A8W4 precedence: special shuffle, interleaved gate/up
A4W4: generic separated shuffle
synthetic weight and scale layouts: A16W4 equals separated and differs from A8W4
```

The combined focused MXFP4/Kimi regression selection finished with 16 passed,
1 skipped and 2 unrelated prerequisite-op failures. Both failures are in the
legacy `moe_route_radix` HIP JIT compile, not the production optional
`moe_route_radix4` path. The actual ROCm Radix-4 suite passes 45/45 tests on
this environment; neither legacy failure reaches the modified MXFP4 layout
code.

## Fresh TP8 validation

```text
KV cache: torch.float8_e4m3fn
max_total_num_tokens: 1868927 (unchanged)
GSM8K 50:  Accuracy 1.000, Invalid 0.000
GSM8K 200: Accuracy 0.980, Invalid 0.005
GSM8K 1319: Accuracy 0.948, Invalid 0.001 (one response)
```

The TP0 trace contains:

```text
gemm1_a16w4_port_a16w4_h3584_i384_ne896_bm32_tn128_situv2_xcd4
gemm1_a16w4_port_a16w4_h3584_i384_ne896_bm32_tn32_situv2_xcd4_kw2
gemm2_a16w4_port_ne896_h3584_i384_bm32_tn128
gemm2_a16w4_port_ne896_h3584_i384_bm32_tn128_xcd4
fuse_qk_rope_concat_and_cache_mla_per_head_kernel<..., Fp8KVCacheDataType, ...>
```

No Opus/A8W4 stage1 or stage2 GEMM appears. Opus sorting remains present and
is shared route preparation, not an A8W4 stage GEMM.

The invalid rate drops to one response over the complete 1,319-question set
and is consistent with the deterministic invalid response already seen in
prior production-profile reruns. The user explicitly accepted this result and
overrode the earlier zero-invalid stop decision.

## Corrected C64 compact trace

The retained configuration completed 64/64 requests at C64 with 8,192 input
tokens and 64 output tokens. The profiler window spans 23.350 seconds on TP0
and contains 204,494 GPU kernels:

```text
MoE GEMM:             27.69% kernel time
dense GEMM:           22.43%
MLA attention:        18.10%
collectives:          16.07%
attention residual:    7.52%
KDA:                   3.48%
```

Runtime dispatch contains four A16W4 stage kernel variants, zero A8W4/Opus
stage GEMMs, and the FP8 `fuse_qk_rope_concat_and_cache_mla_per_head_kernel`.
All eight rank traces pass gzip integrity checks and total approximately
149 MiB compressed. The profiled 64-output benchmark is a trace-capture
workload and is not a replacement for the planned paired 8,192/1,024 endpoint
performance rounds.

## Matched A16W4 endpoint sweep

The full fixed 8,192/1,024 sweep kept FP8 KV, fused Q/KV prep, TP8, memory
settings, warmups and client settings identical to A8W4:

```text
C    A8W4 tok/s   A16W4 tok/s   Delta    A8 TPOT   A16 TPOT   Delta
2        1013.37        981.61   -3.13%      16.88       17.32   +2.61%
4        1811.92       1698.47   -6.26%      18.30       19.50   +6.56%
8        2980.01       2812.07   -5.64%      21.58       22.75   +5.42%
16       4563.04       4352.35   -4.62%      27.09       28.05   +3.54%
32       6356.34       5729.64   -9.86%      36.90       40.79  +10.54%
64       8072.59       7354.88   -8.89%      54.58       59.40   +8.83%
```

All 1,008 requests succeeded and capacity remained 1,868,927 tokens. A16W4
fails both the C32 non-regression requirement and the required C64 gain.
Retain A8W4 for production.

## Artifacts

```text
/workspace/kimi-k3-runs/a16w4-fp8-mla-q-cache-c64-trace-2026-08-13/server.log
/workspace/kimi-k3-runs/a16w4-fp8-mla-q-cache-c64-trace-2026-08-13/gsm8k-50.log
/workspace/kimi-k3-runs/a16w4-fp8-mla-q-cache-c64-trace-2026-08-13/run.sh
/workspace/kimi-k3-runs/a16w4-caller-fix-2026-08-13/server.log
/workspace/kimi-k3-runs/a16w4-caller-fix-2026-08-13/gsm8k-50.log
/workspace/kimi-k3-runs/a16w4-caller-fix-2026-08-13/gsm8k-200.log
/workspace/kimi-k3-runs/a16w4-caller-fix-2026-08-13/dispatch-analysis-tp0.json
/workspace/kimi-k3-runs/a16w4-caller-fix-2026-08-13/dispatch-traces/
/workspace/kimi-k3-runs/a16w4-caller-fix-2026-08-13/gsm8k-1319.log
/workspace/kimi-k3-runs/a16w4-caller-fix-c64-trace-2026-08-13/run.sh
/workspace/kimi-k3-runs/a16w4-caller-fix-c64-trace-2026-08-13/client.log
/workspace/kimi-k3-runs/a16w4-caller-fix-c64-trace-2026-08-13/trace-analysis-tp0.json
/workspace/kimi-k3-runs/a16w4-caller-fix-c64-trace-2026-08-13/traces/
/workspace/kimi-k3-runs/a16w4-fused-qkv-fp8kv-sweep-2026-08-14/
```

Raw traces and logs are retained pending explicit cleanup approval.
