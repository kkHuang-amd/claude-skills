# Kimi-K3.1 numerics rules (AgentX / InferenceX) -- every optimization must respect these

Owner node: mi355-4. Source: guideline screenshot from the user, 2026-11-02. These bind every K3.1 change
(sglang-k31-amd PRs, aiter kernels, the sgl-amd-mono monokernel). Before proposing or merging an optimization,
check it against the table; an idea that needs a "no" cell is out of scope, not a trade-off to argue.

## Rules

| Component | Allowed | Note |
|-----------|---------|------|
| Linear ops (dense GEMMs: attention projections, router, latent up/down, dense MLP) | must match the original model: **BF16** | high probability lower precision hurts accuracy |
| Shared expert | **BF16** (as the original) | quantizing the shared expert hurts accuracy |
| MoE weights | MXFP4 or NVFP4 | |
| MoE activations | a4w4 allowed (so a8w4 too) | AMD cannot do a4w8, hence a4w4 + NVFP4/MXFP4 weights |
| MoE dispatch | MX / NV style quant allowed | e.g. mxfp8 / mxfp4 quant of the routed input |
| MoE combine | **BF16** (as the original) | the weighted sum of expert outputs |
| AllReduce and equivalents (AG / RS) | **BF16** (as the original) | no fp8 / fp4 communication |
| KDA cache (recurrent state) | **FP32** | 16-bit KDA state clearly hurts reasoning accuracy |
| Conv cache | original dtype (probably) | |
| KV cache | FP8 allowed | FP4 KV only if the native checkpoint ships it (K3.1 does not -> not allowed on official upstream AgentX in week 1; unofficial runs must be labelled unofficial and show the full KVV suite) |
| Indexer cache | FP4 allowed | |
| EPLB | dynamic allowed, **no static EPLB** | |

Any of these may change after week 1 of official upstream AgentX if a change is shown to pass KVV and/or TB 4.0
(mainly KVV).

Practical reading: higher precision than the original is fine (f32 accumulation, f32 router logits, f32 merges);
lower precision than the original is allowed ONLY in the cells that say so (MoE weights / activations / dispatch,
KV cache FP8, indexer cache FP4).

## Status of current work against the rules (mi355-4, 2026-11-02)

```text
compliant
  serving config: --kv-cache-dtype fp8_e4m3 (KV FP8), MXFP4 experts, AITER_SITUV2_A8W4 (a8w4)   ok
  upstream SGLANG_ROCM_K3_A4W4_MOE (a4w4 routed experts)                                     ok (allowed cell)
  OPT 12-G4 mxfp8 split quant of the routed input (rejected for speed anyway)                ok (MoE dispatch)
  PR #7 KDA verify fp32 ReplaySSM ring; committed temporal state fp32                        ok
  PR #8 row-sharded tail, PR #9 / #10 sparse MLA (bf16 dots, fp8_ds_mla KV scales)           ok
  mono S1 / wide layer launch: bf16 weights + activations, f32 accumulate; collectives move
    bf16 and sum in f32 ("tagged" and "flag" formats both bf16); router logits f32; shared
    expert gate/up/down bf16                                                                 ok
needs a check
  ReplaySSM ring (--enable-linear-replayssm-spec): rawk / rawv stored bf16 (g, beta f32). They
    are the bf16 activations replayed into the fp32 state, not a 16-bit state -- believed ok;
    confirm with the rule owner if challenged.
  local branch k31/sparse-mla-fp8-q (fp8 query for sparse MLA): not covered by a cell ->
    treat as NOT allowed unless the rule owner says otherwise.
not allowed (do not pursue)
  fp8 / fp4 dense or shared-expert GEMMs (OPT rows 3 / 9 "fp8 experimental"); fp8 / fp4
  all-reduce / all-gather / reduce-scatter payloads; bf16 (or lower) KDA state; FP4 KV cache
  on K3.1; static EPLB.
```
