Owner node: crsuse2-m2m-231

# Proxy MoE (Laguna) FFN-only monokernel prototype

Started 2026-10-22 (user: "直接先做 FFN-only mono 插件的原型"). Feasibility and measurements that motivate it:
`OPT_crsuse2-m2m-231.md` section "Mono (monokernel) feasibility" (clean bs2 trace: ~288 us/layer, MoE ~88,
non-MoE ~200; mega_moe_tp not usable at small M).

## CONTINUE HERE

**Status:** P0 started: reading the SGLang monokernel API (PR #44353, already in our SGLang branch) and the
sgl-amd-mono common runtime + MiniMax-M3 plugin (closest template). No kernel code yet.
**Next:** write the design section below from those reads, then P0 scaffolding.
**Code:** plugin in `/data/proxy_moe/ext/sgl-amd-mono` branch `feat/laguna-ffn-mono` (local only; clone of
github.com/sgl-project/sgl-amd-mono @06b8f56). SGLang = `/data/proxy_moe/wt/sglang` (ed3d8b0089), AITER =
`/data/proxy_moe/wt/aiter` (e9b0b5979). Ask the user before pushing anything.
**Rules:** one variable per experiment; every run recorded with node + date; kernels measured under CUDA graph;
per-stage numerics checked against SGLang's own ops (not against another framework).

## Target (what the FFN launch replaces, per layer, decode rows M <= 16, TP8 EP1)

From the o_proj output onward, today's SGLang path (clean bs2 trace, us/layer): AR + RMSNorm fused (~15),
fp32 router GEMM + post (~14), topk / sort / MX quant-sort (~16), shared expert W4A4 (2 GEMM + reduce + 2
quant, ~27), routed MoE moe1 + moe2 (~75), MoE AR + next input RMSNorm (~15), plus elementwise glue (~25).
≈ 185 us/layer at bs2 is covered by an FFN-only launch.

Model facts (proxy_24l config + laguna.py): hidden 16384; 24 layers, all sparse; 512 routed experts, top-16,
`moe_router_score_func` sigmoid, `e_score_correction_bias` (all 0), renormalize, routed scaling 1.0, no
softcap; router weight fp32 [512, 16384] (replicated); shared expert = LagunaMLP (merged gate_up + down,
silu), inter 4096 → 512 per rank, quark W4A4 (MXFP4 weights + MXFP4 activations, triton
`dynamic_mxfp4_quant` + `gemm_afp4wfp4`); routed experts MXFP4 W4A4 in aiter's shuffled layout, inter 512 per
rank, GateMode SEPARATED (gate rows then up rows); RMSNorm eps 1e-6; attention bf16 (not in the launch).
Activation precision must stay A4 (MXFP4) to match the proxy spec (MiniMax's mono uses A8W4 — not allowed).

## Phases

- **P0 scaffolding:** plugin package `sgl_amd_mono/laguna/` with a spec for LagunaForCausalLM, widths 1..16,
  forward = per-layer SGLang attention + a reference FFN composed of today's ops. Pass: outputs identical
  to the normal path; decode graphs captured on the plugin forward; fail-stop works.
- **P1 stages (standalone 8-GPU harness, CUDA graph, vs SGLang ops):** S1 seam (o_proj AR by peer push,
  rank-ordered sum, + residual + RMSNorm); S2 router GEMV + sigmoid + bias + top-16 + renorm; S3 routed MoE
  A4W4 over the step's expert union; S4 shared expert W4A4; S5 MoE AR. Then fuse S1-S5 into one persistent
  launch per layer.
- **P2 serving:** decode TPOT / ITL at CONC 1-16 (1024/1024, 70000/300) vs today; numerics (per-layer
  rel_l2 vs today's path at INT8 AR noise level).

## Design

### SGLang monokernel API (PR #44353; paths under `/data/proxy_moe/wt/sglang/python/sglang/srt/`)
- `monokernels/spec.py`: `MONOKERNEL_API_VERSION = 1`; `MonoSpec(name, devices={"gfx950": DeviceLimits(widths,
  tp_sizes, resident_blocks)}, features, requires=Requirements(dtypes, attention_backends, server_args,
  hf_config, env), numerics=BITWISE|TOLERANCE, model_disable_reasons, build, api_version, optional
  config_disable_reasons / load_tensor / tokens_per_request / unread_metadata)`; `register(backend,
  architecture, path)`; lazy `find_spec` by HF architecture + ISA. Backend "amd" is the installed plugin's
  (`/sgl-workspace/sgl-amd-mono` @e07665c); we add `LagunaForCausalLM` to its `SPECS` in our clone.
- `runtime.py`: `_install_forward` replaces the WHOLE `LagunaForCausalLM.forward` for served shape keys
  (DECODE / TARGET_VERIFY, rows = bs x tokens_per_request), calls `built.forward(Plan(key), input_ids,
  positions, forward_batch, input_embeds=None, pp_proxy_tensors=None, ...)` and must return what the original
  returns (logits processor output). Obligations: KV + aux writes for every row incl. padded rows; graph
  capturable; warmup calls it eagerly too. Build runs after weight load, before KV sizing; gets (model,
  hf_config, BuildContext(isa, widths, deadline_s <= 60, copies)); KV pools / attn backend fetched lazily in
  forward. `handle_monokernel` aligns decode capture sizes to the spec widths (1..16 with
  `--cuda-graph-max-bs-decode 16`). Ranks must decide identically; cross-rank waits need deadlines.
- Plugin forward may call SGLang modules (reference does; our `_sp_forward` is a template). FFN-only needs an
  unrolled attention (qkv → qk-norm → rope → `self.attn` writes KV → stop before `o_proj`), so o_proj can be
  fused with the AR into the launch. Refuse `layers_to_capture`, speculative, DP-attn, PP, EP>1.
- Env: `/data/proxy_moe/mono_env.sh` (new base + our clone first on PYTHONPATH; verified the clone's
  `sgl_amd_mono` is the one imported).

## Results (append-only; node + date)

| Date | Node | Run | What | Result |
|---|---|---|---|---|
