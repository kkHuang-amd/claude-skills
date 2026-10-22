Owner node: crsuse2-m2m-231

# Proxy MoE (Laguna) FFN-only monokernel prototype

Started 2026-10-22 (user: "直接先做 FFN-only mono 插件的原型"). Feasibility and measurements that motivate it:
`OPT_crsuse2-m2m-231.md` section "Mono (monokernel) feasibility" (clean bs2 trace: ~288 us/layer, MoE ~88,
non-MoE ~200; mega_moe_tp not usable at small M).

## CONTINUE HERE

**Status:** P0 DONE (plugin commit in local branch, see Results). Kernel work not started.
**Next:** S1 seam stage (o_proj partial push + rank-ordered sum + residual + RMSNorm) as a FlyDSL launch with
a check_ffn-style 8-GPU harness, starting from DSV4.1 `stages/seam.py`.
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
### Kernel template and decisions (from the sgl-amd-mono read, clone @06b8f56; paths under `sgl_amd_mono/`)
- Reuse as-is: `common/` (plan BLOCKS=256 x THREADS=512, one CTA per CU; sync mailboxes with tags, PEER over
  XGMI; `peer.PeerBuffer` IPC (allocate in the same order on every rank); `guard` deadline + fail-stop,
  FLAG_WORDS=8 = TP8 max; `ops`, `mx.mfma_scaled` incl. FP4 x FP4).
- Template = DSV4.1's standalone FFN launch (`deepseek_v41/gfx950/layer.py` `build_mono_ffn`, `runner.py:392`,
  stages `moe.py` / `moe_shape.py` / `router.py` / `seam.py`; parameterized experts and top-k; standalone check
  `test/kernels/deepseek_v41/check_ffn.py`: synthetic fp4 weights under torchrun, golden partial, routing
  agreement, all widths in one graph, 500 bit-stable replays). MiniMax-M3 K4 (`minimax_m3/gfx950/post_attn.py`)
  for the TP push / rank-ordered sum, AG_RS mode and the wide (expert-grouped) MoE.
- Changes for Laguna (blockers first): hidden 16384 overflows LDS / VGPR in the M3 layout → K-split up/gate
  tasks, xn staged in L2 scratch not LDS; down row groups 512 > 256 CTAs → loop tasks; 512 experts top-16 →
  route key needs 9 expert bits, 8 experts per lane, 16 pick rounds, renorm butterfly (1,2,4,8), route
  stride >= 17, expert-union compaction over 8 waves; A4W4 → fmt (4,4) MFMA plus an in-kernel MXFP4 quantizer
  that matches aiter's rule (hip per_1x32 for routed = bit-exact with today, triton for the shared expert per
  `SGLANG_OPT_LAGUNA_PREFILL_SP_SHARED_QUANT` semantics); separate W4A4 shared expert pointers (quark layout);
  fp32 router 32 MiB/rank/layer → K-split GEMV, or TP-shard the router (64 experts / rank) and exchange logits
  through the peer mailbox (exact fp32 dot order differs from hipBLASLt → numerics TOLERANCE, check routing
  agreement).
- Plan: P0 spec + reference forward (SGLang ops), then kernel stages in this order, each with a check_ffn-style
  harness: S1 seam (push / sum / residual / norm), S5 MoE AR, S2 router, S4 shared, S3 routed (hardest).
- Env: `/data/proxy_moe/mono_env.sh` (new base + our clone first on PYTHONPATH; verified the clone's
  `sgl_amd_mono` is the one imported).

## Results (append-only; node + date)

| Date | Node | Run | What | Result |
|---|---|---|---|---|
| 2026-10-22 | crsuse2-m2m-231 | c231_mono_p0_1022 | P0: plugin `laguna_ffn_hip` (clone branch feat/laguna-ffn-mono 1st commit), reference FFN = SGLang ops | loads on 8 ranks, widths 1..16, 16/16 decode graphs routed to the plugin; c16 1024/256 TPOT 16.17 ms (normal path ~15.8: reference uses unfused AR + norm); check (FFN-input level): residual exact, FFN out 2.9e-3 = run-to-run noise of MoE atomics + AR |

### Notes
- Whole-model recheck is NOT a usable check here: SGLang's own decode path differs from itself by 2.6-3.1e-2
  at layer-0 attention output when the same step is run twice in one process (attention input identical;
  cause not investigated, possibly the repeated fp8 KV write). The plugin check compares at the FFN input.
