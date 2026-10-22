Owner node: crsuse2-m2m-231

# Proxy MoE (Laguna) FFN-only monokernel prototype

Started 2026-10-22 (user: "直接先做 FFN-only mono 插件的原型"). Feasibility and measurements that motivate it:
`OPT_crsuse2-m2m-231.md` section "Mono (monokernel) feasibility" (clean bs2 trace: ~288 us/layer, MoE ~88,
non-MoE ~200; mega_moe_tp not usable at small M).

## CONTINUE HERE

**Status:** all five stages exist as standalone launches and are correct (S1 seam, S5, S2 router, S4 shared
bit-exact, S3 routed deterministic; optional shared-fused S3). S3 now 48 / 97 / 169 / 288 / 536 us at M
1/2/4/8/16 vs aiter fused_moe ~77 / 90 / 152 / 258 / 452 (UG / DOWN at 5.0-6.5 TB/s; "rest" = U + Q + RQ +
grid syncs + launch 13-27 us, to revisit inside the fused launch). S4 MFMA 16-22 us. Not started: S2
performance, fusion into one launch, plugin wiring, serving.
**Next:** S2 (TP-shard the router: 64 experts / rank + logit exchange), then fuse S1-S5 into one launch and
wire it into the plugin; serving TPOT at CONC 1-16.
**Repro:** `source /data/proxy_moe/mono_env.sh; T=/data/proxy_moe/ext/sgl-amd-mono/test/kernels/laguna;
torchrun --standalone --nproc_per_node=8 $T/check_seam.py; for c in router mxq shared routed; do
HIP_VISIBLE_DEVICES=0 python3 $T/check_$c.py; done`
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

| 2026-10-22 | crsuse2-m2m-231 | S1 seam check (`check_seam.py`, log `c231_mono_p0_1022/s1_seam_check.log`) | `laguna/gfx950/seam.py`: 256 CTAs x 512 thr, CTA = 64 hidden cols; push wave w → peer w; rank-ordered fp32 sum → bf16 + residual; sumsq per CTA → scratch mailbox; norm | vs exact torch reference (rank-order fp32 sum, bf16 round, fused add-RMSNorm semantics): x and residual 0 ulp at M 1/2/4/8/16; 200 graph replays bit-stable, guard words clean; 10.9 / 12.7 / 15.8 / 22.2 / 35.8 us per launch (slowest rank) |

| 2026-10-22 | crsuse2-m2m-231 | S1/S5 chained check (`check_seam.py`, log `c231_mono_p0_1022/s15_seam_check.log`) | seam phases: 0 = S1, 1 = S5 (MoE AR + residual + next input / final norm); own peer region and scratch per phase, tag slot 2*layer+phase; 3 layers S1→S5 chained in one graph | x and residual 0 ulp vs exact reference at M 1-16; stable over 200 replays; guard clean; 8.1 / 10.2 / 13.3 / 19.4 / 33.1 us per launch back to back |
| 2026-10-22 | crsuse2-m2m-231 | S2 router check (`check_router.py`, 1 GPU, log `c231_mono_p0_1022/s2_router_check.log`) | `laguna/gfx950/router.py`: CTA = 2 experts, wave = expert x K quarter, fp32 GEMV; logits via scratch mailbox; top-16 sigmoid + bias, lowest-id ties, renorm (sum + 1e-20) as `_router_triton_kernel` | expert ids identical (same order) for all 8 trials x M tokens at M 1/2/4/8/16; weights rel_l2 <= 1.4e-7; graph stable; 14.4 / 15.8 / 18.8 / 36.6 / 115.3 us (today's router at bs2 ~14 us/layer) → slow at M >= 8: every CTA re-reads all tokens' x from L2 |

| 2026-10-22 | crsuse2-m2m-231 | MXFP4 quantizers (`check_mxq.py`) | `laguna/gfx950/mxq.py`: triton rule (shared expert) and hip rule (routed; amax clamped to 1e-10) in-kernel, packed with v_cvt_scalef32_pk_fp4_f32 | fp4 bytes and scales 100% equal to aiter's triton / hip quant on [16, 16384] incl. all-zero groups; decode exact |
| 2026-10-22 | crsuse2-m2m-231 | S4 shared expert (`check_shared.py`, 1 GPU) | `laguna/gfx950/shared.py`: Q (triton rule) → GU (exact fp4 decode, fp32 FMA, waves in order, bf16) → silu*up → RQ → DOWN | output bit-identical to SGLang's quark W4A4 path (rel 0) at M 1-16; MID bf16 100% (one 1-ulp element at M=4, exp approx); graph stable; 21.4 / 25.3 / 32.5 / 61.4 / 263.9 us (today ~27 us/layer at bs2) |
| 2026-10-22 | crsuse2-m2m-231 | S3 routed experts (`check_routed.py`, 1 GPU, `ep_moe_check.build` weights = quark layout) | `laguna/gfx950/routed.py`: union in LDS, Q (hip rule), UG per union expert (dynamic loop), RQ, DOWN (route-weighted fp32 sum per CTA rows, no atomics) | rel_l2 vs aiter fused_moe 5.4-5.7e-3 at M 1-16, fused_moe vs itself 2.0-5.6e-3 (atomic stage 2); deterministic, graph stable; 178.7 / 297.4 / 653.4 / 1919 / 5641 us vs fused_moe ~77 / 90 / 152 / 258 / 452 us |

| 2026-10-22 | crsuse2-m2m-231 | S3 optimisation (`check_routed.py`, 1 GPU; logs `/data/proxy_moe/f2/mono/s3_*.log`) | (1) routed-only DOWN, 16 B loads, prefetch: 176 / 276 / 509 / 1495 / 4015 us; (2) whole (16,16) shuffle-group loads (the 4-row tasks used 1/8 of each 256 B line), parallel flag polls (RQ 25 → 2 us): 74 / 168 / 319 / 682 / 1612; (3) batched x loads per token: 66 / 137 / 289 / 629 / 1199; (4) all weight loads up front: worse (VGPR pressure), reverted; (6) quarter double-buffering: 68 / 153 / 297 / 564 / 1108; (7) scaled fp4 x fp4 MFMA 16x16x128 in UG and DOWN (A = 16 weight rows of a shuffle group, B = 16 token columns, zero where not routed): **48 / 91 / 170 / 327 / 751 us** | rel_l2 vs fused_moe 5.4-5.7e-3 throughout (fused_moe vs itself 1.9-5.6e-3); deterministic; graph stable. rocprofv3: VGPR 128 + AGPR 128, scratch 0 at M<=2, 1076 B at M=16 (scalar version) |

| 2026-10-22 | crsuse2-m2m-231 | S4 MFMA (`check_shared.py`, log `c231_mono_p0_1022/s4_shared_mfma_check.log`) | GU: CTA = (16-channel block, 1/8 of K), scaled fp4 x fp4 MFMA (A = 16 row-major quark rows, B = 16 token columns), slice partials in fp32 → GR sums slices in order, bf16, silu*up → RQ → DOWN MFMA | output still bit-identical to the quark path at M 1-16; 15.8 / 17.5 / 18.8 / 20.6 / 22.2 us (scalar: 21.4 ... 263.9) |

| 2026-10-22 | crsuse2-m2m-231 | S3 phase timing (`time_routed_phases.py`: run-time skip of UG / DOWN loops, one binary) | after cross-task UG prefetch + DOWN mid prefetch | full / UG / DOWN / rest us: M1 50.4 / 24.5 (5.65 TB/s) / 14.6 (4.76) / 13.3; M2 98.6 / 61.5 (4.22) / 34.2 (3.80) / 15.2; M4 191.3 / 123.8 / 63.4 / 16.8; M8 340.5 / 208.4 (4.86) / 123.1 (4.12) / 20.6; M16 732.0 / 441.4 (4.04) / 264.2 (3.38) / 27.6. DOWN prefetch depth 4 measured no better (kept 2) |
| 2026-10-22 | crsuse2-m2m-231 | shared expert fused into S3 (user request; `build_routed(shared=True)`, `check_fused.py`, log `c231_mono_p0_1022/s3_fused_shared_check.log`) | shared expert = union entry 512 (every token, weight 1.0): quark row-major weights read in place, x and mid quantized with the triton rule, gate/up rounded to bf16 before silu (its GEMM's output), as S4; output = routed + shared summed in fp32 (one rounding) | vs SGLang (fused_moe + quark shared, bf16 add): 5.4-5.7e-3, same as separate S3+S4 (5.4-5.7e-3); fused vs S3+S4 2.8-2.9e-3 (two bf16 roundings vs one). Time fused vs S3 + S4: 66.6 vs 63.3, 113.5 vs 114.8, 191.8 vs 203.8, 362.5 vs 345.4, 743.8 vs 752.9 us (M 1/2/4/8/16): no clear gain; first version forgot the bf16 rounding of the shared gate/up (rel 4.6-9.2e-3 vs S3+S4) |

| 2026-10-22 | crsuse2-m2m-231 | S3 large-M (`time_routed_phases.py`, `check_routed.py`) | (a) DOWN: 4 experts an iteration in straight-line code (loop-carried prefetches were drained at the loop edge, so prefetch depth did nothing): M1 DOWN 14.6 → 10.9 us, large M unchanged; (b) B-operand loads (x in UG, mid in DOWN) predicated by routing with out-of-range buffer offsets (zero, no traffic): at M16 ~15/16 of them were wasted L2 reads | full / UG / DOWN / rest: M1 47.8 / 21.4 (6.46 TB/s) / 12.3 / 13.6; M4 165.6 / 101.6 (5.19) / 56.0 (4.72) / 17.1; M8 291.3 / 179.4 (5.64) / 101.0 (5.02) / 19.4; M16 541.9 / 353.2 (5.04) / 163.7 (5.45) / 27.3. check: 48.2 / 97.4 / 168.8 / 287.8 / 536.4 us, rel_l2 vs fused_moe 5.4-5.7e-3 |

| 2026-10-22 | crsuse2-m2m-231 | S3 items 1 + 2 tried (user): wave-level UG tasks (one wave = (expert, 32-channel block) over all K, no cross-wave reduction / CTA barrier) with the MID requant folded in (no RQ stage, one grid sync fewer); kept as `/data/proxy_moe/f2/mono/routed_waveug.py`, NOT committed | `time_routed_phases.py` M4 / M8 / M16: full 284.5 / 348.8 / 638.7 us (was 165.6 / 291.3 / 541.9): UG 225.3 (2.34 TB/s) / 243.5 (4.05) / 439.9 (4.01) vs 101.6 / 179.4 / 353.2 before; rest 12.5 / 14.7 / 23.3 (was 17.1 / 19.4 / 27.3: the folded requant saves ~4-5 us as expected); DOWN slightly faster. Static unroll of the wave's K loop: worse (UG 345.8 / 374.5 / 704.1, register pressure). Reverted to 50acaf9 | per-wave load → wait → MFMA is serial and 2 waves / SIMD cannot hide it; at M4 only ~1000 tasks for 2048 waves (half idle, a busy wave streams 512 KB alone at ~2.3 GB/s) |

### Notes
- Whole-model recheck is NOT a usable check here: SGLang's own decode path differs from itself by 2.6-3.1e-2
  at layer-0 attention output when the same step is run twice in one process (attention input identical;
  cause not investigated, possibly the repeated fp8 KV write). The plugin check compares at the FFN input.- FlyDSL gotchas met: no `from __future__ import annotations` in a kernel module (`@fx.struct` needs real
  types); Python objects (Mailbox, Addr) used inside traced if/for must live in a dict (traced branches carry
  IR values only); do not reuse a dict's name as a `range_constexpr` loop variable.
- FlyDSL: a dynamic `for x in range(n)` carries any name assigned before the loop and reassigned inside as
  loop state (type must match: reusing a name across stages breaks it; prefix stage locals); a list of fp32
  accumulators is fine as loop state; `buffer_load` takes at most 4 dwords; no 1-element vector store
  (store the scalar); kernel cache keys must include every helper module's source (`kernel_sources(mxq)`),
  or a changed helper keeps the old binary.
- Stage-cutoff timing (`time_routed_stages.py`) stops being reliable once builds differ in register
  allocation (negative deltas); use it only for coarse splits. Scalar fp4 decode + FMA was VALU-bound
  (~7 us per UG task); the 16x16x128 scaled MFMA lane layout equals the (16,16) shuffle tile layout.
- Harness traps met: `epoch.add_(1)` bumps EVERY word (deadline and error words too) → the guard sees a
  failure and stops at the first wait (fault at 0x0); bump only word 0 (`epoch[:1].add_(1)` /
  `epoch[0] += 1`). A stream fetched before `torch.cuda.graph` capture is NOT the capture stream: the
  kernels then run outside the graph and the "graph time" is just the epoch bump (fetch the stream inside).
