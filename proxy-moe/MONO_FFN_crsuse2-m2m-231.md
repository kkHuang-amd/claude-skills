Owner node: crsuse2-m2m-231

# Proxy MoE (Laguna) FFN-only monokernel prototype

Started 2026-10-22 (user: "直接先做 FFN-only mono 插件的原型"). Feasibility and measurements that motivate it:
`OPT_crsuse2-m2m-231.md` section "Mono (monokernel) feasibility" (clean bs2 trace: ~288 us/layer, MoE ~88,
non-MoE ~200; mega_moe_tp not usable at small M).

## CONTINUE HERE

**Status (2026-10-22, second session):** the plugin serves a fused FFN launch per layer (`laguna/gfx950/layer.py`:
S1 seam → `xf` sync → S2 TP router (S4 Q/GU behind the logit exchange) → `rt` sync → S3 routed (RQ folded into
UG, one grid sync) + S4 DOWN → S5 seam), every decode width 1..16 (`SGLANG_AMD_MONO_LAGUNA_MAX_ROWS` caps it).
Kernel: 84.8 / 121.3 / 184.0 / 295.4 / 469.8 us/layer at M 1/2/4/8/16, bit-exact vs the standalone chain.
Serving TPOT vs base: 1024/256 -26 / -12 / -15 / -6 / -3 % at CONC 1/4/6/8/16; ISL 4096 and 8192 / OSL 1024
-25.5 / -11 / -5 / -3 % at CONC 1/4/8/16, TTFT unchanged. Eager check vs SGLang 2.8-3.3e-3 (run-to-run 2.9e-3).
This session (rows of 2026-10-22 "Next 1/2/3/5"): S3+S4down = 75-78 % of the launch at M8-16; UG time steps
with ceil(union/16) rounds (a part-full round costs ≈ a full one, ~20 us); balancing it by stream-K was built
twice (contiguous; round-robin + tail split), correct, but ≤ 5 us best case / losses elsewhere → not adopted.
**Next (pick in order, one variable each, record every run):**
1. TP4 mono (T11: TP4 EP1 serves 1090 vs TP8 934 tok/s): `INTER` is hard-coded 512 (per rank at TP8) →
   1024; seam / router already take the world size. Twice the tasks an expert → relatively smaller round tail.
2. Prefill SP (`SGLANG_OPT_LAGUNA_PREFILL_SP=1`) together with mono: orthogonal in theory (prefill vs decode
   graphs), unverified → check mode + one serving point.
3. S3 DOWN 5.0-5.6 TB/s at M4-16 (~20 us at M8 to the 6.9 ceiling): measure its in-flight sensitivity first
   (a no-prefetch probe as done for UG), then decide (BATCH and the pre-sync W2 prefetch had no effect).
4. UG round tail, cheaper than a split: smaller rounds (512 CTAs, 2 a CU, if registers / LDS allow
   co-residency) halve the quantum; measure occupancy first.
5. M1 latency: S1 / S5 one-shot seams ~8 us each; S2 + exchange ~11 us (rt sync); per-layer fixed costs.
6. Accuracy beyond the per-layer check needs real Laguna weights (user decision); then an upstream PR.
Do NOT retry (recorded as failed): UG tasks-per-iteration 2/4, UG prefetch depth 2/3 (spills), LDS-staged weight
stream (UG not in-flight-bound), seam 2 pairs a lane, DOWN BATCH 2/6/8, per-expert flags polled by DOWN,
DOWN first-batch W2 before the grid sync, UG stream-K (contiguous deal: -5 % at full rounds; tail-only split:
~10 us fixed latency a split round). Code of both stream-K variants: `/data/proxy_moe/f2/mono/routed_streamk_*.py`.
Logs: `/shared_nfs/kk/results/proxy_24l/c231_mono_p0_1022/` (kernels), `.../c231_mono_p1_1022/` (serving).
**Repro:** `source /data/proxy_moe/mono_env.sh; T=/data/proxy_moe/ext/sgl-amd-mono/test/kernels/laguna`;
kernels: `torchrun --standalone --nproc_per_node=8 $T/check_seam.py`, `... $T/check_router_tp.py`,
`... $T/check_layer.py` (fused vs standalone, timing), `... $T/time_layer_stages.py`; 1 GPU:
`HIP_VISIBLE_DEVICES=0 python3 $T/check_{mxq,shared,routed,fused}.py`, `$T/time_routed_phases.py 4,8,16`.
Serving: `/data/proxy_moe/f2/mono/p1_serve.sh` (STEPS=check|base|mono|mono_v2|..., CHECK_MODE=1|2; each step
starts and kills its own server; base TPOT above is from this script on 2026-10-22).
**Code:** plugin in `/data/proxy_moe/ext/sgl-amd-mono` branch `feat/laguna-ffn-mono` (clone of
github.com/sgl-project/sgl-amd-mono @06b8f56), pushed to github.com/kkHuang-amd/sgl-amd-mono
`feat/laguna-ffn-mono` @07f2583 (2026-10-22; remote `kk`; commits authored as kk via GIT_AUTHOR_* /
GIT_COMMITTER_* env, no AI attribution). No upstream PR yet. SGLang = `/data/proxy_moe/wt/sglang` (ed3d8b0089), AITER =
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
| 2026-10-22 | crsuse2-m2m-231 | S2 TP-sharded router (`check_router_tp.py`, torchrun TP8, log `c231_mono_p0_1022/s2_router_tp_check.log`) | `laguna/gfx950/router_tp.py`: rank r owns experts 64r..64r+63 (4 MiB fp32 instead of 32 MiB); CTA = (local expert, K quarter), 8 waves x 512 K; quarter partials via scratch mailbox, summed in order by the quarter-0 CTA and pushed to every rank (wave w -> rank w); every rank runs the same top-16 | expert ids identical (order too) for all 4 trials x M at M 1/2/4/8/16; weights rel_l2 <= 1.1e-7; all 8 ranks agree; graph stable; 13.6 / 14.1 / 15.0 / 16.7 / 21.9 us (was 14.4 / 15.8 / 18.8 / 36.6 / 115.3) → M16 5.3x; small M is launch + one cross-GPU hop, which the fused launch hides |
| 2026-10-22 | crsuse2-m2m-231 | Fused FFN launch, 1st try (`check_layer.py`, torchrun TP8, 3 chained layers, log `c231_mono_p0_1022/layer_check_first_try.log`) | `laguna/gfx950/layer.py`: S1 → `xf` grid sync → S2 TP router → `rt` sync → S3 with shared fused in → S5 in one launch, stage bodies shared with the standalone launches (`emit_*`) | NOT equal to the standalone chain and unstable: H hand-off stored with plain stores, read on other XCDs (L2s are not coherent inside a launch) → fixed by CM_DEV stores of every cross-CTA hand-off (H, RES_MID, ids, wts); then bit-exact, but 117 / 236 us at M1 / M4 vs 104.5 / 218 standalone |
| 2026-10-22 | crsuse2-m2m-231 | Fused FFN launch, S4 behind the logit exchange (`check_layer.py`, TP8, 3 chained layers, log `c231_mono_p0_1022/layer_check.log`; stage marks `layer_stage_marks.log`) | S3 routed-only + S4 split: S4 Q + GU between the router's logit push and its top-k wait, S4 GR / RQ before the `rt` wait, S4 DOWN after S3's DOWN; S5 adds MOE + SH in bf16 before the push (= SGLang's bf16 add) | x and residual bit-exact vs the standalone chain (S1, S2 TP, S3, S4, torch bf16 add, S5) for all 3 layers at M 1/2/4/8/16; graph stable; errors 0. 89.4 / 140.9 / 218.1 / 340.8 / 574.6 us/layer (standalone launches 99.1 / 152.2 / 219.8 / 339.0 / 582.4). Marks (M1): S1 19 (incl. rank skew), xf 3, router + S4 Q/GU 7, top-k + exchange 11, S3 + S4 DOWN 48, S5 9 → S3 is ~55-85 % of the launch. Today's SGLang FFN path ≈ 185 us/layer at bs2 |
| 2026-10-22 | crsuse2-m2m-231 | Plugin serves the fused launch: eager check (`SGLANG_AMD_MONO_CHECK=1`, CONC 1/4/16, logs `c231_mono_p1_1022/check*/server.log`) | `laguna/gfx950/runner.py` + `spec.py` wiring (peer / scratch / epoch, 16 widths precompiled, per-layer weight pointers from SGLang's loaded params) | 1st: next-input rel 5.6e-2..1.3e-1 vs SGLang (guard 0). Per-part check (`=2`) over all 24 layers: routing identical, routed 5.5e-3 (= fused_moe self-noise), S5 alone exact, but (a) shared expert 0 or 2-11e-2: SGLang's `SiluAndMul` is bf16(silu(g)) * u, not fp32 silu*up; (b) H 2e-3 off: SGLang's fused add-RMSNorm (aiter opus, hidden > 8192) squares the fp32 sum, normalizes columns < 8192 from the fp32 sum and the rest from the rounded residual, one rounding. Both matched → next-input rel 2.7e-3..8.7e-3 (rows 1/4; SGLang run-to-run 2.9e-3), guard 0 |
| 2026-10-22 | crsuse2-m2m-231 | Serving A/B, graphs on, TP8, ISL 1024 OSL 256, CONC 1/4/16 (8/16/64 prompts), one variable `--monokernel-backend amd` (`/data/proxy_moe/f2/mono/p1_serve.sh`, logs `c231_mono_p1_1022/{base,mono}`) | fused FFN launch at commit ec9b278 (local) vs the normal path | TPOT median base / mono: c1 5.36 / 4.09 ms (**-23.7 %**, ≈ 53 us/layer saved), c4 7.63 / 7.44 (-2.5 %), c16 15.78 / 18.08 (**+14.6 %**, worse). TTFT c16 385 / 506 ms (prefills wait behind slower decode steps). Output tok/s c1 175.5 / 222.7, c16 934.9 / 819.7. → the launch wins at small M only: S3 (routed) is 55-85 % of it and loses to aiter fused_moe from M ≈ 4 up |
| 2026-10-22 | crsuse2-m2m-231 | Serving sweep CONC 2/3/5/6/8, base vs mono (all widths), same setup (logs `c231_mono_p1_1022/{base,mono}_sweep`) | where the launch stops winning | TPOT median base / mono: c2 6.13 / 5.63 (-8.2 %), c3 7.67 / 6.57 (-14.3 %), c5 9.41 / 8.47 (-10.0 %), c6 9.81 / 9.20 (-6.2 %), c8 10.31 / 10.88 (+5.5 %) → crossover between 6 and 8 |
| 2026-10-22 | crsuse2-m2m-231 | Serving, mono with widths capped at 6 (`SGLANG_AMD_MONO_LAGUNA_MAX_ROWS=6`, now the default; log `c231_mono_p1_1022/mono_cap6`) vs base | wider decode steps on SGLang's normal path | TPOT median (base → cap6): c1 5.36 → 4.09 (-23.7 %), c4 7.63 → 7.44 (-2.5 %), c8 10.31 → 10.28 (=), c16 15.78 → 15.78 (=); TTFT c16 385 → 339. Plugin routes widths [1..6] only (log line) |
| 2026-10-22 | crsuse2-m2m-231 | S3 large M: nontemporal weight loads (`time_routed_phases.py 4,8,16`, 1 GPU; one variable each: UG / DOWN / both) | expert W13 / W2 `buffer_load` cache_modifier 2 (NT, as aiter's stage-1 `b_nt=2`) | full S3 M4/8/16: 166.6 / 285.3 / 541.8 → UG-only NT 157.0 / 272.7 / 536.5, DOWN-only 164.0 / 281.4 / 537.0, both **156.0 / 267.0 / 502.8 (-6..7 %)**; kept (commit 640282b). aiter fused_moe ~152 / 258 / 452. Torch streaming read on this GPU 5.26 TB/s, copy 4.64 |
| 2026-10-22 | crsuse2-m2m-231 | S3: UG tasks per loop iteration 1 / 2 / 4 (straight-line, env experiment) | fewer loop edges (edge drains the in-flight loads) | full M4/8/16: 155.2 / 268.5 / 502.6 vs 164.3 / 284.8 / 532.9 (2) vs 164.3 / 290.4 / 545.2 (4) → worse, reverted (copy `/data/proxy_moe/f2/mono/routed_tpi.py`) |
| 2026-10-22 | crsuse2-m2m-231 | S3: UG prefetch depth 1 / 2 / 3 quarters (register queue) | more loads in flight | 2: 299 / 501 / 856 us, 3: 362 / 607 / 1060 (UG 2.0-2.6 TB/s) → register spill (≈60 VGPR a quarter a lane); reverted (`routed_gdepth.py`). Deeper prefetch needs an LDS-staged stream |
| 2026-10-22 | crsuse2-m2m-231 | S3: UG task tail cost (skip bit 2) and DOWN BATCH 2 / 4 / 6 / 8 | where the rest goes | tails 4.8 / 12.4 / 15.3 us at M4/8/16 (~3 %); BATCH all within ±2 us (no effect, kept 4) |
| 2026-10-22 | crsuse2-m2m-231 | Fused launch after NT (`check_layer.py`, TP8, 3 layers) | | bit-exact vs standalone; 87.4 / 130.6 / 199.9 / 316.1 / 544.0 us/layer at M 1/2/4/8/16 (was 89 / 141 / 218 / 341 / 575) |
| 2026-10-22 | crsuse2-m2m-231 | Serving, NT build, widths 1..16 (`SGLANG_AMD_MONO_LAGUNA_MAX_ROWS=16`, log `c231_mono_p1_1022/mono_nt16`) vs base | did NT move the crossover? | TPOT median base / mono: c6 9.81 / 8.89 (-9.4 %, was -6.2 %), c8 10.31 / 10.48 (+1.6 %, was +5.5 %), c16 15.78 / 17.51 (+11 %) → crossover still 6..8, default cap stays 6 |
| 2026-10-22 | crsuse2-m2m-231 | Read ceiling (`/data/proxy_moe/tools/read_bw.py`, triton int32 sum, .cg) | what streaming can reach | 256 CTAs x 4 warps, 16 KB a CTA an iteration: 6.90 TB/s; 512 x 4: 6.91; finer grids 5.5-5.7. torch sum 5.26 |
| 2026-10-22 | crsuse2-m2m-231 | S3 UG: `rocdl.sched_barrier(0)` after each quarter's MFMAs (found via a run-time branch around the MFMAs that changed codegen) | one scheduling region a quarter | S3 M4/8/16 155 / 268 / 503 → 157 / 272 / **455** us (UG 6.43 TB/s at M16; = aiter 452); correct (check_routed PASS). Same barrier in DOWN: no effect (removed). Commit daf6947. Fused launch (`check_layer`): 87.2 / 124.2 / 190.5 / 303.7 / 556.0 us/layer at M 1/2/4/8/16 (M16 worse than the standalone chain 510: seams S1 40 + S5 33 us at M16 in the marks) |
| 2026-10-22 | crsuse2-m2m-231 | S3 UG probe: no prefetch at all (each quarter loaded right before use) | is in-flight depth the limiter? | UG M4/8/16 97.3 / 166.6 / 281.3 us vs 90.5 / 168 / 274 with prefetch → halving in-flight bytes costs 0-7 %: **not in-flight-bound**, so an LDS-staged stream (only adds in-flight bytes) is not expected to pay; not built |
| 2026-10-22 | crsuse2-m2m-231 | Seam: 2 tagged pairs a lane (16 B stores, `check_seam.py`, TP8) | fewer xGMI write transactions | 8.1 / 10.3 / 15.4 / 21.7 / 34.9 us vs 8.0 / 10.3 / 13.2 / 19.3 / 33.0 (1 pair) → no gain, reverted: the seam is xGMI-byte-bound (one-shot: every rank's full partial, tag words doubling it, to all 7 peers ≈ 3.5 MB out a rank at M16) |
| 2026-10-22 | crsuse2-m2m-231 | Seam two-shot (`check_seam.py` with `LAGUNA_SEAM_TWO_SHOT_MIN` 17 vs 1, TP8, 3 chained layers) | reduce-scatter: a CTA's columns to their owner rank (rank = CTA // 32) only; owner sums the ranks in order, rounds to bf16, all-gathers it; ~4x fewer bytes, one more hop | one-shot 8.0 / 10.2 / 13.1 / 19.2 / 32.9 us, two-shot 8.5 / 9.2 / 10.9 / 13.9 / 20.6 at M 1/2/4/8/16; both within 1 ulp of the reference (residual 0 ulp). Default two-shot from M2. Fused launch (`check_layer`): 87.0 / 123.5 / 187.2 / 295.7 / **468.3** us/layer (was 87.2 / 124.2 / 190.5 / 303.7 / 556.0), bit-exact vs standalone |
| 2026-10-22 | crsuse2-m2m-231 | S3 fold RQ into UG (`check_routed.py`, `LAGUNA_FOLD_RQ` 0 vs 1) | a CTA takes both 16-channel halves of a 32-channel MID group back to back (pairs round-robin) and requantizes the group after the second tail: no RQ stage, no UG→RQ grid sync | S3 M1/2/4/8/16 50.7 / 86.8 / 158.5 / 268.9 / 455.2 → 47.5 / 82.5 / 156.7 / 269.5 / 447.3 us; error vs fused_moe unchanged. Per-expert flags polled by DOWN not built (a poll round trip per batch of 4 experts would cost more than the sync). Fused launch 84.8 / 121.3 / 184.0 / 295.4 / 469.8 us/layer, bit-exact vs standalone |
| 2026-10-22 | crsuse2-m2m-231 | Serving eager check after two-shot seam + fold RQ (`c231_mono_p1_1022/check1`, CONC 1/4/16) | | next-input rel vs SGLang 2.8e-3..3.3e-3 (rows 1 / 4), guard 0 in all 400 checks |
| 2026-10-22 | crsuse2-m2m-231 | Serving, widths 1..16, current build (`c231_mono_p1_1022/mono_v2_16`) vs base (same day) | TPOT median | c1 5.36 → **3.95 (-26.3 %)**, c4 7.63 → 6.73 (-11.8 %), c6 9.81 → 8.30 (-15.4 %), c8 10.31 → 9.74 (-5.5 %), c16 15.78 → 15.29 (-3.1 %); TTFT unchanged (66 / 180 / 229 / 275 / 341 vs 68 / 179 / 227 / 275 / 385) → wins at every CONC: default cap back to 16 |
| 2026-10-22 | crsuse2-m2m-231 | Next 1: stage split at M 1/4/8/16 (`time_layer_stages.py`, TP8 prof build, log `c231_mono_p0_1022/stages_1022b.log`) and `time_routed_phases.py 4,8,16` (`phases_1022b.log`) | where M8-16 goes | layer 93.6 / 196.7 / 318.6 / 480.8 us: S3+S4down 48 / 140 / 239 / 377 (75-78 % at M8-16), S1 16 / 23 / 41 / 50 (includes the CPU-barrier launch skew of this harness, not graph-chained: use `check_seam.py` for the seam), S5 8 / 10 / 15 / 25. S3 alone M4/8/16: UG 92.5 / 172.5 / 274.5 us (5.70 / 5.72 / 6.43 TB/s), DOWN 57 / 95 / 156 (4.6 / 5.2 / 5.7 TB/s) |
| 2026-10-22 | crsuse2-m2m-231 | Next 3: DOWN issues its first BATCH's W2 loads before the UG→DOWN grid-sync wait (`LAGUNA_DN_PRE` 0 vs 1, `check_routed.py` PASS both; 2 runs each, logs `dnpre{0,1}_*`) | hide the first batch's latency | full S3 M4/8/16: 151.9 / 268.9 / 453.2 and 154.9 / 266.1 / 449.7 (off) vs 151.2 / 272.5 / 450.3 and 153.4 / 266.5 / 451.1 (on) → within the ±3 us run-to-run noise (expected ceiling one batch ≈ 3 us); reverted |
| 2026-10-22 | crsuse2-m2m-231 | UG time vs union size at M8 (`/data/proxy_moe/f2/mono/ug_union_sweep.py 8 ...`, log `ug_union_m8.log`; FOLD_RQ pairs: nu·16 pairs over 256 CTAs, rounds = ceil(nu/16)) | is the partial last round the M4/M8 UG gap? | UG us at union 48/49: 73.4/82.9, 64/65: 89.2/103.0, 96/97: 130.6/139.7, 112/113: 148.5/167.4, 120: 167.1, 128: 169.0 → a step per round: one expert past a round boundary costs +9..19 us, 113→128 only +1.6. A part-full round costs ≈ a full one (a CTA's pair-task is bound by its own stream rate, ~20 us, not by aggregate BW). Full rounds run 6.2-6.55 TB/s. Balanced UG at the real unions would be ≈ 1.32-1.36 us/expert: M4 (61) ~83 vs 90.5, M8 (114) ~151 vs 167, M16 (204) ~270 vs 274 |
| 2026-10-22 | crsuse2-m2m-231 | UG stream-K, contiguous (`LAGUNA_UG_STREAMK`, copy `/data/proxy_moe/f2/mono/routed_streamk_contig.py`): CTA b takes units (task, quarter) [bN/256, (b+1)N/256); a cut task finished by the lower CTA with the upper's fp32 partial (tagged mailbox), fold RQ across CTAs via a MID-ready flag | remove the part-full round | correct (`check_routed` PASS, rel_l2 unchanged). M8 sweep (`ug_union_m8_sk{0,1}.log`): step gone (65: 102.6 → 95.5, 113: 166.5 → 158.6) but whole rounds slower (112: 148.3 → 154.2, 128: 167.3 → 173.9) |
| 2026-10-22 | crsuse2-m2m-231 | Probe: old kernel, only the deal changed round-robin → contiguous blocks of pairs (exact unions 64/96/112/128, 2 runs, `ug_contig{0,1}{a,b}.log`) | is the stream-K loss the access order? | 112: 149.0 / 147.7 → 156.1 / 156.2, 128: 167.6 / 167.1 → 175.8 / 173.8 us → a contiguous deal costs ~5 % at full rounds (all CTAs streaming far-apart experts at once; mechanism not measured). A probe of masked-load codegen (`g_k >= 0` predicate) read 0 cost but is invalid: the predicate likely folds to true |
| 2026-10-22 | crsuse2-m2m-231 | UG hybrid: whole rounds round-robin (old loop, unchanged code), stream-K only over the tail tasks (a task split over up to 4 CTAs, finisher adds the partials in quarter order; copy `routed_streamk_tail.py`) | keep the full-round rate, balance the tail | correct (PASS). `check_routed` M1/2/4/8/16 48.1 / 92.9 / 165.1 / 272.5 / 455.8 vs 47.9 / 83.7 / 156.4 / 271.5 / 449.7 (off). M8 sweep (`ug_sktail2_{0,1}.log`) UG off/on: 113 165.9/160.4, 114 162.5/157.7, 116 163.5/159.2, 118 163.3/162.8, 120 166.0/161.6, 124 166.1/167.6, 128 168.2/167.0 → ≤ 5 us at small tails (run-to-run of the same setting here up to 10 us), losses when the tail is large (M2/M4: ~7 units a CTA). A split round has ~10 us fixed latency (two task epilogues, partial hand-off, MID-ready wait), so it cannot recover the ~20 us round. Not adopted (code reverted) |
| 2026-10-22 | crsuse2-m2m-231 | Next 5: serving at long inputs, OSL 1024, base vs mono (widths 1..16), `p1_serve.sh` STEPS="base_long mono_long" (new optional ISL field), logs `c231_mono_p1_1022/{base,mono}_long/` | does the win hold at ISL 4096 / 8192? | TPOT median base → mono: ISL 4096 c1 5.68 → 4.23 (-25.5 %), c4 8.04 → 7.12 (-11.4 %), c8 10.94 → 10.35 (-5.4 %), c16 16.23 → 15.70 (-3.3 %); ISL 8192 c1 5.69 → 4.24 (-25.5 %), c8 10.94 → 10.36 (-5.3 %), c16 16.24 → 15.71 (-3.3 %). TTFT equal (146 / 655-664 / 1130-1143 ms) except 4096 c4 443 → 470 (one run, 8 prompts). No guard / Traceback in the mono server log. Accuracy: the dummy ckpt makes gsm8k / end-to-end logprob meaningless (OPT doc); the per-layer check mode is the numeric check |

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
- `load_ptr64` (peer table) is wave-uniform: per-lane peer indices silently all write lane 0's peer and the
  readers time out (fault at 0x0). Map one wave per peer.
- Inside one launch, a hand-off read by other CTAs must be stored AND loaded with CM_DEV: the XCDs' L2s
  are not coherent (a plain store can sit in the writer's L2; the standalone launches were safe only
  because a launch boundary writes L2 back). Symptom: right values after the launch, wrong consumers.
- Helpers that hold traced `if` / `for range` must be `@flyc.jit` (only kernel / jit bodies are AST-
  rewritten: "cannot evaluate dynamic 'Boolean' as Python bool"). In a body moved into a helper, a loop
  or tuple target named like a parameter (`for c in ...`, `a, b = ...`) rebinds the dict inside a
  traced branch → "Cannot extract IR values from Mailbox"; name the dict parameters `ctx` / `args`.
- Match SGLang's real ops, not the textbook ones: `SiluAndMul` = bf16(silu(gate)) * up; the fused add-RMSNorm
  for hidden 16384 (aiter opus) uses the fp32 sum for the squares and for columns < 8192, the rounded
  residual for the rest. A 1-ulp bf16 change in H flips fp4 bins and shows as 1-10 % at the next input.
