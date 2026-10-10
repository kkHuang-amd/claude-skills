# Mono (fused decode-layer) kernel for DSV4.1 in SGLang: evaluation

Owner node: crsuse2-m2m-255
Created: 2026-10-10. Inputs: vLLM PR #60397 (merged), ROCm/ATOM PR #2479 (merged), vLLM RFC #60904 (open).
Related: OPT_SWEEP_1008.md (our items 0-6), DECODE_MULTISTREAM_1008.md (m2m-259: launch-gap analysis),
FP4_INDEX_PLANE_PORT.md (the index-plane part of the same ATOM PR), TP4_GAP_1006.md.

## CONTINUE HERE

**Status:** evaluation written (overlap map + SGLang design proposal). Nothing implemented or measured yet.
**Next (P0, no GPU needed except the last bullet):**
1. Pin the four layout mismatches below (KV record, dense weight layout, shared-expert fusion, MoE weight shuffle) by reading
   SGLang's DSV4.1 HIP code; decide per mismatch: adapt the kernel, or change SGLang's layout under the mono flag.
2. Decide kernel sourcing (option C below needs the ATOM `atom/mono` + V4.1 kernels importable as a package).
3. Microbench vLLM #60397's FFN launch standalone on this node at SGLang TP4 shapes (it is the least coupled part).
**Decisions (user, 2026-10-10):** follow the recommendation: sourcing option C (SGLang-owned contract + external
kernels); mono runs with `--enforce-shared-experts-fusion` OFF (and item 5 preshuffle off) until a variant exists.
Work continues in a NEW session started with MONO_KERNEL_SESSION_PROMPT.md.
**Pass criteria for any prototype:** GSM8K TP2/TP4 >= 0.895 with real acceptance (EVAL_ONLY); AgentX c1-c8 P90 above the
opt1008 rows; HIP-graph replay stress (>=10k replays/width) bit-stable; no hang (bounded waits).

## 1. What the three sources are

| | vLLM #60397 | ATOM #2479 | vLLM RFC #60904 |
|---|---|---|---|
| Scope | DSV4.1-Flash only, MI355X | V4.1 mono + FP4 index plane + shared `atom/mono` framework (MiniMax-M3 mono came earlier, #2419/#2422) | standard for all models in vllm/models |
| Launches / layer | K1 + K2 on 30 layers; FFN-only launch on the 10 indexer/compressor layers (2,8,14,20,24,28,32,36 + 2 more) | K1 (attn_pre) + K2 (= K2a attn_post + K2b moe) on all layers; compressor runs between K1 and K2 | n/a (contract only) |
| In-kernel | mHC seams + Sinkhorn gate, norms, MXFP8 act quant, wqkv/wq_b GEMV, q/kv norm, RoPE, KV insert, slot gather, sparse attention (split+combine, sink, inv-RoPE), wo_a+wo_b, TP all-reduce, FFN seam, router+top-6, shared expert, routed MXFP4 experts over the expert union (no sort), MoE all-reduce | same, plus indexer query (K1) and indexer scoring + exact top-k (K2a, FP8 or FP4 plane) | whole-layer or FFN-only modes |
| Outside | indexer + compressor (layer gets FFN-only launch), Engram, layer-0 seam, embed/lm_head, DSpark draft layers | compressor, Engram, candidate block table, step-meta write, embed/lm_head, draft LM head/sampler (draft blocks DO run on K1/K2) | prefill / mixed steps, cross-layer launches |
| Technology | FlyDSL, persistent 256 CTA x 512 thr (1 CTA/CU), stage task tables, tagged mailboxes (epoch), scaled MFMA GEMVs | FlyDSL, same execution model; reproduces aiter/Triton rounding bit-exactly | device primitives to move into FlyDSL |
| TP comm | in-kernel peer push (aiter `UncachedIpcHeap`), exact fp32 rank-order sum; TP 2/4 | same; TP 2/4/8; does not use aiter custom AR | `peer.py` (torch symmetric mem / shared buffers) |
| Rows | decode, <= 48 rows (8 req x 6 = 1+5 DSpark), verify rows included | same 48 rows; one build per width, compiled before capture | `widths` declared per spec |
| Graphs | eager or FULL; PIECEWISE declined | FULL or eager; capture ladder 1..8 then x16 | FULL / FULL_DECODE_ONLY; PIECEWISE refused |
| Layout | KV `fp8_ds_mla` (584 B record); dense MXFP8 row-major; routed MXFP4 aiter (16,16) shuffle, A8W4; shared expert separate (MXFP8) | KV bf16; fp8/fp4 index cache; A8W4 interleaved gate/up; weights bound in place | `weight_source: views|convert`; extra quant only as declared option |
| Wiring | env `VLLM_ROCM_MONO_DECODE`; `MonoDecodeLayer.create` in layer `__init__`, try-mono-else-fallthrough in `forward` (+17 lines) | `ATOM_MONO_ENABLE` (default 1); `MonoDecodeModel` installer wraps the compiled model; `step_supported` routing; collective disable on `MonoUnsupported` | `MonoSpec` (refuse_config/refuse_model/pack/bind/eligible/build) + `SupportsMonoKernel` + `mono.forward_layer(...)`; `--monokernel`; default-deny features; fail-stop (no runtime fallback after a failed step); health word |
| Size | ~7.9k lines (kernels 6.6k, runner 437, wiring 376, tests 1.2k) | V4.1 mono ~8k + `atom/mono` ~1.7k | `common/mono/` ~650 lines |
| Reported | kernel 1.10-2.18x/layer; AgentX P90 vs InferenceX run 1.30-1.56x (TP2), 1.38-1.66x (TP4); GSM8K/GPQA/AIME/NIAH at parity | TP4 step 10.9 -> 3.7 ms (bs1), 11.3 -> 5.1-6.3 ms (bs2); GSM8K parity; recipe text partly stale (says refused) | GLM-5.2: 2459 -> 210 kernels/step, bs1 2.31x |

Why it matters here: DECODE_MULTISTREAM_1008.md measured ~1022 kernels/step at c2 with a 1.1-1.5 us dependent-launch gap
(~1.1-1.5 ms/step), and multi-stream overlap is a net loss on ROCm 7.2. Mono attacks exactly that (launch count + weight
reads) at the same small-batch regime where the B200 gap sits (TP4_GAP_1006.md: P90 gap 18-25% after opt1008).

## 2. Q1: overlap with our current optimizations

Mono covers decode steps of <= 48 rows only (AgentX: c1-c8 TP4 decode is almost always <= 8 requests). Everything
prefill-side is untouched; decode steps above 48 rows (c16+ bursts, c32/c64) still use today's path.

| Our item (OPT_SWEEP_1008 / recipe) | Path | With mono | Note |
|---|---|---|---|
| 0 index-Q RoPE + FP4 pack | prefill | **keep** | ATOM K1 has its own decode index query |
| 1 bf16 indexer logits | prefill | **keep** | decode scoring is inside ATOM K2a (vLLM: outside, unchanged) |
| 2 candidate block-max fast path | prefill (+ decode on indexer layers) | **keep**; superseded in decode only with ATOM-style K2a | vLLM style keeps our indexer in decode |
| 3 mHC AR + boundary stats (TP4, <= 12 rows) | decode | **fully replaced** (mono seams + in-kernel AR) | item 3 becomes dead code under mono; keep for mono-off |
| 4 small MoE sort (<= 64 pairs) | decode | **replaced** (mono routes over the expert union, no sort) | 64 pairs ~ 9 rows: entirely inside mono's range |
| 5 FlyDSL MXFP8 dense GEMM (preshuffle) | prefill + decode | **decode replaced; prefill keep** | **conflict**: preshuffle rewrites the dense weights, mono reads row-major MXFP8 views -> either mono reads the preshuffled layout or item 5 must keep an unshuffled copy (memory) / be off |
| 6 wo_a split-K + MXFP8 quant (<= 64 rows) | decode | **replaced for <= 48 rows**; 49-64 rows keep | |
| OPUS sparse prefill | prefill | **keep** | |
| QR INT8 quick-reduce | all TP all-reduces | **bypassed in mono layers** (in-kernel exact fp32 AR) | non-mono steps keep it; mono AR is more accurate than INT8 QR |
| `--enforce-shared-experts-fusion` | all | **conflict** | both mono kernels compute the shared expert separately (MXFP8, top-6 over 384); our recipe folds it as expert 385, top-7, FP4 -> mono needs fusion off, or a variant that consumes the fused layout |
| tokenizer workers, AOT build, PDI/chunk tuning | host / prefill | keep | |
| tuned TP4 MoE CSV (paused) | decode MoE | **moot for <= 48 rows** | |
| decode multi-stream overlap (m2m-259) | decode | **superseded** | |
| SGLang HIP glue: fused qk-norm-rope, q-rope-into-k-store, hc_boundary, wo_a_fp8, gfx95 native MXFP8 skinny GEMV, aiter fused_moe, custom AR | decode | replaced inside mono layers; all still needed for prefill / >48 rows / mono-off | |

Layout mismatches to settle before any port (P0):
1. **KV cache**: SGLang recipe = `fp8_e4m3`, page 256 (SWA + compressed pools, unified radix cache) vs vLLM `fp8_ds_mla`
   584 B record vs ATOM bf16. The K1 KV-insert and K2 slot-gather/attention stages must be rewritten for SGLang's layout
   (largest porting cost), or start with the FFN-only launch, which does not touch KV.
2. **Dense weights**: row-major MXFP8 expected; item 5 preshuffles.
3. **Shared expert**: separate MXFP8 expected; recipe fuses it into the FP4 MoE.
4. **Routed experts**: aiter (16,16)-shuffled MXFP4, A8W4 gate/up (vLLM) / interleaved (ATOM); check SGLang's aiter MoE layout
   (`SGLANG_USE_AITER=1`, `AITER_BF16_FP8_MOE_BOUND=0`) and the DSpark draft (128 experts, top-3) layout.
5. DSpark: SGLang verify = 6 tokens/request (block 5, `speculative_num_draft_tokens` 6) -> 8 req x 6 = 48 rows, same as both.

Expected payoff, for scale: opt1008 moved AgentX P90 vs CI by +4..+17%; vLLM's mono reports 1.30-1.66x P90 vs the same
kind of InferenceX run at c1-c8. Mono would supersede items 3/4/6 and the decode half of 5 in that regime.

## 3. Q2: how to design mono in SGLang

Constraints from SGLang:
- Routing input is `ForwardBatch` (`forward_mode.is_decode()` / `is_target_verify()`, batch size, seq lens) and must be
  rank-uniform; the decision has to be a pure function of the captured batch size under HIP graphs (cuda_graph_runner and
  the DSpark verify graph). Capture sizes <= max width must be mono widths (today `--cuda-graph-max-bs-decode 64`).
- SGLang's model code is not torch.compiled on this path, so per-layer routing in `forward` (vLLM style) is possible; a
  whole-model wrapper (ATOM `MonoDecodeModel`) is not needed.
- Existing HIP hooks: `deepseek_v4.py` -> `deepseek_common/amd/deepseek_v4_hip.py` (+ fused_mhc, gfx95_dense, wo_a_fp8).
- Coexistence: SGLang custom AR / QR buffers, TBO, DP attention, EP, PD disaggregation, hierarchical cache, LoRA -> refuse.

Sourcing options:
- **A. Vendor the kernels into sglang** (vLLM #60397 style): fastest to a prototype, ~8k lines model-specific, duplicates ATOM/vLLM.
- **B. Consume a shared package** (ATOM `atom/mono` + `atom/models/deepseek_v41/mono` as a pip dependency, or kernels
  moved into FlyDSL/aiter): no duplication; SGLang depends on ATOM's layouts (bf16 KV, A8W4 interleave) and release cadence.
- **C. (recommended) SGLang-owned contract + external kernels**: a small `sglang/srt/layers/mono/` runtime that follows
  RFC #60904's shape (so the same per-model spec ports across vLLM/SGLang), and per-model kernels pulled from the shared
  package (B) with an SGLang adapter only for what differs (KV binding, metadata).

Proposed SGLang contract (mirrors the RFC names where it can):
- `MonoSpec` per model (`sglang/srt/models/deepseek_common/amd/mono/spec.py`): `layers(cfg)`, `widths`, `supports`,
  `caches`, `weight_source`, `options`, `refuse_config(server_args)`, `refuse_model(model)`, `pack(model, ctx)`,
  `bind(ctx, kv_pools)`, `eligible(ctx, step)`, `build(ctx, packed, model, width) -> LayerKernel` with
  `launch(step, hidden, residual) -> (hidden, residual)` and `health()`.
- Common runtime (`sglang/srt/layers/mono/`): `attach(model, server_args)` after weight load; `begin_step(forward_batch)`
  once per step (pure decode/verify, width, eligibility, KV re-bind on `(data_ptr, numel)` change);
  `forward_layer(i, layer, positions, hidden, residual)` in the layer loop; peer memory; step epoch; health copy next
  to sampled ids; fail-stop.
- Two layer modes, as in the RFC: whole layer (K1+K2) and FFN-only (SGLang attention with output AR off, kernel does
  AR + seam + MoE + AR) -- the FFN-only mode is the low-risk first step because it needs no KV adapter.
- Flag: one server arg (`--enable-mono-decode`), default off; tuning in the spec `options`; refusals printed all at once
  at startup; no silent self-disable (RFC) -- but SGLang must decide between fail-stop (RFC) and ATOM's collective
  disable-on-build-failure; recommend: refuse at build time, fail-stop at run time.
- Warmup: build/compile every width before graph capture (FlyDSL JIT cost; ties into the AOT work in TP4_GAP).
- Tests: per-stage numerics vs SGLang ops (not vs vLLM's), graph replay stress, GSM8K with EVAL_ONLY (no simulated
  acceptance), NIAH long context, AgentX c1-c8.

Phased plan:
- **P0 (feasibility, ~2 days):** settle the 5 layout points above; run vLLM's FFN launch microbench at SGLang TP4 shapes.
- **P1 (prototype, FFN-only):** wire the FFN-only launch for all layers behind the flag, shared-expert fusion off,
  item 5 off; measure TPOT c1-c8 vs opt1008 and GSM8K.
- **P2 (whole layer):** K1+K2 with an SGLang KV adapter (fp8_e4m3, page 256, SWA + compressed pools); indexer layers
  stay FFN-only first (vLLM split), ATOM-style K2a indexer later.
- **P3 (generalize):** extract the common runtime, align with RFC #60904's final API, propose upstream.

Open questions: KV adapter cost vs switching the mono layers to ATOM's bf16 KV; whether item 5's preshuffle can be the
mono weight layout (one copy); fail-stop vs fallback policy in SGLang; who owns the kernels (ATOM package vs aiter vs
sglang); co-tenancy (a resident 256-CTA grid starves if another process shares the GPU -- our smoke/sweep never shares GPUs).

## Log (append-only)
- 2026-10-10 crsuse2-m2m-255: doc created from the three sources (sub-agent reads of #60397 @193922d6, ATOM #2479 head
  454d0e3 / merge 0873517 via /workspace/ATOM refs/remotes/pr/2479, RFC #60904 at 2026-10-10, no reviewer comments yet).
