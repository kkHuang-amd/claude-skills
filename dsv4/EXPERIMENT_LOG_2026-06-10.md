# DeepSeek-V4-Pro serving perf — experiment log (2026-06-10)

Split from the master `EXPERIMENT_LOG.md` (chronological, by date). See that file for the index and `SKILL.md` for how-to.

---

## Exp: apples-to-apple MoE routing (active-expert gap) — 2026-06-10

Question: ATOM moe1 decode dump had **103 active experts** vs SGLang **216** (over
64 tokens x top_k=6 = 384 selections, n_routed=384). Is ATOM's topk systematically
concentrating?

Method: added env/sentinel-gated **router-stage dump** in both engines (one full
T==64 decode pass, all 61 layers): hidden, router_logits, topk_ids/weights,
e_score_correction_bias, config, layer_id.
- SGLang hook: `TopK.forward_cuda` STANDARD branch (NOT the BYPASSED branch — aiter
  resolves output_format=STANDARD) + `MoERunnerConfig.layer_id`.
- ATOM hook: after `FusedMoE.select_experts` in `model_ops/moe.py` apply(); layer_id
  parsed from `layer.prefix`.
- Gate is a SENTINEL FILE `/workspace/router_dump/.enable` (NOT env) because
  `sglang serve` does NOT propagate inline env / PYTHONPATH to worker procs.
  `router_dump.py` installed into site-packages so workers can import it.
- Fed the **same 64 fixed prompts** to both engines (`/workspace/router_probe.py`,
  temp 0; auto-detects model id from /v1/models; needs long max_tokens so a 64-wide
  decode batch forms).

Findings (`/workspace/analyze_router.py`, `/workspace/cross_hidden.py`):
- Both DSv4 routers are **identical config**: ungrouped `sqrtsoftplus`, top_k=6,
  renormalize=True, routed_scaling_factor=2.5, bias-for-selection.
- A reference `sqrt(softplus(logits))+bias` top-6 selector reproduces BOTH engines'
  actual topk with overlap ~1.000 -> selection algorithm is identical, no kernel bug.
- Cross-engine `router_logits` cosine ~0.997 at every layer.
- NON-HASH active experts essentially EQUAL given same input: ATOM avg 22.5 vs
  SGL 20.5, tracking layer-by-layer (19/19, 23/23, 16/15, ...). (Low ~20 because the
  64 probe prompts were homogeneous; both engines collapse equally.)
- Hash layers 0-2 differ (ATOM ~29 vs SGL ~52) because they route by current
  token-id and the first 64-wide decode step had already diverged generated tokens.

Conclusion: the topk ALGORITHM is identical; ATOM does not intrinsically pick fewer
experts from the same hidden. The original 103-vs-216 came from two INDEPENDENT
dumps on different random data. BUT (user's sharp point) the moe1 time scales ~linearly
with active experts via per-expert M-padding in the sorted grouped GEMM, so a
*reproducible* moe-time gap implies a *systematic* active-expert gap whose root cause,
if real, is UPSTREAM hidden-state homogeneity (ATOM decode hidden more collapsed),
NOT topk. Decisive test pending: same diverse prompts, active-expert DISTRIBUTION
across many real decode steps, + identical-routing replay to subtract the structural part.

### BUG found (user): SGLang intermediate_pad passed as 0 (should be 128)
- `fp8.py process_weights_after_loading_block_quant` (line ~1151, `_use_aiter and
  is_fp4_expert`) DOES pad intermediate 384->512 (like ATOM), but the pad amount is
  never stored on the layer.
- `fp8.py maybe_get_hip_aiter_quant_info` builds `AiterMoeQuantInfo` WITHOUT
  intermediate_pad -> defaults to 0 -> `fused_moe(..., intermediate_pad=0)` while
  ATOM passes 128. SGLang therefore computes the full padded 512 intermediate
  (wasteful, routing-independent structural slowdown). Fix: store
  `layer.intermediate_pad = padded_inter - inter_per_part` and pass it through.

Router-stage hook artifacts: `/workspace/router_dump/` (router_{sgl,atom}.pt,
112-118 MB), `/workspace/moe_dump/router_dump.py` (+ site-packages copy),
`/workspace/router_probe.py`, `/workspace/analyze_router.py`, `/workspace/cross_hidden.py`.

### BUG FIXED + decisive routing test — 2026-06-10 (cont.)
Fix applied in `sglang/python/sglang/srt/layers/quantization/fp8.py`:
- `process_weights_after_loading_block_quant`: store `layer.intermediate_pad =
  padded_inter - inter_per_part` (and `layer.hidden_pad = 0`).
- `maybe_get_hip_aiter_quant_info`: pass `intermediate_pad`/`hidden_pad` into
  `AiterMoeQuantInfo`.
Verified: SGLang now sends `intermediate_pad=128` to fused_moe (was 0) -> no longer
computes the padded 512 intermediate; matches ATOM. (router_dump meta confirms 128.)

Decisive active-expert distribution test (same 64 diverse prompts, 256 decode steps,
58 non-hash layers, 14,840 samples/engine; lightweight per-step `count_step`,
sentinel `/workspace/router_dump/.count`, rank-0 only):
  overall mean active experts:  SGL = 182.1   ATOM = 182.6   (diff +0.5)
  per-layer diffs small & bidirectional (-7.6 .. +6.9).
=> CONCLUSION: ATOM does NOT systematically pick fewer active experts. Given the same
input both engines route to ~the same expert count. The 103-vs-216 was purely a
data artifact (different benchmark data in the two original independent dumps). The
moe1 timing gap is therefore STRUCTURAL (the intermediate_pad=0 bug + scale dtype /
kernel), not routing. Hooks left in code but sentinel-gated (inert; no .enable/.count).
Artifacts: /workspace/router_dump/counts_{sgl_saved,atom}.json.

### Controlled moe1/moe2 timing matrix (resolves routing-vs-structural) — 2026-06-10
`/workspace/microbench_matrix.py` (same aiter.fused_moe; vary weights-engine/routing/pad,
common hidden, AITER_BF16_FP8_MOE_BOUND=0):
                                    moe1      moe2     active
  SGL-w  R216 pad=0  (orig/BUG)    127.1us   98.3us   216
  SGL-w  R216 pad=128 (FIXED)      104.4us   79.7us   216
  ATOM-w R216 pad=128             103.9us   79.5us   216   == SGL-w
  SGL-w  R103 pad=128              55.2us   73.4us   103   == ATOM-w
  ATOM-w R103 pad=128 (ATOM base)  55.1us   73.2us   103
Conclusions:
1. intermediate_pad fix: SGL moe1 127->104us (-18%), moe2 98->80us (-19%).
2. NO engine/kernel structural diff: at matched routing+pad, SGL-w == ATOM-w
   (104 vs 104; 55 vs 55). scale dtype e8m0 vs uint8 costs nothing.
3. routing dominates: R216 vs R103 (same w/pad) = 104 vs 55us (~1.9x).
Reconciliation: the original ATOM55-vs-SGL127 gap = ~23us real bug (pad, fixed) +
~49us DATA artifact (SGL data 216-active vs ATOM data 103-active). With identical data
both route ~182 -> both ~89us moe1 -> EQUAL after fix. No contradiction with the
earlier microbench (which leveraged the 216-vs-103 data difference).

### End-to-end impact of intermediate_pad fix (tp8, cuda graph ON) — 2026-06-10
Valid A/B (pad value baked at graph-capture; live sentinel toggle does NOT work under
cuda graph, must restart). random data, ratio 1.0, num-prompts 128/192, conc 64:
                  pre-fix(pad0)   post-fix(pad128)   delta
  8k/1k c64 tok/s    1346            1413            +5.0%
            TPOT ms   37.30           35.09          -5.9%
            ITL  ms   27.98           26.02          -7.0%
  1k/1k c64 tok/s    2139            2274            +6.3%
            TPOT ms   28.22           26.36          -6.6%
            ITL  ms   27.66           25.48          -7.9%
=> the intermediate_pad fix buys ~5-6% decode throughput / ~6-8% TPOT-ITL at conc 64.
(Earlier "live toggle" A/B showed ~0% because cuda graph had already captured pad=128;
must restart server with sentinel to bake pad=0.) Fix is clean (no toggle) in fp8.py;
sentinel removed. Artifacts: /workspace/bench_fix/*.jsonl.

### Full re-sweep WITH intermediate_pad fix (SGLang) — 2026-06-10
random ratio 1.0; bench_dsv4.py; np=4*conc(1k)/2*conc(8k). Results in
/workspace/bench_fix/{tp8,tp8dp8}/ (summarize_sgl_dsv4.py).
tp8       1k/1k out tok/s: c2=138 c4=269 c8=508 c16=894 c32=1470 c64=2294
tp8       8k/1k out tok/s: c4=249 c8=444 c16=716 c32=1061 c64=1416
tp8dp8    1k/1k out tok/s: c64=2066 c128=3499 c256=5668 c512=8585 c1024=8575(plateau,TTFT67s)
tp8dp8    8k/1k out tok/s: c64=1494 c128=2141 c256=2795 c512=3344
tp8dp8    8k/1k total tok/s: c64=13445 c128=19273 c256=25154 c512=30098
(c1024 1k plateau = known extreme-conc delayer over-throttle, not the fix.)

### PR #27858 applied + GSM8K accuracy parity — 2026-06-11
Applied upstream PR sgl-project/sglang#27858 (same MoE fix; my intermediate_pad
change was incorporated verbatim). Added: drop shuffle_*_a16w4 imports, add
self.gu_intv = envs.SGLANG_USE_AITER_MOE_GU_ITLV.get() (default True == old
hardcoded), use shuffle_scale/shuffle_weight with self.gu_intv. Behavior at
default unchanged.
GSM8K 5-shot, 200 q, TP8, same config, only intermediate_pad toggled:
  before (pad=0):   Accuracy 0.970  Invalid 0.000
  after  (pad=128): Accuracy 0.965  Invalid 0.000
=> within noise (1 question; different GEMM M-tiling), NO accuracy regression.
PR description written to claude-skills/dsv4/PR_27858_description.md.

### Full re-sweep ratio 0.8 (np=conc*8, warm=conc*2), tp8dp8 with delayer+ROCM700A=0
tp8 (ROCM700A=1, no delayer):
  1k/1k out tok/s: c2=136 c4=256 c8=470 c16=801 c32=1254 c64=1867
  8k/1k out tok/s: c4=242 c8=424 c16=692 c32=1025 c64=1360
tp8dp8 (--enable-prefill-delayer --prefill-delayer-max-delay-ms 5000, SGLANG_USE_ROCM700A=0):
  1k/1k out tok/s: c64=1944 c128=3261 c256=5306 c512=8073 c1024=8264(TTFT58s)
  8k/1k out tok/s: c64=1306 c128=1966 c256=2646 c512=3129
  8k/1k total tok/s: c64=11798 c128=17750 c256=23821 c512=28165
Results: /workspace/bench_r08/{tp8,tp8dp8}/.

### SGLang(+fix) vs ATOM-best, matched methodology (ratio 0.8, np=conc*8, warm=conc*2) — 2026-06-11
SGLang = bench_r08 (intermediate_pad fix; dp8 with delayer+ROCM700A=0).
ATOM = Experiment 1 best numbers (ATOM client, same ratio/np/warm). out tok/s, SGL/ATOM%:
tp8:
  1k/1k  c2 136/121 112% | c4 256/228 112% | c8 470/420 112% | c16 801/739 108% | c32 1254/1214 103% | c64 1867/1819 103%
  8k/1k  c4 242/215 112% | c8 424/379 112% | c16 692/634 109% | c32 1025/1001 102% | c64 1360/1397 97%
tp8+dp8:
  1k/1k  c64 1944/1856 105% | c128 3261/3181 103% | c256 5306/5543 96% | c512 8073/8381 96% | c1024 8264/11583 71%
  8k/1k  c64 1306/1455 90% | c128 1966/2212 89% | c256 2646/3085 86% | c512 3129/3783 83%
Takeaways: tp8 SGLang now matches/beats ATOM everywhere except 8k/1k c64 (97%); the
pad fix closed the old ~93% gap. dp8: SGLang wins low conc (c64-128) but ATOM scales
better high conc (dp-attention weakness, not MoE). Outliers 1k c1024 (TTFT 58.6s) &
8k c512 (TTFT 16.3s) are prefill-delayer admission at extreme conc, not decode.
Comparison script: /workspace/compare_atom.py

