# vLLM vs SGLang — DeepSeek-V4.1-Flash on MI355X

## CONTINUE HERE
**Status:** DSpark-sim (AL 3.51) and DSpark-off compared (tables below); GSM8K equal (0.908 vs 0.901).
SGLang wins decode at all loads (TPOT -11..-16%), loses TTFT at c32 (x1.4-2.3) -> near-parity e2e at c32.
vLLM crash with prefix caching ON (OPUS + prefix hit) worked around with --no-enable-prefix-caching.
**Next:** (1) DONE: vLLM DSpark real GSM8K 0.904 (parity). (2) profile SGLang c32 TTFT (queue vs prefill compute) and test mixed-chunk / chunked-prefill settings;
(3) try SGLANG_ROCM_USE_MULTI_STREAM=1 (shared-expert overlap, vLLM default); (4) file vLLM OPUS+prefix-cache bug.

## Comparison protocol (decided 2026-09-24)
- DSpark compared with **simulated acceptance AL=3.51** (InferenceX `golden_al_distribution/dsv41flash_dspark.yaml`,
  thinking_on, 5 draft tokens -- same value AgentX uses). Both engines count the bonus/verify token in AL.
  SGLang: `SIM_AL=3.51` in launch_server.sh -> `SGLANG_SIMULATE_ACC_LEN=3.51` (+ `SGLANG_RAGGED_VERIFY_MODE=static`,
  required verify-all). vLLM: `"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51`.
  SGLang `--speculative-dspark-block-size 5` == vLLM `num_speculative_tokens 5` (gamma).
- Same client for both: `sglang.bench_serving --backend sglang-oai` vs `--backend vllm` (identical
  /v1/completions code path); GSM8K via vLLM's `gsm8k_eval.py` for both (`scripts/run_gsm8k_openai.sh`).
- Accuracy only with real acceptance. Throughput: PR-style bs1 (median of 6, 1000/TPOT) + ISL4096/OSL1024 sweep.
- vLLM config = InferenceX `dsv41flash_fp4_mi355x_vllm_mtp.sh` + `--enable-expert-parallel` + fp8 KV/block 256
  to match SGLang (both are knobs in vllm_launch.sh; InferenceX itself runs neither).

## Setup facts
- vLLM image: `vllm/vllm-openai-rocm:nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984` (ROCm 10.0).
- Source = official `vllm-project/vllm` commit `3df4ae153e` (2026-09-21, main). Checked out (depth 1) at
  `/sgl-workspace/vllm-src`. NOTE: the image's own aiter/triton/torch versions can differ from ours
  (ROCm 10 vs our ROCm 7.2) -- record them when the image is run (`pip list | grep -iE 'aiter|torch|triton'`).
- DSV4.1 code in vLLM: `vllm/models/deepseek_v41/{amd,common,nvidia}/` (33 files, 14.6k lines);
  AMD-specific: `amd/model.py` (1107), `amd/rocm.py` (961), `amd/dspark.py` (548).
- vLLM GSM8K eval configs: `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-AITER-TEP4.yaml`.
- vLLM image stack (pip list in image): vllm 0.3.1.dev190+g3df4ae153.rocm100, torch 2.12.0+rocm10.0.0,
  triton 3.8.0 (rocm10), amd-aiter 0.1.22.post1, flydsl 0.3.2, ROCm 10.0.
  vs SGLang env: torch 2.9.1+rocm7.2.0, triton 3.7.0, aiter dev acf8fdf9 (+#5561/#5802), flydsl 0.3.2, ROCm 7.2.
  => ROCm/torch/aiter all differ; a perf gap may be stack, not engine.
- Container launch: vLLM image ENTRYPOINT is `vllm serve` -> needs `--entrypoint /bin/bash`.
- This container has no docker; the vLLM image must be run elsewhere (or on the host).

## Performance comparison

### DSpark ON, simulated AL 3.51 (vLLM measured 3.52), 4xMI355X TP4+EP4, fp8 KV, 2026-09-24
Client: sglang.bench_serving `sglang-oai` vs `vllm` (same /v1/completions code). ISL 4096 / OSL 1024.
vLLM: image 3df4ae153e, default block size (256 fails, see gotcha), EP on, fp8 KV. SGLang: RUNBOOK state.

| metric | conc | SGLang | vLLM | SGLang vs vLLM |
|---|---|---|---|---|
| PR-style decode tok/s (median 6, 1000/TPOT) | 1 | 378.1 | 341.3 | +11% |
| random-ids out tok/s (incl. TTFT) | 1 / 8 / 32 | 358 / 1623 / 3160 | 306 / 1290 / 3250 | +17% / +26% / -3% |
| ShareGPT out tok/s (incl. TTFT) | 1 / 8 / 32 | 362 / 1564 / 3060 | 319 / 1545 / 3110 | +13% / +1% / -2% |
| random-ids mean TPOT ms | 1 / 8 / 32 | 2.63 / 4.06 / 7.45 | 2.97 / 4.54 / 8.67 | SGLang lower at all conc |
| random-ids mean TTFT ms | 1 / 8 / 32 | 166 / 888 / 2736 | 308 / 1696 / 1175 | SGLang 2.3x WORSE at c32 |

Reading: SGLang has faster decode steps at every concurrency (TPOT -11..-14%) and faster TTFT at low load, but at
c32 its TTFT is 2.3x vLLM's, which erases the TPOT win in end-to-end throughput. Suspect prefill scheduling /
batching under load (vLLM: --max-num-batched-tokens 16384 mixes prefill+decode; SGLang default chunked prefill
and prefill/decode alternation) -- untested hypothesis, top item to investigate. c8 random-ids vLLM (1290 vs
1545 on ShareGPT) looks noisy (32 prompts). Stack differs (ROCm10/torch2.12/aiter 0.1.22.post1 vs ROCm7.2/
torch2.9.1/aiter dev) so gaps are not purely engine.
Raw: /shared_nfs/kk/dsv41/{perf,prstyle}_{sgl_dspark_sim3.51_oai,vllm_dspark_sim3.51}/

### DSpark ON, REAL acceptance -- accuracy
vLLM (block rejection, 5 draft tokens, prefix cache off): GSM8K 0.904 (vLLM gsm8k_eval.py), mean AL on GSM8K 3.54,
0 memory faults. SGLang DSpark real: 0.902-0.911 across 5 runs (SGLang few_shot_gsm8k client). => accuracy parity.

### DSpark OFF
- vLLM try1 (prefix caching ON = vLLM default): GSM8K crashed the engine -- `HSA_STATUS_ERROR_MEMORY_FAULT` on all
  4 GPUs right after first `Using AITER OPUS for large sparse MLA prefill` (`aiter.ops.pa_sparse_prefill_opus`,
  gate: >=1024 queries, `VLLM_ROCM_USE_AITER_MLA` default on). Load: 75 running / 53 waiting, prefix hit 82%.
  Hypothesis: OPUS + prefix-cache hit (queries < KV len). Retry with --no-enable-prefix-caching (try2): full GSM8K
  + perf ran with 0 faults -> consistent with the hypothesis (not proven: OPUS still ran on non-prefix prefills?
  check `Using AITER OPUS` in server_nodspark2.log). Candidate upstream vLLM/aiter bug report.
  Log: /shared_nfs/kk/dsv41/vllm/server_nodspark.log (contains a NUL-filled hole from the old server's shutdown).
SGLang (oai client): PR-style c1 147.6; random-ids 144 / 897 / 2198; ShareGPT 145 / 883 / 2081. vLLM try2 (prefix caching off, EP, fp8 KV), same client:

| metric | conc | SGLang | vLLM | SGLang vs vLLM |
|---|---|---|---|---|
| GSM8K 5-shot 1319 (vLLM gsm8k_eval.py, both) | - | 0.908 | 0.901 | +0.7pt (within ~1pt noise) |
| PR-style decode tok/s | 1 | 147.6 | 123.8 | +19% |
| random-ids out tok/s | 1 / 8 / 32 | 144 / 897 / 2198 | 120 / 762 / 2048 | +20% / +18% / +7% |
| random-ids TPOT ms | 1 / 8 / 32 | 6.77 / 8.10 / 11.86 | 8.08 / 9.64 / 13.74 | SGLang -16% / -16% / -14% |
| ShareGPT out tok/s | 1 / 8 / 32 | 145 / 883 / 2081 | 121 / 749 / 1952 | +20% / +18% / +7% |
| random-ids TTFT ms | 1 / 8 / 32 | 164 / 841 / 2768 | 256 / 888 / 1935 | SGLang 1.43x WORSE at c32 |

Same pattern as DSpark-on: SGLang decode faster at every load; at c32 its TTFT falls behind vLLM (DSpark on: 2.3x,
off: 1.43x). => #1 port candidate: prefill scheduling under load (vLLM mixed prefill+decode batches,
--max-num-batched-tokens 16384). Profile SGLang c32 TTFT breakdown (queueing vs prefill compute) before porting.

## Notes on the inventory (main-session review)
- Candidate "MoE activation dtype BOUND=256" is ALREADY A/B'd on SGLang (NOTES.md step 3): no throughput
  change, GSM8K not better. Low priority.

## Optimization inventory (vLLM AMD path -> SGLang branch status)
Traced 2026-09-24 by reading code only (nothing measured). vLLM `3df4ae153e` against SGLang branch `dsv41-amd-main` `e2e824dc58`.
Paths: vLLM paths are relative to `vllm/` (`csrc/` and `CMakeLists.txt` are at the repo root). `R` = `v1/attention/ops/rocm_aiter_mla_sparse.py`. SGLang paths are relative to `python/sglang/`.
Model facts that decide applicability (read from `/shared_nfs/models/DeepSeek-V4.1-Flash/config.json`): hidden 5120, 384 routed experts (top-6) + 1 shared expert, compress_ratios {0,1,2} (no ratio 4), index_topk 512, linear quant **[32,32] MXFP8 (ue8m0)**, MXFP4 experts, engram layers [1,14], dspark_n_routed_experts 128.

**Main finding:** most of vLLM's hand-written AMD fast paths are either built for V4.0 (DSV4-Pro/Flash with 128-block FP8 linears) or already exist in SGLang in a more fused form. vLLM's V4.1 AMD path is *less* fused than the SGLang branch in the areas that matter for decode (mHC+all-reduce, wo_a/wo_b, indexer, attention, router).

| # | optimization | vLLM location | kernel/lib | phase | SGLang status | effort | impact (bs1-32 decode) |
|---|---|---|---|---|---|---|---|
| 1 | Delayed mHC: previous sublayer's post is folded into the next pre (one kernel); unfused above an AITER crossover (gfx950: >=1024 tokens) | `model_executor/layers/mhc.py:380-445`, `_aiter_ops.py:3955-3967`, `models/deepseek_v41/amd/model.py:260-376` | aiter `mhc_pre_delayed` / `mhc_fused_post_pre` (TileLang fallback) | all / mHC | HAS: `srt/models/deepseek_common/amd/deepseek_v4_fused_mhc.py:237-381` (aiter `mhc_fused_post_pre`, Triton fallback) | - | - |
| 2 | Standalone aiter mHC pre/post (engram seam, model tail) | `mhc.py:520-560` | aiter `mhc_post` / `mhc_pre` | all / mHC+Engram | HAS: aiter mHC path (`test_mhc_aiter_hip.py`), `kernels/ops/layernorm/mhc_boundary_hip.py` | - | - |
| 3 | All-reduce + mHC fusion | `csrc/libtorch_stable/all_reduce_mhc.cu` (**CUDA-only**, `CMakeLists.txt:463`) | - | decode / AR | SGLang AHEAD: `kernels/ops/communication/all_reduce_mhc_hip.py`, `deepseek_v4_fused_mhc.py:510-543` | - | - |
| 4 | Triton hc-collapse for the DSpark/MTP head (V4.1 has no learned hc_head) | `models/deepseek_v41/amd/model.py:255-258`, `amd/dspark.py:220` | Triton `hc_collapse_triton` | DSpark draft | HAS: `kernels/ops/layernorm/mhc_head.py` | - | - |
| 5 | Dense MXFP8 linears (wq_a/wkv, wq_b, wo_b, shared expert): native `tl.dot_scaled` with a gfx950 graph-tuned tile table; separate per-token MXFP8 activation quant | `models/deepseek_v41/quant_config.py:186-207` -> `model_executor/kernels/linear/mxfp8/rocm_native.py:120-269` | Triton `tl.dot_scaled` | all / GEMM | SGLang AHEAD: `kernels/ops/quantization/mxfp8_native_amd_gfx95.py` + small-M GEMV (`mxfp8_gemv_gfx95_configs.json`) + norm->fp8-grid fusion (`rmsnorm_fake_quant_amd_gfx95.py`, `srt/models/deepseek_common/amd/deepseek_v4_gfx95_dense.py:62-129`) | - | - (per-GEMM speed vs vLLM not measured) |
| 6 | aiter B-preshuffle blockscale GEMM for fused_wqa_wkv, wo_b and shared-expert gate_up; aiter `fused_qk_rmsnorm_group_quant` feeding both wq_b GEMMs | `models/deepseek_v41/amd/rocm.py:532-663`, `models/deepseek_v4/amd/model.py:129-170` | aiter `gemm_a8w8_blockscale_bpreshuffle`, `fused_qk_rmsnorm_group_quant` | all / GEMM | **N/A for V4.1 (not used by V4.1):** needs `weight_scale_inv` + a 128-block FP8 kernel. V4.1's [32,32] MXFP8 registers `weight_scale` (`amd/model.py:867-878`), and `prepare_*_preshuffle` is only called from the V4.0 model (`deepseek_v4/amd/model.py:1436-1438`). SGLang has the 128-block equivalents anyway (`srt/layers/quantization/fp8_utils.py:73`, `deepseek_v4.py:564-583`) | - | none for V4.1 |
| 7 | WO_A: dequantize once to a cached BF16 weight + Triton fused inverse GPT-J RoPE + `torch.einsum` (comment says it mirrors SGLang/ATOM) | `R:1570-1690`, `amd/rocm.py:665-682` | Triton + hipBLASLt bmm | all / attn-out | SGLang AHEAD: `SGLANG_DSV41_FUSED_WO_A` (default on) at `srt/models/deepseek_v4.py:1219-1230`, aiter batched GEMM `deepseek_v4.py:380-560`, fp8-grid wo_a->wo_b `deepseek_v4_gfx95_dense.py:195-222`, inverse RoPE folded into the aiter decode combine (`srt/layers/attention/hip_flash_mla.py:176-190`) | - | - |
| 8 | One kernel for Q-RoPE/pad + KV RoPE + UE8M0 FP8 quant + paged SWA insert | `models/deepseek_v41/attention.py:925-1043`, `csrc/libtorch_stable/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | custom HIP (hipified .cu) | all / attn-prep | HAS: HIP KV-store/RoPE fusion `srt/models/deepseek_common/amd/deepseek_v4_hip.py:35-72` (`test_v41_kv_store.py`) | - | - |
| 9 | Multi-stream input projections: fused_wqa_wkv on the default stream; compressor kv_score (fp32 mm) and indexer weights_proj on aux streams, up to 1024 tokens | `attention.py:846-898` | torch streams; `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024` | decode+prefill / attn-prep | HAS: `deepseek_v4.py:2193-2230` + `_forward_prepare_low_ratio_multi_stream` `:1530` (ratio 1/2, decode + verify, no token cap on gfx95). The generic path's 64-token cap never applies to V4.1 on HIP, because ratio 0/1/2 are excluded there | - | - |
| 10 | Overlap Q-proj/KV-insert with the compressor, then overlap indexer prep with compressor cache insert | `attention.py:746-828` | torch streams | all / compressor+indexer | HAS (same side-stream structure, `deepseek_v4.py:1466-1560`) | - | - |
| 11 | Fused compressor: state save + group pool + RMSNorm -> BF16 latent in one kernel; separate RoPE+quant insert that can be scheduled independently | `models/deepseek_v41/compressor.py:246-323`, `common/ops/fused_compress_quant_cache.py:22,224` | Triton | all / compressor | HAS: `srt/layers/attention/dsv4/dsv41_sparse.py:117-127` (fused compress; `test_dsv41_fused_compress.py`) | - | - |
| 12 | Indexer Q: fused RoPE + quant, weights emitted in the scoring dtype; **FP8 indexer K cache on ROCm** (MXFP4 indexer is Blackwell-only) | `models/deepseek_v4/common/ops/fused_indexer_q.py:547`, `v1/attention/backends/mla/indexer.py:58-73` | Triton | all / indexer | SGLang AHEAD: MXFP4 indexer on HIP (`kernels/ops/attention/dsv4/fp4_indexer_hip.py`, `fp4_rope_hip.py`), which halves the indexer K bytes | - | - |
| 13 | Indexer decode logits for ratio 1/2: aiter preshuffled paged MQA logits (ChunkK=256) | `R:788-894` | aiter `deepgemm_fp8_paged_mqa_logits` | decode / indexer | Different design: SGLang uses its own FP4 HIP indexer (see #12) | - | - |
| 14 | Indexer prefill logits: aiter `fp8_mqa_logits`, chunked by `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` (FlyDSL variant is gfx942-only) | `R:964-1022`, `R:1236-1240` | aiter Triton | prefill / indexer | HAS (FP4 HIP prefill indexer + `SGLANG_OPT_DSV4_NONPAGED_INDEXER`) | - | - |
| 15 | Top-k: aiter `top_k_per_row_{prefill,decode}` on gfx950 behind measured gates; for topk=512 and <=384 rows it falls back to native `_C.top_k_per_row_decode` | `R:74-110`, `R:1444-1478` | aiter topk / vLLM `csrc/libtorch_stable/topk.cu` | all / indexer | HAS: sgl-kernel `deepseek_v4_topk_transform_512` (`srt/layers/attention/dsv4/low_ratio_backend_hip.py:52`) | - | - |
| 16 | Two-level candidate-block filtering (source layer publishes candidate blocks; strided mask kernel with a fixed 128-program grid tuned on gfx950) | `R:1025-1127`, `models/deepseek_v41/amd/model.py:413-427` | Triton | all / indexer | HAS: `srt/layers/attention/dsv4/candidate_indexer.py` (+ graph variants). Whether SGLang's mask kernel uses the same fixed-grid trick is **unverified** | S | low |
| 17 | Ragged top-k indices built once per index-source group and reused by the layers below | `amd/rocm.py:760-794` | Triton pack kernel | decode / attention | HAS: per-ratio cache `srt/layers/attention/deepseek_v4_backend_hip_radix.py:382-389` | - | - |
| 18 | Sparse decode attention: gfx950 Triton split-K partial+combine, direct fp8_ds_mla loads, adaptive/one-wave split heuristic, writes bf16 output in place | `R:3257-3297`, `R:3380-3500`, `R:3719-3791` | Triton | decode+verify / attention | HAS (different kernel): aiter gluon `pa_decode_sparse` + swap-AB gluon for small batches, adaptive splits (`hip_flash_mla.py:20-135`, `srt/layers/attention/deepseek_v4_backend_hip_radix.py:2873-2900`, `kernels/ops/attention/dsv4/swapab_gluon_hip.py`) | - | - (relative speed unmeasured) |
| 19 | Skip NaN sanitizing of the extra (compressed) cache when fp8_ds_mla + no KV transfer | `amd/rocm.py:48-59` | kernel flag | decode / attention | UNKNOWN (did not find the equivalent flag; the aiter kernel may not sanitize at all) | S | low |
| 20 | Sparse prefill attention: aiter OPUS kernel for >=1024 query tokens, Triton ragged below that; chunked dequant+gather of compressed+SWA KV into a BF16 workspace | `R:48-67`, `R:3642-3718`, `amd/rocm.py:849-961` | aiter `pa_sparse_prefill_opus`, Triton | prefill / attention | PARTIAL: default HIP path runs the decode-style aiter sparse kernel for prefill too. OPUS (`pa_sparse_prefill_fp8_opus`) is wired only in the opt-in unified-KV path (`kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py:678`, `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`) | M | low for decode (helps TTFT / chunked prefill only) |
| 21 | Routed MoE backend = `AITER_MXFP4_BF16` (a16w4 aiter `fused_moe`, CK/FlyDSL), swiglu clamp fused, hidden/intermediate padding. AITER's own `AITER_BF16_FP8_MOE_BOUND` (default 256) then keeps **bf16 activations for M<256** | `model_executor/layers/fused_moe/oracle/mxfp4.py:700-710`, `experts/rocm_aiter_moe.py:360-432` | aiter `fused_moe` (FlyDSL/CK) | all / MoE | PARTIAL/DIFFERENT: SGLang forces fp8 activations everywhere (`AITER_BF16_FP8_MOE_BOUND=0`, a8w4 FlyDSL), `srt/layers/moe/moe_runner/aiter.py` | S (env A/B) | **uncertain, possibly med**: the a16w4 path skips the per-layer activation quant, but the kernel-speed difference at M<=160 is unmeasured |
| 22 | Shared expert on an aux stream, overlapped with routed MoE, for <=256 tokens (on by default on ROCm) | `model_executor/layers/fused_moe/runner/shared_experts.py:60-120`, `utils/torch_utils.py:854-863` | torch streams; `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256`, `VLLM_DISABLE_SHARED_EXPERTS_STREAM=0` | decode / MoE | PARTIAL: opt-in `SGLANG_ROCM_USE_MULTI_STREAM` (default False) at `srt/models/deepseek_v4.py:2745-2755`; PR notes call it "mixed/slower" | S | low-med (shared expert is 1 small FP8 MLP per layer; the gain depends on whether MoE fills all CUs at bs<=32) |
| 23 | Fuse the shared expert into the routed MoE (`VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS`, default False, **disabled under EP**); heterogeneous fp8-shared + fp4-routed variant | `models/deepseek_v4/amd/model.py:205-260,356-512` | aiter `fused_moe` with `shared_w1/w2` | decode / MoE | MISSING, but **N/A for our config**: heterogeneous FSE hard-requires hidden 7168/TP8/no-EP (DSV4-Pro), and plain FSE is off under EP in both engines (`srt/models/deepseek_v4.py:5209-5235`) | L | n/a at TP4+EP4 |
| 24 | DSV4 router: `_C` sqrt-softplus top-k with bias, hash layers, and vision bias (`dsv4_topk` fast path is non-aiter only) | `model_executor/layers/fused_moe/router/fused_topk_bias_router.py:145-275` | vLLM `_C` op | all / MoE router | SGLang AHEAD: fused gate GEMV + top-k `kernels/ops/moe/rocm_router_gate.py` | - | - |
| 25 | Engram: fused post-wkv gate kernel; Triton n-gram hash with slot cache; tables in host memory addressed through UVA | `models/deepseek_v41/common/engram.py:247,588,754-1056`, `nvidia/engram.py:92-330` | Triton | all / Engram | HAS: `kernels/ops/embeddings/engram_gate.py:51`, `srt/layers/engram.py`, host table `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE` | - | - |
| 26 | Engram row gather prefetched on a side stream for all engram layers before the decoder loop | `models/deepseek_v41/amd/model.py:580-586`, `nvidia/engram.py:350-398` | torch stream | all / Engram | MISSING in the branch (grep finds no engram prefetch; the PR text mentions an opt-in prefetch with mixed results that is no longer in the branch) | S-M | low (2 layers; the lookup at bs<=32 x 5 is tiny) |
| 27 | Custom all-reduce via aiter (default on); QuickReduce off by default | `envs.py:1247,1368` | aiter CustomAllreduce | all / AR | HAS (+ AR+mHC fusion, #3) | - | - |
| 28 | Breakable CUDA graph: only the sparse indexer + attention run eagerly; FULL graphs for decode | `models/deepseek_v41/attention.py:718,899`, `compilation/breakable_cudagraph.py:66` | - | prefill (graph) | HAS: `--cuda-graph-backend-prefill breakable` | - | - |
| 29 | DSpark context-KV precompute: wkv slice of fused wqa_wkv + fused RoPE/quant/insert op per draft layer | `amd/dspark.py:152-182,227-295` | custom HIP (#8) | DSpark draft | HAS (branch DSpark draft model, `kernels/ops/speculative/dspark/dspark_draft_model.py`) | - | - |
| 30 | DSpark adaptive verification: per-(request, position) survival score from a confidence head, global token budget from a startup cost profile (off by default; on in the `-confidence-TEP4` CI config) | `v1/worker/gpu/spec_decode/adaptive_verification.py`, `docs/features/speculative_decoding/adaptive_verification.md` | host + CUDA graphs | DSpark verify | HAS (probably): `srt/speculative/dspark_components/dspark_planner.py:133-209` (`HostConfidenceBudgetPlanner`). Needs a checkpoint with a confidence head; whether it works on AMD is unverified | S (test) | none at bs1; med at bs8-32 (it trims wasted verify tokens as the GPU saturates) |
| 31 | DSpark top-k-gathered Markov head (bias only the top-k base logits) | `v1/worker/gpu/spec_decode/dspark/speculator.py:204-252`, `config/speculative.py:596` | torch | DSpark draft | MISSING, but **N/A**: documented as a Qwen3-DSpark option, and V4.1 config has `dspark_draft_topk=None` | - | none |
| 32 | DSpark block length: CI/docs use `num_speculative_tokens: 7` + probabilistic draft sampling (SGLang recipe uses block size 5) | `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-AITER-TEP4.yaml` | config | DSpark | DIFFERENT CONFIG | S (flag) | uncertain: +/- at bs1 depending on the acceptance tail; likely negative at bs32 without adaptive verification |

Counts: 32 rows. HAS (incl. SGLang AHEAD) 23 (1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,24,25,27,28,29,30). PARTIAL/DIFFERENT 4 (20,21,22,32). MISSING 3 (23 and 31, both N/A for this config; 26). N/A 1 (6: code that exists in vLLM but is not used by V4.1). UNKNOWN 1 (19).

### Top port candidates
1. **MoE activation-dtype A/B (#21)**: try vLLM's default route, i.e. leave `AITER_BF16_FP8_MOE_BOUND` unset (256) so M<256 runs a16w4. Local aiter already has #5802, which fixes the crash that forced BOUND=0.
   S effort (env only). MoE is the largest decode kernel family, so this is the cheapest experiment that could move bs1-32. Check GSM8K and graph capture at num_tokens 240.
2. **Shared-expert aux-stream overlap (#22)**: vLLM turns it on by default at <=256 tokens. Re-measure `SGLANG_ROCM_USE_MULTI_STREAM=1` with the PR-style bs1/bs8/bs32 method, DSpark on and off.
   S effort. The earlier "mixed" result may predate the current FlyDSL MoE and AR+mHC fusion.
3. **DSpark verify length / adaptive verification (#30, #32)**: A/B block size 5 vs 8 at bs1, and turn on SGLang's confidence budget planner for bs8-32, if the checkpoint has a confidence head.
   S effort (flags). The PR's real-text bs32 DSpark gain is only 1.03x, which is exactly the regime adaptive verification targets.
4. **OPUS sparse prefill in the default HIP path (#20)**: reuse the `pa_sparse_prefill_fp8_opus` wrapper already in `unified_kv_kernels/runtime.py` for extend batches with >=1024 query tokens.
   M effort. TTFT / chunked-prefill only; negligible for steady-state decode.
5. **Engram side-stream prefetch (#26)**: low value (2 layers); port only if a profile shows the engram lookup on the critical path.

Nothing else in vLLM's V4.1 AMD path is both missing in SGLang and applicable to TP4+EP4 V4.1-Flash.

### vLLM launch settings/env for DSV4.1 ROCm
- CI config `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-AITER-TEP4.yaml` (threshold 0.92, 5-shot, 1319 q): `--tokenizer-mode deepseek_v4 --trust-remote-code --max-model-len 8192 --tensor-parallel-size 4 --enable-expert-parallel --block-size 256 --gpu-memory-utilization 0.5 --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype=fp8 --max-num-batched-tokens 16384 --max-num-seqs 128 --moe-backend aiter --speculative-config '{"method":"dspark","model":"deepseek-ai/DeepSeek-V4-Flash-DSpark","num_speculative_tokens":7,"draft_sample_method":"probabilistic","enable_adaptive_verification":false}'`; env `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`.
- The `-confidence-TEP4` variant: same, but `indexer_kv_dtype=mxfp4` (Blackwell-only per `indexer.py:67`), draft `attention_backend=FLASH_ATTN`, `enable_adaptive_verification: true`. This looks like a CUDA config.
- Attention backend forced to `ROCM_FLASHMLA_SPARSE_DSV4` (`amd/model.py:139-149`). Sequence parallel is off on ROCm (`:152-154`). The model is not torch.compiled (no `@support_torch_compile`), so compile fusion passes such as the AR+RMSNorm pass do not apply.
- Relevant env defaults (`envs.py`): `VLLM_ROCM_USE_AITER=False` (must set 1), `_LINEAR/_MOE/_MLA/_RMSNORM/_CUSTOM_AR=True`, `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=False`, `VLLM_ROCM_USE_AITER_MOE_SITUV2=0`, `VLLM_ROCM_AITER_MOE_DISPATCH_POLICY=0`, `VLLM_ROCM_MOE_PADDING=1`, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE`, `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024`, `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256`, `VLLM_ADAPTIVE_VERIFICATION_PROFILE_CONTEXT_LEN` (8192 per docs).
- Neither `AITER_BF16_FP8_MOE_BOUND` nor `TRITON_HIP_USE_ASYNC_COPY` is set in vLLM configs, so AITER's defaults apply. `rocm_native.py:123-131` notes that Triton 3.8 enables async copy by default on gfx950.

### Open questions (need runtime profiling)
- Per-kernel decode time, vLLM vs SGLang at bs1/8/32: MXFP8 `tl.dot_scaled` vs SGLang MXFP8 GEMM/GEMV; vLLM Triton split-K sparse decode vs aiter gluon/swap-AB; FP8 aiter paged MQA logits vs SGLang's FP4 HIP indexer. Only a trace from the vLLM image can show whether vLLM wins any of these.
- a16w4 vs a8w4 FlyDSL MoE at M = 1..160 (#21): kernel time and accuracy.
- Does vLLM decode actually run as a FULL graph for DSV4.1 on ROCm, or break at `@eager_break_during_capture` every layer? The indexer code references `CUDAGraphMode.FULL` (`R:1446`). Check this in the vLLM server log.
- Whether vLLM DSpark (block 8, probabilistic) gets a higher bs1 acceptance length than SGLang block 5 on the same prompts.
- Does the DSV4.1-Flash DSpark checkpoint ship a confidence head? It gates #30 in both engines.
- The aiter/triton/torch versions in the vLLM image (ROCm 10) differ from ours, so any kernel-level win may come from the library version rather than vLLM code.
