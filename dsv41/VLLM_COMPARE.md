# vLLM vs SGLang — DeepSeek-V4.1-Flash on MI355X

## CONTINUE HERE
**Status: CLOSED (2026-09-24/25).** The project moved to porting ATOM optimizations -> see ATOM_PORT.md (and the
2026-09-27 ATOM gap survey in NOTES.md). Final outcome: SGLang decode is faster than vLLM at every load (TPOT
-11..-16%). The c32 TTFT gap was a benchmark artifact: lock-step waves plus SGLang's batch-wide simulated acceptance.
With `--enable-mixed-chunk` (DSpark off) or per-request simulated acceptance (DSpark), SGLang beats vLLM on TTFT.
GSM8K is at parity. Full pre-condensation text: `/shared_nfs/kk/dsv41/doc_backup_20260929/VLLM_COMPARE.md`.
Leftover (not pursued): report the vLLM OPUS + prefix-cache crash upstream.

## Comparison protocol (decided 2026-09-24)
- DSpark compared with **simulated acceptance AL=3.51** (InferenceX `golden_al_distribution/dsv41flash_dspark.yaml`,
  thinking_on, 5 draft tokens -- same value AgentX uses). Both engines count the bonus/verify token in AL.
  SGLang: `SIM_AL=3.51` in launch_server.sh -> `SGLANG_SIMULATE_ACC_LEN=3.51` (+ `SGLANG_RAGGED_VERIFY_MODE=static`,
  required verify-all). vLLM: `"rejection_sample_method":"synthetic","synthetic_acceptance_length":3.51`.
  SGLang `--speculative-dspark-block-size 5` == vLLM `num_speculative_tokens 5` (gamma).
- GOTCHA: the simulators differ. SGLang sets ONE acceptance per batch per step (lock-step, e2e std ~34 ms); vLLM
  samples per request and per draft position (requests desync). This alone explains the DSpark c32 TTFT gap (NOTES.md).
- Same client for both: `sglang.bench_serving --backend sglang-oai` vs `--backend vllm` (identical
  /v1/completions code path); GSM8K via vLLM's `gsm8k_eval.py` for both (`scripts/run_gsm8k_openai.sh`).
- Accuracy only with real acceptance. Throughput: PR-style bs1 (median of 6, 1000/TPOT) + ISL4096/OSL1024 sweep.
- vLLM config = InferenceX `dsv41flash_fp4_mi355x_vllm_mtp.sh` + `--enable-expert-parallel` + fp8 KV to match
  SGLang (knobs in vllm_launch.sh). vLLM block size 256 fails -> default block size used.

## Setup facts
- vLLM image `vllm/vllm-openai-rocm:nightly-rocm100-3df4ae153eb385e27b52f26c81f8edb9e20b9984`; source =
  `vllm-project/vllm` 3df4ae153e (2026-09-21) at `/sgl-workspace/vllm-src` (also `/sgl-workspace/vllm-ref`).
  ENTRYPOINT is `vllm serve` -> run with `--entrypoint /bin/bash`. This container has no docker.
- Image stack: vllm 0.3.1.dev190+g3df4ae153.rocm100, torch 2.12.0+rocm10.0.0, triton 3.8.0, amd-aiter 0.1.22.post1,
  flydsl 0.3.2, ROCm 10.0. SGLang: torch 2.9.1+rocm7.2.0, triton 3.7.0, aiter dev acf8fdf9 (+#5561/#5802),
  flydsl 0.3.2, ROCm 7.2 => stack differs, so gaps are not purely engine.
- vLLM DSV4.1 code: `vllm/models/deepseek_v41/{amd,common,nvidia}/`; AMD: `amd/model.py`, `amd/rocm.py`, `amd/dspark.py`.
- vLLM GOTCHA: prefix caching ON (default) + GSM8K -> `HSA_STATUS_ERROR_MEMORY_FAULT` on all 4 GPUs right after
  the first `Using AITER OPUS for large sparse MLA prefill` (prefix hit 82%). `--no-enable-prefix-caching` -> 0 faults.
  Hypothesis: -1 / negative CSR entries after a prefix hit (OPUS_PORT.md). Log: /shared_nfs/kk/dsv41/vllm/server_nodspark.log.

## Final numbers (4xMI355X TP4+EP4, fp8 KV, random-ids ISL4096/OSL1024 unless noted, 2026-09-24)
Accuracy (real acceptance): DSpark off GSM8K SGLang 0.908 vs vLLM 0.901; DSpark on vLLM 0.904 (AL 3.54),
SGLang 0.902-0.911 over 5 runs => parity.

DSpark OFF (SGLang baseline = before OPUS):
| metric | conc | SGLang | vLLM | SGLang vs vLLM |
|---|---|---|---|---|
| PR-style decode tok/s | 1 | 147.6 | 123.8 | +19% |
| out tok/s | 1 / 8 / 32 | 144 / 897 / 2198 | 120 / 762 / 2048 | +20% / +18% / +7% |
| ShareGPT out tok/s | 1 / 8 / 32 | 145 / 883 / 2081 | 121 / 749 / 1952 | +20% / +18% / +7% |
| TPOT ms | 1 / 8 / 32 | 6.77 / 8.10 / 11.86 | 8.08 / 9.64 / 13.74 | -16% / -16% / -14% |
| TTFT ms | 1 / 8 / 32 | 164 / 841 / 2768 | 256 / 888 / 1935 | c32 1.43x worse |

DSpark ON, sim AL 3.51 (vLLM measured 3.52), SGLang baseline:
| metric | conc | SGLang | vLLM | SGLang vs vLLM |
|---|---|---|---|---|
| PR-style decode tok/s | 1 | 378.1 | 341.3 | +11% |
| out tok/s | 1 / 8 / 32 | 358 / 1623 / 3160 | 306 / 1290 / 3250 | +17% / +26% / -3% |
| ShareGPT out tok/s | 1 / 8 / 32 | 362 / 1564 / 3060 | 319 / 1545 / 3110 | +13% / +1% / -2% |
| TPOT ms | 1 / 8 / 32 | 2.63 / 4.06 / 7.45 | 2.97 / 4.54 / 8.67 | SGLang lower at all conc |
| TTFT ms | 1 / 8 / 32 | 166 / 888 / 2736 | 308 / 1696 / 1175 | c32 2.3x worse |

c32 after fixes (SGLang branch opus-prefill; vLLM unchanged):
| config | SGLang TTFT ms | out tok/s | vs vLLM |
|---|---|---|---|
| DSpark off, OPUS + QR=INT8 | 2273 | 2337 | TTFT 1.17x worse, tok/s +14% |
| DSpark off, OPUS(+MIXED) + QR=INT8 + `--enable-mixed-chunk` | 1520 | 2341 | TTFT 1.27x better, tok/s +14% |
| DSpark sim, OPUS | 2485 | 3340 | TTFT 2.1x worse, tok/s +2.8% |
| DSpark sim, OPUS + `--enable-mixed-chunk` | 2214 (mean) | 3354 | e2e std only 34 -> 119 ms |
| DSpark, OPUS, per-request sim acceptance (experiment, rolled back) | 885 | 3160 | vLLM 1175 / 3250: TTFT better, tok/s -3%, TPOT 9.15 vs 8.67 |
Raw: /shared_nfs/kk/dsv41/{perf,prstyle}_{sgl_dspark_sim3.51_oai,vllm_dspark_sim3.51}/. c8 random-ids vLLM
DSpark (32 prompts) looks noisy.

## Optimization inventory (vLLM AMD path -> SGLang status)
Traced 2026-09-24 by code reading, vLLM 3df4ae153e vs SGLang `dsv41-amd-main` e2e824dc58 (file:line locations
for both engines are in the backup). V4.1 model facts that decide applicability: hidden 5120, 384 routed experts
(top-6) + 1 shared, compress ratios {0,1,2}, index_topk 512, linears [32,32] MXFP8 (ue8m0), MXFP4 experts,
engram layers [1,14]. Main finding: vLLM's V4.1 AMD path is *less* fused than the SGLang branch where decode
matters (mHC+AR, wo_a/wo_b, indexer, attention, router).

Status: **have** = already in SGLang (same or more fused); **N/A** = not needed for V4.1 at TP4+EP4;
**ported** = done here; **open** = not done.
- have: #1 delayed mHC (post folded into next pre); #2 standalone aiter mHC pre/post; #3 AR+mHC fusion (vLLM
  CUDA-only; SGLang ahead); #4 hc-collapse for DSpark head; #5 dense MXFP8 linears (SGLang ahead: GEMV + norm->fp8
  fusion); #7 WO_A (SGLang ahead, `SGLANG_DSV41_FUSED_WO_A`); #8 RoPE+quant+SWA insert fusion; #9/#10
  multi-stream projections and compressor/indexer overlap; #11 fused compressor; #12 indexer Q (SGLang ahead:
  MXFP4 indexer on HIP); #13/#14 indexer logits (different FP4 HIP design); #15 top-k (sgl-kernel
  `deepseek_v4_topk_transform_512`); #16 candidate-block filtering (fixed-grid trick unverified); #17 ragged top-k
  reuse; #18 sparse decode attention (aiter gluon + swap-AB); #24 router (SGLang ahead); #25 engram gate/hash/
  host table; #27 aiter custom AR; #28 breakable prefill graph; #29 DSpark context-KV precompute.
- ported: #20 OPUS sparse prefill for large extends -> OPUS_PORT.md (now superseded by the opt branch's
  `SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1`).
- N/A / not needed: #6 128-block FP8 preshuffle GEMMs (V4.0 only); #21 MoE bf16 activations for M<256
  (`AITER_BF16_FP8_MOE_BOUND=256` A/B'd: no throughput change, GSM8K not better -> keep BOUND=0); #23 shared-expert
  fusion (off under EP; heterogeneous variant is DSV4-Pro-only); #31 DSpark Markov head (`dspark_draft_topk=None`).
- open (never measured): #22 shared expert on aux stream (`SGLANG_ROCM_USE_MULTI_STREAM=1`, default off; PR
  notes "mixed/slower"); #26 engram side-stream prefetch (low value; ATOM also has engram overlap, see ATOM
  survey); #30 DSpark adaptive verification (`HostConfidenceBudgetPlanner`; needs a confidence-head checkpoint);
  #32 DSpark block 7-8 + probabilistic draft vs our block 5; #19 skip-NaN-sanitize flag (unknown, low).

## vLLM launch reference (DSV4.1 ROCm)
- CI config `tests/evals/gsm8k/configs/DeepSeek-V4-Flash-DSpark-AITER-TEP4.yaml`: `--tokenizer-mode deepseek_v4
  --trust-remote-code --max-model-len 8192 --tensor-parallel-size 4 --enable-expert-parallel --block-size 256
  --gpu-memory-utilization 0.5 --kv-cache-dtype fp8 --attention_config.indexer_kv_dtype=fp8
  --max-num-batched-tokens 16384 --max-num-seqs 128 --moe-backend aiter --speculative-config
  '{"method":"dspark","model":"deepseek-ai/DeepSeek-V4-Flash-DSpark","num_speculative_tokens":7,
  "draft_sample_method":"probabilistic","enable_adaptive_verification":false}'`; env `VLLM_USE_V2_MODEL_RUNNER=1
  VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_MOE=1`.
- `VLLM_ROCM_USE_AITER` defaults False (must set 1); QuickReduce off (`VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=NONE`);
  `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD=1024`, `VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=256`. Attention backend
  forced to `ROCM_FLASHMLA_SPARSE_DSV4`; the model is not torch.compiled. Neither `AITER_BF16_FP8_MOE_BOUND` nor
  `TRITON_HIP_USE_ASYNC_COPY` is set (Triton 3.8 enables async copy by default on gfx950).
