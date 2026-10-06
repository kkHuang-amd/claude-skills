# B200 vLLM TP4 profile + microbench for DSV4.1-Flash (answer to B200_REQUEST_1006.md rev 2)

Owner node: dgx-025

## CONTINUE HERE

**Status:** request is now rev 3 (decode = cold pass + cache-warm pass, report/profile the warm one). P1a done, P2 done
(both unaffected by rev 3). **P1, P2, P3 all done** (decode c1/c8/c16 with the rev 3 method). Open item: B200 c8 TPOT here 3.00 ms vs 1.92 ms
quoted from InferenceX run 37070984585 (see P1b "Gap to flag"). The c1 server (entry idx 0) is left running on GPU 0-3.
**Next:** nothing requested; possible follow-up is reconciling the c8 TPOT gap with the InferenceX artifacts.
**Files:** `dsv41/scripts/b200_tp4_1006/` (`serve.sh`, `prefix_sweep.py`, `decode_run.sh`, `extend_prof.py`,
`kernel_summary.py`, recipe copy).
**Repro (P1a):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash dsv41/scripts/b200_tp4_1006/serve.sh 2 c8 > server_c8.log 2>&1 &
# wait for "OpenAI server is ready" (Rust frontend; there is no "Application startup complete")
python3 dsv41/scripts/b200_tp4_1006/prefix_sweep.py
```

## Environment

- Container `vllm/vllm-openai:nightly-ac9126e58aa7bbab1856ba6593ba4d5003fea516` (same vLLM commit as the requested
  `nightly-dev-x86_64-cu130-ac9126e58aa7`; image digest is not visible from inside the container).
  `vllm 0.30.1rc1.dev493+gac9126e58`, torch 2.13.0+cu130, flashinfer 0.7.0.post1.
- 4x B200 (GPU 0-3 of 8), driver 580.173.02, power limit 1000 W, max clocks SM 1965 / mem 3996 MHz.
- Weights: `deepseek-ai/DeepSeek-V4.1-Flash` from HF (48 shards).

### Recipe deviations (read before comparing)

- Commit `a8504a430` is not in the public InferenceX repo, so args come from origin/main `5ff11ab20`
  (`inferencex-e2e/benchmarks/single_node/srt-slurm-recipes/dsv41flash/vllm/b200-fp4-mtp/agentic.yaml`, copied into
  the scripts dir): base args + `zip_override_tp4` entry for the conc. That includes `--tokenizer-mode deepseek_v41`,
  the deepseek_v41 tool/reasoning parsers and `--kernel-config '{"enable_flashinfer_autotune":true}'`.
- Speculative config: `rejection_sample_method: synthetic`, `synthetic_acceptance_length: 3.51` (as requested; the
  InferenceX throughput harness makes the same substitution, see `runners/test_dsv41flash_capture.py`).
- On main the TP4 zip maps conc -> max-num-seqs as c1:8, c8:16, c16:32 (the request text says 8|16 for c8|c16).
  I follow main: the **c8 server** is entry idx 2 (`max-num-seqs 16`, `max-num-batched-tokens 8192`,
  capture size 8190) and also serves P1a and P2; c16 uses entry idx 3 (`max-num-seqs 32`, compilation config adds
  `decoder_replay_cudagraph_capture_sizes`); c1 (P3) uses entry idx 0 (`max-num-seqs 8`, `max-num-batched-tokens 4096`).
- `--profiler-config` (torch, with_stack off, record_shapes on, ignore_frontend, max_iterations 40) is always passed;
  it is inactive outside `/start_profile`..`/stop_profile`.
- Startup: weights 253 s, CUDA graph capture 216 s, then FlashInfer autotune ~20 min; KV cache 78.4 GiB/GPU
  (46.2M tokens).

## P1a -- cached-prefix TTFT sweep (Deliverable 1b), conc 1, c8 server

Method: for each (L, N) a fresh random-token-id prefix P_L (ids in [1000, 120000)), warm with max_tokens 1, then
3x `P_L + N fresh tokens` (max_tokens 1, streamed `/v1/completions`), client-side TTFT, median reported.
Cache hit verified per request from `vllm:prefix_cache_hits_total` / `queries_total` deltas: every measured request hit
exactly L tokens out of L+N.

Median TTFT (ms), 2026-10-06, dgx-025:

| L \ N | 512 | 4096 |
|------:|----:|-----:|
| 32k   | 58.6 | 95.7 |
| 64k   | 63.5 | 113.7 |
| 128k  | 88.6 | 143.4 |
| 256k  | 147.9 | 199.9 |

Linear fit TTFT = a + b x L:
- N=512: **a = 40.3 ms, b = 0.412 ms per 1k prefix**
- N=4096: a = 83.0 ms, b = 0.460 ms per 1k prefix

This matches the InferenceX-trace fit for B200 (58 ms + 0.39 ms/1k). The 3.6k extra new tokens cost ~43 ms at every L.
Raw CSV: `/shared_nfs/kk/dsv41_b200/p1a_sweep.csv` on dgx-025 (first 64k/512 rep was 102.7 ms, an outlier; median
unaffected).

## P1b -- decode D64 (Deliverables 1a, 2.1, 3), request rev 3 method

Method (rev 3): `decode_run.sh <conc> cold|warm|warmprof` = `vllm bench serve --dataset-name random
--random-input-len 65536 --random-output-len 1024 --ignore-eos --seed 0 --num-prompts <conc> --max-concurrency <conc>`
three times on the same server: **cold** (fills the prefix cache, not reported), **warm** (reported: every prefill a
cache hit, all conc requests decode together), **warmprof** (same prompts; /start_profile once all conc requests have
their first token, profiler stops itself after 40 worker steps). The warm and warmprof runs are separate so the reported
numbers carry no profiler overhead.

### Clean numbers (warm run, no profiler), 2026-10-06, dgx-025

| conc | server entry | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | ITL mean/p50/p90 ms | out tok/s | AL |
|---:|---|---|---|---|---:|---:|
| 16 | idx 3 (max-num-seqs 32) | 361 / 355 / 428 | **3.62 / 3.61 / 3.67** | 12.64 / 12.53 / 12.97 | 3979 | 3.51 |
| 8 | idx 2 (max-num-seqs 16) | 265 / 259 / 294 | **3.00 / 3.00 / 3.03** | 10.49 / 10.42 / 10.86 | 2417 | 3.51 |

c16 server's last `SpecDecoding metrics` line: `Mean acceptance length: 3.51, ... Per-position acceptance rate: 1.000,
1.000, 0.510, 0.000, 0.00`. (Cold c16 pass for reference: TTFT p50 6066 ms, TPOT p50 7.91 ms.) c8 server's last line: `Mean acceptance length:
3.52, ... Per-position acceptance rate: 1.000, 1.000, 0.521, 0.000, 0.000` (cold c8 pass: TTFT p50 3325 ms, TPOT p50 4.92 ms).

### Profile c16 (warmprof), 40 steps

Decode batch size = 16: all 40 profiled steps are `execute_context_0(0)_generation_16(96)` (16 requests x 6 tokens:
target verify of 1 + 5 DSpark draft tokens). On B200 **every rank's trace contains the CUDA-graph kernels**
(rank 0-3: 13.166 / 13.171 / 13.167 / 13.173 ms/step, GPU idle 3.1-3.2%), so rank 0 is summarized.
Profiled step 13.17 ms vs 12.53 ms unprofiled ITL (~5% profiler inflation).

trace `dp0_pp0_tp0_dcp0_ep0_rank0.1791278450062936490.pt.trace.json.gz`; steps (execute_* annotations): 40 [('execute_context_0(0)_generation_16(96)', 40)]
wall GPU window 526.65 ms -> **13.166 ms/step** over 40 steps; summed kernel time 14.711 ms/step (multi-stream overlap); GPU idle 3.2%

| group | total ms | ms/step | pct |
|---|---:|---:|---:|
| MoE (incl. routing) | 244.79 | 6.120 | 41.6 |
| dense GEMM | 122.35 | 3.059 | 20.8 |
| mHC (hyper-connections) | 56.47 | 1.412 | 9.6 |
| norm/rope/elementwise/quant | 44.95 | 1.124 | 7.6 |
| all-reduce/comm | 39.01 | 0.975 | 6.6 |
| sparse MLA attention | 30.58 | 0.764 | 5.2 |
| indexer (logits + top-k) | 29.78 | 0.744 | 5.1 |
| KV compressor / compressed-KV | 7.30 | 0.182 | 1.2 |
| other | 5.85 | 0.146 | 1.0 |
| sampling/draft-specific | 5.21 | 0.130 | 0.9 |
| engram | 2.16 | 0.054 | 0.4 |

```csv
kernel_name,group,calls,total_us,us_per_step,pct
"bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x32x256_s5_et128x32_m256x32x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp",MoE (incl. routing),1720,138083,3452.1,23.47
"bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x32x256_s4_et128x32_m256x32x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M",MoE (incl. routing),1720,80031,2000.8,13.60
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_",dense GEMM,7400,57635,1440.9,9.79
"void deep_gemm::sm100_mega_mhc_impl<5120u, 40u, 148u, true, true, false, 0u>(CUtensorMap_st, CUtensorMap_st, CUtensorMap",mHC (hyper-connections),3160,49850,1246.2,8.47
"void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4, __nv_bfloat16, true, float4>(flashinfe",all-reduce/comm,3520,29843,746.1,5.07
"fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1VarSeqQ16Kv128PersistentSwapsAbForGen",sparse MLA attention,1640,24614,615.4,4.18
"kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o9616512div85121_tensorptri64gmemo961_tensorptrf32gmemo64641_cutlas",dense GEMM,1600,19000,475.0,3.23
"void moe::dev::finalize::finalizeKernelVecLoad<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, float, 2, true> >(m",MoE (incl. routing),1600,18218,455.4,3.10
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_",dense GEMM,1760,15639,391.0,2.66
"nvjet_sm100_tss_32x64_64x16_4x2_2cta_h_bz_splitK_TNN",dense GEMM,1640,10028,250.7,1.70
"ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",all-reduce/comm,240,9172,229.3,1.56
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,3560,7834,195.8,1.33
"void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom::KernelParams<__nv_b",MoE (incl. routing),1600,7418,185.4,1.26
"_dsv4_topk_kernel",indexer (logits + top-k),1600,6710,167.8,1.14
"void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal",dense GEMM,1880,6137,153.4,1.04
"void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, true, true, false>(c10::BFloat16*, unsig",KV compressor / compressed-KV,1720,5975,149.4,1.02
"_build_flashinfer_mixed_sparse_indices_kernel",sparse MLA attention,1720,5197,129.9,0.88
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,1720,4539,113.5,0.77
"_q_kv_norm_quant_kernel",norm/rope/elementwise/quant,1720,4056,101.4,0.69
"void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::T",norm/rope/elementwise/quant,320,4021,100.5,0.68
"void vllm::sampled_topk::sampled_topk_kernel<512>(float const*, int const*, int*, long, int)",indexer (logits + top-k),160,3941,98.5,0.67
"void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloa",norm/rope/elementwise/quant,1720,3748,93.7,0.64
"void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(i",norm/rope/elementwise/quant,1720,3605,90.1,0.61
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorptrbf16gmema",norm/rope/elementwise/quant,1720,3395,84.9,0.58
"void at::native::mbtopk::computeBlockDigitCounts<float, unsigned int, unsigned int, 2>(at::cuda::detail::TensorInfo<floa",indexer (logits + top-k),160,3000,75.0,0.51
"mhc_post_tilelang_kernel",mHC (hyper-connections),280,2696,67.4,0.46
"void deep_gemm::sm100_paged_mqa_logits<1u, 32u, 128u, 64u, true, true, true, 3u, 10u, 256u, 16u, 128u, 256u, cutlass::fl",indexer (logits + top-k),120,2561,64.0,0.44
"void deep_gemm::sm100_paged_sparse_mqa_logits<128u, 8u, 2u, 5u, 5u, 5u, 148u, 2u, true>(unsigned int, unsigned int, __nv",indexer (logits + top-k),160,2448,61.2,0.42
"nvjet_sm100_tst_512x24_64x3_4x1_v_bz_TNT",dense GEMM,200,2419,60.5,0.41
"nvjet_sm100_tst_256x104_64x4_4x1_v_bz_TNT",dense GEMM,40,2393,59.8,0.41
```

largest 'other': Kernel 0.84ms; void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::A 0.74ms; _compute_local_logits_stats_kernel 0.74ms; _expand_candidates_kernel 0.49ms; memcpy32_post 0.44ms; _compute_swa_indices_and_lens_kernel 0.43ms; _post_update_kernel 0.31ms; _hash_ids_kernel 0.22ms

`dsv41/scripts/trace_kernel_summary.py` (the MI355X-side script) was also run on this trace
(`mi355x_fmt_rank0.{md,csv}` next to it), but its first-match regexes misfile B200 kernel names: the trtllm fp4 MoE
GEMMs `bmm_MxE4m3_MxE2m1*` / `bmm_Bfloat16_MxE2m1*` land in dense_gemm (its moe = 3.2%) and its sparse_mla_attn
(25.7%) and comm (1.6%) are also off. Use the table above for B200 groups.

### Profile c8 (warmprof), 40 steps

Decode batch size = 8: all 40 steps are `execute_context_0(0)_generation_8(48)`. All 4 ranks contain the CUDA-graph
kernels (10.670 / 10.698 / 10.692 / 10.695 ms/step); rank 0 summarized. Profiled 10.67 ms/step vs 10.42 ms unprofiled
ITL (~2% inflation).

trace `dp0_pp0_tp0_dcp0_ep0_rank0.1791280746764604913.pt.trace.json.gz`; steps (execute_* annotations): 40 [('execute_context_0(0)_generation_8(48)', 40)]
wall GPU window 426.82 ms -> **10.670 ms/step** over 40 steps; summed kernel time 12.243 ms/step (multi-stream overlap); GPU idle 4.6%

| group | total ms | ms/step | pct |
|---|---:|---:|---:|
| MoE (incl. routing) | 168.04 | 4.201 | 34.3 |
| dense GEMM | 116.73 | 2.918 | 23.8 |
| mHC (hyper-connections) | 50.89 | 1.272 | 10.4 |
| norm/rope/elementwise/quant | 44.45 | 1.111 | 9.1 |
| sparse MLA attention | 34.94 | 0.874 | 7.1 |
| all-reduce/comm | 31.19 | 0.780 | 6.4 |
| indexer (logits + top-k) | 23.84 | 0.596 | 4.9 |
| KV compressor / compressed-KV | 7.16 | 0.179 | 1.5 |
| other | 5.84 | 0.146 | 1.2 |
| sampling/draft-specific | 5.13 | 0.128 | 1.0 |
| engram | 1.52 | 0.038 | 0.3 |

```csv
kernel_name,group,calls,total_us,us_per_step,pct
"bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x32x256_s5_et128x32_m256x32x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp",MoE (incl. routing),1600,94170,2354.3,19.23
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_",dense GEMM,9160,76236,1905.9,15.57
"bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x32x256_s5_et128x32_m256x32x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M",MoE (incl. routing),1600,52946,1323.7,10.81
"void deep_gemm::sm100_mega_mhc_impl<5120u, 40u, 148u, true, true, false, 0u>(CUtensorMap_st, CUtensorMap_st, CUtensorMap",mHC (hyper-connections),3160,44551,1113.8,9.10
"fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",sparse MLA attention,1520,28526,713.1,5.82
"void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4, __nv_bfloat16, true, float4>(flashinfe",all-reduce/comm,3520,24054,601.4,4.91
"kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o4816512div85121_tensorptri64gmemo481_tensorptrf32gmemo64641_cutlas",dense GEMM,1600,11291,282.3,2.31
"nvjet_sm100_tss_64x16_64x16_2x4_2cta_h_bz_splitK_TNT",dense GEMM,1640,9963,249.1,2.03
"void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, float, 2, true> >(moe::dev",MoE (incl. routing),1600,8644,216.1,1.76
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,3560,7478,186.9,1.53
"ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",all-reduce/comm,240,7133,178.3,1.46
"void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom::KernelParams<__nv_b",MoE (incl. routing),1600,6983,174.6,1.43
"void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal",dense GEMM,1880,6895,172.4,1.41
"_dsv4_topk_kernel",indexer (logits + top-k),1600,6737,168.4,1.38
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,1720,6215,155.4,1.27
"void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, true, true, false>(c10::BFloat16*, unsig",KV compressor / compressed-KV,1720,5868,146.7,1.20
"_build_flashinfer_mixed_sparse_indices_kernel",sparse MLA attention,1720,4954,123.8,1.01
"void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloa",norm/rope/elementwise/quant,1720,4836,120.9,0.99
"_q_kv_norm_quant_kernel",norm/rope/elementwise/quant,1720,4285,107.1,0.87
"void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(i",norm/rope/elementwise/quant,1720,3238,80.9,0.66
"void vllm::cooperative::cooperative_topk_cs2<512u>(vllm::cooperative::CooperativeTopKParams<512u>)",indexer (logits + top-k),160,3189,79.7,0.65
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorptrbf16gmema",norm/rope/elementwise/quant,1720,3002,75.1,0.61
"bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x32x256u2_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transOut_schPd2x1x2x3_bias",MoE (incl. routing),120,2736,68.4,0.56
"mhc_post_tilelang_kernel",mHC (hyper-connections),280,2710,67.7,0.55
"void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::T",norm/rope/elementwise/quant,320,2387,59.7,0.49
"nvjet_sm100_tst_256x48_64x5_4x1_v_bz_TNT",dense GEMM,40,2312,57.8,0.47
"nvjet_sm100_tst_256x40_64x5_4x1_v_bz_TNT",dense GEMM,40,2298,57.5,0.47
"void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetC",norm/rope/elementwise/quant,1600,2234,55.9,0.46
"void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, ",norm/rope/elementwise/quant,1720,2210,55.2,0.45
"mhc_pre_big_fuse_with_norm_tilelang_kernel",mHC (hyper-connections),280,1843,46.1,0.38
```

largest 'other': _compute_local_logits_stats_kernel 1.36ms; void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::A 0.68ms; Kernel 0.50ms; _expand_candidates_kernel 0.48ms; _compute_swa_indices_and_lens_kernel 0.44ms; memcpy32_post 0.38ms; _post_update_kernel 0.32ms; _hash_ids_kernel 0.22ms

### c8, rev 2 method (superseded by the rev 3 rerun above; kept for the record)

Ran before rev 3: clean = 24 prompts at conc 8 (cold 64k prefills interleaved with decode) -> TPOT p50 6.47 ms, ITL p90
84 ms; an extra all-together run (8 prompts, out 8192, cache-warm) -> **TPOT 2.81 ms p50, ITL p50 9.80 ms, 2793 tok/s**.
Its profile already had batch 8 in all 40 steps (`generation_8(48)`), 11.16 ms/step profiled vs 9.8 ms unprofiled.
Group shares (rank 0, ms/step): MoE 4.20, dense GEMM 2.62, all-reduce 1.45, mHC 1.27, norm/elementwise/quant 1.06,
sparse MLA 0.87, indexer 0.59, KV compressor 0.18, sampling/draft 0.10, engram 0.03; GPU idle 3.8%.

**Gap to flag:** steady-state c8 TPOT here (rev 3 warm 3.00 ms; rev 2 out-8192 run 2.81 ms) vs 1.92 ms p50 quoted for B200 from InferenceX run 37070984585.
Candidates: recipe differences (5ff11ab20 vs a8504a430, max-num-seqs) or aiperf vs `vllm bench serve` TPOT
definitions. Not resolved here.

## P2 -- extend over a cached prefix (Deliverable 2.2 / 3), conc 1, c8 server

Warm P_L, one unprofiled warm extend, then profile `P_L + 512` (max_tokens 1). Unprofiled-equivalent TTFT under the
profiler: 96.1 ms (L=64k), 188.5 ms (L=256k); cache hit L of L+512 in both. Each trace has 3 `execute_*` steps; the
real one is `execute_context_1(640)_generation_0(0)` (640 scheduled tokens).

**Use rank 1, not rank 0.** On ranks 0/2/3 all-reduce is ~46 ms of a 81 ms window (L=256k), dominated by 3 single calls
(19.6, 6.0, 4.8 ms) = waiting for rank 1, which is the slowest rank in this window (its all-reduce total is 1.6 ms).
Rank 1 therefore shows the real work:

| group (rank 1) | L=64k ms | L=256k ms | delta |
|---|---:|---:|---:|
| MoE (incl. routing) | 7.76 | 7.74 | 0.0 |
| indexer (logits + top-k) | 1.33 | 3.86 | **+2.53** |
| dense GEMM | 3.32 | 3.31 | 0.0 |
| norm/rope/elementwise/quant | 2.07 | 2.85 | +0.78 |
| mHC | 2.09 | 2.08 | 0.0 |
| all-reduce/comm | 1.60 | 1.62 | 0.0 |
| sparse MLA attention | 1.58 | 1.59 | 0.0 |
| other | 0.64 | 1.00 | +0.36 |
| engram | 0.34 | 0.34 | 0.0 |
| KV compressor / compressed-KV | 0.28 | 0.28 | 0.0 |
| sampling/draft | 0.07 | 0.05 | 0.0 |
| **summed kernel time** | **21.07** | **24.72** | **+3.65** |
| GPU wall window | 54.89 | 75.90 | +21.0 |
| GPU idle in window | 64.4% | 69.4% | |

Takeaways for the MI355X comparison:
- On B200 the extend over a cached prefix is **mostly not GPU kernel time**: GPU idle 64-69%. Of the +21 ms wall growth
  from 64k to 256k, only +3.7 ms is kernels (indexer +2.5 ms is the only kernel group that scales with L; sparse MLA
  attention is flat because it reads a fixed top-k).
- The CPU-side `execute_context_1(640)` annotation grows 49.3 -> 67.8 ms (+18.5 ms), but no individual aten op accounts
  for it (largest: `aten::slice` +3.7 ms over 1386 calls); the rest is Python/host time not covered by aten ops
  (with_stack is off, so not attributed further).
- Rank 0/2/3 tables (all-reduce-dominated) are in `summary.md` next to each trace.

## P3 -- decode D64 c1 (Deliverables 1a, 2.1, 3), c1 server (entry idx 0: max-num-seqs 8, max-num-batched-tokens 4096, capture 4092)

Same rev 3 method (cold / warm / warmprof, `--seed 0`, 1 prompt).

| run | TTFT p50 ms | TPOT mean/p50/p90 ms | ITL mean/p50/p90 ms | out tok/s | AL |
|---|---:|---|---|---:|---:|
| warm (reported) | 92.9 | **1.64 / 1.64 / 1.64** | 5.70 / 5.71 / 5.90 | 578 | 3.49 |
| cold (reference) | 1075 | 1.66 / 1.66 / 1.66 | 5.73 / 5.41 / 5.71 | 369 | 3.47 |

c1 server's last `SpecDecoding metrics`: `Mean acceptance length: 3.50, ... Per-position acceptance rate: 1.000, 1.000,
0.498, 0.000, 0.000`.

### Profile c1 (warmprof), 40 steps -- summarize rank 3

Batch size 1: all 40 steps are `execute_context_0(0)_generation_1(6)`. All ranks contain the CUDA-graph kernels, but at
c1 the profiler slows the host side enough that the step is 9.89 ms profiled vs 5.71 ms unprofiled ITL. Rank 3 is the
slow rank (GPU idle 44.6%, ranks 0-2 4.3-4.4%): ranks 0-2 spend 4.09 ms/step in all-reduce waiting for it. Rank 3's
busy time per step, 9.875 x (1 - 0.446) = 5.5 ms, matches the unprofiled ITL, so **rank 3's kernel table is the real
per-step GPU work** and is the one below (rank 0's is all-reduce-inflated; it is in `summary_rank0.md`).

trace `dp0_pp0_tp3_dcp0_ep3_rank3.1791282761563803751.pt.trace.json.gz`; steps (execute_* annotations): 40 [('execute_context_0(0)_generation_1(6)', 40)]
wall GPU window 395.00 ms -> **9.875 ms/step** over 40 steps; summed kernel time 7.715 ms/step (multi-stream overlap); GPU idle 44.6%

| group | total ms | ms/step | pct |
|---|---:|---:|---:|
| dense GEMM | 78.59 | 1.965 | 25.5 |
| MoE (incl. routing) | 63.93 | 1.598 | 20.7 |
| mHC (hyper-connections) | 63.65 | 1.591 | 20.6 |
| norm/rope/elementwise/quant | 38.39 | 0.960 | 12.4 |
| sparse MLA attention | 23.89 | 0.597 | 7.7 |
| indexer (logits + top-k) | 17.90 | 0.448 | 5.8 |
| KV compressor / compressed-KV | 7.26 | 0.182 | 2.4 |
| all-reduce/comm | 6.40 | 0.160 | 2.1 |
| other | 4.98 | 0.124 | 1.6 |
| sampling/draft-specific | 2.76 | 0.069 | 0.9 |
| engram | 0.84 | 0.021 | 0.3 |

```csv
kernel_name,group,calls,total_us,us_per_step,pct
"bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x8x256_s6_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schedS_biasFp32M_bN_tm",MoE (incl. routing),1600,34104,852.6,11.05
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_",dense GEMM,5840,31284,782.1,10.14
"void deep_gemm::sm100_tf32_hc_prenorm_gemm_impl<24u, 20480u, 64u, 32u, 64u, 16u, 128u, 12u, 128u, 128u>(unsigned int, CU",mHC (hyper-connections),3200,20575,514.4,6.67
"mhc_pre_big_fuse_with_norm_tilelang_kernel",mHC (hyper-connections),3360,19358,484.0,6.27
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100_splitkSm100BlockScaledSplitKGemmKernel_object_at",dense GEMM,3320,19196,479.9,6.22
"bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x8x256_s4_et128x8_m128x8x32_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M_bN",MoE (incl. routing),1480,18029,450.7,5.84
"fmhaSm100fKernel_QkvE4m3OBfloat16H512HVPerCta128PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ8Kv128StaticSwapsAbFor",sparse MLA attention,1520,17662,441.6,5.72
"kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_LamportMHCDeviceKernel_object_at__tensorptrb",mHC (hyper-connections),3080,14300,357.5,4.63
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,3560,8764,219.1,2.84
"kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o616512div85121_tensorptri64gmemo61_tensorptrf32gmemo64641_cutlassc",dense GEMM,1600,7457,186.4,2.42
"kernel_cutlass_kernel_vllmmodel_executorkernelslinearcute_dsl_ll_bf16_splitkLLBf16SplitK_object_at__tensorptrbf16_gmem_a",dense GEMM,1720,7064,176.6,2.29
"_dsv4_topk_kernel",indexer (logits + top-k),1600,6125,153.1,1.98
"void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, true, true, false>(c10::BFloat16*, unsig",KV compressor / compressed-KV,1720,6071,151.8,1.97
"void moe::dev::routing::routingCustom::routingIndicesDynBlockKernel<moe::dev::routing::routingCustom::KernelParams<__nv_",MoE (incl. routing),1600,6009,150.2,1.95
"_build_flashinfer_mixed_sparse_indices_kernel",sparse MLA attention,1720,5008,125.2,1.62
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,1720,4597,114.9,1.49
"_q_kv_norm_quant_kernel",norm/rope/elementwise/quant,1720,4589,114.7,1.49
"void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloa",norm/rope/elementwise/quant,1720,3999,100.0,1.30
"ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",all-reduce/comm,240,3719,93.0,1.21
"kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_QuadFinalizePublishDeviceKernel_object_at__t",mHC (hyper-connections),1480,3436,85.9,1.11
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorptrbf16gmema",norm/rope/elementwise/quant,1720,3005,75.1,0.97
"void flashinfer::trtllm_mnnvl_allreduce::oneshotAllreduceFusionKernel<(unsigned char)4, __nv_bfloat16, false, (flashinfe",all-reduce/comm,440,2681,67.0,0.87
"void cutlass::Kernel2<cutlass_80_tensorop_s16816gemm_bf16_64x64_64x6_tn_align8>(cutlass_80_tensorop_s16816gemm_bf16_64x6",dense GEMM,320,2629,65.7,0.85
"kernel_cutlass_kernel_vllmmodelsdeepseek_v41nvidiaopscute_dslall_reduce_mhc_SharedOnlyPublishDeviceKernel_object_at__ten",mHC (hyper-connections),1600,2556,63.9,0.83
"void deep_gemm::sm100_mega_mhc_impl<5120u, 40u, 148u, true, true, false, 0u>(CUtensorMap_st, CUtensorMap_st, CUtensorMap",mHC (hyper-connections),200,2425,60.6,0.79
"void vllm::cooperative::cooperative_topk_cs8<512u>(vllm::cooperative::CooperativeTopKParams<512u>)",indexer (logits + top-k),160,2265,56.6,0.73
"nvjet_sm100_tst_256x16_64x6_4x1_v_bz_TNT",dense GEMM,40,2252,56.3,0.73
"nvjet_sm100_tst_128x8_64x12_2x1_v_bz_splitK_TNT",dense GEMM,40,2141,53.5,0.69
"void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetC",norm/rope/elementwise/quant,1600,2125,53.1,0.69
"void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, ",norm/rope/elementwise/quant,1720,1978,49.4,0.64
```

largest 'other': void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::A 0.66ms; void at::native::reduce_kernel<128, 4, at::native::ReduceOp<c10::BFloat16, at::n 0.61ms; _expand_candidates_kernel 0.48ms; memcpy32_post 0.47ms; Kernel 0.46ms; _compute_swa_indices_and_lens_kernel 0.42ms; _compute_local_logits_stats_kernel 0.35ms; _post_update_kernel 0.23ms

## Decode groups across conc (ms per step; c1 = rank 3, c8/c16 = rank 0, profiled)

| group | c1 | c8 | c16 |
|---|---:|---:|---:|
| MoE (incl. routing) | 1.598 | 4.201 | 6.120 |
| dense GEMM | 1.965 | 2.918 | 3.059 |
| mHC (hyper-connections) | 1.591 | 1.272 | 1.412 |
| norm/rope/elementwise/quant | 0.960 | 1.111 | 1.124 |
| all-reduce/comm | 0.160 | 0.780 | 0.975 |
| sparse MLA attention | 0.597 | 0.874 | 0.764 |
| indexer (logits + top-k) | 0.448 | 0.596 | 0.744 |
| KV compressor / compressed-KV | 0.182 | 0.179 | 0.182 |
| other | 0.124 | 0.146 | 0.146 |
| sampling/draft-specific | 0.069 | 0.128 | 0.130 |
| engram | 0.021 | 0.038 | 0.054 |
| **step (profiled wall)** | 9.894 (5.5 busy) | 10.670 | 13.166 |
| **unprofiled ITL p50** | 5.71 | 10.42 | 12.53 |

c1 -> c8 the step nearly doubles while tokens per step go 6 -> 48; MoE (1.6 -> 4.2 -> 6.1 ms) and dense GEMM grow with
batch, mHC and attention barely move. The c8/c16 all-reduce rows include ~0.5 ms/step of rank skew (see c8 rev 2 note).

## Traces (dgx-025, not in git)

| trace set | path | size |
|---|---|---:|
| decode c8 (rev 2 method), 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c8/` | 22 MB |
| decode c16 (rev 3 warmprof), 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c16/` | 22 MB |
| decode c8 (rev 3 warmprof), 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c8_rev3/` | 22 MB |
| decode c1 (rev 3 warmprof), 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c1/` | 22 MB |
| extend L=64k + 512, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/extend_65536/` | 1.1 MB |
| extend L=256k + 512, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/extend_262144/` | 1.2 MB |

Each dir holds `*rank{0..3}*.pt.trace.json.gz`, vLLM's `profiler_out_*.txt`, and `summary.md` (`kernel_summary.py`
output; extend dirs also `summary_rank1.md`).
