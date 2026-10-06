# B200 vLLM TP4 profile + microbench for DSV4.1-Flash (answer to B200_REQUEST_1006.md rev 2)

Owner node: dgx-025

## CONTINUE HERE

**Status:** request is now rev 3 (decode = cold pass + cache-warm pass, report/profile the warm one). P1a done, P2 done
(both unaffected by rev 3). P1b: c16 done with the rev 3 method; c8 being redone with it (c8 server restarting).
**Next:** `decode_run.sh 8 cold|warm|warmprof` -> kernel_summary -> restart `serve.sh 0 c1` -> P3 (same 3 passes, conc 1).
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
| 8 | idx 2 (max-num-seqs 16) | _rerun with rev 3 method in progress_ | | | | |

c16 server's last `SpecDecoding metrics` line: `Mean acceptance length: 3.51, ... Per-position acceptance rate: 1.000,
1.000, 0.510, 0.000, 0.00`. (Cold c16 pass for reference: TTFT p50 6066 ms, TPOT p50 7.91 ms.)

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

### c8, rev 2 method (superseded by the rev 3 rerun; kept until it lands)

Ran before rev 3: clean = 24 prompts at conc 8 (cold 64k prefills interleaved with decode) -> TPOT p50 6.47 ms, ITL p90
84 ms; an extra all-together run (8 prompts, out 8192, cache-warm) -> **TPOT 2.81 ms p50, ITL p50 9.80 ms, 2793 tok/s**.
Its profile already had batch 8 in all 40 steps (`generation_8(48)`), 11.16 ms/step profiled vs 9.8 ms unprofiled.
Group shares (rank 0, ms/step): MoE 4.20, dense GEMM 2.62, all-reduce 1.45, mHC 1.27, norm/elementwise/quant 1.06,
sparse MLA 0.87, indexer 0.59, KV compressor 0.18, sampling/draft 0.10, engram 0.03; GPU idle 3.8%.

**Gap to flag:** steady-state c8 TPOT here (2.81 ms) vs 1.92 ms p50 quoted for B200 from InferenceX run 37070984585.
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

## Traces (dgx-025, not in git)

| trace set | path | size |
|---|---|---:|
| decode c8 (rev 2 method), 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c8/` | 22 MB |
| decode c16 (rev 3 warmprof), 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c16/` | 22 MB |
| extend L=64k + 512, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/extend_65536/` | 1.1 MB |
| extend L=256k + 512, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/extend_262144/` | 1.2 MB |

Each dir holds `*rank{0..3}*.pt.trace.json.gz`, vLLM's `profiler_out_*.txt`, and `summary.md` (`kernel_summary.py`
output; extend dirs also `summary_rank1.md`).
