# B200 vLLM TP4 profile + microbench for DSV4.1-Flash (answer to B200_REQUEST_1006.md rev 2)

Owner node: dgx-025

## CONTINUE HERE

**Status:** P1a done; P1b c8 done; P2 done (run early on the c8 server to save a restart). P1b c16 in progress
(server restarted with the c16 entry); then P3 (c1 server, entry idx 0).
**Next:** `decode_run.sh 16 clean|puredecode|prof` on the c16 server -> kernel_summary -> restart `serve.sh 0 c1` -> P3.
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

## P1b -- decode D64 (Deliverables 1a, 2.1, 3), c8 server (entry idx 2, max-num-seqs 16)

### Clean numbers (no profiler), 2026-10-06, dgx-025

Three runs, because the requested D64 run mixes 64k prefills into the decode:
- **clean** = as requested: random in 65536 / out 1024, ignore-eos, 3 x conc prompts, max-concurrency conc.
  Every new request's 64k prefill (8 chunks of 8192) interleaves with the others' decode, so TPOT/ITL include prefill
  stalls (ITL p90 84 ms).
- **puredecode** (extra): conc prompts, out 8192, all requests start together -> steady-state decode TPOT.
  Prompts repeat the clean run's (same bench seed), so prefill hits the prefix cache; decode is unaffected.
- **prof**: same as puredecode with the profiler window inside; its TPOT includes profiler overhead + trace dump, so it
  is not a performance number.

| run | conc | prompts | TTFT mean/p50/p90 ms | TPOT mean/p50/p90 ms | ITL mean/p50/p90 ms | out tok/s |
|---|---:|---:|---|---|---|---:|
| clean | 8 | 24 | 2316 / 1463 / 5233 | 6.05 / 6.47 / 7.03 | 21.15 / 10.30 / 84.17 | 955 |
| puredecode | 8 | 8 | 375 / 401 / 410 | **2.81 / 2.81 / 2.82** | 9.86 / 9.80 / 10.41 | 2793 |

SpecDecoding (server, during these runs): `Mean acceptance length: 3.51`, per-position 1.000, 1.000, ~0.51, 0, 0
(bench: acceptance length 3.51, 7002 drafts).

**Gap to flag:** steady-state c8 TPOT here is 2.81 ms (step ~9.8 ms / AL 3.51), vs. 1.92 ms p50 that the request
quotes for B200 from InferenceX run 37070984585. Not resolved here; candidates are recipe differences
(5ff11ab20 vs a8504a430, max-num-seqs 16) or a different TPOT definition in aiperf vs `vllm bench serve`.

### Profile (Deliverable 2.1 / 3), c8, 40 steps

Step = one `execute_context_0(0)_generation_8(48)` worker step (8 requests x 6 tokens = target verify + DSpark draft).
Profiled step 11.16 ms (rank 0) / 11.11 ms (rank 1) vs 9.8 ms unprofiled ITL, i.e. ~14% profiler inflation; group
shares are still representative. GPU idle within the window: 3.8% (rank 0) / 9.0% (rank 1) -> GPU-bound, not launch-bound.
All-reduce per step: rank 0 1.45 ms, rank 1 0.81 ms, rank 2/3 1.30/1.35 ms, so ~0.5 ms/step of rank 0's all-reduce is
waiting on the slowest rank.

trace `dp0_pp0_tp0_dcp0_ep0_rank0.1791275905776403364.pt.trace.json.gz`; steps (execute_* annotations): 40 [('execute_context_0(0)_generation_8(48)', 40)]
wall GPU window 446.20 ms -> **11.155 ms/step** over 40 steps; summed kernel time 12.501 ms/step (multi-stream overlap); GPU idle 3.8%

| group | total ms | ms/step | pct |
|---|---:|---:|---:|
| MoE (incl. routing) | 168.02 | 4.200 | 33.6 |
| dense GEMM | 104.80 | 2.620 | 21.0 |
| all-reduce/comm | 58.02 | 1.450 | 11.6 |
| mHC (hyper-connections) | 50.67 | 1.267 | 10.1 |
| norm/rope/elementwise/quant | 42.22 | 1.055 | 8.4 |
| sparse MLA attention | 34.92 | 0.873 | 7.0 |
| indexer (logits + top-k) | 23.78 | 0.594 | 4.8 |
| KV compressor / compressed-KV | 7.21 | 0.180 | 1.4 |
| other | 5.13 | 0.128 | 1.0 |
| sampling/draft-specific | 4.01 | 0.100 | 0.8 |
| engram | 1.26 | 0.031 | 0.3 |

```csv
kernel_name,group,calls,total_us,us_per_step,pct
"bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x32x256_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp",MoE (incl. routing),1600,93914,2347.8,18.78
"bmm_Bfloat16_MxE2m1MxE4m3_Fp32_Ab32_Bb32_t128x32x256_s4_et128x32_m256x32x32_c2x1x1_rM_TN_transOut_schPd2x1x2x3_biasFp32M",MoE (incl. routing),1720,56095,1402.4,11.22
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_",dense GEMM,7320,48529,1213.2,9.71
"void deep_gemm::sm100_mega_mhc_impl<5120u, 40u, 148u, true, true, false, 0u>(CUtensorMap_st, CUtensorMap_st, CUtensorMap",mHC (hyper-connections),3160,44405,1110.1,8.88
"void flashinfer::trtllm_mnnvl_allreduce::twoshotAllreduceKernel<(unsigned char)4, __nv_bfloat16, true, float4>(flashinfe",all-reduce/comm,3520,44147,1103.7,8.83
"fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparseP1MultiCtasKvVarSeqQ16Kv128StaticSwapsAbForGen",sparse MLA attention,1520,28472,711.8,5.69
"kernel_cutlass_kernel_flashinfergemmkernelsdense_blockscaled_gemm_sm100Sm100BlockScaledPersistentDenseGemmKernel_object_",dense GEMM,1840,17140,428.5,3.43
"ncclDevKernel_AllGather_RING_LL(ncclDevKernelArgsStorage<4096ul>)",all-reduce/comm,240,13869,346.7,2.77
"kernel_cutlass_device_kernel_tensorptrbf16gmemalign16o4816512div85121_tensorptri64gmemo481_tensorptrf32gmemo64641_cutlas",dense GEMM,1600,11194,279.8,2.24
"nvjet_sm100_tss_64x16_64x16_2x4_2cta_h_bz_splitK_TNT",dense GEMM,1640,9818,245.5,1.96
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,3560,7431,185.8,1.49
"void moe::dev::finalize::finalizeKernel<moe::dev::finalize::KernelParams<cutlass::bfloat16_t, float, 2, true> >(moe::dev",MoE (incl. routing),1600,7369,184.2,1.47
"void moe::dev::routing::routingCustom::routingIndicesClusterKernel<moe::dev::routing::routingCustom::KernelParams<__nv_b",MoE (incl. routing),1600,7105,177.6,1.42
"_dsv4_topk_kernel",indexer (logits + top-k),1600,6620,165.5,1.32
"void vllm::deepseek_v4_fused_ops::fusedDeepseekV4FullCacheKernel<c10::BFloat16, true, true, false>(c10::BFloat16*, unsig",KV compressor / compressed-KV,1720,5895,147.4,1.18
"void cublasLt::splitKreduce_kernel<32, 16, int, float, float, float, float, false, float, float, float, true, false, fal",dense GEMM,1880,5871,146.8,1.17
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeSwizzledKernel_object_at__tensorptrbf16gme",norm/rope/elementwise/quant,1720,4998,124.9,1.00
"_build_flashinfer_mixed_sparse_indices_kernel",sparse MLA attention,1720,4991,124.8,1.00
"_q_kv_norm_quant_kernel",norm/rope/elementwise/quant,1720,4214,105.3,0.84
"void vllm::act_and_mul_kernel<c10::BFloat16, __nv_bfloat162, &(c10::BFloat16 vllm::silu_kernel<c10::BFloat16>(c10::BFloa",norm/rope/elementwise/quant,1720,3659,91.5,0.73
"void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(i",norm/rope/elementwise/quant,1720,3442,86.1,0.69
"kernel_cutlass_kernel_flashinferquantizationkernelsmxfp8_quantizeMXFP8QuantizeLinearKernel_object_at__tensorptrbf16gmema",norm/rope/elementwise/quant,1720,3286,82.1,0.66
"void vllm::cooperative::cooperative_topk_cs2<512u>(vllm::cooperative::CooperativeTopKParams<512u>)",indexer (logits + top-k),160,3187,79.7,0.64
"mhc_post_tilelang_kernel",mHC (hyper-connections),280,2717,67.9,0.54
"bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_Bb32_Cb32_t128x32x256u2_s5_et128x32_m128x32x32_c1x1x1_rM_TN_transOut_schPd2x1x2x3_bias",MoE (incl. routing),120,2666,66.7,0.53
"nvjet_sm100_tst_256x48_64x5_4x1_v_bz_TNT",dense GEMM,40,2315,57.9,0.46
"nvjet_sm100_tst_256x40_64x5_4x1_v_bz_TNT",dense GEMM,40,2294,57.4,0.46
"void at::native::unrolled_elementwise_kernel<at::native::CUDAFunctor_add<int>, std::array<char*, 3ul>, 4, TrivialOffsetC",norm/rope/elementwise/quant,1600,2233,55.8,0.45
"void at::native::elementwise_kernel<128, 4, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::T",norm/rope/elementwise/quant,320,2227,55.7,0.45
"void at::native::vectorized_elementwise_kernel<8, at::native::FillFunctor<unsigned char>, std::array<char*, 1ul> >(int, ",norm/rope/elementwise/quant,1720,2206,55.1,0.44
```

largest 'other': void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::A 0.68ms; _compute_local_logits_stats_kernel 0.57ms; Kernel 0.49ms; _expand_candidates_kernel 0.49ms; memcpy32_post 0.45ms; _compute_swa_indices_and_lens_kernel 0.44ms; _post_update_kernel 0.32ms; _hash_ids_kernel 0.22ms

Groups: MoE = trtllm fp4 block-scale MoE `bmm_*` + routing/finalize; dense GEMM = flashinfer blockscaled dense GEMM,
nvjet/cublasLt; mHC = deep_gemm `sm100_mega_mhc_impl` + tilelang mhc pre/post (listed separately; the request has no
mHC group); norm/rope/elementwise/quant includes the mxfp8 activation quantize kernels.

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
| decode c8, 40 steps, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/decode_c8/` | 22 MB |
| extend L=64k + 512, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/extend_65536/` | 1.1 MB |
| extend L=256k + 512, 4 ranks | `/shared_nfs/kk/dsv41_b200/traces/extend_262144/` | 1.2 MB |

Each dir holds `*rank{0..3}*.pt.trace.json.gz`, vLLM's `profiler_out_*.txt`, and `summary.md` (`kernel_summary.py`
output; extend dirs also `summary_rank1.md`).
