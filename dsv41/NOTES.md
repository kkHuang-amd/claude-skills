# DSV4.1 notes (newest first)

## 2026-09-27 ATOM gap survey (image rocm/atom-dev:nightly_202609250902)

Image label org.opencontainers.image.revision = ATOM 4685e3cf (2026-09-25, #2400); aiter = nightly copy in
/app/aiter-test = aiter main e2d019f15 (0.1.23.dev173, = #5750 merge commit, 2026-09-25;
contains #5749, NOT #5561/#5802; 173 commits past our global v0.1.22.post1, 38 past our pr/5750 1053c79bb base). Blobless clone at /workspace/atom-survey (checked out 4685e3cf).
ATOM V4.1 is 6 PRs, 9/21-9/25: #2314 native, #2352 decode launch budget + native FP8, #2373 TP2 DSpark + AgentX
recipe (recipes/DeepSeek-V4.1-Flash-Agentic.md), #2377 index paging + gfx950 gluon decode, #2392 block maxima,
#2400 FP8 tail faults. Recipe uses --spec-decode-acceptance-length 3.51 = same fixed AL as our
SGLANG_SIMULATE_ACC_LEN=3.51 (our log accept len 3.45-3.55) -> comparison fair on that axis.
ATOM recipe's own TP2 table (c1/2/8/16/32/64 TTT): 10,134/10,743/27,095/46,969/83,359/103,358 -- slightly below
the public screenshot (10,820/11,267/29,261/53,993/91,002/102,390); ours ties it at c8 and beats it at c16-c64.

Gaps vs /sgl-workspace/sglang-rolao-opt (spot-checked = verified by rg; others from subagent reads):
- [verified] no gfx950 gluon fused sparse decode with LDS-shared KV panels (#2377 paged_decode_gluon) -- low-conc decode.
- [verified] no DualRMSNormMXFP8 (aiter fused_dual_rmsnorm_mxfp8_quant exists in aiter-5750) -- decode launches.
- [verified] no rope_quant_window (RoPE + FP8 quant + window write in one kernel) -- decode launches.
- [verified] DSpark draft attn fuse_wqa_wkv=False (deepseek_v4_dspark.py:115); ATOM draft prologue ~3 kernels,
  write_context_kv packed GEMM + fused_draft_kv_tail (21.73 -> 6.67 launches/draft step, TP4).
- [verified] Engram side-stream overlap: ATOM_ENGRAM_OVERLAP default 1; SGLang none. Our TP2 also uses HOST table.
- [verified] candidate block maxima grid = rows x full table width (low_ratio_backend_hip.candidate_block_scores);
  ATOM #2392 bounds by row context: decode c64 2k 66.8->2.5 us, prefill 8k 1347->10.4 us. Small port.
- index plane paged at candidate block length, score only candidate blocks (#2377, needs aiter#5749 -- present in
  aiter-5750 but unused). Large port. Also: fused decode normalize-once (small), mhc_pre_delayed (medium),
  draft torch.compile (medium), state checkpoints @8192 for prefix cache (large).
Recipe diffs: mem 0.9 vs 0.70-0.85, chunk 16384 vs 4096, graph capture to 128 vs max-bs 64, block 16.

## Closed investigations (2026-09-24/25, condensed)
Full logs with every intermediate run: `/shared_nfs/kk/dsv41/doc_backup_20260929/NOTES.md`. Setup: 4xMI355X TP4,
random-ids ISL4096/OSL1024, GSM8K 5-shot 1319 q. Run-to-run GSM8K spread at temp 0 is ~±1pt (batching nondeterminism).

- **09-24 Gap vs PR (branch e2e824dc58).** Baseline (aiter acf8fdf93 unpatched, BOUND=0, sgl-kernel 0.4.7) GSM8K
  off 0.905 / on 0.905 (PR 0.9045 / 0.9022). The apparent -5% (off) / -21% (DSpark) c1 gap was mostly measurement
  method: our bench's `Output token throughput` includes TTFT; the PR uses ignore_eos, warm-up + 6 runs MEDIAN,
  flush_cache, seed 42, and first-to-last streamed token. PR-style (`scripts/run_pr_style_c1.sh`): off 147.7 vs
  PR 155.73 (-5.2%), on 621.1 vs 641.88 (-3.2%). PR env list = TRITON_HIP_USE_ASYNC_COPY=0, SGLANG_USE_AITER=1,
  SGLANG_USE_ROCM700A=0 (container sets 1; only affects DP-attention), ROCM_QUICK_REDUCE_QUANTIZATION=NONE,
  AITER_BF16_FP8_MOE_BOUND=0. No "SGLANG_OPT_HIP_*" names exist on this branch. PR model revision dba1be0a40aa45a94ad051997016db3960a90277.
- **09-24 aiter #5561 / #5802 steps.** +#5561 (LDS-DMA drain) at BOUND=0: GSM8K 0.908/0.911, throughput unchanged
  (off c1/c8/c32 144.6/897/2191) -> keep it (correctness fix). Side note: c1 TPOT 6.63 -> 6.76-6.77 ms after #5561
  (~2% slower, never A/B'd PR-style). +#5802 (bf16 SiLU) at BOUND=256: throughput unchanged (off 144.6/901/2235),
  GSM8K x3 off mean 0.896, on 0.903 -> no gain. Keep BOUND=0 (= PR config).
- **09-24 sgl-kernel rebuilt from branch (AOT top-k sort_output).** Built via pyproject_rocm.toml +
  `AMDGPU_TARGET=gfx950 python setup_rocm.py install`. GOTCHA: the .egg is SHADOWED by the existing
  site-packages/sgl_kernel dir -> replace that dir. Verify: `torch.ops.sgl_kernel.deepseek_v4_topk_transform_512.default._schema`
  contains sort_output. 0.4.7 backup: /shared_nfs/kk/dsv41/sgl_kernel_backup_0.4.7. Result: GSM8K 0.907/0.902,
  throughput unchanged -> sort fusion was not the gap.
- **09-24 Simulated-AL "slowdown" = false alarm.** SIM_AL=3.51 PR-style c1 378 tok/s vs real 621: per-step cost is
  identical (~10.0-10.2 ms/DSpark step at bs1, vs plain decode 6.77 ms); the seed-42 prompt simply has real AL 5.3-5.7.
  Rule: compare DSpark throughput only at equal AL (server-log `accept len`, or SIM_AL).
- **09-24 c32 TTFT vs vLLM (DSpark off 2768 vs 1935 ms).** 16k-chunk prefill profile (`scripts/profile_prefill.sh`,
  ~500 ms): comm 30% (NCCL all-reduce, 168 MB/layer, ~210 GB/s busbw = xGMI bandwidth-bound), sparse attention
  21% (decode kernel used for prefill), MoE 16%, norm/mHC/quant 15%, GEMM 14%. Levers: QR=INT8 (`QR=` knob in
  launch_server.sh) c32 TTFT -6.6%, GSM8K 0.907, loses bit-determinism; OPUS prefill -10.2% (OPUS_PORT.md);
  combined 2273 ms. The main cause was request DESYNC: the fixed ISL/OSL + ignore_eos bench runs lock-step waves
  (SGLang e2e std 34-45 ms vs vLLM 683 ms), and SGLang prefills a whole wave before decoding. `--enable-mixed-chunk`
  (+ OPUS on MIXED batches): c32 TTFT 1520 ms (vLLM 1935), 2341 tok/s (vLLM 2048), TPOT 12.18 ms (+6%), GSM8K 0.911.
- **09-24 mHC prefill roofline -- no change made.** hc_boundary_prefill_kernel ~3.1-3.6 TB/s, capped by 1 CTA/CU
  (LDS); CTA sweep (`scripts/bench_hc_boundary_ctas.py`) best is <= 1.7% on the kernel (~0.2% of prefill) -> heuristic
  kept (`kernels/ops/layernorm/mhc_boundary_hip.py:612`). `_rmsnorm_sinkhorn_kernel` is fused rmsnorm+MXFP8 quant +
  sinkhorn, ~2.6 TB/s; changing num_warps breaks the bitwise contract. More would need an mHC kernel redesign
  (<= ~3% of prefill).
- **09-24 DSpark c32 TTFT (2485 vs vLLM 1175 ms).** Root cause = simulator, not engine: SGLang DSpark sim fills one
  acceptance for the whole batch (`speculative/dspark_components/dspark_verify.py:222`) -> perfect lock-step; vLLM
  samples per request/position. A per-request Bernoulli experiment (rolled back) gave TTFT 885 ms, 3160 tok/s, TPOT
  9.15 ms (vLLM 1175 / 3250 / 8.67). DSpark DOES support mixed chunk (`spec_info.py:146`): TTFT 2485 -> 2214 ms mean.
  Implication: AgentX's `SGLANG_SIMULATE_ACC_LEN` keeps batches in lock-step, so its latency distribution differs
  from real acceptance.
- **09-25 AgentX TP2 3600 s faults -- NOT OPUS.** c16/c64 faulted ("Write access to a read-only page", both GPUs,
  under prefill-graph replay); OPUS=0 c64 faulted too. Root cause = int32 `loc` loads in the Triton DSv4 KV-store
  (`triton_store_cache.py`): writes go out of bounds past ~3.67M tokens (TP2 has ~50M). Fixed by upstream #41159
  (cherry-pick c65a3acad7; `patches/sglang_share_opus_prefill_0002.patch`). kvfix reruns are fault-free
  (results/agentx.md). Runs before the fix with loc > 3.67M may have silently corrupted KV.
- **09-24 AgentX TP2 bring-up.** In the old container, TP2 (EP2 and EP1) hit a GPU memory fault during prefill-graph
  capture (M >= 2048) in untuned aiter fused_moe shapes. In a fresh container (`setup_env.sh`: aiter acf8fdf93 +
  #5561 + #5802, rebuilt sgl-kernel) it does NOT reproduce, and `scripts/repro_moe_tp2.py` passes. Cause unknown
  (stale aiter / FlyDSL JIT cache suspected). Gotcha: export AIPERF_PYTHON_VERSION=3.11 after agentx_env.sh.
  Smoke c4 (600 s, unofficial): TTT 9,318 tok/s/GPU, but output_actual mean 778 vs 1978 expected.
