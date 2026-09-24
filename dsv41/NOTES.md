# DSV4.1 notes (newest first)

## 2026-09-24 baseline: branch e2e824dc58, aiter acf8fdf93 UNPATCHED, BOUND=0, sgl-kernel 0.4.7 (no sort_output), 4xMI355X

GSM8K 5-shot 1319: DSpark off 0.905, on 0.905 (PR 0.9045 / 0.9022) -> accuracy OK.

Output tok/s, ours vs PR (4xMI350X):
- DSpark off: random-ids c1 147.5 (PR 155.7, -5%); ShareGPT c1 147.9 (155.9, -5%), c8 884 (952, -7%), c32 2091 (2182, -4%).
- DSpark on:  random-ids c1 508 (642, -21%, accept 3.51); ShareGPT c1 563 (335), c8 1900 (1447), c32 3427 (2242).
- ShareGPT DSpark-on is NOT comparable to PR "real text": our accept len 4.3-4.5 vs PR's implied much lower;
  PR's real-text dataset is unknown. Compare on random-ids and DSpark-off only.
- Caveats: c1/c8 use only 8/32 prompts (noisy); both servers benchmarked concurrently on GPU0-3 / GPU4-7.
- Suspects for the -5% off / -21% on gap: missing AOT top-k sort_output fusion (extra sort launch per layer),
  unpatched aiter (#5561 commit claims throughput-neutral though).

## 2026-09-24 step 2: + aiter #5561 (LDS-DMA drain), BOUND=0

GSM8K: off 0.908, on 0.911 (baseline 0.905/0.905; ~+0.5pt, within 1319-q noise ~±0.8pt).
Throughput unchanged within noise: off random-ids c1 144.6 (base 147.5), c8 897 (908), c32 2191 (2213);
ShareGPT c1 145 (148), c8 883 (884), c32 2081 (2091). On random-ids c1 518 (508), c8 2155 (1963), c32 3401 (3403).
=> #5561 does NOT explain the gap vs PR (as its commit said: throughput-neutral). Keep it applied (correctness fix).

## 2026-09-24 step 3: + aiter #5802 (bf16 SiLU route), BOUND=256 (M<256 MoE -> bf16 activation), #5561 kept

Throughput unchanged vs step 2 (BOUND=0): off random-ids c1 144.6 (TPOT 6.76 = identical), c8 901, c32 2235;
ShareGPT c1 145, c8 888, c32 2124. On random-ids c1 516, c8 2020, c32 3307.
GSM8K x3: off 0.890/0.897/0.902 (mean 0.896), on 0.911/0.892/0.905 (0.903).
BOUND=0 runs so far: off 0.905/0.908, on 0.905/0.911. Run-to-run spread at temp 0 is ~±1pt (batching
nondeterminism), so bf16 route is at best equal, maybe ~0.5-1pt lower -- not conclusive.
=> Using BOUND=0 instead of merging #5802 costs NO throughput. Keep BOUND=0 (= PR's measured config).
Gap vs PR (-3..5% off, -20% DSpark random-ids c1) is not from aiter #5561/#5802. Next suspect: sgl-kernel
0.4.7 lacking AOT top-k sort_output (needs rebuild from branch), and MI350X vs MI355X / dataset differences.

## 2026-09-24 PR env / measurement audit

PR body env (complete list): TRITON_HIP_USE_ASYNC_COPY=0, SGLANG_USE_AITER=1, SGLANG_USE_ROCM700A=0,
ROCM_QUICK_REDUCE_QUANTIZATION=NONE, AITER_BF16_FP8_MOE_BOUND=0. Ours matches; SGLANG_USE_ROCM700A: CORRECTION -- container env sets it to 1 (not default off); only affects DP-attention
(dp_attention.py get_bool_env_var) so omitting it is equivalent. Commits mention "SGLANG_OPT_HIP_*" switches but
no such names exist in the tree -> nothing to set. Model revision in PR: dba1be0a40aa45a94ad051997016db3960a90277.
Effective server args (from log) already match PR: kv_cache_dtype fp8_e4m3 (auto-set for DSV4), page_size 256
(auto), shared-experts fusion auto-disabled under EP, radix off.
PR measurement differs from ours: ignore_eos, 6 timed runs after 1 warm-up, MEDIAN, "output tok/s from FIRST TO
LAST streamed event" (= excludes TTFT), cache flushed per run, random prompt seed 42; "real text" dataset undefined.
Our bench `Output token throughput` includes TTFT. Decode-only equivalent at c1 = 1000/TPOT:
baseline off 1000/6.63 = 151 (PR 155.7, -3%); DSpark on random-ids 1000/1.80 = 556 (PR 642, -13%).

## 2026-09-24 sgl-kernel rebuilt from branch (aot, gfx950)
Built in /tmp/aot_build copy (Dockerfile recipe: pyproject_rocm.toml + `AMDGPU_TARGET=gfx950 python setup_rocm.py
install`, ~few min). GOTCHA: `setup.py install` creates an .egg that is SHADOWED by the pre-existing
site-packages/sgl_kernel dir -> must replace that dir with the egg's sgl_kernel. Verify:
`torch.ops.sgl_kernel.deepseek_v4_topk_transform_512.default._schema` contains sort_output.
Old 0.4.7 backed up at /shared_nfs/kk/dsv41/sgl_kernel_backup_0.4.7 (restore: rm -rf + cp -a sgl_kernel back).

## 2026-09-24 rebuilt sgl-kernel (sort_output) + #5561 + #5802(unused at BOUND=0), BOUND=0

GSM8K: off 0.907, on 0.902 (noise). Standard sweep unchanged vs old kernel (off random-ids c1 144.3/c8 897/c32 2191;
on random-ids c1 505/c8 2063/c32 3436) => top-k sort fusion is NOT the gap either.
PR-style bs1 (`scripts/run_pr_style_c1.sh`: warm-up + 6 runs median, flush_cache, seed 42, 1000/TPOT):
- DSpark off 147.7 (runs 147.5-147.7) vs PR 155.73 -> -5.2%
- DSpark on  621.1 (runs 613.5-625.0) vs PR 641.88 -> -3.2%
Most of the earlier "-21%" was measurement method (TTFT included + few prompts), not a real gap.
Side observation: DSpark-off c1 TPOT 6.63 ms in the unpatched baseline vs 6.76-6.77 ms in EVERY run after
#5561 was applied (~2% slower decode) -- #5561 may cost ~2% in bs1 decode; untested A/B with PR-style method.

## 2026-09-24 simulated AL "slowdown" = false alarm
SIM_AL=3.51 PR-style c1 = 378 tok/s vs real-acceptance PR-style c1 = 621 tok/s looked like sim overhead. It is not:
per-step cost is identical (~10.0-10.2 ms/DSpark step at bs1: real 520 tok/s / AL 5.30, sim 351 / 3.51, from
server-log "Decode batch" lines). The seed-42 random-ids prompt yields REAL AL 5.3-5.7 (cap 6 for block 5), so the
PR's 641.88 "random bs1" number is a high-AL case. The sweep's random-ids acc 3.50 used different (unseeded) prompts.
Rule: always compare DSpark throughput at equal AL -- read `accept len` from the server log, or use SIM_AL.
DSpark step at bs1 ~10 ms vs plain decode 6.77 ms (x1.48).

## 2026-09-24 c32 TTFT investigation (vs vLLM) -- step 1
- c32 bench is lock-step waves (fixed ISL/OSL + ignore_eos): 32x4096 = 131k prompt tokens arrive together, SGLang
  prefills in 16384-token chunks (chunked_prefill_size = max_prefill_tokens = 16384, same as vLLM
  max-num-batched-tokens) at ~36k tok/s (~0.46 s/chunk) -> P99 TTFT 3.8 s, mean 2.75 s. Mixed-chunk irrelevant
  here (nothing decoding during a wave's prefill). => TTFT gap = large-batch PREFILL THROUGHPUT, not scheduling.
  vLLM: c1 TTFT slower (256 vs 164 ms) but c32 faster (1935 vs 2754) -> vLLM scales better at big prefill batches.
- Profile of one 16384-token prefill (scripts/profile_prefill.sh, TP0, GPU busy/span 98%, ~500 ms):
  comm 30% (ncclDevKernel_Generic 111 ms 22% + aiter cross_device_reduce_2stage 38 ms), sparse attention 21%
  (_pa_decode_sparse = DECODE kernel used for prefill, 101 ms), MoE 16%, norm/mHC/quant 15%
  (hc_boundary_prefill_kernel 43 ms), GEMM 14%.
  Candidates: (a) OPUS sparse prefill for >=1024 queries (vLLM does this; inventory #20); (b) the NCCL 22% --
  pending: which collective/size, and rank-skew check (NCCL kernel time can include waiting).
- Rank skew ruled out: TP1 profile identical (NCCL 108 vs 111 ms). NCCL = ~80 per-layer all-reduces of
  16384x5120 bf16 = 168 MB each (pynccl, not in torch record_param_comms), p50 1.2 ms -> ~210 GB/s busbw =
  near 4-GPU xGMI limit => comm is BANDWIDTH-bound; only lever is less volume: quick-reduce quantization
  (recipe forces ROCM_QUICK_REDUCE_QUANTIZATION=NONE for determinism; container default INT8). Testing QR=INT8
  (launch_server.sh `QR=` knob) for c32 TTFT + GSM8K.
- OPUS is NOT a drop-in for the default HIP path: default = packed fp8 page rows (528/288 B, `aiter_sparse_decode_fwd`
  in srt/layers/attention/hip_flash_mla.py -> aiter pa_decode_sparse for prefill AND decode); OPUS
  (`pa_sparse_prefill_fp8_opus`) needs a two-pool KV (fp8 NoPE [pages,512] + bf16 RoPE [pages,64]) = the existing
  opt-in unified-KV path `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`
  (kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py:678), which the DSV4-Pro ROCm cookbook uses.
  Next: A/B that env end-to-end (c1/c32 TTFT, TPOT, GSM8K) before any porting.
