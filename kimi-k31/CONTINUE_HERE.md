# Kimi-K3.1 — CONTINUE HERE

Owner node: mi355-4. Last updated 2026-11-02 04:30.

**Status:** T0-T3 done; optimization loop in `OPT_TRACKER.md`. GPUs free, no server running.
Four PRs open on kevin-mii/sglang-k31-amd against `dev_amd` (all behind default-off flags):

```text
#7  k31/kda-verify-fp32-ring      5c  HIP KDA verify with fp32 ReplaySSM ring
    SGLANG_ROCM_KDA_VERIFY_HIP_FP32_RING=1          decode TPOT -2.2..-2.8%
#8  k31/prefill-row-sharded-tail  1   row-sharded latent-MoE tail + FFN read (sglang#44981 port)
    SGLANG_ROCM_K3_ROW_SHARDED_TAIL_MIN_TOKENS=4096  prefill TTFT -3..-7%, c48 TPOT -3%
#9  k31/sparse-mla-scale-post     2   fp8_ds_mla group scales after the bf16 dots
    SGLANG_ROCM_SPARSE_MLA_SCALE_POST=1             sparse MLA -12.5%, prefill TTFT ~-2%
#10 k31/sparse-mla-prefill-hip   1   HIP sparse MLA prefill kernel (fp8_ds_mla, H <= 16)
    SGLANG_ROCM_SPARSE_MLA_PREFILL_HIP=1            sparse MLA 5.74 -> 2.47 ms, prefill TTFT -7.5..-9%
```
#9 and #10 touch the same long-prefill path: with #10 on, #9 only matters where #10 declines.
Not yet measured: #7 + #8 + #10 on together (different code; expect additive).

**Next: decode mono kernel, new session (user, 2026-11-01): read `MONO_KERNEL.md` CONTINUE HERE.**
S1 (c1) done in the real config: TPOT 4.96 -> 4.49 ms; code on fork kkHuang-amd/sgl-amd-mono `k31/mono-s1`.
2026-11-01: gsm8k1319 gate PASSES (mono on 0.9757 vs off 0.9765, discordant 2/3). P2a step 1 (tile GEMM
v0) measured; aiter hgemm with HTI off at 8 waves ties/beats tgemm (in_proj 27.9 vs 30.1 us), 16 waves no
better and spills PR #13's K2 -> GEMMs go inside the 512-thread launch (`MONO_KERNEL.md` section 10.4).
2026-11-01 13:00: P2a step 2 column-slice collectives done (bit-exact, 8 x 10^5 stress launches clean, tagged
format default; section 10.7); next is wiring them to the tile GEMMs.
2026-11-01 14:30: fused tile GEMM -> RS (o_proj) one launch, -7..-28% vs hipBLASLt + RCCL (section 10.8);
next is AG -> in_proj.
2026-11-01 15:30: fused AG -> in_proj one launch, -14..-32% vs RCCL + F.linear (section 10.9); next: parity
scheme, then the planner.
2026-11-01 17:30: parity by launch index (epoch hazard reproduced, fixed, stressed; section 10.10); next: planner.
2026-11-01 19:00: planner v0 + post-attention chain launch (o_proj -> RS -> RMSNorm -> AG -> MoE front),
-17..-46% vs the standard path (section 10.11); next: chain timeline, then T1 / T2 / T3.
2026-11-01 20:00: chain timeline: stages serial, each near standalone cost; front bound by tile-GEMM speed
(section 10.12); next: T1 / T2 backwards, tile GEMM speed.
2026-11-01 21:00: AttnRes seam on column-sharded state, one scalar round, in the chain (section 10.13);
next: T1 (MoE tail), T2, T3.
2026-11-01 22:00: pre-attention chain (MoE tail + AttnRes + in_proj) one launch (section 10.14); next: T3 KDA.
2026-11-01 23:00: KDA stage in the pre-attention launch, correct but +59 us at m 128 (K1 task split); next: wide
KDA task a (request, head) (section 10.15).
2026-11-02 00:00: wide KDA task: m 128 pre_attn + KDA 86 us (KDA +21, standard ~24) (section 10.16); next: one
KDA-layer launch (pre_attn + KDA + post_attn).
2026-11-02 01:00: ONE wide-row KDA layer launch T1..T6 works: 82 / 136 / 272 us at m 16 / 128 / 384 (section
10.17); next: T6 tail (router top-16, latent quant, shared down), then SGLang integration + endpoint A/B.
2026-11-02 02:00: T6 tail in the layer launch (route + top-16 + shared down); layer 105 / 167 / 335 us at m 16 / 128
/ 384 (section 10.18); next: DSpark captures, then SGLang integration + endpoint A/B.
2026-11-02 03:00: DSpark capture from the wide layer launch (section 10.19); next: SGLang integration + endpoint
A/B + gsm8k gate.
2026-11-02 04:00: REAL-MODEL check of the wide layer launch passes at rows 32 / 128 (layer 8, t1 off; section
10.20); next: T1 + shared expert on real data, then integration.
2026-11-02 04:30: real-model check covers the whole launch incl. T1 and the shared expert (section 10.20); next:
SGLang integration (serve wide verify steps with the launch) + endpoint A/B + gsm8k.

**Previous section (`SPARSE_MLA_PREFILL.md`):** R1 HIP kernel passes E2E behind
`SGLANG_ROCM_SPARSE_MLA_PREFILL_HIP=1`: kernel 2.5 ms vs Triton 5.0 (seq 16384), prefill TTFT
-5.5..-6.6% vs SCALE_POST, greedy gsm8k 0.985 with 191/200 identical to control. PR #10
(k31/sparse-mla-prefill-hip @ 91ca03666b, pushed 2026-10-30).
Not for decode: decode/verify sparse MLA is opus MXFP8, 1.42 ms = 3.0% of the 47.85 ms VERIFY
step, with DCP-gathered heads (> 16) and partial outputs this kernel does not do.
Kernel is now compute (VALU) bound: 64 cvt floor + ~110 other VALU per wave-tile.
Repro: `python scripts/sparse_mla_hip_microbench.py 16384 clustered,random 2` (from /tmp).

**Decode, #3 bf16 GEMM** (row 3 in `OPT_TRACKER.md`): custom skinny-M GEMM streams at the same ~4.3 TB/s as
tgemm in the real GEMM and pays ~3 us more fixed cost (31 vs 28 us for 6016x7168). Recommended to stop #3;
branch `k31/decode-skinny-gemm` kept uncommitted. Next candidate: 12-G4 MoE prologue.

**Decode, 12-G4 (3) split stage-1 mxfp8 quant: REJECTED E2E, discarded by the user (2026-10-31).** Branch
`k31/moe-mxfp8-split-quant` deleted; the clean diff is kept at
`/shared_nfs/kimi-k31/results/moe_split_quant_wip_20261031/split_quant_clean_rejected.diff`. The earlier capture-memory blowup was MY bug: the helper was inserted
between `@functools.cache` and `def apply_aiter_small_moe_sort_patch`, so the cache moved onto the helper (kept
every call's tensors alive + could return stale results) and the patch lost its once-only guard (92 nested
wrappers). Lesson: never insert code directly above a `def` without checking for decorators. After the fix,
capture mem is normal (prefill graph 5.69 GB) but A/B (results/ab/i12g4_split_quant2_20261031) vs control:
TPOT c4 7.72/7.78, c16 15.62/15.58, c48 31.42/31.34, prefill 8k 526.7/528.6 -> noise; the ~4 us/layer microbench
gain does not show.
Control on current dev_amd (results/ab/i12g4_control_20261031): TPOT c4 7.78 / c16 15.58 / c48 31.34, prefill 8k 528.6.

Re-estimates (2026-10-31): #5d with PR #7 on leaves verify 16.1 + f_b 4.2 + norm 3.9 us/layer x69; small fixes
(compact conv window, LDS pipelining) ~0.1-0.14 ms/step each, f_b/norm fusion needs MFMA f_b or no V split
(~0.3-0.5 ms), only a chunked-MFMA redesign reaches ~0.8 ms. #6 attn_res `_agg_kernel` (191 x 8.3 us = 1.58 ms):
one CTA per token = 128 CTAs at verify, ~16 GB/s per CU; three full-H reductions per token (row RMS, score dot,
out RMSNorm) make an H split need cross-CTA sync -> ~0.8 ms if it reached ~4 us, redesign-level work.

**Other open items (`OPT_TRACKER.md`):** 12-G4 MoE prologue (single-CTA sort rejected;
multi-CTA one-launch sort open), 7 decode waits for whole-batch prefill, 8b bonus-row-only
logits, 11 hipBLASLt fallbacks, 3/9 bf16 GEMM (+fp8 experimental). Upstream PR overlap with
these items is mapped in `AITER_PRS.md` ("SGLang K3-on-gfx950 PR queues").

**Code (node-local, mi355-4):**
```text
/sgl-workspace/sglang-k31-amd   on k31/sparse-mla-prefill-hip @ 91ca03666b (off origin/dev_amd,
                                  pushed, PR #10): R1 HIP kernel + flag + dispatch + unit test
                                  (24 pass). Old k31/sparse-mla-splitk-long has a stale stash
                                  (pre-split copy of the same edits) -> droppable.
/sgl-workspace/aiter-k31        k31/main-20261030 @ f0b96f49e (aiter main, gfx950 build)
flydsl 0.4.1, sgl-eval 0.1.3
Local-only branches: k31/kda-verify-fused-fb-wip, k31/moe-sort-one-launch, k31/sparse-mla-fp8-q
```
Golden trees untouched: /sgl-workspace/sglang, /sgl-workspace/aiter. Editable installs point
at the k31 trees.

**Always first:** `source /workspace/claude-skills/kimi-k31/env.sh` (`/etc/bash.bashrc` puts
the old /sgl-workspace/aiter on PYTHONPATH). Run tools from `/tmp` (cwd inside
`sglang/benchmark` shadows HF `datasets`). Kill servers by the python PID in
`logs/server_<tag>.pid`, never by `$!` of a wrapper; never `pkill -f` a pattern that is in
your own command line. No GPU microbench while a server is up.

## Scripts (`/workspace/claude-skills/kimi-k31/scripts/`)

```text
launch_server.sh <tag> [args]    A/B base server (DCP8 + DSPARK, sim acc 3.84, no radix);
                                 REPLAYSSM=0 / SIM=0 env toggles; extra env passes through
ab_bench.sh <tag>                decode 8k/1k c4/c16/c48 + prefill 8k  -> results/ab/<tag>
prefill_sweep.sh <tag>           prefill TTFT 4k..128k c1              -> results/prefill/<tag>
capture_trace.sh <tag> <log> ... late-prefill + 2 s decode trace, 1 file/rank -> traces/<tag>
trace_breakdown.py <TP0 trace>   per-step op-class breakdown (EXTEND_BODY = prefill graphs)
trace_principles.py / trace_layer_seq.py   principles pass, per-layer kernel order
gsm8k_gate.sh <tag> <log>        TEMP=0 for the low-noise greedy gate  -> results/acc/<tag>
*_microbench.py                  KDA verify, MoE prologue, sparse MLA prefill (1 GPU);
                                 sparse_mla_splitk_microbench.py = C3 split sweep + rel L2
```

## Baselines (mi355-4, 2026-10-30, `launch_server.sh` config, all flags off)

```text
decode  TPOT c4 7.94 / c16 16.18 / c48 31.94 ms; VERIFY step 47.85 ms (c16 trace)
prefill TTFT 4k 301 / 8k 528 / 16k 990 / 32k 1988 / 128k 8237 ms
gsm8k200 greedy, real draft: 0.98, accept len req-weighted 5.105; run-to-run band
  188-190/200 identical generations; judge T=1.0 runs only with long-tail checks
```

## Cookbook recipe (for accuracy / AgentX-style runs; differs from the A/B config by
HiCache + radix on and no simulated acceptance)

```bash
source /workspace/claude-skills/kimi-k31/env.sh
SGLANG_USE_AITER=1 SGLANG_AITER_K3_OPT=1 AITER_FLYDSL_FORCE=1 AITER_SITUV2_A8W4=1 \
SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1 HSA_NO_SCRATCH_RECLAIM=1 HSA_USERPTR_FOR_PAGED_MEM=0 \
SGLANG_ROCM_RUNTIME_RESERVE_GB=10 SGLANG_ENABLE_KIMI_K31_PREFILL_BCG=1 \
PYTORCH_HIP_ALLOC_CONF=garbage_collection_threshold:0.95 \
python -m sglang.launch_server --model-path $K31_MODEL --trust-remote-code \
  --tp-size 8 --dcp-size 8 --dcp-comm-backend a2a \
  --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 \
  --mamba-full-memory-ratio 0.65 --mamba-radix-cache-strategy extra_buffer_lazy \
  --max-running-requests 48 --cuda-graph-max-bs-decode 48 \
  --speculative-algorithm DSPARK --speculative-draft-model-path /shared_nfs/Kimi-K3-DSpark \
  --speculative-draft-attention-backend triton --enable-linear-replayssm-spec \
  --enable-hierarchical-cache --hicache-ratio 3.4 \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 256 512 768 1024 1536 2048 3072 4096 \
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3
```
Weights load ~33 min cold, ~4 min with a warm page cache.

## Results index

Detailed rows per item live in `OPT_TRACKER.md`; trace analysis in
`TRACE_c16_isl8k_20261030.md`; artifacts under `/shared_nfs/kimi-k31/{logs,results,traces}`.
