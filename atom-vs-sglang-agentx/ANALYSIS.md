Owner node: crsuse2-m2m-287 (took over from crsuse2-m2m-243 on 2026-10-14; the node that runs setup_node.sh and continues replaces this line)

# ATOM vs SGLang -- DSv4-Pro AgentX c256/c384 gap analysis (2026-10-14)

## CONTINUE HERE

**Status (2026-10-14 ~04:00, crsuse2-m2m-249):** analysis + task table done; T2/T4/T5/T3 implemented as patches; GSM8K gate:
baseline 0.937, T4+T5 0.936 (OK). **T2 HANGS** (see "T2 hang root cause"). No AgentX c256 A/B has run yet.
**Rebuild env on a new node (same image):** `bash /workspace/claude-skills/atom-vs-sglang-agentx/scripts/setup_node.sh 2>&1 | tail -30`
(sglang worktrees from patches/ on base 684fea8af3, aiter_pin 02575f1c3a + patches/aiter, aiperf venv, InferenceX-agentx 92966d7).
Do NOT trust any PID / GPU / /tmp path in this doc from another node -- re-check with hostname, rocm-smi, ps.
**Next (pending user decision, in this order):**
1. AgentX c256 A/B, 3600 s each, POINTs under runner/points/: `c256-1007` (baseline) vs `c256-1007-t4t5`
   (PYTHONPATH sglang-atomgap), then `c256-1007-t4t5-coal` (PYTHONPATH sglang-coalescer; adds --prefill-delayer-target-fill 0.7).
   Never add SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR until T2 is redesigned: it hangs.
2. T2 redesign (proposed "option 2", ATOM-like): make all DP ranks run decode at one DP-agreed padded height so one CUDA
   graph per size bakes one MegaMoE instance consistently; check whether SGLang's dp_padding_mode max_len /
   require_mlp_tp_gather can be enabled for megamoe behind a flag. Option 1 (two graph sets, pick at replay from the
   scheduler's gathered max) doubles small-bs graph memory.
**One-shot A/B driver:** `setsid nohup bash /workspace/claude-skills/atom-vs-sglang-agentx/scripts/ab_c256.sh > /shared_nfs/kk/results/DeepSeek-V4-Pro/atom-gap-impl-1014/ab_c256_$(hostname).log 2>&1 &` (~4 h for 3 arms).
**Runner (single arm):** `PYTHONPATH=<sglang worktree>/python:/sgl-workspace/aiter_pin:/sgl-workspace/mori POINT=<pt>
RESULT_DIR=/shared_nfs/kk/results/DeepSeek-V4-Pro/atom-gap-impl-1014/agentx/<pt>-<node> bash /workspace/claude-skills/atom-vs-sglang-agentx/runner/agentx_e2e.sh`
(copy to a frozen file first). GSM8K gate: `scripts/gsm8k_point.sh <tag> <point_dir> [ENV=VAL...]` (needs PYTHONPATH exported;
GSM8K_TIMEOUT env, normal run ~40 s after ready). Compare: `python3 scripts/cmp_bmk.py <dir with */bmk_*/*.json>`.
**Pass criteria:** GSM8K >= 0.93 and no hang; AgentX client exit 0, errors=0; compare tput/gpu, P90 intvty, TTFT vs baseline.

## T2 hang root cause (2026-10-14, crsuse2-m2m-249)

Bisect (GSM8K, c256 point): T4+T5 OK; T2 hangs with fixed-slot AND with compact path -> the second instance itself.
Debug log (patches/debug_mega_select.diff, env SGLANG_MEGA_DEBUG_SELECT=1) showed `get_dp_global_num_tokens()` is a
1-element LOCAL list in this config (megamoe EP a2a => require_mlp_tp_gather False; `_update_gather_batch` in
managers/scheduler_components/dp_attn.py:281 stores only `[mlp_sync_info.num_tokens]`, although the scheduler did gather
the full list). E.g. r0 g=[666] -> MTPR 8192 while r1..r7 g=[1] -> MTPR 512 => different p2p wire format => all ranks spin.
Even with a DP-agreed max, per-rank CUDA graphs bake the instance at capture by the rank's own bs, so ranks replaying
different graph sizes still disagree. ATOM avoids this: decode runs at one DP-unified running_tokens height
(model_runner.py `_padded_decode_inputs`, `running_tokens_are_unified`), one graph set, extra memory = only the small
instance's buffers.

## Implementation (2026-10-14, crsuse2-m2m-249) -- high-priority rows

Worktree `/sgl-workspace/sglang-atomgap`, branch `amd/dsv4-agentx-atom-gap` (local only, not pushed), base origin/main 684fea8af3.
aiter: `/sgl-workspace/aiter_pin` = 02575f1c3a (P8 pin) + resolved #6042 patch + Dockerfile seds/.co/torch_utils patch;
select with `PYTHONPATH=/sgl-workspace/sglang-atomgap/python:/sgl-workspace/aiter_pin:/sgl-workspace/mori`.

| T | commit | knob (default off) |
|---|---|---|
| T2 (HANGS, see root cause) | e742549603 | `SGLANG_AMD_FLYDSL_MEGA_MOE_DECODE_MTPR=512` + aiter `AITER_MEGA_FIXED_SLOT_MAX_MTPR=1023` |
| T4 | 6ea22333c0 | `SGLANG_AITER_MEGA_RANK_SYNC_ALLOW_OVERLAP=1` |
| T5 | e11d87febe | `SGLANG_EPLB_MAX_REBALANCES=4` |
| T3 | 8fc256b57b (branch `amd/dsv4-prefill-coalescer`, worktree `/sgl-workspace/sglang-coalescer`, on top of T2/T4/T5) | `--enable-prefill-delayer --prefill-delayer-target-fill 0.7` (unit tests: 5 new + 64 existing pass, CPU) |

Tests: `scripts/gsm8k_point.sh` on point `runner/points/c256-1007` (= CI 1007 c256 argv/env, SIMULATE_ACC dropped).
Results append to /shared_nfs/kk/results/DeepSeek-V4-Pro/atom-gap-impl-1014/gsm8k/results.md.
GSM8K results (append-only): /shared_nfs/kk/results/DeepSeek-V4-Pro/atom-gap-impl-1014/gsm8k/results.md (gap_all,
bis_t2_fixed, bis_t2_compact rows with empty accuracy = HANG, killed by timeout). AgentX: .../agentx/<point>-<node>/.

## CI measured data (pulled 2026-10-14 with gh; node crsuse2-m2m-249)

Artifacts: /shared_nfs/kk/results/DeepSeek-V4-Pro/atom-vs-sglang-agentx-1014/ci/{atom,sglang}/ (bmk json, c256 server
logs). Compare script: `python3 scripts/cmp_bmk.py <ci dir>`. 3600 s per cell, 8x MI355X.

| metric | ATOM c256 | SGL c256 | ATOM c384 | SGL c384 |
|---|---|---|---|---|
| tput tok/s/gpu | 73,232 | 67,295 (-8.1%) | 84,180 | 74,093 (-12.0%) |
| P90 intvty | 32.5 | 27.5 | 19.1 | **22.4 (SGL better)** |
| ITL mean ms | 25.9 | 32.5 | 46.1 | 40.7 |
| TTFT mean / p90 s | 21.3 / 39.1 | 20.8 / 49.4 | 21.7 / 39.9 | 40.0 / 89.7 |
| requests done | 17,079 | 15,926 | 21,162 | 18,201 |
| GPU prefix hit (server) | 0.928 | 0.959 | 0.865 | 0.954 |

Findings:
- Spec acceptance is forced on both (InferenceX golden curve): ATOM "Forced speculative acceptance ON, mean 3.77",
  SGLang log accept len mean 3.771. So ITL gaps are pure step-time gaps.
- At c384 SGLang has the better P90 intvty; ATOM wins there on throughput and TTFT (2x better mean TTFT).
- SGLang's on-GPU prefix hit is HIGHER than ATOM's, and computed (non-cached) prefill tokens over the whole run are
  about equal (~0.165B each; ATOM figure estimated from per-engine "Avg prompt throughput" x 11 s interval). ATOM
  finished the run's logged work in 5,342 s vs SGLang 6,526 s. => the gap is SPEED (prefill + decode step), not cache
  efficiency. T1 (session affinity) is downgraded to L.
- SGLang c256 effective per-request ITL is flat at 32-35 ms for decode batch 0..31 per rank (from log gen throughput /
  running-req). Flat ITL vs batch = fixed per-step cost dominates: DP lockstep with prefill on some rank (~21k
  prefill batches, one somewhere every ~0.3 s), MegaMoE 8192-MTPR compact path, host overhead. Supports T2, T3, T4,
  T6, T8 as the main levers.

## Headline

Both sides run the same core kernels: aiter MegaMoEV2 a8w4, asm `mla_decode_fwd_v4_nm`, OPUS sparse prefill,
FlyDSL FP4 indexer, fp8 two-pool KV. The gap is mostly **how they are sized/scheduled**: session stickiness, a
decode-sized MegaMoE instance, prefill coalescing, shared-expert overlap, EPLB churn. Attention kernels are near parity.

Verified in code (2026-10-14): sglang-router `cache_aware` never reads the routing key (only `manual.rs` /
`consistent_hashing.rs` do); `SGLANG_AITER_MEGA_RANK_SYNC=1` makes `_should_overlap_shared_and_routed` return
False (`layers/moe/mega_moe_flydsl.py`); P6 mxscale is whitelisted and off by default.

## Task table

Type: R = recipe/env only, C = SGLang code, K = kernel. Impact: H/M/L on c256/c384 (tput/GPU, P90 intvty, TTFT).

| T | Item | ATOM | SGLang main | Impact | Type | Our PR | Next action |
|---|------|------|-------------|--------|------|--------|-------------|
| T1 | DP session affinity | `ATOM_DP_SESSION_AFFINITY=1`, routes by x-dynamo-session-id + parent-session-id (agent fan-out to one rank); session never moves after turn 1 | router `cache_aware` ignores `X-SMG-Routing-Key`; past `balance-abs-threshold 32` turns go to least-loaded rank and lose on-chip prefix | L (CI data: SGLang GPU hit already higher, 0.959 vs 0.928) | R (+C for parent-id lineage) | none | A/B router `policy: manual` (min_load assignment) at c256; then parent-session lineage in router |
| T2 | Decode-sized MegaMoE instance | #2209/#2463: `ATOM_MEGA_DECODE_MTPR` 512 (c256) / 1024 (c384) + `AITER_MEGA_FIXED_SLOT_MAX_MTPR` -> fixed-slot path | one MTPR-8192 instance for all passes -> compact path in decode | H (P90, tput); aiter #6127: 436 vs 589 us/layer @224 tok/rank | C | P8 (aiter pin 02575f1c3a has #6127) gives capability only | build 2nd small instance, select for decode/verify/draft; needs P8 |
| T3 | Prefill coalescer (fill target) | #1611 cross-DP coalescer, #2238 TP coalesce; `ATOM_ENABLE_PREFILL_DELAYER=1` | PrefillDelayer aligns ranks + watermark only | M-H (tput, TTFT) | C | none; port exists on branch `feat/prefill-coalescer-sum-fill-target` (skill sglang-prefill-coalescer) | A/B the branch on this recipe at c256/c384, then upstream |
| T4 | Shared expert overlap with MegaMoE | overlaps shared expert on alt stream up to 1024 tokens | `SGLANG_AITER_MEGA_RANK_SYNC=1` (recipe) disables overlap -> serial | M-H (P90) | C | none | find why rank_sync blocks overlap; allow overlap under rank_sync |
| T5 | EPLB rebalance cap | #2482 `ATOM_EPLB_MAX_REBALANCES` 4 (c256) / 5 (c384) | rebalances every 200 iters all run, all layers at once; known deadlock risk | M (P90, TTFT) | C (small) | none | add max-rebalances knob (+ optional balancedness gate) |
| T6 | Prefill chunk per rank | 16384/rank (`max-num-batched-tokens`, `attn-prefill-chunk-size`) | 65536/dp8 = 8192/rank, tied to MegaMoE MTPR 8192 | M (TTFT) | R | none | try chunked-prefill 131072 + MTPR 16384 (check mori heap / memory) |
| T7 | DP load-balance metric | `dp-load-balance least_tokens`, `ATOM_DP_LB_REQ_EQUIV=512`, charges only per-turn new tokens | `load-balance-method total_requests`; router min_load also counts requests | M | C | none | token-weighted LB in scheduler/router; do after T1 |
| T8 | Prefill/decode interleave | `ATOM_PREFILL_DECODE_INTERVAL` 40 (c256) / 20 (c384) | `prefill-decode-interval` 20 / 15 | M (P90, guess) | R | none | sweep 20/40 at c256 |
| T9 | KV offload thresholds | LMCache lmcache_mp; `OFFLOAD_MIN_LOAD/SAVE_TOKENS=8192`, lookups off scheduler path (#2454) | HiCache write_through every node; loads hits down to ~10 tokens | M (guess) | R/C | none | add min load/save token threshold to HiCache; measure host-load stalls |
| T10 | e8m0 scale on dense FP8 linears | `blockscale_e8m0_scale` default on for gfx950 (all dense FP8 linears) | none on main | M | R+C | **P6 #44007** partial: whitelist, `SGLANG_DSV4_DENSE_MXSCALE` default off | turn on in recipe, widen whitelist to all pow2-scale shapes |
| T11 | FlyDSL fused compressor | `ATOM_FUSED_COMPRESS_USE_FLYDSL=auto` for C4 main / indexer / HCA, 2-3x vs Triton at N>=4096 | Triton only | M (TTFT, tput) | K+C | none | port FlyDSL compressor (check aiter/flydsl availability) |
| T12 | aiter #6127 MegaMoE knobs | combine mask, `S1_LAUNCH_HANDSHAKE=0`, persistent Stage2 (`CFG_OVERRIDE`) | old pin e7d2453 lacks them | L-M | R+C | **P8 #44035** (pin) | after P8: wire and A/B each knob; pairs with T2 |
| T13 | FP8 decode split plan + HCA persistent kernel | #2462 aiter-latency-model split plan; `ATOM_V4_HCA_PERSIST=1` | own split heuristic; no HCA persist | L-M (P90 at long ctx) | C | **P1 #44016** full | land P1 |
| T14 | Host metadata path | #2101 device indptrs, #2399 metadata reuse, #2383 block-table appends, #2368 deferred-id kernel | partial; DSv4 metadata builders use `.tolist()`/`.item()` -- not verified whether on steady-state decode | L-M (P90, guess) | C | none | host-side trace of one decode step at c256 first |
| T15 | Inductor compile | `level: 3` + `cudagraph-mode FULL` (model and DSpark draft) | no DSv4 torch.compile | L-M (guess) | C | none | profile glue-kernel time first; only then consider compile |
| T16 | Verify SWA ring write fused into norm/rope/quant | yes | no | L | C | **P2 #44003** full | land P2 |
| T17 | C4 indexer q_lora quant reuse | yes | no | L | C | **P4 #44005** full | land P4 |

PRs not mapped to an ATOM gap: P3 #44004 (MoE small-batch sort: helps TP arms, MegaMoE has its own dispatch, so it
does not apply at c256/c384); P5 #44006 (FP4 indexer rowgroup) and P7 #44008 (in-tree GEMM tune) have no ATOM
counterpart identified -- own perf work, not gap closing.

Checked, not a gap at c256/c384: #2271 (gfx1250 only), #2191/#2120/#2170/#2215/#2216/#2348 (MLA only),
#2367/#2220/#2148 (EAGLE/MTP drafters, not DSpark), 0873517 (V4.1/gfx1250), #2347/#2018/#2182 (mori-v2/gfx1250),
#2114 (TP only), `ATOM_MOE_GU_ITLV`, pad-row masking, fused EPLB map/record, `AITER_BF16_FP8_MOE_BOUND`,
act+quant fusion. CUDA graph buckets: parity (SGLang finer).

## Suggested order

1. Recipe-only A/Bs at c256 (cheap, one var each): T8, T6, T10 (needs P6); T1 last.
2. Code, high impact: T2 (needs P8), T4, T5, then T3 A/B.
3. Medium: T7, T9, T11; trace-first: T14, T15.
