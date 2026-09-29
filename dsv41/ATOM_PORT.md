# Porting ATOM V4.1 optimizations into SGLang (rolao opt-branch)

Full pre-condensation history: /shared_nfs/kk/dsv41/doc_backup_20260929/ATOM_PORT.md.

## CONTINUE HERE

**Status (2026-09-29):** best config pushed = rolao/dsv41/opt-branch **026da361c0** (T2 + requant fix + draft
metadata in graph) + `--enforce-shared-experts-fusion` + local tuned FMoE CSV. Best sweep done (below).
Remaining gap to ATOM = low-concurrency P90 (c1/c2/c8 -13/-14/-17%).
**RUNNING (started 2026-09-29 07:49 +08):** PDI sweep RUNS "8:32 8:4 2:32 2:4 1:4", PREFIX atomport_best_pdi,
chain PID 554036, ETA ~13:15 +08, progress `tail -3 /shared_nfs/kk/dsv41/agentx/series.txt`.
Done so far: c8 PDI32 28,643.3 / P90 212.3 (vs PDI16 +0.3% / +6.5%; TTFT p50 0.45 -> 0.54 s, p90 1.42 -> 1.78 s;
vs ATOM -2.1% / -11.5%).
**Next:** finish PDI sweep -> if larger PDI keeps helping P90, try c8/c2 PDI 64; then capture an ATOM trace under
AgentX-like load (c1/c8) and compare step-by-step with ours (the proxy host gap is only ~0.85 ms/step, so most of
the AgentX c1 gap is NOT host gap).
**Files:** tree `/sgl-workspace/sglang-rolao-opt` (branch atomport-mxfp8-producers, tracks rolao/dsv41/opt-branch;
worktree also holds the uncommitted SGLANG_HIP_SPEC_EVENT_WAIT experiment in managers/overlap_utils.py -- do not ship),
aiter `/sgl-workspace/aiter-5750` (+ untracked `aiter/configs/model_configs/dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv`,
+ local tuner patch), flydsl `/sgl-workspace/pydeps-flydsl-0341`, ATOM reference `/workspace/atom-survey` @ 4685e3cf.
**Repro (best config, c1 server; drop SERVER_ONLY to run AgentX; EVAL_ONLY=true for GSM8K):**
```bash
cd /shared_nfs/kk/dsv41/agentx && PYTHONPATH=/sgl-workspace/pydeps-flydsl-0341:/sgl-workspace/aiter-5750:/sgl-workspace/mori \
SRC=/sgl-workspace/sglang-rolao-opt/python SGLANG_OPT_HIP_OPUS_SPARSE_PREFILL=1 \
EXTRA_ARGS="--fp8-gemm-backend aiter --enforce-shared-experts-fusion" \
SERVER_ONLY=1 OPUS=0 TP=2 EP_SIZE=1 GPUS=4,5 CONC=1 PREFILL_DECODE_INTERVAL=16 TAG=<tag> \
setsid nohup bash /workspace/claude-skills/dsv41/scripts/agentx_colleague_run.sh > <tag>.nohup 2>&1 < /dev/null &
```
Series: `SCRIPT=agentx_colleague_run.sh RUNS="conc:pdi ..." PREFIX=<p> bash scripts/agentx_series.sh` (same env).
**Pass criteria (project):** AgentX TP2 c1-c8 P90 within 5% of ATOM (337/324/-/240), TTT not below ATOM at c1-c16,
no regression at c32/c64 (89,715 / 111,506), GSM8K >= 0.89.
**Workflow per change:** unit parity -> GSM8K 1319 x3 (EVAL_ONLY, real acceptance) -> step span -> AgentX.
GPU 4,5 = servers, GPU 6 = unit tests. Never force-push; fetch before push (gh account kkHuang-amd has write access).

## Results

Commits on rolao/dsv41/opt-branch (all pushed, fast-forward, author wunhuang):
- **8c67e0bd51** T2: MXFP8 operands emitted by their producers (knobs SGLANG_HIP_SHARED_ACT_MXFP8 / _WO_A_MXFP8 /
  _FFN_NORM_MXFP8, EnvBool(_default_hip)). 1376 -> 1256 kernels/step, GPU span 9.458 -> 8.879 ms (-6.1%), bit-exact
  unit parity, GSM8K 0.904.
- **ba5eb0f439** shared-expert MXFP4 requant: aiter dynamic_mxfp4_quant scale + round-to-nearest-even
  (fp8_utils.quantize_block_fp8_weight_to_mxfp4; test_ties_round_to_even). Makes `--enforce-shared-experts-fusion`
  accurate: GSM8K 0.891 -> 0.906.
- **026da361c0** DSpark draft block metadata built inside the draft CUDA graph (knob
  SGLANG_HIP_DSPARK_DRAFT_RAW_METADATA; DSV4RawDSparkDraftMetadata in deepseek_v4_backend_hip_radix.py, same pattern
  as DSV4RawVerifyMetadata). Host critical path -~0.2 ms/step.

AgentX TP2 c1 PDI16 progression (TTT / P90, 3600 s, single runs): rolao all-opts 9,818.6 / 267.5 -> T2 9,977.9 /
277.5 -> + fusion (fixed requant) 10,036.3 / 282.5 -> + tuned FMoE 10,269.1 / 287.5 -> + draft metadata in graph
10,306.1 / 292.8 (ATOM 10,820.2 / 337).
**Best-config sweep 2026-09-29 (rows "BEST sweep" in results/agentx.md, all 0 faults):** c1 10,306.1/292.8,
c2 10,476.9/279.6, c4 15,680.7/267.0, c8 28,552.6/199.3, c16 51,219.5/137.8, c32 PDI16 90,876.4/102.0,
c32 PDI4 93,855.8/68.7, c64 PDI16 90,671.3/78.8, c64 PDI4 117,189.7/39.7. vs rolao all-opts: TTT +3.8..+7.6%,
P90 +3.1..+10.3% at every point. vs ATOM: c1/c2/c8 TTT -4.8/-7.0/-2.4%, P90 -13.1/-13.7/-17.0%; c16 -5.1%/+7.7%;
c32 PDI4 +3.1%/+3.5%; c64 PDI4 +14.5%/+71.9%.
GSM8K 1319 5-shot with the best config (CONC=32 eval server): 0.901-0.908 over 6 runs.

## Closed work (summaries)

- **Gap attribution (P0, 2026-09-27):** c1 decode is context-insensitive (ctx 2k vs 64k equal) -> fixed per-step
  cost dominates; indexer only 0.09 ms/step. Small kernels cost 4.3-7.2 us each regardless of work -> lever =
  launch count. mHC seam is NOT a gap vs ATOM (ours 2 kernels/seam, ATOM >= 3). Index-build fusion not worth it.
- **T2 (MXFP8 producer folds):** (c) shared SwiGLU -> Triton silu_and_mul_clamp fp8 grid (1376 -> 1335);
  (a) wo_a split-K reduce EMIT_FP8 epilogue (-> 1296, span 9.099); (b) FFN rmsnorm_sinkhorn emit_fp8 for the shared
  gate_up (-> 1256, 8.879). Parity scripts: test_shared_act_mxfp8.py, test_wo_a_mxfp8.py, test_ffn_norm_mxfp8.py.
- **T3 (shared + routed add):** no cheap fold (aiter fused_moe output= overwrites, ar_ll has no input add);
  superseded by shared-expert fusion, which removes the add.
- **Shared-expert fusion:** upstream #32340 path (FP8 shared expert requantized to MXFP4 at load, appended as
  routed slot -> top-7; loader FusedMoE._maybe_load_fp8_shared_expert_as_fp4, layer.py:891). Saves 0.25 ms/step
  (shared gate_up/clamp/down/add gone; moe1/moe2 slightly longer). Accuracy drop 1.3 pt was a REAL BUG: the old
  MXFP4QuantizeUtil.cast_fp4 rounded e2m1 ties toward zero; FP8 sources hit midpoints often -> every matrix gain
  0.949, shared MLP output gain 0.850 (scripts/shared_expert_requant_check.py, all 43 shared experts). Fix -> MLP
  gain 0.937 / rel err 0.190 (residual = inherent MXFP4). Same bug hits DSV4-Pro fusion (ue8m0 [128,128] scales),
  accuracy only: Pro MLP gain 0.85 -> 0.94 (upstream reported no GSM8K drop on Pro). FHMoE (keep shared FP8, #35074)
  no longer needed for accuracy; possible follow-up: per-layer gain fold into the shared topk weight.
- **FMoE tuning of the fused shapes (2026-09-28):** 385/7 tokens 1-4096 and DSpark draft 129/4 tokens 1-512
  (5120/1152, a8w4 per_1x32) had no tuned config. Tuned 23/23 shapes on GPU 6 (all ksplit 0, err 0.0%), CSV
  /shared_nfs/kk/dsv41/moe_tune/ -> deployed into aiter-5750 model_configs/ (backup
  patches/aiter_local_dsv41_tp2_sef_fp8fp4_tuned_fmoe.csv). Tuner needed a local fix: generate_v2_stage1_data
  quantized E x 2I x H weights on CPU per token (42 min without progress) -> `with torch.device(device):`
  (patches/aiter_local_moe_tune_gpu_datagen.patch). Span 1176 -> 1135 kernels, 8.701 -> 8.389 ms; GSM8K unchanged.
  Caveat: unfused 384/6 + 128/3 are still untuned, so fusion-vs-unfused A/Bs slightly favour fusion.
- **Host gap (2026-09-28):** SGLang HIP host-syncs the spec publish event (overlap_utils.resolve_seq_lens_cpu,
  `publish_ready.synchronize()` workaround for #26672). Findings: (1) ATOM also syncs on step N's accept counts before
  preparing N+1; its DSpark draft runs after verify so the sync overlaps the draft. Our dspark_worker_v2 already
  publishes right after accept (dspark_worker_v2.py:912), before the ~0.3 ms commit_hidden tail -> reordering gives
  ~nothing; the DSpark draft here has no MoE kernels and is cheap. (2) Real c1 idle is ~0.85 ms/step (9.2 ms step vs
  8.39 ms GPU); the 1.6-1.9 ms trace gap was profiler-inflated. Event.synchronize wake-up is ~13 us. (3) py-spy per
  step: sync wait 5.51 ms, critical host path after the sync ~0.55 ms (0.43 = draft metadata) -> fixed by
  026da361c0 (proxy c1 384.5 -> 393.6 tok/s, P90 ~328 -> ~336; AgentX c1 +0.4% / +1.8%).
  SGLANG_HIP_SPEC_EVENT_WAIT=1 (device-side wait; host really runs ahead because our backend's
  needs_cpu_seq_lens = SGLANG_OPT_USE_ONLINE_COMPRESS = False) is ~10% SLOWER on the GPU; cause unknown
  (plain graph-after-event-wait microbench is not slower). Hypothesis: TP2 rank skew makes ar_ll spin.
- **PDI survey (2026-09-29):** PDI = scheduler iterations to skip prefill after each prefill batch/chunk
  (scheduler.py _should_defer_prefill; counts iterations even with no decode running -> c1 ~unaffected).
  Smaller PDI: prefill sooner (TTT/TTFT up), decode interrupted more (P90 down). Our c32/c64 16 -> 4: TTT
  +3%/+29%, P90 -33%/-50%. Recipes: 16 (ours c<64, B200/H200/GB300), 4 (ours TP2 c64, B300 c32+), 24/20
  (InferenceX MI355X DSV4 SGLang), ATOM DSV4 script ATOM_PREFILL_DECODE_INTERVAL=10.

## Open candidates

- ATOM trace under AgentX-like load (next, user-approved).
- (B) Explain/fix the EVENT_WAIT=1 GPU slowdown: rocprofv3 --kernel-trace on both ranks, EVENT_WAIT 0 vs 1,
  compare per-kernel durations (esp. ar_ll) -- ceiling ~0.85 ms/step at c1.
- Per-step GPU breakdown (fusion+tuned, c1, ms/step): MoE 2.25 (moe1 0.92, moe2 0.50, moe_sorting 0.49,
  topk_gating 0.27, router gemv 0.22, _fused_append_shared_experts 0.17, fused_mx_quant_moe_sort 0.17), gemm 1.83,
  sparse attn 1.22 (sparse_mla 0.76 + decode_reduce 0.18 + reduce_partials 0.17), mHC 0.55, ar_ll 0.50, wo_a
  split-K 0.35 + 0.17. Cheapest: fold _fused_append_shared_experts into topk (~-0.17 ms); larger: ATOM #2377 gluon
  fused sparse decode (removes the 2 reduces, ~-0.3 ms); wo_a split-K retune.
- Original plan items not done: P1 #2392 row-bounded block maxima (no c1 value), P2 DualRMSNormMXFP8 /
  rope_quant_window, P3 DSpark draft kernelization (ATOM 21.73 -> 6.67 launches/draft step), P4 gluon sparse
  decode, P5 Engram side-stream overlap (our TP2 uses the host table), P6 recipe knobs (mem 0.9, cuda-graph-max-bs
  128, chunk 16384).

## Measurement methods and traps

- **Proxy:** scripts/atomport_proxy_bench.py (random ids, c1, ctx 2k, OSL 1024, 3 reps) for relative A/Bs only;
  proxy P90 is higher than AgentX P90. 1-2% effects need the span metric.
- **Span:** proxy with `--profile-dir`, then scripts/atomport_step_spans.py (kernels/step, GPU span, host gap);
  scripts/atomport_step_sequence.py prints one step's kernel sequence; atomport_trace_breakdown.py groups by module.
  torch profiler: GPU activities only, no stack/shapes, few steps -- CPU+GPU hung both TP ranks in RCCL. Even GPU-only
  profiling occasionally hangs (NCCL watchdog 1 s after /start_profile); retry on a fresh server. The profiled host
  gap is inflated ~2x; use step time minus GPU span without the profiler for real idle.
- **Host:** `py-spy record --nonblocking -r 1000 -d 10 --format raw --pid <sglang::scheduler_TP0>` during a long
  decode, then scripts/pyspy_tree.py (tree under event_loop_overlap).
- **Evals:** GSM8K only on EVAL_ONLY=true servers (SIMULATE_ACC commits unverified drafts -> 0.39). Compare GSM8K on
  the same CONC setting (CONC=32 eval = mem 0.85 / max-running 64; CONC=1 = max-running 2, 5x slower).
  Greedy token parity across servers is unusable: two fresh servers differ at token 0 (target prefill
  nondeterminism); a second run on the same server hits the radix cache. Use GSM8K + real accept length instead.
- **Ops:** kill servers by PID AND their renamed children (`ps -eo pid,comm | awk '$2 ~ /^sglang::/'`), otherwise
  scheduler processes keep ~227 GB of VRAM. rocm-smi GPU[7] == HIP device 5 here and host PIDs show as UNKNOWN.
  Never `pkill -f` a pattern contained in your own command line. Do not write a script and launch it in the same
  parallel tool batch (the launch can run before the file exists).
