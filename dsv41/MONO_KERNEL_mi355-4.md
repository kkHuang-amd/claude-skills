# DSV4.1 mono kernel in SGLang: mi355-4 work log (P0 -> P1)

Owner node: mi355-4
Parent: MONO_KERNEL_EVAL.md (owner crsuse2-m2m-255; design + decisions). This doc carries the mi355-4 implementation;
findings are appended to the parent's Log, never to its CONTINUE HERE.

## CONTINUE HERE

**Status (mi355-4, 2026-10-13 07:40 +08):** all 40 decoder layers on mono (K1+K2; the 8 indexer layers are "hybrid":
SGLang indexer, and the compressor on 2/8/14/20, run between K1 and K2) on production KV layouts + 9b nosync + task 8
draft on mono + 9c sim-accept fold (SGLANG_DSPARK_SIMULATE_FOLD=1). All in /sgl-workspace/sglang, uncommitted; frozen
copy /sgl-workspace/snap_task9c_1012. GSM8K TP4 0.906, TP2 0.904.
AgentX (night 10-12, section "AgentX night 10-12"): P90 vs ATOM 1010 (run 38056462998) TP2 c1/4/8/16 +3.2/+18.8/+17.3/+24.4%,
TP4 c1 -2.2%, c4 +11.2%, c8 +9.6% (rerun +14.6%); TTT/GPU within +-2% of ATOM; 0 Tracebacks.
Only gap left: TP4 c1 P90 (our ITL p90 1.665 vs ATOM 1.628), tail = long-ISL requests -> context-dependent indexer cost.
No host bubble left at c1 (py-spy, task 9c).
**Task 17 S1 (mi355-4, 2026-10-13): hybrid wired** (section "Task 17 S1"): ATOM's tree was 2-2.7x slower than SGLang's
top-k, so per the user K2A=1 = K2i FP4 scorer + SGLang's top-k / candidate kernels (+ device walk plan replacing the
aiter decode schedule). Microbench c1 -46 us/step at 128k, -167 at 512k; real-data K2A_CHECK Jaccard min 1.0 (FULL) /
0.996 (REINDEX); GSM8K 100 0.940, NIAH 120k 4/4.
**A/B done (10-13, snap_k2a_s1_1013, TP4, task17_ab.sh):** GSM8K 1319 k1 0.904 / k0 0.903, NIAH 60/60 both, 0 Tracebacks;
TPOT mean of 3 (ms) k0 -> k1: 8k c1 1.460 -> 1.451 (-0.6%), 8k c4 2.128 -> 2.178 (+2.3%), 128k c1 1.497 -> 1.457 (-2.7%),
128k c4 2.189 -> 2.223 (+1.5%; k0 alone spans 2.09-2.25). **Next (ask the user):** c4 does not match the microbench
(-125 us/step predicted) -> more reps / GPU profile at c4; 512k A/B; AgentX TP4 c1.
**Pushed (10-13):** HaiShaw/sglang perf/v41-mono @3c5ea0778c (5 commits on upstream +170; worktree /sgl-workspace/sglang_v41mono;
snapshot snap_v41mono_rebased_1013). AgentX TP4 on it (K2A=1): c1 7,444 / P90 617.1 (vs ATOM 1010 +0.5% / +0.2%),
c4 10,146 / 529.1 (+5.0% / +22.1%); 0 Tracebacks. **Next: task 20** (user 10-13, new session; prompt dsv41/MONO_COMPARE_SESSION_PROMPT_1013.md, doc
dsv41/MONO_COMPARE_mi355-4.md): compare our mono design with zijiecode/sglang-mono-amd m3-mono. Then (ask the user):
512k A/B, S3 (index q in K1), tasks 15/16/19.
Then (ask the user): tasks 15, 16, 18 (commit prep), 10-13.
Nothing committed (sglang dsv41-mono-ffn, aiter dsv41-opt-1008, skills dsv41/*): ask first; no AI attribution.
**Run:** on perf/v41-mono >= 134919476c mono = `--layer-backend mono` alone (no env); older snapshots: `--enable-mono-kernel` + mono env = `SGLANG_MONO_DSV41_WHOLE=1 SGLANG_MONO_DSV41_INDEXER=1 SGLANG_MONO_DSV41_EDGE=1 SGLANG_MONO_DSV41_DRAFT=1 SGLANG_MONO_DSV41_K2A=1`. A/B of snapshots: scripts/fw_ab.sh (per-variant MONO_ARG)
```bash
S=/workspace/claude-skills/dsv41/scripts; cd /tmp   # never launch from inside sglang/srt/models (transformers.py shadows)
<mono env> VARIANTS="x:/sgl-workspace/snap_task9c_1012" SOAK=none GSM=x bash $S/task9b_ab.sh   # c1/c4 TPOT x3 + GSM8K
<mono env> INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423 \
  SRC=<frozen snapshot>/python LANE=tp4 CONCS="1 2 4 8" TAGP=<tag> \
  SWEEP_EXTRA_ARGS="--fp8-gemm-backend aiter --disable-shared-experts-fusion --layer-backend mono" bash $S/sweep_ci37423_opt.sh
python3 -I $S/agentx_join_itl.py <atom agentic dir> <our run dir>   # per-request ITL vs ATOM by ISL bin
```
AgentX and A/B runs ALWAYS from a frozen copy of /sgl-workspace/sglang/python. Profiles: PROF_ACT=GPU (CPU activity at c1
hung once); with the nosync patch the profiler drops verify-graph kernels -> gaps only estimable from clean TPOT.
**Pass criteria:** GSM8K TP4 >= 0.895 (EVAL_ONLY, parallel 8); real accept len ~3.36; AgentX vs task8 rows; no Traceback.

## Task list (mi355-4; status as of 2026-10-13 20:50 +08 -- keep in sync)

Target = ATOM #2479 coverage (all 40 layers whole + draft on mono), not vLLM #60397's split. Nothing is committed yet
(sglang dsv41-mono-ffn, aiter dsv41-opt-1008, skills dsv41/*). Latest frozen snapshot: /sgl-workspace/snap_k2a_s1_1013 (task9c + K2A hybrid).

### Done

| # | Done on | Item | Result |
|---|---|---|---|
| 1 | 10-11 | AgentX TP4 sweeps, whole-layer mono (V4 KV) c1-c8 + same-branch no-mono baseline | mono-only attribution vs SA CI 37423021942 |
| 2 | 10-11 | V41 phase 1 (main KV V41, compressed fp8) | GSM8K 0.902, TPOT = V4 speed |
| 3 | 10-12 | V41 phase 2: production layouts, K2 reads the V41_FP4 compressed cache | GSM8K 0.907, no env overrides needed |
| 4 | 10-11 | 8 indexer layers whole (K1 + K2), **hybrid**: SGLang indexer runs between K1 and K2 | all-40 GSM8K 0.900; indexer itself -> task 17 |
| 5 | 10-11 | Compressor + indexer layers 2/8/14/20 whole, **hybrid** (compressor between launches, as in ATOM) | all-40 run |
| 6 | 10-11 | Layer 1 (Engram, ratio 0) whole | EDGE flag |
| 7 | 10-11 | Layer 0 whole (one-hot pre) | EDGE flag |
| 8 | 10-12 | DSpark draft stages on mono (DRAFT_SPEC, `SGLANG_MONO_DSV41_DRAFT=1`) | c1 TPOT -5.0%, c4 -5.6%; GSM8K 0.904; real accept len 3.398 (base 3.362) |
| 9 | 10-12 | c1 step profile vs ATOM | gap was the per-step host bubble at that time |
| 9b | 10-12 | Host bubble part 1: skip HIP `publish_ready.synchronize()` when no CPU seq_lens are needed (nosync) | c1 TPOT -6.8% |
| 9c | 10-12 | Host bubble part 2: simulated accept folded into the verify graph (`SGLANG_DSPARK_SIMULATE_FOLD=1`, default on) | c1 TPOT -1..-3%, GSM8K 0.906; py-spy: no host bubble left at c1 (host ~3.9 ms/step ahead); residual ~0.45 ms/step is GPU-side |
| 14 | 10-12 | New ATOM baseline (run 38056462998, nightly 1010) analysis | c1 P90 577 -> 616 is a p90-on-a-cliff artifact (median ITL unchanged); ATOM #2513 is memory / BF16-only, not a speed port |
| 17 S1 | 10-13 | K2a S1 **hybrid** (`SGLANG_MONO_DSV41_K2A=1`): ATOM FP4 scorer (K2i) + SGLang top-k / candidate kernels, device walk plan, REINDEX on compact candidate logits (ATOM's tree was 2-2.7x slower than SGLang's top-k -> not used) | GSM8K 0.904 (k0 0.903), NIAH 60/60; real-data Jaccard min 1.0 / 0.996; AgentX TP4 c1 P90 612.7 (vs ATOM -0.5%, was -2.2%), c4 511.5 (+18.1%, was +11.2%); per-request ITL vs task9c 0.986 / 0.988 |
| 18 | 10-13 | Commit prep + push: DRAFT_CHECK debug removed, ruff/ruff-format clean, 5 commits; rebased onto HaiShaw perf/v41-mono (upstream +170) and pushed b4822bae03..3c5ea0778c. Upstream #43623 already folds the simulated correct_len; our fold now only adds the in-graph sampled bonus (temperature > 0, AgentX default 1.0) | rebased snap_v41mono_rebased_1013 K2A=1: GSM8K 0.907, NIAH 120k 20/20, TPOT = old base within noise; temp 1.0 128k fold on/off c1 1.484/1.465, c4 2.261/2.329; 0 Tracebacks |
| 20 | 10-13 | Compare with zijiecode/sglang-mono-amd m3-mono + framework merge (MONO_COMPARE_mi355-4.md): `--layer-backend mono` (validated set default, per-feature env removed except K2A for A/B), MonoStep.query_len + decline, check mode (SGLANG_MONO_CHECK), MonoSpec.state_bindings, kernels -> `kernels/ops/mono/gfx950/{common,deepseek_v41}`, DSV4.1 private common + attention dups merged into the shared common (84 kernels ISA-identical), JIT key covers common, CI tests | commit 134919476c on perf/v41-mono-fw, fast-forwarded perf/v41-mono, pushed to HaiShaw 10-13 20:45. Final vs base: GSM8K 1319 0.901 / 0.912, NIAH 60/60, TPOT +-1.4%, AgentX TP4 c1 7,401 / 610.3 (-0.6% / -1.1%), c4 9,561 / 516.0 (-5.8% / -2.5%), per-request ITL 1.005 / 1.002 |
| -- | 10-12 night | AgentX night 10-12 on snap_task9c_1012 (TP4 c1/c4/c8/c8_r2/c16, TP2 c1/c4/c8/c16) | P90 vs ATOM 1010: TP2 +3..+24%, TP4 c1 -2.2%, c4 +11.2%, c8 +9.6/+14.6%; TTT/GPU within +-2% of ATOM; TP2 mono GSM8K 0.904; c8 rerun = variance, no PDI change; 0 Tracebacks (table: section "AgentX night 10-12") |

### Open

| # | Item | Status / next step |
|---|---|---|
| 17 | **K2a S2 / S3** (plan: K2A_PORT_PLAN_mi355-4.md): S2 fuse the indexer into K2 (needs a faster in-kernel top-k first: ATOM's tree is 2-2.7x slower than SGLang's), S3 index query in K1 (`SGLANG_MONO_DSV41_K2A_IQ=1`, est. ~-100 us/step at any ISL) | S1 done (hybrid). Open: 512k TPOT A/B of S1; S3 next candidate |
| 15 | V4 vs V41 layout TPOT: decode-load ISL 4096, V41 +1.7% c2 / +3.4% c4 (c1/c8 slightly faster; noise ~+-2%) | cause not found; hypothesis FP4 compressed-KV dequant in K2. Plan: same snapshot, `SGLANG_DSV4_KV_LAYOUT=v4` vs default, c1/2/4/8 x3, ISL 4096 + 131072; if real, c4 GPU-only profile of K2 on the 18 ratio-2 layers |
| 16 | High concurrency: mono only runs steps <= 48 rows (8 req x 6 DSpark rows; ATOM same); larger steps fall back | partly done: TP4 c16 and TP2 c16 ran clean (P90 +27% / +19% vs CI). Still open: TP2 c32/c64 vs base, mono vs fallback step counts, GSM8K at parallel 128. Raise MAX_ROWS only if high-conc gains are wanted |
| 19 | **Port the HaiShaw perf/v41-* opts into the mono branch.** Finding 10-13: none of the 7 branches is in dsv41-mono-ffn (tips not ancestors of HEAD, patches not applied); SGLANG_DSV41_PREFILL_LOGITS_BF16 / ROCM_MHC_ALL_REDUCE_STATS / AITER_SMALL_MOE_SORT_MAX_PAIRS / ROCM_MXFP8_AITER_PRESHUFFLE are read by no code in sglang or aiter -> every mono and same-branch base run had them OFF; --tokenizer-worker-num is 1. Only upstream's SGLANG_HIP_WO_A_MXFP8 epilogue is active | candidates: index-q prefill fuse, indexer bf16 logits (aiter side cc1773508 already in), candidate block-max fast path (all apply cleanly), --tokenizer-worker-num 4; mhc-ar boundary stats conflicts and only helps prefill under mono; small-moe-sort / flydsl mxfp8 gemm / woa fuse mostly dead under mono. Prefill opts should cut the prefill interference that makes the c8/c16 P90 tail. Validate: GSM8K, AgentX TP4 c8 (+c1) vs task9c |
| 10 | Safety: bounded waits + error words in the vendored kernels (fail-stop reports) | not started |
| 11 | Per-stage numerics tests vs SGLang ops (K1 insert bytes vs V41 store kernel, K2 vs aiter_sparse, FFN stages) | not started |
| 12 | MiniMax-M3 on our framework: kernels = their aiter PR6047 copy (user 10-13), wrap as a MonoSpec, bring their prerequisites (aiter sparse PA, aiter indexer, SHUFFLE KV); weights /shared_nfs/MiniMaxAI/MiniMax-M3 | not started; user to decide: us vs zijie on our framework. Prep done: their common.py mapped onto gfx950/common (22/30 identical) |
| 13 | GSM8K A/B x3 to settle the -1 pt (mono vs no-mono) | not started |
| 21 | DSV4.1 mono MoE does not mask CUDA-graph padding rows (they route like real rows; SGLang's module path masks them via num_token_non_padded) -> extra experts loaded at non-captured batch sizes | found 10-13 (task 20 F7); perf only. Plan: pass the live row count to the router stage, padded rows select no expert; measure c3/c5/c7 + AgentX c4 |
| 22 | Check mode: layer 17 hidden max\|diff\| 15% of ref max vs module path (median over 40 layers 2.4%) | found 10-13; accuracy fine. Look at which stage diverges (MXFP8 shared expert / quant order) |
| 23 | Unreachable shared-common (L0) code: keep or delete | deferred to the M3 decision (M3 uses some, e.g. fp8x8_bf16_pk); per-step clear + fence protocol goes with task 10 |
| -- | Optional: GPU-side gap profile (~0.45 ms/step at c1) | needs an env switch to force the nosync path off so the torch profiler sees verify-graph kernels |
| -- | Optional: AgentX TP4 c2 on snap_task9c_1012 | dropped by the user 10-12 |

## Environment (mi355-4)

- GPUs: 8x MI355X (gfx950); this work uses GPUs 4-7 only, exclusive (user, 2026-10-10).
- sglang: branch `dsv41-mono-ffn` from `HaiShaw/sglang perf/v41-mono` @6e8f3e09dd (= upstream main snapshot, no mono code
  yet; opt-1008 items 0-6 NOT cherry-picked). Remote `haishaw` added. Image edits kept uncommitted: python/pyproject.toml
  (image ROCm version), kernels/aot pyproject, hipified AOT sources. AOT sources unchanged vs previous HEAD -> no rebuild.
- aiter (user 2026-10-10: upgrade to the opt-1008 version): `/sgl-workspace/aiter` branch `dsv41-opt-1008` =
  kkHuang-amd `perf/v41-indexer-bf16-logits` @cc1773508 (= origin/main 6264f8f5c + bf16 logits commit; remote `kkhuang`).
  Uncommitted image steps from sglang docker/rocm.Dockerfile: #6042 cherry-pick 042a2891 (pa_decode_sparse conflict =
  comment only, kept the existing `row_tiles`), mla_v4 `.co` copied from sglang, torch.Stream fix in torch_utils.py.
  Not needed at this commit: #5967 CSVs (merged), #6007 group32 `other=127` (gone), lru_cache sed (already @cache).
  Old e7d2453f2 image edits: `git stash` "e7d2453 image edits (pre-opt1008 1010)"; old JIT .so + jit/build + flydsl_cache
  -> /sgl-workspace/aiter_jit_backup_e7d2453_1010. Build: `scripts/aiter_aot_build.sh` (full AOT, gfx950), log
  /shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/aiter_aot_build_mi355-4_1010.log.
  Note: sglang perf/v41-mono's Dockerfile pins aiter e7d2453f2, so this is a newer aiter than the base expects.
- flydsl 0.3.4.1. Model: `/shared_nfs/deepseek-ai/DeepSeek-V4.1-Flash`.
- AgentX (old env differed: InferenceX main 57bec457c, aiperf 754356e): new InferenceX checkout
  `/workspace/InferenceX-ci37423` @47adb59a (CI recipe; layout has `inferencex-e2e/`, aiperf submodule 89b2186) and new
  runtime `/workspace/agentx-runtime-ci37423` (uv + cache hardlinked from the old one, venv built, aiperf 0.12.0).
  Old `/workspace/InferenceX` and `/workspace/agentx-runtime` untouched. Select it with
  `INFMAX_CONTAINER_WORKSPACE=/workspace/InferenceX-ci37423/inferencex-e2e AGENTX_RUNTIME_ROOT=/workspace/agentx-runtime-ci37423`
  (new knobs in agentx/agentx_env.sh and scripts/agentx_colleague_run.sh; defaults unchanged for other nodes).
  Every benchmark_lib function agentx_colleague_mi355x_sglang.sh calls exists at 47adb59a.
- Backups of the previous state (HEADs, local diffs, untracked tarball, pip freeze):
  /shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/repo_backup_mi355-4_1010/ (previous sglang HEAD 1203eb9181 on main).

## Decisions

- 2026-10-10 user: base = HaiShaw perf/v41-mono; compare against SA CI data (run 37423021942), no local opt1008 baseline;
  GPUs 4-7 exclusive; this separate doc; rebuild AgentX env if it differs, keep the old one.

## Design: SGLang mono framework, multi-model from day one (user, 2026-10-10)

Goal: ATOM-level framework (ATOM: shared `atom/mono` used by MiniMax-M3 and V4.1 kernels), not a DSV4.1 one-off.
Interface follows vLLM RFC #60904 (`MonoSpec`); DSV4.1 is the first spec, and a second model validates the interface
in the same phase: MiniMax-M3 (ATOM kernels on L0; SGLang `minimax_m3.py`).

### Layers

| Layer | What | Where | Model-specific? |
|---|---|---|---|
| L0 device + plan library | mailboxes, bounded waits, peer push / rank-order sum, MX/FP4 decode helpers, select/top-k, stamps; plan: shard/tiles, build keys (source digest), region contracts (trace + check) | external, NOT copied into sglang: ATOM `atom/mono/{device,plan}` (906 + 364 lines, imports only flydsl + aiter) | no |
| L0 runtime helpers | `PeerBuffer` (aiter UncachedIpcHeap), `WidthBuilds`, `tp_agree` consensus, `step_fence`, compile-before-capture, `KernelAbi` | same package, `atom/mono/runtime` (779 lines; only `deployment.py` imports atom.plugin) | no |
| L1 SGLang host runtime | contract + routing + safety, below | `python/sglang/srt/layers/mono/` (new, ~800 lines) | no |
| L2 model spec + adapter | `MonoSpec` instance, weight views/convert, cache bind, eligibility, build per width | `python/sglang/srt/models/<model>/mono/` (e.g. `deepseek_common/amd/mono/`) | yes, ~300-400 lines each |
| L3 model kernels | FlyDSL layer kernels | external, pinned: DSV4.1 = vLLM `amd/mono/` @193922d6 (stages/attention/layer/runner); MiniMax-M3 = ATOM `atom/models/minimax_m3/mono` | yes |

**Packaging (user, 2026-10-10): in-tree.** Users install nothing beyond sglang (+ flydsl/aiter already in the ROCm image).
L0 is vendored to `python/sglang/srt/layers/mono/device/` (+ `plan/`, runtime helpers into L1) from ATOM @454d0e3 (MIT);
model kernels to `python/sglang/srt/models/deepseek_common/amd/mono/kernels/` from vLLM @193922d6 (Apache-2.0) and
`python/sglang/srt/models/minimax_m3_mono/kernels/` (path TBD) from ATOM (MIT). Each vendored file keeps an origin header
(repo, commit, path, license); internal imports rewritten to sglang paths; JIT keys keep the source digest. DSV4.1 kernels
are re-pointed at the vendored L0 where APIs match, else keep their private copy until converged. Upstream path later:
L0 -> standalone package / FlyDSL, then the in-tree copy becomes an import. The namespace shim below is dev-only (P0).

(superseded for users) L0/L3 loading: `sglang/srt/layers/mono/sources.py` resolves pinned source dirs (env `SGLANG_MONO_SRC_<NAME>` or an
installed package) and maps their namespaces (`atom.mono`, `vllm.models.deepseek_v41.amd.mono`) without importing the
host frameworks (`atom/__init__.py` pulls atom.plugin.sglang; vllm is not installed). Every JIT key includes the
source digest (L0 `build_key.source_digest`). Target state: L0 becomes a standalone package (or moves into FlyDSL, as the
RFC proposes); the shim then reduces to a normal import. Known gap: vLLM's DSV4.1 kernels import their PRIVATE copy of
L0 (`amd/mono/common/`, adapted from ATOM @97359d6), not `atom.mono` -- L1 must not depend on that copy; converging
DSV4.1 kernels onto L0 is a later upstream task (vLLM or ATOM side), tracked as a risk, not a blocker.

### L1 modules (`python/sglang/srt/layers/mono/`)

| File | Holds |
|---|---|
| `spec.py` | `MonoSpec` (name, layers(cfg), widths, supports: frozenset[Feature], caches: CacheReq map, weight_source views/convert, options, refuse_config(server_args, model_config), refuse_model(model), pack, bind, eligible, build), `LayerKernel` protocol (`launch`, `health`), `MonoStep`, `MonoContext`, `Feature`, `LayerMode` (WHOLE / FFN) |
| `registry.py` | model arch -> spec import path (`"sglang.srt.models.deepseek_common.amd.mono.spec:MONO_SPEC"`); config-time code reads the spec without importing the model |
| `guards.py` | `configure(server_args)`: default-deny features (TBO, DP attention, EP/EPLB, PP, PD disagg, LoRA, hicache, KV connectors, weight updates, partitioned GPUs, non-TP spec features...), capture-size rewrite (sizes <= max(width) must be widths), residency (256-CTA grid fits the CUs), all refusals reported at once, startup raises when refused |
| `layer.py` | `attach(model, server_args)` after weight load; `forward_layer(state, i, layer, *args)`; `begin_step(forward_batch)` |
| `routing.py` | per-step decision for all layers at once: pure decode / target-verify, row count = a width, `spec.eligible`; rank-uniform (only ForwardBatch / graph bs); graph-safe (pure function of the captured bs); cache re-bind on `(data_ptr, numel)` change |
| `failstop.py` | health words copied next to sampled ids (scheduler output path), raise before returning that step's tokens, worker exit, `SGLANG_MONO_UNSAFE_WARN` (test-only), fault-inject hook `SGLANG_MONO_FAULT_INJECT` |
| `peer.py` | peer buffers via L0 `PeerBuffer`; coexistence with SGLang custom AR / quick-reduce buffers (separate allocation, no sharing) |
| `abi.py` | `MONO_ABI_VERSION`, error-word layout (re-exports L0 `KernelAbi`) |
| `metrics.py` | counters: mono steps, fallback steps by reason, incidents; one startup log line (spec, widths, refusals) |
| `test/registered/mono/` | test kit: fake spec + fake kernel, refusal / routing / param-name / view-check / re-bind / fail-stop CPU tests; multi-GPU stage numerics vs SGLang ops, graph replay >= 300x bit-identical, fault injection |

Wiring per model is ~15 lines: model `post_load_weights` -> `self.model.mono = mono.attach(...)`; layer loop calls
`mono.forward_layer(...)`; a step not routed to mono calls the layer exactly as before (module tree, param names,
load_weights untouched; CI checks named_parameters equal with the switch on/off). Switch: one server arg
`--enable-mono-kernel`, default off, same for every model; tuning only in the spec's `options`.

Decisions that differ from ATOM (ATOM: whole-model wrapper `MonoDecodeModel` + `step_supported`, collective disable on
`MonoUnsupported`): SGLang uses per-layer `forward_layer` (no torch.compile on this path; layer and kernel paths can
interleave in a step) and fail-stop at run time (RFC), refusal only at config / load / bind time.

### Keeping L1 model-agnostic (rules)

- No model constants in L1: max rows (48), tokens per request (DSpark 6), residual streams (mHC 4), hidden size, layer
  lists, FFN-only layer sets -- all come from the spec (`widths`, `layers`, `LayerMode` per layer, `options`).
- `launch` takes and returns exactly what the SGLang layer returns (DSV4.1: hidden + mHC residual/mixes;
  MiniMax-M3: its own convention) -- L1 never interprets layer state.
- Draft / speculative layers are a spec feature (`Feature.SPEC_DRAFT_LAYERS`), not L1 logic.
- Second spec lands in the same phase as DSV4.1 FFN-only, so any DSV4.1 assumption in L1 fails its tests.

### Second model: MiniMax-M3 (user, 2026-10-10)

| Candidate | Kernel exists | SGLang model | Weights on mi355-4 | Exercises |
|---|---|---|---|---|
| MiniMax-M3 | yes: ATOM `atom/models/minimax_m3/mono` (25 files), built directly on L0 `atom.mono` | `minimax_m3.py` | not found under /shared_nfs (incl. hf_cache) | L0 reuse as-is, different layer convention; needs weights |
| GLM-5.2 | vLLM #60889 (open, not merged) | `glm5_next.py` (check it is the same arch) | not found | RFC's "structurally different second model" |
| Kimi-K3 | none (would be written from scratch on L0) | `kimi_k3.py` | `/shared_nfs/Kimi-K3` | everything, but kernel authoring is a large project |
| Fake spec only | test kit fake kernel | -- | -- | L1 contract/routing/fail-stop only; no real second layout |

### Revised phases

- P0 (in progress): DSV4.1 layouts (done), sourcing (done; now L0 = atom/mono, L3 = vLLM kernels), FFN microbench.
- P1: L1 runtime + `sources.py` shim + test kit (fake spec) + DSV4.1 spec in FFN-only mode for all 40 layers +
  second-model spec skeleton (refusal/routing/views on CPU). Validate: CPU kit, DSV4.1 stage numerics, GSM8K, TPOT, AgentX.
- P2: DSV4.1 whole-layer (K1+K2, KV adapter: SGLang 584 B record = vLLM layout) + second model end-to-end.
- P3: upstream: L1 to sglang, L0 as a standalone package / FlyDSL, DSV4.1 kernels converged onto L0.

## P0 findings

### P0.1 layout check (mi355-4, 2026-10-10; sglang dsv41-mono-ffn vs vLLM mono @193922d6 / ATOM @454d0e3)

Shapes (config): hidden 5120, moe_intermediate 2304 -> TP4 576/rank (routed padded 640), 384 experts top-6,
sqrtsoftplus + e_score_correction_bias, x1.5; checkpoint quant_method fp8 (ue8m0 block scales), experts fp4.

| # | Item | Verdict | Action under the mono flag | Evidence (sglang python/sglang/srt/) |
|---|---|---|---|---|
| 1 | KV record | CORRECTED 2026-10-10: consistent with vLLM's fp8_ds_mla only under SGLang KVLayout.V4 (448 e4m3 + 64 bf16 RoPE + 8 UE8M0). Production on gfx950 resolves to KVLayout.V41 (512 e4m3 incl. RoPE + 16 UE8M0; `select_dsv4_kv_layout`: auto + aiter_sparse -> V41), confirmed on the live mi355-4 server env (no SGLANG_DSV4_KV_LAYOUT / SGLANG_HACK_FLASHMLA_BACKEND set). Whole-layer needs SGLANG_DSV4_KV_LAYOUT=v4 or a V41 port of the kernel's KV insert / key load stages. ATOM bf16: needs adapter. Not used by FFN-only | none for P1; P2 = wire SWA/compressed block tables to mono slots | mem_cache/deepseek_v4_memory_pool.py:276-284; vLLM attention/plan.py:29-32 |
| 2 | Dense MXFP8 (shared expert in FFN-only; wqkv/wq_b/wo_b in P2) | needs SGLang layout change: aiter `(16,16)` bpreshuffle is in place and shape-preserving | set `keep_plain_weight_layout` on shared gate_up/down (and P2 attn linears); assert `aiter_bpreshuffled` is unset; bind `weight_scale_inv` as E8M0 [N/32,K/32] | layers/quantization/fp8.py:880-905; models/deepseek_v4.py:949 (wo_a precedent); vLLM runner.py:122-147 |
| 2d | mHC params + Sinkhorn | consistent (hc_*_fn [24,4*5120] f32, scale [3], base [24], 20 iters) | bind views | models/deepseek_v4.py:580-594 |
| 3 | Shared expert | needs adapter: fusion OFF -> DeepseekV2MLP FP8 block [32,32] ue8m0, fused gate_up [1152,5120], down [5120,576] (shapes match mono sgu/sw2) | fusion OFF (decided) + item 2 | models/deepseek_v2.py:821-830 |
| 4 | Routed MXFP4 | consistent: pad 576->640, shuffle_scale + shuffle_weight(16,16) with GU_ITLV=1 (default) -> GUGU = mono's layout; w13 [384,1280,2560], w2 [384,5120,320]. Only name differs (`w13_weight_scale_inv`) | bind by name; refuse if SGLANG_USE_AITER_MOE_GU_ITLV=0. AITER_FORCE_A8W4 (GGUU) is quark-only, not this checkpoint | layers/quantization/fp8.py:1791-1922; environ.py:907 |
| 4r | Router | consistent (bf16 gate [384,5120], f32 bias, sqrtsoftplus, top-6, renorm x1.5) | none | models/deepseek_v2.py:513-529,769-785 |
| 5 | DSpark rows | needs adapter: verify = 6 rows/req; decode graphs captured up to bs 64 (=384 rows) | mono only for captured widths <= 48 rows (bs <= 8), rank-uniform decision from the graph bs; draft layers keep SGLang path (vLLM style; ATOM's draft mono later) | speculative dspark_config.py:51-58; model_executor decode_cuda_graph_runner.py:302-378 |

FFN-only split point: after `self_attn`, before the FFN seam/MoE (models/deepseek_v4.py:3192-3259; hc_pre_from_prev
path: run_attn_hc 3322-3330 / run_ffn_hc). Attention-output AR to disable: `_project_wo_b` (2622-2640) already has a
`skip_all_reduce` / `mhc.AttnOutput` deferral -> under mono return the unreduced wo_b partial and do NOT take
`all_reduce_mhc_post_combine_norm` (layers/.../deepseek_v4_mhc.py:614-631); mono pushes + reduces it. Risks: a peer
disagreeing on mono vs fallback deadlocks the step (decision must be rank-uniform); double / missing reduce if both
paths active.

### P1 results (mi355-4, 2026-10-10)

GSM8K TP4, 1319 q, 5-shot, parallel 8 (every decode step <= 8 req; captures 6..48 rows logged "-> mono"), EVAL_ONLY,
fusion off in both (scripts/mono_smoke.sh):

| config | accuracy | invalid | latency | accept len (median, running>=6) | gen tok/s (median, running>=6) |
|---|---|---|---|---|---|
| mono (`--enable-mono-kernel`) | 0.901 | 0.001 | 122.1 s | 3.36 | 1182 |
| no mono (same flags) | 0.911 | 0.001 | 111.0 s | 3.355 | 1138 |

Accuracy passes the 0.895 gate; -1.0 pt vs A/B is within the DSpark run-to-run spread seen before (0.89-0.91) but needs
repeats. Decode throughput +3.9% at equal acceptance; the end-to-end latency gap is outside decode (not explained yet).
TPOT with REAL acceptance on random-token prompts (mono_tpot_ab.sh EVAL_ONLY=true, tag mi355-4_tpot_*) is dominated by
acceptance noise (c1 prod 1.46 / nomono 2.99 / mono 1.17 ms): not usable; rerun with simulated acceptance (tag
mi355-4_tpotsim_*, OSL 1024) below.

V4 KV layout (SGLANG_DSV4_KV_LAYOUT=v4, needed by the whole-layer launch), no mono: GSM8K parallel 8 = 0.908
(109.7 s); TPOT sim-AL prod c1/c2/c4/c8 = 2.134 / 2.493 / 2.686 / 4.003 ms, nomono = 2.318 / 2.563 / 2.897 / 3.545 ms
(V41 prod: 2.166 / 2.494 / 2.660 / 3.267) -> V4 itself costs nothing at c1-c4; prod c8 4.003 looks like an outlier
(TTFT 634 ms vs 423), rerun before using it.

Decode TPOT, simulated golden acceptance (as CI), ISL 4096 / OSL 1024, one server per config with the CONC=8 recipe args
(scripts/mono_tpot_ab.sh, tag mi355-4_tpotsim_{prod,nomono,mono}):

| conc | prod (fusion ON) | no mono (fusion off) | mono (fusion off) | mono vs prod | mono vs no-mono |
|---|---|---|---|---|---|
| 1 | 2.166 | 2.336 | 1.931 | -10.8% | -17.3% |
| 2 | 2.494 | 2.534 | 2.229 | -10.6% | -12.0% |
| 4 | 2.660 | 3.336 | 2.678 | +0.7% | -19.7% |
| 8 | 3.267 | 3.397 | 3.278 | +0.3% | -3.5% |

(TPOT ms.) Matches P0.3: FFN-only mono wins at c1-c2, parity with production at c4-c8 (where production's fused
shared expert and SGLang's fused_moe are strong); the whole-layer launch (P2) is where c4-c8 gains have to come from.

### Whole-layer mono (P2a, V4 KV layout) -- mi355-4, 2026-10-11

30 layers whole (K1 + K2: ratio 1/2, no compressor / indexer / Engram), 10 FFN launch ("layers=40 {ffn: 10, whole: 30}"),
SGLANG_MONO_DSV41_WHOLE=1 SGLANG_DSV4_KV_LAYOUT=v4, fusion off; decode graphs 6..48 rows captured -> mono.
- GSM8K 1319, parallel 8, EVAL_ONLY: **0.901**, invalid 0.000, 101.8 s (V4 no-mono 0.908 / 109.7 s; FFN-only mono on V41
  0.901 / 122.1 s). Log gsm8k_mi355-4_monowhole_gsm8k_p8_r2.log.
- Decode TPOT, sim AL, ISL 4096 / OSL 1024 (tag mi355-4_tpotv4whole_r2_mono):

| conc | prod V41 (fusion ON) | FFN-only mono (V41) | whole-layer mono (V4) | whole vs prod |
|---|---|---|---|---|
| 1 | 2.166 | 1.931 | 1.734 | -19.9% |
| 2 | 2.494 | 2.229 | 1.919 | -23.1% |
| 4 | 2.660 | 2.678 | 2.451 | -7.9% |
| 8 | 3.267 | 3.278 | 3.130 | -4.2% |

(TPOT ms.) The V4 layout alone does not explain it (V4 prod c1-c4 2.134 / 2.493 / 2.686). Worth the V41 port.

### Remaining gap to ATOM's full mono (mi355-4, 2026-10-11)

Config: kv_source_layer_ids [2, 8, 14, 20], index_source_layer_ids [2, 8, 14, 20, 24, 28, 32, 36], engram_layer_ids
[1, 14], compress_ratios 0 for layers 0-1, 2 for 2-19, 1 for 20-39.

| layers | role | ours | ATOM #2479 | to do |
|---|---|---|---|---|
| 30 consumers (ratio 1/2, no own compressor / indexer) | read shared compressed KV + top-k | whole (K1+K2), V4 KV; V41 phase 1 untested | whole | V41 validation, V41_FP4 compressed reads |
| 24, 28, 32, 36 | own indexer (ratio 1) | FFN only | whole, indexer scoring + exact top-k in K2a | port ATOM K2a onto SGLang's FP4 index cache |
| 2, 8, 20 | own compressor + indexer | FFN only | K1, compressor (between launches), K2 | compressor hand-off + K2a |
| 14 | compressor + indexer + Engram | FFN only | same, Engram outside | + Engram hand-off |
| 1 | Engram (ratio 0) | FFN only | whole, Engram outside | whole after Engram; ratio-0 (SWA only) build |
| 0 | embedding-broadcast seam (ratio 0) | FFN only | whole | layer-0 seam input + ratio 0 |
| DSpark draft | draft blocks | SGLang path | K1/K2 via DraftMonoRunner (LM head / sampler outside) | draft spec + runner |
| step metadata | -- | host scheduler | device-written step meta + candidate block table | -- |

ATOM recipe for run 37597985033 (InferenceX d3ba89f): rocm/atom-dev nightly 20261007, kv_cache_dtype bf16,
index-cache fp4, block 16, level 3, FULL graphs, DSpark K5 golden AL 3.51 (same AL and dataset
semianalysis_cc_traces_weka_062126 as our sweeps). User reports our c1 P90 ~10% behind ATOM; leading suspect: the 8
indexer layers (only context-length-dependent part; 1M-context corpus), then draft + per-step host overhead. To be
confirmed by a c1 step profile (deferred by the user, 2026-10-11).

### Indexer / compressor layers: design (tasks 4 + 5) -- mi355-4, 2026-10-11

Full ATOM K2a port (subagent study): ~1.6-2.5k vendored + 0.7-1.2k glue; risks: FP4 plane page 8 (ATOM) vs SGLang
page 64 split buffers, key table must move from K1 to K2, host walk plan vs CUDA graphs, top-k not bit-identical to
SGLang's HIP indexer. Chosen first step (hybrid, like ATOM's compressor between its launches): K1 exports bf16 x
(attn_norm output) and q_lora (q_norm output) -> SGLang's own indexer (+ compressor on 2/8/14/20) between the launches
-> K2 builds the key table from the fresh top-k (stage_kt moved into K2 under a build flag), then attention + FFN.
Same top-k selection as today (same kernels), ~300-500 lines. ATOM K2a later replaces the middle if the profile says
the indexer itself is the gap.

### V41 whole-layer phase 1 (task 2) -- mi355-4, 2026-10-11

Default main KV layout (V41 on gfx950) + SGLANG_DSV4_COMPRESSED_KV_LAYOUT=fp8, 30 whole / 10 FFN ("layers=40 {ffn: 10,
whole: 30}"): GSM8K 1319 parallel 8 **0.902**, invalid 0.001, 101.7 s -> the ported K1 insert / K2 loads are
numerically sound (same accuracy as V4 0.901). TPOT sim AL (ms, tag mi355-4_tpotv41whole): c2 1.953 / c4 2.479 / c8 3.085
(V4 30-whole 1.919 / 2.451 / 3.130): V41 matches V4 speed, no V4 needed for the whole layer anymore (except V41_FP4
compressed, phase 2).

### All 40 layers whole on PRODUCTION layouts (task 3 done) -- mi355-4, 2026-10-12

No layout env (main V41 + compressed V41_FP4), SGLANG_MONO_DSV41_WHOLE=1 INDEXER=1 EDGE=1: "layers=40 {whole: 40}".
GSM8K 1319 parallel 8 **0.907**, invalid 0.000, 101.1 s. TPOT sim AL (ms) c1 1.663 / c2 1.971 / c4 2.468 / c8 3.070 vs prod
(no mono) 2.166 / 2.494 / 2.660 / 3.267 = -23.2 / -21.0 / -7.2 / -6.0%. The FP4 compressed reads work first try.

### All 40 layers whole (tasks 4-7, V4 KV) -- mi355-4, 2026-10-11

SGLANG_MONO_DSV41_WHOLE=1 INDEXER=1 EDGE=1, SGLANG_DSV4_KV_LAYOUT=v4: "layers=40 {whole: 40}"; indexer / compressor layers run
K1 -> forward_low_ratio_sources -> key-table kernel -> K2. GSM8K 1319 parallel 8: **0.900**, invalid 0.001, 101.1 s.
TPOT sim AL (ms): c1 1.678 / c2 1.938 / c4 2.386 / c8 3.084 vs 30-whole 1.734 / 1.919 / 2.451 / 3.130 vs prod V41 2.166 /
2.494 / 2.660 / 3.267 (-22.5 / -22.3 / -10.3 / -5.6% vs prod). Small gain over 30-whole at ISL 4096 (indexer cheap at short
context); the long-context effect needs AgentX c1.

### V41 port plan (P2b) -- mi355-4, 2026-10-11

Production gfx950 = main KVLayout.V41 + compressed (ratio 1/2) KVLayout.V41_FP4 (resolve_compressed_kv_layout: auto ->
FP4 for ratio 1/2, "lossless for the e2m1 latents"). The vendored kernels read/write V4 (+ V4 compressed). Byte layouts
(verified in kernels/jit/include/sgl_kernel/deepseek_v4/kv_layout.cuh):
- V41 main: data[d] = page + slot*512 + d (512 e4m3, RoPE dims quantized too); scale[g] = page + page_size*512 + slot*16 + g
  (16 UE8M0, one per 32). Writer: bf16(norm) -> RoPE on the last 64 -> bf16 -> per-32 amax, exponent =
  max((bits>>23) - 8 + (mant > 0x600000), 114), e4m3 RNE of x * 2^-exp (store_row_fp8 L147-173).
- V41_FP4 compressed: 256 B e2m1 (two per byte, even index low nibble) + 32 e4m3 scales (one per 16).
Kernel changes (sites from the vendored code): attention/plan.py constants (NOPE 448 / DATA 576 / SCALE_BYTES 8);
front.py KV insert (per-64 scales + bf16 RoPE -> per-32 over all 512, V41 exponent formula, unified e4m3 store);
back.py key_rec + QK/PV loads (7x64 fp8 + bf16 RoPE -> 16x32 fp8; compressed path: FP4 e2m1 + e4m3/16 decode);
runner.py asserts (584) and spec RECORD / _record_view / the SGLANG_DSV4_KV_LAYOUT=v4 gate. Attention math unchanged
(512-d dot), only loads / stores. Tests: K1 insert bytes == SGLang V41 store kernel on identical inputs (exact); K2
output vs SGLang's V41 aiter_sparse attention; then GSM8K + TPOT on default layouts (no env overrides).

### AgentX TP4 vs SA CI run 37423021942 (append-only; FFN-only mono, fusion off, opt1008 sweep flags otherwise)

CI ref on mi355-4: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_ci_37423021942/agg_bmk_reconstructed.json, rebuilt
from the CI column of OPT_SWEEP_1008.md (mi355-4's /shared_nfs is a different NFS: the original agg_bmk.json is not
here). Compare: `agentx_agg_table.py --ref <that> --row <label> <point json>`.

| node / date / cfg | tp | conc | TTT/gpu | P90 intvty | TPOT p50 ms | TTFT p50 s | ok/total |
|---|---|---|---|---|---|---|---|
| mi355-4 10-10 mono_tp4_c1 rc=0 | 4 | 1 | 6,371.8 / 6,019.4 (+5.9%) | 473.8 / 406.5 (+16.6%) | 2.020 / 2.230 (-9.4%) | 0.74 / 0.88 (-16.4%) | 302/313 |
| mi355-4 10-10 mono_tp4_c2 rc=0 | 4 | 2 | 6,979.9 / 6,169.7 (+13.1%) | 452.6 / 350.6 (+29.1%) | 2.010 / 2.240 (-10.3%) | 0.41 / 0.59 (-29.9%) | 492/514 |
| mi355-4 10-10 mono_tp4_c4 rc=0 | 4 | 4 | 9,126.8 / 8,807.1 (+3.6%) | 397.1 / 331.6 (+19.8%) | 2.020 / 2.300 (-12.2%) | 0.38 / 0.50 (-24.8%) | 686/730 |
| mi355-4 10-10 mono_tp4_c8 rc=0 | 4 | 8 | 15,621.9 / 15,361.7 (+1.7%) | 347.7 / 282.2 (+23.2%) | 2.220 / 2.720 (-18.4%) | 0.29 / 0.30 (-3.4%) | 1401/1488 |
| mi355-4 10-11 monowhole_tp4_c1 rc=0 (whole-layer, V4) | 4 | 1 | 6,965.1 / 6,019.4 (+15.7%) | 523.4 / 406.5 (+28.7%) | 1.820 / 2.230 (-18.4%) | 0.65 / 0.88 (-26.3%) | 315/326 |
| mi355-4 10-11 monowhole_tp4_c2 rc=0 (whole-layer, V4) | 4 | 2 | 6,947.1 / 6,169.7 (+12.6%) | 511.4 / 350.6 (+45.9%) | 1.810 / 2.240 (-19.2%) | 0.39 / 0.59 (-33.8%) | 481/503 |
| mi355-4 10-11 monowhole_tp4_c4 / c8: INVALID (rc=1, crashed at start: back.py edited while the sweep ran; rerun from a frozen snapshot, TSUF=_r2) | | | | | | | |
| mi355-4 10-11 monowhole_tp4_c4_r2 rc=0 (whole-layer, V4, frozen snapshot) | 4 | 4 | 9,367.4 / 8,807.1 (+6.4%) | 469.8 / 331.6 (+41.7%) | 1.820 / 2.300 (-20.9%) | 0.39 / 0.50 (-21.8%) | 698/742 |
| mi355-4 10-11 monowhole_tp4_c8_r2 rc=0 | 4 | 8 | 16,062.2 / 15,361.7 (+4.6%) | 367.1 / 282.2 (+30.1%) | 2.020 / 2.720 (-25.7%) | 0.31 / 0.30 (+4.2%) | 1421/1508 |
| mi355-4 10-11 monoall40_tp4_c1 rc=0 (all 40 whole, V4, frozen snapshot) | 4 | 1 | 7,046.5 / 6,019.4 (+17.1%) | 536.6 / 406.5 (+32.0%) | 1.760 / 2.230 (-21.1%) | 0.68 / 0.88 (-23.3%) | 317/328 |
| mi355-4 10-11 monoall40_tp4_c2 rc=0 | 4 | 2 | 7,077.2 / 6,169.7 (+14.7%) | 530.7 / 350.6 (+51.4%) | 1.760 / 2.240 (-21.4%) | 0.37 / 0.59 (-36.5%) | 472/494 |
| mi355-4 10-11 monoall40_tp4_c4 rc=0 | 4 | 4 | 9,413.9 / 8,807.1 (+6.9%) | 494.0 / 331.6 (+49.0%) | 1.780 / 2.300 (-22.6%) | 0.33 / 0.50 (-33.5%) | 710/754 |
| mi355-4 10-11 monoall40_tp4_c8 rc=0 | 4 | 8 | 16,178.5 / 15,361.7 (+5.3%) | 380.3 / 282.2 (+34.8%) | 1.950 / 2.720 (-28.3%) | 0.29 / 0.30 (-4.9%) | 1425/1512 |
| mi355-4 10-12 nosync_tp4_c1 rc=0 (all 40 whole, PRODUCTION layouts, + task 9b nosync patch, snap_task9b_1012) | 4 | 1 | 7,230.8 / 6,019.4 (+20.1%) | 570.0 / 406.5 (+40.2%) | 1.670 / 2.230 (-25.1%) | 0.67 / 0.88 (-23.4%) | 321/332 |
| (ref) ATOM run 37597985033 tp4 c1 | 4 | 1 | 7,282.9 / 6,019.4 (+21.0%) | 577.3 / 406.5 (+42.0%) | 1.520 / 2.230 (-31.8%) | 0.92 / 0.88 (+4.1%) | 323/334 |
| mi355-4 10-12 task8_tp4_c1 rc=0 (production layouts, all 40 whole + nosync + draft mono, snap_task8b_1012) | 4 | 1 | 7,331.2 / 6,019.4 (+21.8%) | 587.9 / 406.5 (+44.6%) | 1.580 / 2.230 (-29.1%) | 0.71 / 0.88 (-19.5%) | 324/335 |
| mi355-4 10-12 task8_tp4_c2 rc=0 | 4 | 2 | 7,405.3 / 6,169.7 (+20.0%) | 538.7 / 350.6 (+53.6%) | 1.570 / 2.240 (-29.9%) | 0.41 / 0.59 (-30.3%) | 495/517 |
| mi355-4 10-12 task8_tp4_c4 rc=0 | 4 | 4 | 9,268.7 / 8,807.1 (+5.2%) | 487.2 / 331.6 (+46.9%) | 1.590 / 2.300 (-30.9%) | 0.34 / 0.50 (-32.1%) | 712/756 |
| (ref) ATOM run 37597985033 tp4 c4 | 4 | 4 | 9,209.5 / 8,807.1 (+4.6%) | 429.0 / 331.6 (+29.4%) | 1.530 / 2.300 (-33.5%) | 0.43 / 0.50 (-13.7%) | 704/748 |
| mi355-4 10-12 task8_tp4_c8 rc=0 | 4 | 8 | 16,335.2 / 15,361.7 (+6.3%) | 393.9 / 282.2 (+39.6%) | 1.820 / 2.720 (-33.1%) | 0.31 / 0.30 (+4.3%) | 1429/1516 |
| mi355-4 10-12 task9c_tp4_c1 rc=0 (task8 + 9c sim-accept fold, snap_task9c_1012) | 4 | 1 | 7,405.4 / 6,019.4 (+23.0%) | 602.4 / 406.5 (+48.2%) | 1.54 (ITL p50 1.543) / 2.230 | 0.73 / 0.88 | 324 |
| mi355-4 10-12 task9c_tp4_c8 rc=0 | 4 | 8 | 16,345.7 / 15,361.7 (+6.4%) | 381.7 / 282.2 (+35.3%) | 1.80 (ITL p50 1.802) / 2.720 | 0.28 / 0.30 | 1428 |
| (ref) ATOM run 37597985033 tp4 c8 | 4 | 8 | 16,102.4 / 15,361.7 (+4.8%) | 330.5 / 282.2 (+17.1%) | 1.940 / 2.720 (-28.7%) | 0.48 / 0.30 (+58.7%) | 1424/1511 |

30 vs 40 layers whole (same V4 layout, node, flags; base = same branch no mono):

| conc | P90 base / 30 / 40 (40 vs 30) | TPOT p50 ms base / 30 / 40 | TTT/gpu base / 30 / 40 |
|---|---|---|---|
| 1 | 399.2 / 523.4 / 536.6 (+2.5%) | 2.280 / 1.820 / 1.760 | 6,251 / 6,965 / 7,047 |
| 2 | 372.4 / 511.4 / 530.7 (+3.8%) | 2.280 / 1.810 / 1.760 | 6,339 / 6,947 / 7,077 |
| 4 | 337.0 / 469.8 / 494.0 (+5.2%) | 2.320 / 1.820 / 1.780 | 9,078 / 9,367 / 9,414 |
| 8 | 291.8 / 367.1 / 380.3 (+3.6%) | 2.710 / 2.020 / 1.950 | 15,314 / 16,062 / 16,179 |
| mi355-4 10-11 base_tp4_c1 rc=0 (same branch, NO mono, prod flags) | 4 | 1 | 6,251.0 / 6,019.4 (+3.8%) | 399.2 / 406.5 (-1.8%) | 2.280 / 2.230 (+2.2%) | 0.63 / 0.88 (-28.7%) | 299/310 |
| mi355-4 10-11 base_tp4_c2 rc=0 | 4 | 2 | 6,338.8 / 6,169.7 (+2.7%) | 372.4 / 350.6 (+6.2%) | 2.280 / 2.240 (+1.8%) | 0.38 / 0.59 (-35.1%) | 472/494 |
| mi355-4 10-11 base_tp4_c4 rc=0 | 4 | 4 | 9,077.7 / 8,807.1 (+3.1%) | 337.0 / 331.6 (+1.6%) | 2.320 / 2.300 (+0.9%) | 0.37 / 0.50 (-25.3%) | 683/727 |
| mi355-4 10-11 base_tp4_c8 rc=0 | 4 | 8 | 15,313.5 / 15,361.7 (-0.3%) | 291.8 / 282.2 (+3.4%) | 2.710 / 2.720 (-0.4%) | 0.30 / 0.30 (-1.2%) | 1392/1479 |

Mono-only attribution (same node, branch, flags; mono vs base): FFN-only P90 c1 +18.7% (473.8 vs 399.2), c2 +21.5%,
c4 +17.8%, c8 +19.2%; whole-layer (V4) P90 c1 +31.1% (523.4 vs 399.2), c2 +37.3%, c4 +39.4% (469.8 vs 337.0), c8 +25.8% (367.1 vs 291.8); TPOT c1 2.280 -> 1.820 ms (-20.2%).
The base itself is ~CI (P90 -1.8..+6.2%): the earlier CI deltas are essentially mono.

(For reference, m2m-255 opt1008_tp4_c1_r2: P90 434.7, TPOT 2.110 -- different node and code base.)
Caveat: these deltas are NOT mono alone: vs CI the run also differs in code base (HaiShaw perf/v41-mono = newer
upstream main), OPUS sparse prefill on, QR INT8, fusion off. The mono-only attribution needs the same sweep with
--enable-mono-kernel removed (fusion on) on this node -- not run yet. The sim-AL TPOT A/B (mono vs prod: c1-c2 -11%,
c4-c8 parity) is the clean mono-only number.

### Task 9: c1 gap vs ATOM run 37597985033 (mi355-4, 2026-10-12)

ATOM tp4 c1 artifacts (gh run download): /shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_atom_37597985033/tp4_c1/
(agentic/ = aiperf, server_logs/x/ = untarred srt logs). ATOM log: `V4.1 mono decode on` + `V4.1 mono draft on`,
acceptance rate 50.2% at K5 = AL 3.51 (same as our SIMULATE_ACC_LEN 3.51), so ITL ratio = step-time ratio.

agg (vs CI ref): ATOM P90 577.3, TPOT p50 1.520 ms; ours all-40 (V4) P90 536.6 (-7.1% vs ATOM), TPOT 1.760 (+15.8%).
Per-request join (`scripts/agentx_join_itl.py <atom agentic> <ours>`, 311 joined requests, median ITL ms by ISL):

| ISL | n | ATOM | all-40 | ratio | base (no mono) ratio |
|---|---|---|---|---|---|
| <32k | 8 | 1.642 | 1.725 | 1.049 | 1.453 |
| 64-128k | 63 | 1.510 | 1.733 | 1.142 | 1.487 |
| 128-256k | 43 | 1.501 | 1.738 | 1.152 | 1.496 |
| 256-512k | 128 | 1.521 | 1.765 | 1.163 | 1.489 |
| >=512k | 63 | 1.529 | 1.805 | 1.179 | 1.510 |

LSQ fit ITL = a + b*ISL: ATOM 1.681 - 0.032/100k, ours 1.761 + 0.008/100k. Reading: the gap is ~0.22 ms/token
(~0.8 ms/step at AL 3.51) and almost ISL-independent; the context-dependent part (our indexer path) is at most
~0.07 ms/token at >=512k (~4% of ITL). So the bulk is per-step fixed cost: draft (ATOM runs DSpark draft on mono,
we do not), host / step metadata, LM head / sampling, or the 40-layer launches themselves -> profile decides.
Profiles (task9_profile_chain.sh, frozen snapshot /sgl-workspace/snap_task9_1012, GPUs 4-7, c1, sim AL 3.51,
production layouts): profile_tp4/mi355-4_task9_{mono_d128k,base_d128k,mono_d8k}_c1{,_gpuonly}/ (+ mono_d128k_c1_hipwait).

Clean c1 TPOT (decode_load_profile, 1 req x 1024 tok): mono 128k 1.717 / 8k 1.701 ms (context 8k->128k = +0.016 ms);
base (no mono, prod flags) 128k 2.212 / 2.231; ATOM AgentX 1.52. Per step (TP0, GPU-only trace, `task9_step_breakdown.py`,
`atomport_step_spans.py`), mono 128k:

| part | us/step | what |
|---|---|---|
| pre | 354 (62 kernels) | step metadata: schedule prep, compressed-attn meta, page-table pad/expand, engram hash, copies |
| mono K1/K2/kt (40 layers) | 3,369 (88) | K2 r1 64 us x20, r2 65 x18; K1 ~20 x40 |
| between (non-mono inside the 40 layers) | 768 (127) | indexer + compressor ~500-550 (pa_mqa_logits 81, topk_* 123, idx gemm/quant ~140, index_q_pack 33, candidate/compact ~40, flash_c1/c2 17), engram ~20, elementwise glue ~130, router_gemv/nccl/mhc_post ~95; 8k: 661 |
| post | 396 (67) | LM head (Cijk 55) + logits gather (nccl 49) + DSpark draft (SGLang path) |
| GPU busy total | 4,947 | base: 7,184 (1,129 kernels/step) |
| host gap between steps | 1,661 profiled (base 1,615) | GPU idle; clean estimate ~0.65-1.1 ms (clean step at AL 3.51 = 6.03 ms vs 4.95 GPU) |

Host gap cause: every run_batch starts with `future_map.resolve_seq_lens_cpu` -> on HIP `publish_ready.synchronize()`
(overlap_utils.py, #26672 workaround "Event.wait() regresses TPOT on MI355"); publish fires between verify and
draft_extend, so the host only starts the next step's prep (~350 us + step[DRAFT] ~350 + step[VERIFY] ~210 host,
2 graph launches, multinomial, 3x c10d broadcast, mono-only eager aiter gemm_a8w8_blockscale x2 + rmsnorm ~200 us)
when the GPU has ~0.4 ms (draft_extend) left -> bubble. Same size with and without mono (not a mono issue).
Test: snapshot with `.wait()` on HIP (snap_task9_hipwait): TPOT 1.709 vs 1.717 = no gain (another host sync remains
downstream; trace splits the step differently) -> not a one-line fix.

Attribution vs ATOM (step 5.34 ms at AL 3.51; ours 6.03): our GPU work alone (4.95 ms) is already below ATOM's whole
step, so the gap is the per-step host bubble, not kernels. Indexer: context-dependent cost ~0.11 ms/step (128k vs
8k) and K2a could replace ~0.4-0.5 ms/step of indexer kernels on 8 layers (net ~0.3 ms after K2a's own cost, ~5%);
draft on mono: <= ~0.25 ms GPU + part of the host draft prep. Decision (pending user): K2a NOT first; first remove
the host bubble (device-written step metadata / no per-step publish sync, as ATOM), then task 8, then K2a.

Task 9b A/B (mi355-4, 2026-10-12, `scripts/task9b_ab.sh`, no profiler, all-40 mono, sim AL, ISL 131072 OSL 1024, 3 reps):
patch = /sgl-workspace/snap_task9_nosync (scheduler.py run_batch: when `FutureMap.needs_cpu_seq_lens` is False --
DeepseekV4HipRadixBackend target + DSpark draft without SGLANG_OPT_USE_ONLINE_COMPRESS -- call
`resolve_seq_lens_cpu(batch, stream_ordered=True)` inside forward_stream_ctx after wait_stream; overlap_utils.py skips
the HIP `publish_ready.synchronize()` in that mode: publish_ready is recorded on forward_stream, so stream order fences
the new_seq_lens_buf gather). Not in /sgl-workspace/sglang yet.

| | c1 TPOT ms (r1/r2/r3, mean) | c4 TPOT ms (mean) |
|---|---|---|
| orig (snap_task9_1012) | 1.731 / 1.695 / 1.678, 1.701 | 2.486 / 2.400 / 2.362, 2.416 |
| nosync | 1.585 / 1.595 / 1.578, 1.586 (-6.8%) | 2.347 / 2.305 / 2.328, 2.327 (-3.7%) |

Soak on nosync (ISL 32768 OSL 4096, c1/2/4/8 TPOT 1.564/1.833/2.346/2.928): no hang, 0 Tracebacks; GSM8K 1319 (real
acceptance, parallel 8) 0.901 (pass >= 0.895). vs ATOM ITL 1.52: c1 gap ~11-14% -> ~4% (if AgentX follows the
decode-load numbers). The earlier hang was the profiler (/start_profile at c1), same signature as 10-10.

### Task 9c: fold the simulated accept into the verify graph (mi355-4, 2026-10-12)

Finding: AgentX throughput runs set SGLANG_SIMULATE_ACC_LEN=3.51 (agentx_dsv41_mi355x_sglang.sh, also ATOM's AL), and
`fold_eligible` (dspark_worker_v2.py) required `simulate_acc_len <= 0` -> every AgentX step ran the eager accept
(accept_draft_tokens + 3 TP broadcasts + finalize + BuildOutTokens + simulated bonus) and the eager commit_hidden
(draft write_target_hidden_kv: the "mono-only eager gemm_a8w8_blockscale x2 + rmsnorm" of task 9), on top of the verify
graph's own (gate-0) epilogue. AgentX sends no temperature and the checkpoint has no generation_config -> requests are
temperature 1.0; with SIMULATE_ACC_GREEDY (HIP default) draft/accept run greedy and only the bonus is sampled, so the
draft proposal already folds; only the accept side was eager.
Patch (in /sgl-workspace/sglang, snapshot /sgl-workspace/snap_task9c_1012; switch SGLANG_DSPARK_SIMULATE_FOLD, default 1):
- simulated_bonus.py: `simulated_bonus_sample_graph` -- seed and per-request temperature read from device buffers
  (temperature <= 0 = argmax), only the row at correct_len is sampled (eager samples all bs*5 rows).
- DsparkVerifyEpilogue(simulate_fold=...): in `_accept`, after accept_greedy_triton and BEFORE the TP broadcasts,
  correct_len := sim_correct_len_buf, bonus := graph sample -> finalize / out tokens / commit inject use the simulated
  values; `begin_sim_step(bs, sim_acc_len, temperatures)` writes the buffers (1-3 fills) before the verify replay.
- worker: fold_eligible allows sim when the epilogue has simulate_fold; draws sample_simulated_acc_len per step.
Unit test (/tmp/t9c_bonus_test.py, GPU 4): temp 0 == argmax at the selected row; graph replay with new seeds gives new
draws; peaked row p=0.716 -> empirical 0.720 over 2000 replays.

A/B round 1 (task9b_ab.sh, same snapshot, env switch, ISL 131072 OSL 1024, temperature 0, 3 reps; log
profile_tp4/task9c_ab_mi355-4.log):

| | c1 TPOT ms (r1/r2/r3, mean) | c4 TPOT ms (mean) |
|---|---|---|
| eager (FOLD=0) | 1.520 / 1.483 / 1.519, 1.507 | 2.258 / 2.226 / 2.191, 2.225 |
| fold (FOLD=1) | 1.436 / 1.459 / 1.478, 1.458 (-3.3%) | 2.298 / 2.264 / 2.258, 2.273 (+2.2%?) |

fold: soak c1/2/4/8 (ISL 32k OSL 4k) 1.550/1.742/2.098/2.896, no hang, 0 Tracebacks; GSM8K 0.906 (real acceptance ->
fold path not used there, regression check only). c4 +2.2% is within server-to-server spread -> round 2 (ABBA order,
temperature 0 and 1, c1/4/8 x2; profile_tp4/task9c_ab2_mi355-4.log).

Round 2 (order fold_t0, fold_t1, eager_t1, eager_t0; TPOT ms mean of 2; 0 Tracebacks everywhere):

| | c1 | c4 | c8 |
|---|---|---|---|
| fold, temp 0 | 1.477 | 2.200 | 3.072 |
| eager, temp 0 | 1.496 | 2.260 | 3.036 |
| fold, temp 1 | 1.507 | 2.279 | 3.096 |
| eager, temp 1 | 1.526 | 2.261 | 3.195 (r1 3.302 outlier, r2 3.088) |

Verdict: fold is a small, consistent c1 gain (round 1 -3.3%, round 2 -1.3% at both temps); c4/c8 differences change
sign between rounds = server-to-server noise (~±2%). So the eager accept + commit is only ~0.02-0.05 ms/token
(~0.1-0.2 ms/step), far below the estimated ~0.6 ms/step bubble -> the bulk of the remaining bubble is elsewhere
(host draft/verify prep, scheduler, 2 graph launches + c10d broadcasts). Keep the patch (default on, harmless at
c4/c8). Next for 9c: a CPU-side trace of one c1 step on the fold snapshot (py-spy/record_function, not the torch GPU
profiler which drops verify-graph kernels under nosync) to rank host work between graph replays.

Host trace (scripts/task9c_hosttrace.sh + pyspy_rank.py; fold snapshot, 1 req ISL 131072 OSL 16384, TPOT 1.467 ms,
py-spy --idle --threads 10 s on TP0 scheduler; agentx/mi355-4_t9c_trace/pyspy_tp0.txt): MainThread 75.5% in
`copy_done.synchronize()` (process_batch_result_decode, waiting for batch N while N+1 is already launched:
event_loop_overlap launches before it processes, overlap is only disabled for consecutive prefills / grammar);
host work ~24% ~= 1.2 ms per 5.15 ms step (run_batch 17%: draft propose 4.9%, verify prep+launch 4.2%,
spec_prepare_for_decode 1.9%, graph load_batch/fill_from 4.7%; get_next_batch 2.6%, recv/broadcast_pyobj 1.6%).
-> At steady c1 decode the host is ~3.9 ms/step AHEAD of the GPU: no host bubble remains on the critical path
after nosync + draft mono + fold. The old "6.03 vs 4.95 ms" bubble estimate is stale; what is left is GPU-side
(kernel time + idle gaps inside/between the draft and verify graphs).
rocprofv3 does not work here: `--attach` (rocm 7.2) reports success but writes no files; wrapping the server
(SERVER_CMD_PREFIX in agentx_colleague_mi355x_sglang.sh, MODE=rocwrap in task9c_hosttrace.sh) aborts every rank at
the first decode step with HSA_STATUS_ERROR_INVALID_PACKET_FORMAT (interception vs mono / graph dispatch).
Estimate without a new trace: task 9 GPU busy 4.95 ms/step minus draft-on-mono (<= 0.25 ms) ~= 4.7 ms vs clean
step 1.467 x 3.51 = 5.15 ms -> ~0.45 ms/step of GPU idle that is NOT host-bound (launch gaps between graph nodes /
between the draft and verify graphs). To measure it: torch GPU-only profile with the nosync path forced off (needs an
env switch in decide_needs_cpu_seq_lens; kernels identical, only the step-boundary gap changes) -> intra-step gaps.

### Task 14: new ATOM baseline run 38056462998 (mi355-4, 2026-10-12)

Image rocm/atom-dev:nightly_202610100929 (old ref 37597985033 = nightly_202610070333). Artifacts tp4 c1/c4/c8 in
/shared_nfs/kk/results/DeepSeek-V4.1-Flash/ref_atom_38056462998/ (names.txt; gh here has no `-p`, use `-n <name>`).
bmk agg old -> new: c1 P90 577.3 -> 615.8 (+6.7%), TPOT p50 1.52 -> 1.53, TTFT p50 0.92 -> 0.71 s; c4 429.0 -> 433.1,
TPOT 1.53 -> 1.56; c8 330.5 -> 348.3 (+5.4%), TPOT 1.94 -> 1.94.
"intvty p90" = 1000 / p90 of per-request ITL (slow tail), not a median metric: ITL p90 old 1.737 / new 1.628 / ours
(task8 c1) 1.702 ms -> 577 / 615 / 588. Per-request join old vs new ATOM: median ITL ratio 1.005 (no per-token
speedup); the new ATOM only shrank its slow tail (ITL std 0.30 -> 0.18 ms).
Who is in the slow 15% (/tmp/itl_tail.py): ATOM = short-OSL requests (median OSL 165-262, per-request start cost;
OSL<64 med ITL 1.80-1.87 ms); ours = long-ISL requests (median ISL 340k), our ITL is flat in OSL (1.57-1.58) but grows
with ISL (>=512k 1.618 vs ATOM 1.541, 1.048x). -> To pass the new ATOM c1 P90 (ITL p90 < 1.626) the context-dependent
cost (indexer, K2a) matters as much as the fixed per-step cost (9c).
What changed: recipe config_tp4_c1.yaml identical except the image. ATOM origin/main (/tmp/mono_src/ATOM, fetched)
commits in the window: main suspect fbb89df0 "perf(deepseek_v41): packed FP4 index logits and a two-source BF16 decode
(#2513)" -- the new server log prints `V4.1 score workspace: packed logits, 1.55 GiB` and compiles an extra
`backbone_late` graph. (1) Index scorer FP32 logits packed per row to its own visibility (was a [rows, max_model_len]
plane, 6-8 GiB at 1M) + ScorePlanner step planner; needs aiter#6334 (top_k_per_row_decode(row_starts=, plane_width=),
flydsl_pa_mqa_logits_fp4(out_offsets=)). (2) Decode attention reads two sources (selection from pool + window via the
layer ring), `sparse_attn_v4_paged_2src`, OPUS split-K when the step has too few tokens to fill the GPU. Others: a526f0d5
(decode block tables at padded running_bs), 0ed4975b (FP4 sparse indexer output contract), 7fc0926c (mono mailboxes
fence in one FlyDSL launch).
Verdict (mi355-4, 10-12): the c1 P90 jump is a cliff artifact, not a speedup. ATOM's per-request ITL distribution
has a gap: bulk <= 1.66 ms, then a ~10% slow cluster >= 1.70 ms (short-OSL / overlapped requests: OSL<16 ITL
1.9-2.7, 8-10 of the slow 32 have OSL<64). p90 sits on that cliff: old 321 reqs -> rank lands above (1.737, P90 577),
new 325 reqs with one fewer slow -> below (1.628, P90 616); without OSL<16 both are 1.645 vs 1.620. Per-request
median unchanged (join 1.005). EPLB is off in both (eplb_enable False). #2513 is not a decode-speed change for us:
packed logits = memory only (available_for_kv 145.0 -> 149.0 GB, cache hit identical, compute reads the same
`columns < seen`), two-source decode = BF16 pool only (AgentX runs the FP4 pool). -> Do NOT port #2513 for speed.
ATOM c1 P90 is expected to jump between ~577 and ~616 run to run; compare by median / per-request join.
Ours: smooth distribution (p50 1.581, p90 1.70, no cliff), slow tail = long ISL. Beating ATOM's c1 P90 robustly
needs our ITL p90 < ~1.62, i.e. ~5% off long-ISL requests -> context-dependent indexer cost (K2a) is the lever.
Scripts: agentx_overlap.py (decode vs other prefills: 10-12 reqs overlap, not the driver), /tmp/short_osl.py.

### AgentX night 10-12 (task9c = task8 + 9c fold, snap_task9c_1012; TTT/GPU tok/s / P90 intvty; scripts/agentx_table.py)

| TP | conc | task9c | mi355-4_task8 | CI 37423 | ATOM 1010 | ATOM 1007 | task9c vs CI (TTT / P90) | vs ATOM 1010 (TTT / P90) |
|---|---|---|---|---|---|---|---|---|
| 2 | 1 | 13,834 / 508.2 | - | 11,288 / 345.0 | 13,563 / 492.3 | - | +22.6% / +47.3% | +2.0% / +3.2% |
| 2 | 4 | 18,834 / 410.5 | - | 16,575 / 297.9 | 18,457 / 345.6 | - | +13.6% / +37.8% | +2.0% / +18.8% |
| 2 | 8 | 31,165 / 306.2 | - | 29,765 / 239.3 | 30,894 / 261.0 | - | +4.7% / +27.9% | +0.9% / +17.3% |
| 2 | 16 | 56,486 / 187.0 | - | 54,414 / 157.4 | 55,648 / 150.4 | - | +3.8% / +18.8% | +1.5% / +24.4% |
| 4 | 1 | 7,405 / 602.4 | 7,331 / 587.9 | 6,019 / 406.5 | 7,410 / 615.8 | 7,283 / 577.3 | +23.0% / +48.2% | -0.1% / -2.2% |
| 4 | 4 | 9,612 / 481.5 | 9,269 / 487.2 | 8,807 / 331.6 | 9,661 / 433.1 | 9,210 / 429.0 | +9.1% / +45.2% | -0.5% / +11.2% |
| 4 | 8 | 16,346 / 381.7 | 16,335 / 393.9 | 15,362 / 282.2 | 16,211 / 348.3 | 16,102 / 330.5 | +6.4% / +35.2% | +0.8% / +9.6% |
| 4 | 8_r2 | 16,124 / 399.3 | - | 15,362 / 282.2 | 16,211 / 348.3 | 16,102 / 330.5 | +5.0% / +41.5% | -0.5% / +14.6% |
| 4 | 16 | 29,208 / 273.2 | - | 28,106 / 214.6 | - | - | +3.9% / +27.3% | - / - |

TP2 mono GSM8K 0.904 (first TP2 mono validation, 40 whole, 24 widths -> mono). 0 Tracebacks on all 9 points. c8 rerun 399.3 vs first 381.7 and task8 393.9 -> the first c8 P90 drop was run-to-run variance (prefill interference), no PDI change needed. ATOM 1010 has no tp4 c16 artifact.

### P0.3 FFN-only launch microbench (mi355-4, 2026-10-10)

`scripts/mono_ffn_microbench.py` (vLLM #60397 kernels via namespace shim, no vllm): TP4 on GPUs 4-7, 8 layers with their
own random weights (cold, ~1.9 GB/layer), FFN launches back to back in one HIP graph, 50 replays; UNION=k biases the
router so the step's expert union is <= k (real decode at M=24 has 16-31 distinct experts/layer, ATOM_PORT.md);
UNION=0 = unbiased random routing (union ~ 6M, pessimistic). All outputs finite. Results:
/shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/p0_ffn_bench/ffn_tp4_union{0,8,32}_mi355-4.csv

| M (rows) | union 0 | union 8 | union 32 | (us / layer) |
|---|---|---|---|---|
| 1 | 38.0 | 37.9 | 38.0 | |
| 6 (bs1 x DSpark 6) | 58.9 | 42.7 | 48.9 | |
| 12 (bs2) | 86.2 | 50.8 | 62.2 | |
| 24 (bs4) | 154.7 | 74.7 | 81.3 | |
| 48 (bs8) | 242.0 | 104.4 | 109.2 | |

One launch covers attention-output AR + FFN mHC seam + router/top-6 + shared expert + routed MoE + MoE AR.
Extra unions (M=6/12/24/48): union 48: 52.4/66.5/91.6/121.4; union 64: 53.5/70.3/101.2/129.3; union 96: 55.7/77.2/114.8/150.9.

Baseline = SGLang (dsv41-mono-ffn + aiter opt-1008, production flags: fusion ON, DSpark sim AL, ISL 4096) TP4 decode
profile, `scripts/trace_ffn_span.py` on TP0: wall time from the attention-output AR start to the MoE AR end (the span the
FFN launch replaces; graph replay, no gaps inside: kernel sum == span).

| | baseline span (median, p10-p90) | kernels | mono union 32 / 64 / 96 | speedup (union 32 / 64 / 96) |
|---|---|---|---|---|
| c1, M=6 | 75.2 us (73.2-78.5) | 11: AR+mHC post fused x2, sinkhorn, hc_boundary, router x2, sort, mx-quant-sort x2, moe1 15.6, moe2 8.5 | 48.9 / 53.5 / 55.7 | 1.54x / 1.41x / 1.35x |
| c8, M=48 | 134.7 us (113.6-144.4) | 12: aiter cross_device_reduce x2 (20.7), moe1 45.9, moe2 23.3, multiphase sort x2, ... | 109.2 / 129.3 / 150.9 | 1.23x / 1.04x / 0.89x |

Profiles: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_tp4/mi355-4_monobase_tp4_d4k_c{1_r2,8_r1}/ (8.84 / 12.13 ms
per step). Reading: c1 is a clear win (~1.1-1.3 ms/step of 8.8 over 40 layers); at M=48 the FFN-only launch is
break-even unless the real per-layer expert union is <= ~48 -- SGLang's fused_moe scales well with the union and the
mono shared expert is a separate MXFP8 pass. Next: measure the real union per layer at c4/c8 (routing probe), and
remember the whole-layer launch (P2) also removes the attention-side kernels (~20 more per layer).
- 2026-10-10 mi355-4: baseline profile c1 (tp4_decode_profile.sh, ISL 4096, fusion ON, no mono code) HUNG right after
  /start_profile (07:52:48): ranks diverged (TP1 in next-step broadcast_pyobj, TP0/TP2 in synchronize, TP3 in graph
  replay), GPUs 100%. Server itself fine before (TPOT c1 2.19 ms). Stacks: profile_tp4/mi355-4_monobase_tp4_d4k_c1/pyspy_*.txt.
  Same class as the earlier c1 profile hang noted in tp4_decode_profile.sh. Rerun as _r2.

### P0.2 kernel sourcing / dependencies (mi355-4, 2026-10-10)

- ATOM #2479 mono is NOT importable standalone: atom/mono + atom/models/deepseek_v41/mono import
  atom.models.deepseek_v41.* (~91 sites), atom.model_ops.*, atom.utils.forward_context, atom.plugin. Using it = depending
  on the whole ATOM package (and its model classes / layouts: bf16 KV, A8W4 interleave).
- vLLM #60397 `vllm/models/deepseek_v41/amd/mono/` (common 803 + stages 2768 + attention 2461 lines + layer/runner/sources)
  imports NOTHING from vLLM outside itself; only the wiring file `amd/mono_decode.py` does (envs, CUDAGraphMode,
  distributed, forward_context, model classes, Mxfp4MoeBackend). External deps: flydsl (47 imports), aiter (4 symbols:
  `UncachedIpcHeap`, `buffer_ops`, `update_dpp_i32`, `kernel_signature`; all in aiter e7d2453f2 and in cc1773508), torch.
  Catch: its internal imports are absolute (`vllm.models.deepseek_v41.amd.mono.*`, 32 sites; 25 relative). vllm is not
  installed here.
- => Option C minimal dependency set: the vLLM `mono/` package (pinned @193922d6) + flydsl 0.3.4.1 + aiter. SGLang writes
  only the wiring (the `mono_decode.py` equivalent) + weight/metadata adapters. To import it without vendoring: a tiny
  import shim that maps the `vllm.models.deepseek_v41.amd.mono` namespace onto the pinned source dir (no real vllm), or a
  pip-installable wheel of that subpackage. Both keep the ~7.8k kernel lines out of sglang.
- vLLM mono covers 30 of 40 layers with K1+K2; FFN-only launch on layers 0, 1 (Engram), 2, 8, 14 (Engram), 20, 24, 28, 32,
  36. FFN launch contract: in x (reduced bf16 [M,5120]), residual bf16 [M,4,5120], post_mix [M,4,1], res_mix [M,4,4],
  pre_mix f32 [M,4]; it consumes the UNREDUCED wo_b output (pushes it to peers itself).

### Task 17 S1: K2i standalone indexer launch (mi355-4, 2026-10-13)

Agreed with the user (10-13): split 1 skeleton + switch, 2 walk plan, 3 scorer, 4 top-k tree, 5 candidates, 6 wire into
decode, 7 GSM8K / A/B; unit tests on synthetic q/K in SGLang's real layouts (no server dump), real data only via
`K2A_CHECK` once wired; hook = a `K2A=1` branch inside `low_ratio_index_topk_hip_decode` (after `_indexer_inputs`).
Steps 1-5 done, 6-7 not started.

Code (uncommitted, sglang dsv41-mono-ffn), `python/sglang/srt/models/deepseek_common/amd/mono/kernels/indexer/`:
- `index_plan.py` (ATOM 68506823, minus FP8 `score_tasks`), `topk.py` (verbatim), `score.py` (index_score.py: trees +
  FP4 driver; FP8 scorer dropped; pad `irow` = -1 like SGLang), `score_fp4.py` (only `_keys` changed: SGLang page-64 plane
  read through virtual 8-row pages, option (a) of the plan), `candidate_table.py` (ATOM kernel, caller-owned outputs).
- `walk_plan.py`: device Triton version of `fill_walk_plan` (1 CTA, 8 warps, ~6.4 us; first version 22 us).
- `k2i.py`: 256x512 persistent launch, own LDS + scratch, mailbox tag = epoch + 1 (K2 bumps the epoch once a layer);
  `timeline=True` build stamps each phase (`_stamp` after the scorer and every tree level).
- `runner.py` `K2iRunner`: persistent ilog/ibmax/icout/cand table, owners = a row's request's first row (kmeta word 2 =
  owner x page-table stride, so ATOM's tree code is unchanged), kinds full / producer / reindex.
- `spec.py`: `k2a_mode()` (`SGLANG_MONO_DSV41_K2A`, asserts INDEXER=1); not read by any caller yet.
- Test: `test/registered/mono/test_dsv41_k2a_indexer.py` (11 pass, GPU 4).

Numerics (synthetic, rows = 6 a request): (e) walk plan == host plan on 1,000 random steps; (a) logits vs
`flydsl_pa_mqa_logits_fp4` max rel 1.4-2.0e-7 and vs fp64 1.1-1.5e-7 at 8k/128k/512k (not bitwise, as expected); exact
top-512 (chosen min >= unchosen max, ascending, -1 pad, pool rows through the page table); (b) same logits ->
`topk_transform_paged_hip` / `select_candidate_blocks_hip` sets identical; (c) full pipelines (FULL, producer blocks,
REINDEX on the producer's plane) vs SGLang: per-row Jaccard min 1.0 at 8k+20k / 128k / 300+512k.

Speed (HIP graph, GPU only, 1 request = 6 rows; us a call; K2i includes walk plan 6.4 + weight copy + epoch bump ~8,
REINDEX also the candidate table 3.7; SGLang includes the decode schedule ~31 it runs per call here but once a step and
ratio in production). Logs: /shared_nfs/kk/results/DeepSeek-V4.1-Flash/mono/task17_s1_k2i_microbench_1013/.

| ctx | K2i full | K2i producer | K2i reindex | SGLang full+cand | SGLang reindex |
|---|---|---|---|---|---|
| 8k | 28.2 | 32.5 | 29.4 | 44.9 | 42.8 |
| 64k | 42.0 | 52.5 | 41.5 | 62.6 | 48.5 |
| 128k | 47.9 | 72.6 | 41.6 | 69.8 | 50.4 |
| 512k | 82.9 | 126.1 | 41.5 | 105.6 | 72.6 |

Per step (3 FULL at ctx/2, producer at ctx, 4 REINDEX; schedules / plans once a ratio): 8k K2i ~147 vs SGLang ~159 us,
128k ~260 vs ~250, 512k ~383 vs ~468. Task 9's real 128k figure for SGLang (logits+top-k+cand 244) agrees.
Timeline (128k FULL): scorer done 8 us, then each tree level ~10-12 us (poll 1, keys 1.3, 4 radix passes ~6, emit ~3.5);
128k = 2 sel levels, 512k = 3, the producer adds 2-3 block levels. Tried (exact, tests pass, no speedup, reverted):
LDS-only barriers in `lds_topk`, skipping uncounted keys' atomics, early exit when the k-th's bin is taken whole.
Open question: why a radix pass is ~1.5 us (check ISA / VGPRs / spills of the K2i build).

Top-k alone, same logits (10-13, 6 rows, us; SGLang = HIP graph per kernel call, K2i = timeline tree phase):

| ctx | SGLang topk | SGLang cand | SGLang within | K2i full tree | K2i producer tree | K2i reindex tree | K2i score |
|---|---|---|---|---|---|---|---|
| 8k | 5.5 | 6.9 | 10.1 | 12.5 | 16.8 | 12.3 | 6.6 |
| 64k | 13.6 | 12.4 | 12.4 | 26.6 | 37.5 | 24.3 | 6.8 |
| 128k | 14.6 | 15.8 | 12.5 | 30.8 | 55.8 | 24.1 | 7.8 |
| 512k | 20.1 | 24.8 | 12.6 | 53.3 | 94.4 | 23.8 | 21.0 |

ATOM's tree is 2-2.7x slower than SGLang's top-k; K2i's scorer is the part that wins (REINDEX scores 16k columns
flat). ISA of the K2i build (S=6, bm=131072): 245 VGPRs, 106 SGPRs, **609 SGPR spills** (v_readlane refills after every
radix barrier), LDS 36.7 KB; each lds_topk is ~800 lines over 4 passes.

**Decision (user, 10-13): hybrid** -- K2i = ATOM's FP4 scorer only (`K2iBuild.tree=False`), the selection stays SGLang's
kernels. Wired as `SGLANG_MONO_DSV41_K2A=1`:
- `indexer/hook.py`: `step_plans` (in `init_forward_metadata_in_graph`, once a step and scored ratio: walk plan; the
  aiter decode schedule is then skipped for that ratio; result in `DSV4Metadata.k2i_planned`), `full_logits` (FULL
  layers and the producer: K2i logits -> SGLang's `select_candidate_blocks_hip` / `topk_transform_paged_hip` unchanged),
  `reindex_select` (consumers 24-36: `_reindex_table_kernel` moves the newest block to the last kept slot so ATOM's
  bound skips its unseen rows, K2i scores only the 16384 candidate columns = SGLang's compact row, then
  `topk_transform_paged_hip` on it + `_map_compact_selection_kernel`; no gather, no full-context logits).
- `low_ratio_backend_hip.py` `low_ratio_index_topk_hip_decode`: the K2A branch after `_indexer_inputs`; any step K2i did
  not plan (> 48 rows, other page size) takes SGLang's path. `SGLANG_MONO_DSV41_K2A_CHECK=1` (eager only) also runs
  SGLang's logits / selection and logs `[K2A_CHECK]` Jaccard + logits error every 64 layers.
- Tests: 14 pass (3 hybrid tests: Jaccard min 1.0 vs SGLang's path, 20k / 128k / 512k+70k).

Hybrid vs SGLang a layer (HIP graph, us; SGLang with a prebuilt decode workspace; schedule / plan a step and ratio):

| rows | ctx | FULL sg / k2 | producer sg / k2 | REINDEX sg / k2 | sched sg / plan k2 |
|---|---|---|---|---|---|
| 6 | 8k | 8.2 / 12.7 | 14.8 / 19.5 | 12.9 / 16.5 | 29.2 / 6.1 |
| 6 | 128k | 23.0 / 23.4 | 38.8 / 39.2 | 20.1 / 18.8 | 30.2 / 6.2 |
| 6 | 512k | 49.7 / 41.7 | 72.9 / 66.4 | 42.0 / 19.4 | 30.1 / 6.2 |
| 24 | 8k | 9.4 / 13.3 | 16.0 / 20.0 | 13.8 / 19.0 | 30.0 / 6.3 |
| 24 | 128k | 42.5 / 39.9 | 57.9 / 55.8 | 38.3 / 24.8 | 30.2 / 6.3 |
| 24 | 512k | 128.1 / 95.6 | 167.0 / 134.1 | 106.0 / 25.3 | 30.1 / 6.3 |

Per step (3 FULL at ctx/2, producer, 4 REINDEX, 2 schedules): c1 8k 149 -> 136, 128k 237 -> 191, 512k 410 -> 243 us;
c4 128k ~382 -> ~257 us. Snapshot for all runs: /sgl-workspace/snap_k2a_s1_1013 (scorer-only hybrid, 10-13).

Real-data check (10-13, eager, `K2A_CHECK=1`, GSM8K 100 + NIAH 120k chars x4; run mi355-4_task17_chk_gsm): FULL rows
16,824 Jaccard median / p1 / min 1.0 / 1.0 / 1.0; REINDEX rows 66,720 1.0 / 1.0 / 0.9961; logits max rel err 2.0e-7;
GSM8K 0.940 (100 q), NIAH 4/4, 0 Tracebacks.

## Log (append-only; node + date)

- 2026-10-10 mi355-4: env set up (branch, refs, InferenceX 47adb59a, backups).
- 2026-10-12 mi355-4: overnight: all-40 AgentX c4 P90 494.0 (+5.2% vs 30-whole), c8 380.3 (+3.6%); task 3 validated: all 40 whole on V41 + V41_FP4, GSM8K 0.907, TPOT -23/-21/-7/-6% vs prod.
- 2026-10-11 mi355-4: all-40 AgentX c1: P90 536.6 vs 30-whole 523.4 (+2.5%), TPOT 1.760 vs 1.820 (-3.3%), vs base +34.4% P90 -> fusing the indexer / compressor layers (indexer still SGLang's kernels) recovers only ~2.5% of the ~10% c1 gap to ATOM.
- 2026-10-11 mi355-4: whole-layer AgentX c4_r2 / c8_r2 done (P90 vs base +39.4% / +25.8%). Task 3 written (untested): BackBuild /
  MonoBuild.cfp4; key_rec returns is_c, V41_FP4 rows at off*256 / blk*256 + off*32; QK loads fp4 lanes at half offset and
  decodes with rocdl.cvt_scalef32_pk_bf16_fp4 x E4M3 scale (exact in bf16), per-lane select vs fp8; LDS raw row keeps an fp4
  row's 8 B at the fp8 position and its group's E4M3 scale in the free word 2; PV reads is_c per key from the key table and
  decodes with cvt_scalef32_pk_f32_fp4; runner.comp_record 288; spec resolves the compressed layout via
  resolve_compressed_kv_layout and no longer refuses V41 + FP4. CPU kit 21/21.
- 2026-10-11 mi355-4: base sweep done -> mono-only attribution (whole-layer P90 c1 +31%, c2 +37% vs same-branch no-mono).
  V41 / indexer / all-40 chains all crashed at start: (a) back.py V41 PV branch reused the name `nv` -> FlyDSL scf.if
  rewrite error (also broke V4; fixed: nv41); (b) DeepseekV41Indexer has no forward -> between() now calls
  backend.forward_low_ratio_sources (compressor then indexer, SGLang's serial path); (c) whole-layer AgentX c4/c8
  invalid (code edited mid-sweep). Rerun chain scripts/mono_rerun_1011.sh PID 635876 (all-40, V41, then AgentX c4/c8 from
  the frozen snapshot /sgl-workspace/sglang_snap_1011; tp4_gap_lane.sh got a SRC knob).
- 2026-10-11 mi355-4: layers 0/1/14 enabled behind SGLANG_MONO_DSV41_EDGE=1 (ratio 0 allowed, Engram allowed, layer-0 one-hot pre); all-40 chain queued. Correction: these were never SGLang limitations -- only vLLM's split, copied.
- 2026-10-11 mi355-4: tasks 4/5 hybrid written (untested): MonoBuild.export_idx (K1 writes bf16 x via store_x8 and
  q_lora in stage_qkv to runner.xn / runner.ql), build_mono_kt (front.stage_kt as a plain per-token launch, rebuilds the
  key table from the fresh top-k), runner.forward(between=...), spec: indexer/compressor layers whole when
  SGLANG_MONO_DSV41_INDEXER=1 (Packed.between), fused_mhc _mono_indexer_between (indexer then compressor, SGLang's serial
  order). Also fixed: whole_layer_meta cached the top-k view per step (would go stale if a copy were made); now re-read per
  layer with an int32 / contiguous assert. CPU kit 21/21.
- 2026-10-11 mi355-4: added the task list (gap to ATOM coverage was missing: I had taken vLLM's whole/FFN split as the target).
- 2026-10-11 mi355-4: P2b phase 1 (V41 main cache) written, untested on GPU: `kv41` build flag (MonoBuild/FrontBuild/
  BackBuild -> JIT key); K1 `_store_v41_record` (16 x 32-groups over all 512 post-RoPE dims, store_row_fp8 exponent, data at
  slot*512, 16 scales at block*512 + slot*16); K2 key_rec / 8 fp8 loads / per-32 scale bytes in QK and PV, no bf16 RoPE path
  (V4 code kept unchanged under const_expr); runner record 528/584; spec picks the layout via select_dsv4_kv_layout at pack,
  refuses V41 + FP4 compressed (phase 2). CPU kit 21/21. Validation queued (mono_v41_chain.sh).
- 2026-10-11 mi355-4: whole-layer r2 OK: GSM8K 0.901, TPOT vs prod c1 -19.9%, c2 -23.1%, c4 -7.9%, c8 -4.2%.
- 2026-10-11 mi355-4: overnight: AgentX mono c4 P90 +19.8%, c8 +23.2% vs CI (confounded, see caveat); V4 baseline GSM8K 0.908 + TPOT done; whole-layer runs crashed at pack: SGLang wo_a is bf16 on this base (checkpoint blocks 32x32 -> no fp8 absorb GEMM, dequantized at load). Fix: exact bf16 -> MXFP8 re-quant in pack (_mxfp8_blocks, checked + CPU test, kit 21/21). Rerun queued (/tmp/whole_retry.sh, log mono/whole_retry_mi355-4.log).
- 2026-10-10 mi355-4: whole-layer wiring written (spec MonoKernel.launch_whole + whole_layer_meta, Route.cache per step, _mono_whole in fused_mhc; SGLANG_MONO_DSV41_WHOLE=1 + SGLANG_DSV4_KV_LAYOUT=v4 required; 30 layers: ratio 1/2, no compressor/indexer/engram); CPU kit 20/20; GPU validation queued (mono_whole_chain.sh). User chose: V4 first, then V41 port if worth it.
- 2026-10-10 mi355-4: AgentX mono TP4 c1 vs CI: P90 +16.6%, TPOT -9.4%, TTT +5.9%.
- 2026-10-10 mi355-4: whole-layer metadata mapped (positions, swa_out_cache_loc, swa_page_indices, swa_topk_lengths, c{1,2}_sparse_raw_indices, page_table all available; token_to_req = arange); BLOCKER: production KV layout is V41 on gfx950, the vendored kernels read V4 (fp8_ds_mla) -> decision needed.
- 2026-10-10 mi355-4: TPOT sim-AL A/B: mono vs prod c1 -10.8%, c2 -10.6%, c4 +0.7%, c8 +0.3%.
- 2026-10-10 mi355-4: P1 step 4: MiniMax-M3 spec skeleton (models/minimax_m3_mono/spec.py: 60 layers WHOLE, widths 1..16, TP4, refuses at load: kernels not vendored) + test_mono_specs.py (two specs through one runtime; runtime sources free of model constants); CPU kit 20/20.
- 2026-10-10 mi355-4: GSM8K parallel-8 A/B: mono 0.901 (122.1 s) vs no-mono 0.911 (111.0 s); decode tok/s +3.9% at equal accept len.
- 2026-10-10 mi355-4: P1 step 3: DSV4.1 FFN-only spec + wiring; smoke r3 GSM8K 0.915 (200 q, parallel 128, mostly fallback). r1 refused (registry arch), r2 crashed (keep_plain -> cuda_fp8.h JIT), both fixed.
- 2026-10-10 mi355-4: P1 step 2: L1 runtime + --enable-mono-kernel + mono_hook + CPU kit (15/15).
- 2026-10-10 mi355-4: P1 step 1: vendored L0 -> sglang/srt/layers/mono/common/{device,plan,runtime} (ATOM @454d0e3, no deployment.py; dir not `lib/`: sglang .gitignore ignores **/lib/) and DSV4.1 kernels -> sglang/srt/models/deepseek_common/amd/mono/kernels/ (vLLM @193922d6), 9.5k lines, via scripts/mono_vendor.py (re-runnable, origin header per file). Vendored kernels bench identical to the shim run (union 32: 38.1/48.4/81.5/109.1 us at M=1/6/24/48).
- 2026-10-10 mi355-4: packaging = in-tree (user).
- 2026-10-10 mi355-4: P0.3 FFN microbench + baseline profiles (c1 75.2 us vs mono 49-56; c8 134.7 vs 109-151 by union). Baseline c1 profile hung once (rank divergence after /start_profile), rerun ok.
- 2026-10-10 mi355-4: aiter build rc=0, MiniMax-M3 downloaded, import check (L0 + L3 via shim) ok.
- 2026-10-10 mi355-4: design switched to multi-model (L0 atom/mono, L1 sglang/srt/layers/mono, L2 specs, L3 external kernels); second model MiniMax-M3, weights downloading.
- 2026-10-12 mi355-4: task 9: ATOM run 37597985033 tp4 c1 artifacts downloaded (ref_atom_37597985033/); per-request join: ITL flat vs ISL (ATOM 1.52, ours 1.73-1.81 ms), AL 3.51 both. c1 profiles 128k/8k mono + base (+GPU-only, +hipwait): GPU 4.95 ms/step (mono 3.37, indexer/glue 0.77, pre 0.35, post 0.40) + host bubble ~0.65-1.1 ms (HIP publish_ready.synchronize, same without mono). Gap vs ATOM = host bubble; K2a deferred (net ~0.3 ms). .wait() on HIP: no gain.
- 2026-10-12 mi355-4: task 9b first try = snap_task9_nosync (scheduler.py: when FutureMap.needs_cpu_seq_lens is False -- true for DeepseekV4HipRadixBackend target + DSpark draft without online compress -- resolve_seq_lens_cpu runs inside forward_stream_ctx and skips the HIP publish_ready.synchronize(); stream order fences the gather). Clean c1 128k TPOT 1.615 vs 1.717 ms (-5.9%, 1 request). The following /start_profile run HUNG (GPU6/TP2 0%, others 100%; TP0 broadcast_pyobj, TP1/TP3 synchronize in process_batch_result_decode, TP2 graph replay) = same signature as the 10-10 baseline c1 profile hang (no mono, no patch), so not attributed to the patch yet; stacks profile_tp4/mi355-4_task9_mono_d128k_c1_nosync/pyspy_*.txt; killed by PGID. Next: PROFILE=0 A/B x3 + GSM8K + a long c1/c4 load on nosync to rule out a patch-induced hang.
- 2026-10-12 mi355-4: task 9b A/B x3 (no profiler): nosync c1 TPOT 1.586 vs 1.701 ms (-6.8%), c4 2.327 vs 2.416 (-3.7%); soak c1-c8 32k/4096 no hang, 0 Tracebacks; GSM8K 0.901. Patch only in /sgl-workspace/snap_task9_nosync.
- 2026-10-12 mi355-4: nosync patch ported into /sgl-workspace/sglang (scheduler.py, overlap_utils.py; uncommitted), frozen snap_task9b_1012. AgentX TP4 c1 (production layouts): P90 570.0 vs ATOM 577.3 (-1.3%), TPOT p50 1.670 vs 1.520, ITL ratio 1.09x; 0 Tracebacks. GPU-only profile of nosync drops the verify-graph kernels (2.2 mono/step) -> remaining bubble only estimated (~0.6 ms).
- 2026-10-12 mi355-4: task 8 (in /sgl-workspace/sglang, uncommitted; snapshot snap_task8_1012): DSpark draft stages as whole-layer K1+K2 on a second runner (DRAFT_SPEC in amd/mono/spec.py: 128 experts / top-3 / ug_groups 1, swa_width 192 = window 128 + block 5 padded to 64; wqkv = [wq_a; wkv] copy; kernels: MonoBuild/FrontBuild experts/topk/ug_groups/swa_width, stage_kt reads swa_width slots; layers/mono/layer.py attach_spec; deepseek_v4_dspark.py lazy attach + _forward_stages_mono via _mono_whole). Switch SGLANG_MONO_DSV41_DRAFT=1. First try failed at capture (server_args.speculative_num_draft_tokens None -> use model.gamma). Sim-AL c1 128k TPOT 1.528/1.463/1.522 (mean 1.504 vs nosync 1.586, -5.2%), c4 2.152/2.172/2.113 (2.146 vs 2.327, -7.8%), 0 Tracebacks. Pending: real-acceptance GSM8K + accept len vs baseline 3.362 (draft correctness).
- 2026-10-12 mi355-4: task 8 debug: first draft-mono GSM8K 0.912 but real accept len 1.325 (baseline 3.362) = draft numerically wrong. Per-stage check (SGLANG_MONO_DSV41_DRAFT_CHECK=1, eager, scripts/task8_draft_check.sh): MoE 128/3 ok (SGLang attn + mono FFN 1e-2), K1 xn/q_lora/q exact, KV records = fp8 rounding, keys/sink/scale/rope/wo_a/wo_b ok -> bug: K1's front context lacked "swa_width" (added only to build_mono_kt's), stage_kt read rows at stride 128 vs 192. Fixed (front.py c["swa_width"]). After fix: stage hidden rel err median 6.4e-2 (stage 0) / 4.6e-2 (stage 2), accept len 3.333 (eager, 20 q). Rerunning A/B from snap_task8b_1012.
- 2026-10-12 mi355-4: task 8 fixed A/B (snap_task8b_1012, CUDA graph, no check): c1 128k TPOT 1.510/1.516/1.495 (mean 1.507, -5.0% vs nosync 1.586), c4 2.242/2.144/2.205 (2.197, -5.6% vs 2.327), 0 Tracebacks; GSM8K 1319 real acceptance 0.904, accept len 3.398 (n=156; nosync baseline 3.362), latency 96 s (baseline 99 s).
- 2026-10-12 mi355-4: AgentX TP4 c1/c2/c4/c8 sweep started 04:36 UTC from snap_task8b_1012 (production layouts, all 40 whole + nosync + draft mono, TAGP=mi355-4_task8; log mono/agentx_sweep_mi355-4_task8.log); both mono specs on, SIMULATE_ACC_LEN 3.51.
- 2026-10-12 mi355-4: AgentX task8_tp4_c1: P90 587.9 vs ATOM 577.3 (+1.8%), vs nosync-only 570.0 (+3.1%); TPOT p50 1.580 vs ATOM 1.520; per-request ITL median 1.037x ATOM (64k-512k bins 1.02-1.04x, >=512k 1.06x); TTFT p50 0.71 s vs ATOM 0.92; 0 Tracebacks. c2/c4/c8 running.
- 2026-10-12 mi355-4: AgentX task8_tp4_c2: P90 538.7 (CI 350.6 +53.6%, base 372.4 +44.7%, all-40 V4 530.7 +1.5%), TPOT p50 1.570; 0 Tracebacks. No ATOM tp4 c2 in run 37597985033 (c4/c8 bmk downloaded to ref_atom_37597985033/tp4_c{4,8}/).
- 2026-10-12 mi355-4: AgentX task8_tp4_c4: P90 487.2 vs ATOM 429.0 (+13.6%), vs all-40 V4 494.0 (-1.4%), base 337.0; TPOT p50 1.590 vs ATOM 1.530 / all-40 V4 1.780; TTFT 0.34 vs ATOM 0.43; 0 Tracebacks.
- 2026-10-12 mi355-4: AgentX task8_tp4_c8: P90 393.9 vs ATOM 330.5 (+19.2%), all-40 V4 380.3 (+3.6%), base 291.8; TPOT p50 1.820 vs ATOM 1.940; 0 Tracebacks. Sweep c1-c8 complete.
- 2026-10-12 mi355-4: task 9c: AgentX sim AL (3.51) forced the eager DSpark accept + commit_hidden every step (fold_eligible needed simulate_acc_len <= 0). Patch folds the simulated correct_len / temperature bonus into the verify graph epilogue (SGLANG_DSPARK_SIMULATE_FOLD=1, snap_task9c_1012). A/B x2 rounds: c1 TPOT 1.458/1.477 vs eager 1.507/1.496 (-1.3..-3.3%), c4/c8 within noise; soak ok, GSM8K 0.906, 0 Tracebacks. Bubble mostly elsewhere -> next: host-side trace of a c1 step.
- 2026-10-12 mi355-4: task 14 added: ATOM run 38056462998 (nightly_202610100929) c1 P90 615.8 (old 577.3), c4 433.1 (429.0), c8 348.3 (330.5); median ITL unchanged (join ratio 1.005), only the slow tail shrank (ITL p90 1.737 -> 1.628). intvty p90 = 1000/ITL p90. Our c1 tail = long-ISL requests (ISL-dependent), ATOM's = short-OSL. Suspect ATOM fbb89df0 (#2513, packed FP4 index logits + two-source BF16 decode, needs aiter#6334).
- 2026-10-12 mi355-4: task 9c host trace (py-spy TP0, fold snapshot, c1 ISL 128k): host waits 75.5% in copy_done.synchronize with N+1 already launched -> ~3.9 ms/step slack, no host bubble left; residual ~0.45 ms/step is GPU-side. rocprofv3 unusable (attach writes nothing; wrapped server aborts with HSA INVALID_PACKET_FORMAT at first decode).
- 2026-10-12 mi355-4: task 14 verdict: ATOM c1 P90 577 -> 616 is a p90-on-a-cliff artifact (bulk <= 1.66 ms, slow ~10% cluster >= 1.70 ms); per-token speed unchanged. #2513 (packed index logits / 2-source BF16 decode) = memory / BF16-only -> not a speed port. Our lever for c1 P90 is long-ISL cost (K2a).

- 2026-10-12 mi355-4: AgentX task9c_tp4_c1 (fold): P90 602.4 vs task8 587.9 (+2.5%), old ATOM 577.3 (+4.3%), new ATOM 38056462998 615.8 (-2.2%); ITL p50 1.543 / p90 1.665 (task8 1.581 / 1.702, new ATOM 1.534 / 1.628); per-request join vs new ATOM median 1.010 (task8 1.032), >=512k 1.028 (1.048); TTFT p50 0.73 s; 0 Tracebacks.
- 2026-10-12 mi355-4: AgentX task9c_tp4_c8 (fold): P90 381.7 vs task8 393.9 (-3.1%), new ATOM 348.3 (+9.6%), old ATOM 330.5 (+15.5%); ITL p50 1.802 (task8 1.821, new ATOM 1.943), p90 2.621 (task8 2.539, ATOM 2.876); join vs new ATOM median 0.963 (>=512k 1.023). agentx_overlap.py: slow 10% = 94% decodes overlapping another request's prefill; non-overlapped ITL 1.603 vs task8 1.644 (-2.5%, fold gain), overlapped 2.119 vs 1.993 (+6.3%, overlap frac 0.54 vs 0.47) -> the P90 drop is prefill-interference / scheduling variance, not kernels; 0 Tracebacks.
- 2026-10-13 mi355-4: night queue done (task9c, fold): TP4 c1/c4/c8/c8_r2/c16 P90 602.4/481.5/381.7/399.3/273.2, TTT/GPU 7,405/9,612/16,346/16,124/29,208; TP2 c1/c4/c8/c16 P90 508.2/410.5/306.2/187.0, TTT/GPU 13,834/18,834/31,165/56,486. vs ATOM 1010: TP2 P90 +3.2/+18.8/+17.3/+24.4%, TP4 c1 -2.2%, c4 +11.2%, c8 +9.6/+14.6%. TP2 GSM8K 0.904. c8_r2 confirms the first c8 drop was variance -> no PDI test. 0 Tracebacks.

- 2026-10-13 mi355-4: opt audit: none of HaiShaw perf/v41-{index-q-prefill-fuse, mhc-ar-boundary-stats, indexer-bf16-logits, candidate-blockmax-fastpath, small-moe-sort, flydsl-mxfp8-gemm, woa-fuse-mxfp8-quant} is in dsv41-mono-ffn; their env flags are unread (set by sweep_ci37423_opt.sh but no effect); tokenizer_worker_num 1. All mono / base runs = without these opts. Task 19 added.- 2026-10-13 mi355-4: task 17 S1: K2i (ATOM FP4 scorer + exact top-k tree + candidate table, device walk plan) vendored under mono/kernels/indexer/, 11 unit tests pass (logits rel 2e-7 vs flydsl and fp64; Jaccard 1.0 vs SGLang at 8k-512k); microbench per step K2i vs SGLang 8k 147/159, 128k 260/250, 512k 383/468 us -> no gain at 128k (tree levels ~10-12 us each). Not wired into decode; waiting on user decision.
- 2026-10-13 mi355-4: task 17 S1 -> hybrid (user): K2A=1 = K2i scorer + SGLang top-k/candidates, REINDEX on compact candidate logits, device walk plan replaces aiter decode schedule; 14 tests pass; microbench c1 128k 237->191, 512k 410->243 us/step, c4 128k 382->257; real-data K2A_CHECK (eager) Jaccard FULL min 1.0 / REINDEX min 0.9961, logits rel 2e-7, GSM8K(100) 0.940, NIAH 120k 4/4. Full GSM8K/NIAH/TPOT A/B running (task17_ab.sh, snap_k2a_s1_1013).
- 2026-10-13 mi355-4: task 17 A/B (snap_k2a_s1_1013): GSM8K k1 0.904 / k0 0.903, NIAH 60/60 both; TPOT k0->k1 8k c1 1.460->1.451, 8k c4 2.128->2.178, 128k c1 1.497->1.457 (-2.7%), 128k c4 2.189->2.223 (noisy); logs agentx/task17_ab_mi355-4.log, mi355-4_task17_k{0,1}_{gsm,ab}/.
- 2026-10-13 mi355-4: AgentX task17k1_tp4_c1 (K2A=1 hybrid, snap_k2a_s1_1013): TTT/GPU 7,389 / P90 612.7 (task9c 7,405 / 602.4; ATOM 1010 7,410 / 615.8 -> -0.5%, was -2.2%); per-request ITL vs task9c median 0.986 (256-512k 0.982, >=512k 0.977), slope 0.0066 -> 0.0020 ms/100k ISL; vs ATOM 1010 median 0.993 (>=512k 0.999, 64-128k 0.993). c4 running.
- 2026-10-13 mi355-4: AgentX task17k1_tp4_c4: TTT/GPU 9,494 / P90 511.5 (task9c 9,612 / 481.5 = +6.2% P90, -1.2% TTT; ATOM 1010 9,661 / 433.1 -> +18.1% / -1.7%); per-request ITL vs task9c median 0.988 (>=512k 0.966), vs ATOM median 0.977; 0 Tracebacks.
- 2026-10-13 mi355-4: task 18 done: 5 commits (framework / DSV4.1 layers / DSpark draft / host sync / K2A scorer), rebased onto haishaw/perf/v41-mono in worktree /sgl-workspace/sglang_v41mono (branch perf/v41-mono), fold ported onto upstream #43623 (sampled bonus in graph), pushed to https://github.com/HaiShaw/sglang/tree/perf/v41-mono at 3c5ea0778c. Local /sgl-workspace/sglang dsv41-mono-ffn holds the same 5 commits on the old base 6e8f3e09dd (old fold). Rebased validation: GSM8K 0.907, NIAH 20/20, TPOT within noise of old base. Not yet run on the new base: AgentX.
- 2026-10-13 mi355-4: AgentX TP4 c1/c4 on the rebased base started (snap_v41mono_rebased_1013, K2A=1, TAGP=mi355-4_task18reb, sweep PID 1114580, GPUs 4-7; log mono/agentx_sweep_mi355-4_task18reb.log).
- 2026-10-13 mi355-4: AgentX task18reb_tp4_c1 (rebased base, K2A=1): TTT/GPU 7,444 / P90 617.1 (task17k1 7,389 / 612.7; ATOM 1010 7,410 / 615.8 -> +0.5% / +0.2%, first time ahead at TP4 c1); per-request ITL vs task17k1 median 0.990, vs ATOM 0.982 (>=512k 0.990); 0 Tracebacks. c4 running.
- 2026-10-13 mi355-4: AgentX task18reb_tp4_c4 (rebased base, K2A=1): TTT/GPU 10,146 / P90 529.1 (task17k1 9,494 / 511.5 = +6.9% / +3.4%; ATOM 1010 9,661 / 433.1 -> +5.0% / +22.1%); per-request ITL vs task17k1 median 0.997 (decode unchanged; the TTT gain comes with the upstream +170), vs ATOM 0.972; 0 Tracebacks. GPUs 4-7 free.
- 2026-10-13 mi355-4: task 20 added (user): compare our mono design with zijiecode/sglang-mono-amd m3-mono (acf209105); next session, prompt MONO_COMPARE_SESSION_PROMPT_1013.md.
- 2026-10-13 mi355-4: task 20 comparison written to MONO_COMPARE_mi355-4.md (read-only; no code changes). Waiting for the user to approve the merge plan.
- 2026-10-13 20:50 mi355-4: task 20 done (MONO_COMPARE_mi355-4.md): commit 134919476c on perf/v41-mono-fw, perf/v41-mono fast-forwarded
  locally (not pushed). New open items 21-24.
- 2026-10-13 20:46 mi355-4: pushed HaiShaw/sglang perf/v41-mono 3c5ea0778c..134919476c; skills c5d88f4.
