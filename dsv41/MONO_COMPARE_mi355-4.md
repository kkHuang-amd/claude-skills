# Task 20: our SGLang mono design vs zijiecode/sglang-mono-amd `m3-mono`

Owner node: mi355-4
Parent: MONO_KERNEL_mi355-4.md (task 20). Findings and decisions go here; the parent only gets a status line + Log row.

## CONTINUE HERE

**Status (mi355-4, 2026-10-13 17:15):** branch `perf/v41-mono-fw` (worktree /sgl-workspace/sglang_v41mono, uncommitted;
ask before committing). F1-F7 done in the tree except the final validation. CPU tests 29 pass (run with
PYTHONPATH=/sgl-workspace/sglang_v41mono/python; the default import is /sgl-workspace/sglang); ruff (repo select) +
format + isort 7.0.0 (/tmp/isort_pkg) clean.
- F1 validated (A/B 10-13, same session, see "Framework branch results"): no regression.
- F5 moves: DSV4.1 kernels -> `python/sglang/kernels/ops/mono/gfx950/deepseek_v41/`, L0 device/plan -> `gfx950/common/`,
  L0 runtime -> `srt/layers/mono/runtime/`, indexer hook -> `srt/models/deepseek_common/amd/mono/indexer_hook.py`.
- F6 shared common: vLLM's private `common/` deleted; `attention/device.py` 505 -> ~190 lines (only attention-specific
  helpers + `kernel_symbol` (aiter namespace) + `stamp(c, point)`); index_plan / attention plan / gemv / front / back
  duplicates moved to common; JIT key = digest of `deepseek_v41` + `common` (before: own dir only, L0 edits did not
  rebuild K2i). **ISA check: 84 kernels (K1/K2 r0-2 +export, FFN, draft, K2i; widths 1,2,6,12,24,48) disassemble
  identically before/after F6** (scripts/mono_isa_dump.py; /shared_nfs/kk/results/DeepSeek-V4.1-Flash/isa_task20_f6/).
  The check caught one real bug on the way: front/back called `Mailbox(tag)` positionally = shared `layer` -> tag+1;
  fixed to `Mailbox(tag=tag)`.
- F6(c) NOT done on purpose: unreachable L0 code kept (M3 uses some of it, e.g. fp8x8_bf16_pk); decide with M3.
- F7: CI registration for the K2i GPU test; new CPU tests (switch, capture-width, check mode, state bindings,
  shared-common guard). Padding audit: DSV4.1 MoE does not mask CUDA-graph padding rows (they route like real rows;
  SGLang's module path masks them via num_token_non_padded) -> extra experts loaded at non-captured batch sizes; perf
  follow-up, not a correctness issue (padding KV goes to slot 0). Checked 20:50: no padding rows in our runs (every bs 1..8 captured, max_running_requests 8) -> no effect today.
**Running (mi355-4, GPUs 4-7, started 17:40 +08):** final validation, chain PID 1222260, log /shared_nfs/kk/fw_final.log,
script scripts/fw_final_chain.sh: fw6 (snap_fw_f6_1013) vs base (snap_v41mono_rebased_1013) GSM8K 1319 + NIAH 32/64/120k x20
+ TPOT x3, then AgentX TP4 c1/c4 on fw6 (TAGP mi355-4_fwfinal; compare mi355-4_task18reb 617.1 / 529.1). ~3-4 h.
Kill: `kill -- -1222260` then the GPU 4-7 sglang processes.
**Done (20:45 +08):** committed 134919476c on perf/v41-mono-fw (author wunhuang, no attribution; python/ equals the validated snap_fw_f6_1013) and fast-forwarded perf/v41-mono to it locally. Pushed to HaiShaw/sglang 10-13 20:45 (3c5ea0778c..134919476c). Next: user decides push and M3 (task 12).
**User decisions (2026-10-13 15:40):** (1) `--layer-backend mono`; (2) new branch off perf/v41-mono, cherry-pick their 3
commits [superseded 15:46, see below]; (3) [superseded 15:46: M3 kernels = their aiter PR6047 copy]; (4) M3 weights at /shared_nfs/MiniMaxAI/MiniMax-M3
(present 10-13: 796G, 60 safetensors, arch MiniMaxM3SparseForConditionalGeneration). ATOM M3 source on mi355-4:
/tmp/mono_src/ATOM @454d0e30, atom/models/minimax_m3/mono = 10 files (the P1 note said 25: check the newest ATOM commit).
**Plan revised by the user (15:46):** M3 kernel source = their aiter PR6047 copy. A new branch off perf/v41-mono does
ONLY the framework part of adopt / keep / drop (no M3 port, no cherry-pick of their whole commits); validate DSV4.1 for no
regression; then merge that into perf/v41-mono. M3 (ours vs zijie building it on our framework) is decided afterwards.
Framework steps on the new branch (proposal, see "Framework-only plan" below).
**Next:** after sign-off, step M0 of the merge plan (new branch off perf/v41-mono, cherry-pick their 3 commits side by side).
**Files:**
- Ours: `/sgl-workspace/sglang_v41mono` (branch perf/v41-mono @3c5ea0778c, pushed to HaiShaw/sglang).
- Theirs: `/tmp/mono_src/sglang-mono-amd` (m3-mono @acf209105b; commits 2649708e80, 7068473e8d, acf209105b on base
  16a23a672b, which **is an ancestor of our HEAD**: our tree is newer, so their commits rebase onto ours).
**Repro:**
```bash
git clone -b m3-mono https://github.com/zijiecode/sglang-mono-amd /tmp/mono_src/sglang-mono-amd
git -C /tmp/mono_src/sglang-mono-amd diff --stat HEAD~3..HEAD | tail -3   # 41 files
```
**Pass criteria:** every row filled with file:line evidence on both sides (done); a decision per row (done); a merge plan
the user signs off (pending).

## Size (mi355-4, 2026-10-13)

- Theirs: runtime `srt/layers/mono/` ~1,300 lines (contracts 90, families 59, dispatch 136, runner 54, peer_buffer 49,
  adapters/minimax_m3 111, minimax_m3/{config 52, metadata 45, runner 376, weights 316}); kernels
  `kernels/ops/mono/gfx950/` ~4,400 lines (common 370, minimax_m3/post_attn 2,583, pre_attn 681, index_score 378,
  layout 220, config 85, reduction 46); M3 prerequisites outside mono: aiter sparse PA (139) + aiter indexer (226) +
  minimax_sparse_backend (+199) + memory_pool SHUFFLE store (+23) + env flags; tests ~1,800 lines.
- Ours: L1 `srt/layers/mono/` 3,470 lines incl. the vendored ATOM L0 `common/{device,plan,runtime}` (~2,300);
  DSV4.1 kernels + spec `models/deepseek_common/amd/mono/` 10,632; MiniMax-M3 spec skeleton 65; tests 794.

## Comparison

Path prefixes: `O/` = ours `python/sglang/srt/`, `Om/` = `O/models/deepseek_common/amd/mono/`,
`T/` = theirs `python/sglang/srt/`, `Tk/` = theirs `python/sglang/kernels/ops/mono/gfx950/`.

| axis | ours | theirs | better / why | decision |
|---|---|---|---|---|
| package layout | L1 `O/layers/mono/` (spec, registry, routing, guards, failstop, layer) + vendored L0 `O/layers/mono/common/`; kernels inside the model tree `Om/kernels/` | runtime `T/layers/mono/` (contracts, families, dispatch, runner, peer_buffer, `adapters/<model>.py`, `<model>/`); device code in `Tk/` (`common.py` + `minimax_m3/`), following SGLang PR40204 / PR37465 (SOURCE.md:9) | theirs for kernels: SGLang's own convention for device code, separates kernels from host glue. Ours for the L1 split (model-agnostic runtime, model code outside `layers/mono`) | **merge**: L1 stays ours in `layers/mono/`; all device code moves to `kernels/ops/mono/gfx950/{common,deepseek_v41,minimax_m3}/`; model host glue stays next to the model |
| enablement flag, startup refusals | `--enable-mono-kernel` (`O/arg_groups/mono_hook.py:15-34`); default-deny over a `Feature` enum, every reason at once (`O/layers/mono/guards.py:23-40,43-65,82-87`); covers TBO, DP attn, EP, PP, PD disagg, LoRA, HiCache, EPLB, torch.compile, deterministic, spec verify | `--layer-backend {module,mono}` in `exec.kernel` (exec_.py diff); `validate_layer_backend_mono` raises the first failure only (`T/arg_groups/layer_backend_hook.py:35-82`): arch, HIP, TP, PP, EP / DP attn, LoRA, TBO; deprecated env alias with warning (`T/layers/mono/families.py:45-59`); model ctor raises off gfx950 / import failure (minimax_m3.py `_mono_decode`) | flag: theirs (an enum leaves room for other layer backends and reads better). Refusals: ours (complete list, all reasons at once, misses nothing they check) | **merge**: adopt `--layer-backend mono`, drop `--enable-mono-kernel` and the env alias (neither is upstream); keep our guards / `Feature` default-deny behind it |
| spec / contract API | one declarative `MonoSpec` per model: architectures, `layers(cfg) -> {id: LayerMode WHOLE/FFN}`, widths, `supports`, caches, `refuse_config/refuse_model/pack/bind/eligible/build` (`O/layers/mono/spec.py:83-105`); `LayerKernel.launch/health` (`:73-80`); arch -> spec import path, readable at config time (`O/layers/mono/registry.py:10-28`) | `MonoCapabilities(family, max_query_rows)`, `StepSpec(query_len, is_verify)`, `PreparedStep`, `LayerStateBinding` (HiCache per-layer wait), `ModelAdapter` protocol (attach / step_spec / build_runner / check_layer) (`T/layers/mono/contracts.py:20-90`); family table arch -> (name, tp_sizes) (`T/layers/mono/families.py:20-37`) | ours for the declaration (per-layer modes, features, widths, refusal hooks, a second spec for the draft model); theirs has three ideas we lack: `StepSpec` (query_len makes linear verify explicit), `LayerStateBinding` (HiCache), `check_layer` | **merge**: keep `MonoSpec` + registry as the single model entry; add `query_len` to `MonoStep`, a `state_bindings` hook (HiCache wait) and an optional `check` hook; their `families.py` folds into `registry.py` |
| dispatch into forward, CUDA-graph capture | decided once per step at model level: `begin_step` (`O/models/deepseek_v4.py:4486-4487`, `O/layers/mono/layer.py:56-70`), per layer `routed(i)` (`O/models/deepseek_common/amd/deepseek_v4_hip.py:247`) -> `launch / launch_whole` (`deepseek_v4_fused_mhc.py:706,766`); routing pure on (mode, rows, bs) (`O/layers/mono/routing.py:24-60`); build is **TP-agreed** (`layer.py:100-115`, `tp_agree` at :106) and raises if first reached inside a capture (`layer.py:95-98`) | decided in the decoder layer at slot 0 (minimax_m3.py diff `forward`, `mono.wants` + `begin`); per-step input checks (`T/layers/mono/dispatch.py:62-88`: n == bs x query_len, int64 positions / slots, row tables, not-yet-built in capture -> decline); `MonoUnsupported` at build turns mono OFF **on that rank only** (`dispatch.py:107-116`) | ours on safety: a rank-local OFF with in-kernel all-reduces can split ranks (one rank in the kernel, peers on modules -> hang). Theirs on input hygiene (cheap host checks before anything mutates) and the linear-EAGLE recogniser (`T/layers/mono/adapters/minimax_m3.py:36-50`) | **keep ours** (model-level decision + tp_agree); **adopt** their `_declined` checks into `routing.decide_for_batch` and their EAGLE recogniser as the M3 `eligible` |
| runner: per-width builds, scratch, epochs | per-width lazy build cached in `MonoRuntime._kernels` (`layer.py:87-115`); width scratch allocated outside capture (`Om/spec.py:357`); kernels advance their own mailbox epoch (`Om/kernels/runner.py:13-14,187-188`) | `MonoRunner.kernel(n)` lazy cache (`T/layers/mono/runner.py:28-33`); per-n row tables preallocated 1..16 (`T/layers/mono/minimax_m3/runner.py:196-203`); host step counter `step.add_(1)` after the last layer, tags = step x 128 + layer + 1 (`runner.py:371-375`, `Tk/minimax_m3/config.py:85`); layer-parity double buffers baked into per-layer arg arrays (`runner.py:166-195`) | equivalent; epoch scheme is kernel-family specific. Their runner owning all per-model state behind an ABC is a clean shape for `MonoSpec.build` | **keep both per model**; their `MonoRunner` ABC becomes what an M3 `MonoSpec.build` returns (lightweight; no L1 change needed beyond the `LayerKernel` protocol) |
| peer buffer / in-kernel AR | L0 `PeerBuffer` vendored from ATOM @454d0e3 with step fence + tp_agree on mapping (`O/layers/mono/common/runtime/peer_memory.py:26-70`, `step_fence.py`), but **DSV4.1 uses its private vLLM copy** (`Om/kernels/runner.py:181,203` -> `Om/kernels/common/peer_memory.py`); L0 `WidthBuilds` / `step_fence` / `compile` have no caller outside L0 | 49-line `PeerBuffer` over aiter `UncachedIpcHeap`, no fence, no agreement on a failed open (`T/layers/mono/peer_buffer.py:10-49`); doc of the half-layer double-buffer argument (`:11-20`) | ours (L0) has the safety parts; nobody uses them yet. Three PeerBuffers across the two trees | **merge**: one PeerBuffer = L0 `peer_memory.py`; port DSV4.1 and M3 runners onto it; drop their `peer_buffer.py` and our `Om/kernels/common/peer_memory.py`; keep their docstring |
| weights / load-time hooks | `pack(model, ctx)` after load, views where possible, wo_a requant to MXFP8 (`Om/spec.py:301-340`, `:151`); attach at first forward, refuses inside capture (`O/models/deepseek_v4.py:4128-4146`) | `SparseMoeLayer.from_layer` reads modules at runner build; attention converted to PTPC FP8 + aiter (16,16) shuffle (`T/layers/mono/minimax_m3/weights.py:59-81`); experts must already be aiter-shuffled MXFP4 (`:236-237`) | model-specific; both read loaded modules without touching param names. Theirs costs a second attention-weight copy (FP8) | **keep per model**; M3's `from_layer` becomes the M3 spec's `pack` |
| step metadata | `whole_layer_meta` per layer from ForwardBatch, memoised per step (`Om/spec.py:417-478`, `Route.cache` `layer.py:29-33`) | `prepare_token_rows`: in-graph block table from `req_to_token`, causal verify lengths (prefix + j + 1), clamp to max context, padding rows (slot <= 0) zeroed so they cannot join the MoE expert union (`T/layers/mono/minimax_m3/metadata.py:7-45`); GPU tests for replay refresh / padding (`test/registered/amd/test_minimax_mono_verify.py:39-135`) | theirs is better tested; the padding rule is worth checking on our side (graph padding rows in DSV4.1 MoE) | **keep ours for DSV4.1, adopt theirs for M3**; action: audit DSV4.1 padding rows vs their rule, add the same replay / padding GPU tests for `whole_layer_meta` |
| kernel sourcing | DSV4.1: vLLM #60397 @193922d6 (Apache-2.0, SPDX header per file, `Om/kernels/layer.py:2`), adapted from ATOM (`Om/kernels/README.md`); L0 + K2i scorer vendored from ATOM @454d0e3 / @68506823 with per-file origin headers (`O/layers/mono/common/device/sync.py:2`, `Om/kernels/indexer/score_fp4.py:2`) | M3: aiter PR6047 @e94e7613 + cache-addressing from aiter @7edb5dca, MIT, per-family `SOURCE.md` + `LICENSE` (`Tk/minimax_m3/SOURCE.md`, `Tk/LICENSE`); build options trimmed to SGLang's one config, indexer CP kept | both trace origin; theirs adds a readable per-family SOURCE.md. Note: their M3 source is aiter PR6047, not ATOM `atom/models/minimax_m3/mono` as our P1 plan assumed | **merge**: per-file origin headers (ours) + one `SOURCE.md` per family dir (theirs); M3 source = their aiter-derived copy (the validated one) |
| MiniMax-M3 coverage | skeleton only: config refusals, widths 1..16, TP4, `refuse_model` always refuses (`O/models/minimax_m3_mono/spec.py:10-65`) | full: TP4 / EP1 / PP1, 256-CU check, BF16 attn -> FP8, MXFP4 experts, FP8 K/V/index-K, page-128 alloc + page-16 SHUFFLE KV, decode + linear EAGLE <= 16 rows, HiCache waits, EAGLE3 aux capture (`T/layers/mono/minimax_m3/runner.py:96-302`, SOURCE.md:11-13; model wiring minimax_m3.py diff) | theirs. Our skeleton's shape constants match theirs (16 rows, TP4, 6144 hidden) so the spec slot fits | **adopt theirs** for task 12, wrapped as the M3 `MonoSpec`; needs their non-mono prerequisites (aiter sparse PA, aiter indexer, SHUFFLE store, env) - none of which are in our tree (checked: `aiter_sparse_pa.py`, `aiter_indexer.py` absent) |
| indexer | K2i: ATOM FP4 scorer + SGLang top-k / candidate kernels, device walk plan, runs between K1 and K2 (`Om/kernels/indexer/{k2i,score_fp4,hook,walk_plan}.py`); in-kernel fused version is task 17 S2 (blocked on a fast in-kernel top-k) | index scores computed inside K1 (`Tk/minimax_m3/index_score.py`), block top-k inside K4, context parallel over heads past 288 index blocks (`Tk/minimax_m3/config.py:53`, `index_score.py:89-107`, `post_attn.py:184-189`) | different models (DSV4.1 token-level FP4 indexer vs M3 128-token block indexer); not interchangeable. Their in-kernel block top-k is a reference point for our S2 | **keep both, model-specific**; action for task 17 S2: measure their in-kernel top-k approach before writing ours |
| debug / fail-stop / timeline | fail-stop framework (`O/layers/mono/failstop.py:23-44`, `layer.py:82-83`) but **not wired**: `failstop.check` has no caller outside tests and the DSV4.1 kernel's `health()` returns None (`Om/spec.py:401-402`); L0 `timeline.py` / `debug.py` unused; env checks per feature (K2A_CHECK) | per-layer check mode vs the module path for the first 4 steps, max abs diff logged (`T/layers/mono/adapters/minimax_m3.py:57-97`, `SGLANG_MINIMAX_M3_MONO_CHECK`); waits are unbounded `s_sleep` spins (`Tk/common.py:250-257`); `[mono][rank r]` decline reasons at debug level (`dispatch.py:56-57`) | check mode: theirs (simple, model-generic idea, catches numerics regressions in the real server). Fail-stop: neither works today | **adopt** check mode into L1 (`SGLANG_MONO_CHECK`, spec supplies the reference call); **keep** our fail-stop design and finish it (task 10: bounded waits + error words in both kernel families, wire `check` into the output path) |
| tests | CPU: refusals, routing purity, cache rebind, lazy build, TP-refused width, fail-stop, "L1 has no model constants" (`test/registered/mono/test_mono_runtime.py`, `test_mono_specs.py:129-145`); GPU: K2i indexer vs SGLang (`test_dsv41_k2a_indexer.py`) | CPU: dispatch declines, deferred runner build outside capture, close idempotent, flag / alias / rejections (`test/registered/unit/models/test_mono_runtime.py:95-271`), residual-boundary handoff (`test_minimax_mono_boundary.py`); GPU: metadata replay / padding (`test/registered/amd/test_minimax_mono_verify.py`); CI-registered (`register_cpu_ci` / `register_amd_ci`) | complementary; theirs are CI-registered, ours are not | **merge**: keep both sets, register ours in CI the same way; drop their flag / alias tests that the unified flag replaces |
| validation reported | DSV4.1 TP4: GSM8K 0.907, NIAH 120k 20/20, AgentX c1 P90 617.1 (vs ATOM 1010 +0.2%), c4 529.1 (+22.1%) (parent doc, task 18) | "validated M3 baseline" / "validated on Crusoe 264" in commit titles only; no accuracy or speed numbers in the repo | ours documented; theirs unverifiable from here, and there are no M3 weights on mi355-4 (`/shared_nfs`, `/shared_nfs/models`) | **ask** the author for M3 GSM8K / TPOT numbers and the run setup; validate M3 on a node with weights |

## Framework branch results (append-only; node + date)

| date | node | run | GSM8K | NIAH 120k | TPOT ms 8k c1 / c4, 128k c1 / c4 (mean of 2) | Tracebacks |
|---|---|---|---|---|---|---|
| 10-13 | mi355-4 | fwf1 fw (snap_fw_f1_1013, `--layer-backend=mono`, no mono env) | 0.900 (200) | 4/4 | 1.451 / 2.145, 1.450 / 2.186 | 0 |
| 10-13 | mi355-4 | fwf1 base (snap_v41mono_rebased_1013, old flag + env set + K2A=1), same session | 0.900 (200) | 4/4 | 1.451 / 2.105, 1.487 / 2.206 | 0 |
| 10-13 | mi355-4 | fwf6 fw6 (snap_fw_f6_1013, F1-F7) | 0.890 (200) | 4/4 | 1.445 / 2.150, 1.472 / 2.164 | 0 |
| 10-13 | mi355-4 | fwf6 fw6chk (SGLANG_MONO_CHECK=1, eager): 640 layer checks, 40 layers, 6 rows; hidden rel max\|diff\| median 2.4%, max 15.0% (layer 17); pre max\|diff\| 0.020 | 0.920 (100) | 2/2 (32k) | - | 0 |
| 10-13 | mi355-4 | K2i GPU test (test_dsv41_k2a_indexer.py) on snap_fw_f6_1013 | 14 passed | | | |
| 10-13 | mi355-4 | **final** fwfinal fw6 (snap_fw_f6_1013) | 0.901 (1319) | 60/60 (32/64/120k) | 1.433 / 2.124, 1.455 / 2.204 (mean of 3) | 0 |
| 10-13 | mi355-4 | **final** fwfinal base (snap_v41mono_rebased_1013), same session | 0.912 (1319) | 60/60 | 1.414 / 2.154, 1.466 / 2.194 (mean of 3) | 0 |
| 10-13 | mi355-4 | AgentX TP4 c1 fwfinal (fw6) vs task18reb (base, 10-13 earlier) | TTT/GPU 7,401 vs 7,444 (-0.6%) | P90 610.3 vs 617.1 (-1.1%); vs ATOM 1010 -0.1% / -0.9%; per-request ITL fw6/base median 1.005 | | 0 |
| 10-13 | mi355-4 | AgentX TP4 c4 fwfinal (fw6) vs task18reb | TTT/GPU 9,561 vs 10,146 (-5.8%) | P90 516.0 vs 529.1 (-2.5%); vs ATOM 1010 -1.0% / +19.1%; per-request ITL fw6/base median 1.002 (every ISL bin 0.997-1.007); c4 TTT run-to-run spread seen before: task17k1 9,494 vs task18reb 10,146 (+6.9%) | | 0 |

## Adopt / keep / drop

**Adopt from theirs**
- Whole MiniMax-M3 implementation (kernels, runner, weights, metadata, adapter wiring) and its non-mono prerequisites
  (aiter sparse PA, aiter indexer, SHUFFLE KV store) -> closes task 12.
- `--layer-backend {module,mono}` as the switch.
- Device code under `python/sglang/kernels/ops/mono/gfx950/<family>/` with a `SOURCE.md` per family.
- `StepSpec.query_len` (explicit linear verify), `LayerStateBinding` (HiCache per-layer wait), per-step input checks
  (`_declined`), check mode (per-layer diff vs modules for the first N steps), EAGLE3 aux capture hook.
- Their metadata padding rule and GPU replay tests as a pattern (audit DSV4.1 against it).

**Keep ours**
- `MonoSpec` + registry as the one model-side declaration (per-layer WHOLE / FFN modes, widths, features, refusal hooks,
  second spec for the draft model).
- Default-deny `Feature` refusals with every reason at once; covers PD disagg, EPLB, torch.compile, deterministic,
  which theirs does not check.
- Model-level once-per-step decision + `tp_agree` on builds (no rank-local OFF).
- L0 `PeerBuffer` with step fence + agreement on mapping; the "L1 has no model constants" test; cache rebind on
  `(data_ptr, numel)`.
- Fail-stop design (to be wired, task 10). DSV4.1 kernels, K2i indexer, DSpark draft spec.

**Drop**
- `--enable-mono-kernel`, `SGLANG_MINIMAX_M3_MONO_DECODE` alias, their `families.py` (folded into the registry), their
  `dispatch.py` (replaced by `layer.py` + their declines), their `peer_buffer.py`, our DSV4.1 private
  `kernels/common/peer_memory.py` (after porting onto L0), our M3 skeleton `refuse_model`.
- Unused L0 runtime pieces that still have no caller after the merge (`WidthBuilds`, `timeline`, `compile`): decide at M4
  whether to wire or delete.

## Merge plan (proposal; nothing done until the user approves)

- **M0 side by side.** New branch off perf/v41-mono; cherry-pick their 3 commits (base is our ancestor; the only path
  both create is `layers/mono/__init__.py`). Both switches work; nothing of ours changes. Check: our CPU tests + theirs
  pass; DSV4.1 GSM8K unchanged.
- **M1 one switch.** `--layer-backend mono` drives our guards: their arch table -> registry entries, their checks ->
  `Feature` / `supported_tp` / spec refusals; delete `--enable-mono-kernel` and the env alias; update AgentX scripts
  (`SWEEP_EXTRA_ARGS`). Check: refusal tests, DSV4.1 GSM8K.
- **M2 one contract.** `MonoStep` gains `query_len`; `MonoSpec` gains `state_bindings` + optional `check`; M3 becomes a
  `MonoSpec` (`pack` = `SparseMoeLayer.from_layer`, `build` returns the M3 runner, `eligible` = linear-EAGLE rule);
  `minimax_m3.py` calls `begin_step` / `routed` like DSV4.1; their `dispatch.py` goes, its declines move to `routing`.
  Check: their boundary + metadata tests, M3 check mode on a node with weights.
- **M3 one device tree + one PeerBuffer.** Move DSV4.1 kernels to `kernels/ops/mono/gfx950/deepseek_v41/`, L0
  device/plan to `gfx950/common/` (merge with their `common.py`), DSV4.1 + M3 runners onto L0 `PeerBuffer`. Check:
  DSV4.1 GSM8K + TPOT x3 within noise, AgentX TP4 c1/c4.
- **M4 safety.** Task 10 on both families (bounded waits, error words, `failstop.check` in the output path); check
  mode generic in L1; register our tests in CI.

**Framework + shared-common plan (user 15:46, revised 16:20; M3 port deferred)**

Branch `perf/v41-mono-fw` off perf/v41-mono @3c5ea0778c. Goal: adopt / keep / drop AND one systemic shared common (ATOM
already has it: `atom/models/deepseek_v41/mono` has no private common, 23 files import `atom.mono`). Pieces from their
code are copied per file (no whole-commit cherry-pick); commit messages credit zijiecode.
- F1 switch + defaults: `--layer-backend {module,mono}` (their exec_.py field) driving our guards; drop
  `--enable-mono-kernel`. **User 16:20: mono on = the validated set** (whole layers, indexer layers, edge layers 0/1,
  DSpark draft, K2A=1). The per-feature env switches existed only for A/B; validated ones are removed (WHOLE, INDEXER,
  EDGE, DRAFT). K2A: on by default; keep its env only while S2/S3 A/B is open. Check what the FFN-only DSV4.1 path still
  serves once every layer is whole (if nothing: dead code -> remove; L1 `LayerMode.FFN` stays for other models).
  Update scripts (`SWEEP_EXTRA_ARGS`, task9b_ab.sh / task17_ab.sh, the mono env line in the parent doc).
- F2 routing: `MonoStep.query_len`; their per-step declines in `routing.decide_for_batch`.
- F3 check mode in L1 (`SGLANG_MONO_CHECK`), DSV4.1 reference call.
- F4 `MonoSpec.state_bindings` hook (HiCache per-layer wait), CPU tests only.
- F5 device tree: `python/sglang/kernels/ops/mono/gfx950/common/` (= L0 device + plan) and `.../deepseek_v41/`; host
  runtime pieces stay in `srt/layers/mono/`. Pure moves + import rewrites; `SOURCE.md` per dir.
- F6 one shared common (was task 21): DSV4.1 kernels import `gfx950/common` instead of their private copy.
  (a) identical defs (ops 29/32, mx 6/6, plan 6/11, sync 3/7): import switch only; check the compiled kernels are
  unchanged (compare generated ISA / binary hash before and after, if FlyDSL exposes it) -> zero perf risk.
  (b) diverged defs: sync `Mailbox` / `Addr` / `preg`, plan `sel` / `cdiv` / `source_digest`, `PeerBuffer` (class only,
  per-step protocol unchanged; ATOM clear + fence evaluated in task 10): reconcile one by one (common gets a variant or
  the kernel adapts), each with numerics + TPOT A/B.
  (c) delete the private `kernels/common/`; delete L0 code still unreachable after (a)-(b), listed in SOURCE.md with its
  ATOM commit; write the mapping of their M3 `common.py` onto `gfx950/common` (prep only, no M3 code).
  (d) guard: a CPU test that a family dir (`gfx950/<model>/`) does not redefine a name `gfx950/common` exports (like
  the existing "L1 has no model constants" test).
- F7 tests: their CPU-style tests that apply to L1; ours registered in CI; DSV4.1 metadata padding audit (+ GPU test).
- M3 skeleton docstring: kernel source -> aiter PR6047 (still refuses at load).
Not in scope: M3 port and its prerequisites, fail-stop wiring + per-step protocol (task 10).
Validation (vs snap_v41mono_rebased_1013, which ran with the env set = the new default): CPU tests; GSM8K TP4 >= 0.895
(base 0.907); TPOT x3 c1/c4 at 8k + 128k within +-2%; AgentX TP4 c1/c4 vs 617.1 / 529.1; NIAH 120k; 0 Tracebacks.
Quick GSM8K 200 + TPOT after F1, F5, each F6(b) item. Then merge into perf/v41-mono (fast-forward); push only on the
user's go.

**Open questions for the user** (answered 15:40 / 15:46, kept for the record)
1. Flag: `--layer-backend mono` (proposed) or keep `--enable-mono-kernel`?
2. Where the merge lands: our perf/v41-mono (proposed, newer base) or their m3-mono; and is zijiecode in the loop?
3. M3 validation: no MiniMax-M3 weights on mi355-4. Which node / weights, and can the author share M3 numbers?
4. M3 kernel source: their aiter PR6047 copy (proposed) instead of ATOM `atom/models/minimax_m3/mono`?

## Log (append-only; node + date)

- 2026-10-13 mi355-4: doc created (task 20 defined by the user).
- 2026-10-13 mi355-4: comparison filled (14 axes, file:line both sides), adopt / keep / drop + merge plan M0-M4.
  Key findings: their base 16a23a672b is an ancestor of ours; their M3 source is aiter PR6047 (not ATOM); their build
  failure turns mono off rank-locally (hang risk, ours tp_agrees); our fail-stop is not wired and DSV4.1 uses a private
  PeerBuffer, so L0 peer fence / WidthBuilds are unused; M3 prerequisites (aiter sparse PA / indexer) are absent from our
  tree; no M3 weights on mi355-4. Waiting for the user's sign-off.
- 2026-10-13 15:40 mi355-4: user answered the 4 open questions (status block). User asked whether perf/v41-mono should first get the
  adopt/keep/drop changes + a no-regression check before branching; answer proposed: freeze perf/v41-mono @3c5ea0778c as the
  validated baseline, branch + cherry-pick first (refactor needs their code), validate each step M0-M4 against that baseline.
- 2026-10-13 15:46 mi355-4: user: M3 kernels = aiter PR6047; framework-only branch first, validate, merge into perf/v41-mono,
  M3 later (us or zijie on our framework). Framework-only plan F1-F7 written; waiting for go.
- 2026-10-13 16:05 mi355-4: dead-code scan of perf/v41-mono @3c5ea0778c (scripts /tmp/reach.py = import reachability from
  non-mono production code, /tmp/deadfn.py = unused top-level defs). Unreachable modules: 16 files / 1,064 lines, all in the
  vendored ATOM L0 `layers/mono/common/`: device/{ranks,select}, plan/{build_key,check,shard}, runtime/{abi,compare,compile,
  debug,lifecycle,mailboxes,peer_memory,route,step_fence,timeline,widths}. Reachable L0 = device/{ops,sync,mx,stamps},
  plan/{arith,execution,layout,trace}, runtime/consensus (used by the K2i indexer + tp_agree). Unused defs inside reachable
  files: ~36 lines (device/ops rows_to_lds, mfma_fp8, fp8x8_bf16_pk, lds_bytes; plan/trace note_unnamed). Wired-but-inert:
  failstop.check / MonoRuntime.health have no production caller; DSV4.1 health() returns None. DSV4.1 kernels + spec: no
  unreachable module. Env defaults: SGLANG_MONO_DSV41_{WHOLE,INDEXER,EDGE,DRAFT,K2A} all default 0, so --enable-mono-kernel
  alone runs the FFN-only mode; every validated run set them to 1 (K2A=1). Candidate F1 item: make the validated set the default.
- 2026-10-13 16:20 mi355-4: user: new branch also does the systemic shared common (task 21 folded in as F6); mono on =
  validated set by default, per-feature env switches only for A/B. Plan rewritten ("Framework + shared-common plan").
- 2026-10-13 17:15 mi355-4: F1 A/B no regression; F2-F7 implemented; F6 ISA-identical (84 kernels) after fixing a Mailbox(tag)
  positional bug the ISA check caught; padding-row audit finding logged; F6 GPU run started.
- 2026-10-13 17:40 mi355-4: F6 GPU run clean (TPOT = F1 / base within noise), check mode works on a real server, K2i GPU test 14 pass;
  final validation chain started (PID 1222260). Layer 17's 15% check diff noted for a later look.
- 2026-10-13 20:05 mi355-4: final A/B done: GSM8K 1319 fw6 0.901 / base 0.912; NIAH 60/60 both; TPOT within +-1.4%; AgentX TP4 c1
  -0.6% / -1.1% vs task18reb (c1 run-to-run spread ~1%: task17k1 612.7 vs task18reb 617.1). c4 running (~20:20 +08).
- 2026-10-13 20:40 mi355-4: AgentX TP4 c4 fw6 9,561 / 516.0 vs task18reb 10,146 / 529.1 (-5.8% / -2.5%); per-request ITL equal
  (median 1.002), so decode speed unchanged; c4 TTT varies ~7% between runs of the same tree class. Waiting for the user.
- 2026-10-13 20:45 mi355-4: user chose option 1: commit 134919476c, perf/v41-mono fast-forwarded (local only).
- 2026-10-13 20:50 mi355-4: padding-row finding has no effect on current runs (all bs 1..8 captured); task 21 low priority.
