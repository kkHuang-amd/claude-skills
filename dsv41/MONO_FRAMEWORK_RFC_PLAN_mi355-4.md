# Mono framework -> RFC #43949 shape (mi355-4) -- plan

Owner node: mi355-4. Parent: MONO_KERNEL_mi355-4.md. Status 2026-10-14 17:10 +08: steps 0-5 + kevin-mii's PR
review items B / C done and committed locally; A (step 5's TPOT cost) in validation.

## CONTINUE HERE (mi355-4, 2026-10-14 21:55 +08) -- supersedes the 17:10 block below where they differ

Top-level mono task table (keep in sync when a task changes): MONO_KERNEL_mi355-4.md, section "Task list -- TOP TABLE".

**PR split (10-15 12:40 +08, user: 4 PRs, close #43984 with a pointer, drop the M3 skeleton):** branch
`perf/v41-mono-pr` in worktree /sgl-workspace/sglang_v41mono_split (local, not pushed), on b4822bae03:
af845a2400 [1/4] DSpark host sync (= d9b33f55f1, 6 files +164) -> da56e802d3 [2/4] framework + common_hip (33 files
+3035; no spec registered; srt 7 files; env CHECK / CHECK_STEPS / FAULT_INJECT; CPU 40 pass) -> 7e9d8e12a4 [3/4]
DSV4.1 + skip_metadata hook (40 files +10262; env DSV41_K2A*; CPU 62 pass) -> c4208c0e9e [4/4] DSpark draft stages
(3 files +107). Final tree = e009cb4dad = 020f4c2584 minus the M3 skeleton (minimax_m3_gfx950/, 2 registry entries,
test_mono_specs M3 cases -> OTHER config / DSV4.1-only), so kernels = the GPU-validated h snapshot. All 4 commits
author+committer wunhuang, no AI trailer, lint clean. Branch perf/v41-mono-rfc-split = 020f4c2584 + e009cb4dad.
**DONE 10-15 13:20 +08:** rebased onto origin/main d4daf52369 (no conflicts): 021fc9bb72 / d9b2188ed1 / 24b1993be8 /
e4f3665be4, pushed to HaiShaw:perf/v41-mono-pr1..4, PRs #44405 / #44406 / #44407 / #44408 (titles in the CI's
"[Type] [AMD] scope: title (k/4)" form: pr-title.yml rejects "[AMD][V4.1][k/N]"); #43984 commented + closed.
10-15 13:10 (user): #44405 + 0a603c0670 "limit the host-sync skip and the fold to HIP" (scheduler gpu_only_lens and
simulate_fold gated on _is_hip; worktree /sgl-workspace/sglang_v41mono_pr1, branch pr1-hip); #44406..#44408 still
carry the old 1/4 commit (fine until #44405 merges). The host-sync skip is model-agnostic on HIP (radix backend
needs_cpu_seq_lens = SGLANG_OPT_USE_ONLINE_COMPRESS, default False; publish_ready exists for any spec algo), so it
also hits V4; the fold is V4.1-only (ROCm epilogue gated on target_is_dsv41) and only with SGLANG_SIMULATE_ACC_LEN.
V4-Pro A/B (user: TP8 OK, GPUs 0-7): scripts/v4pro_pr44405_ab.sh, main d4daf52369 vs pr1, TP8 + attn-DP8 + EAGLE,
GSM8K 1319 + TPOT 8k c1/c4 x3, log /shared_nfs/kk/v4pro_pr44405_1015.log, PGID 1835378, results
/shared_nfs/kk/results/DeepSeek-V4-Pro/pr44405_1015/{main,pr1}. RESULT (done 13:56 +08): GSM8K main 0.951 / pr1
0.948, 0 Tracebacks / faults, overlap on; TPOT median of 3 (ms) c1 5.794 -> 5.747 (-0.8%), c4 6.379 -> 6.305 (-1.2%),
both within the run spread (pr1 c1 one run 6.313, main c4 6.34-6.75) -> V4-Pro: correct, no measurable speedup (step
~5.8 ms GPU-bound; the skipped sync hides behind it). Whether the skip fired was not logged. Rebased tip (snap_rfc_pr_1015, tags mi355-4_rfcpr_*): GSM8K 0.905, NIAH 60/60, TPOT 8k c1 1.449 / c4 2.128, 128k c1
1.503 / c4 2.162 ms, 0 Tracebacks / faults. Next: CI and review feedback; a review fix = a new commit on the PR's
branch (never force-push; later PRs then need the fix too -> push the restacked branches as new commits).

**Branch now:** perf/v41-mono-rfc = 24d4379e6e / d9b33f55f1 / 2d998c2496 / **020f4c2584 `Skip aiter's decode schedule
for K2i-scored ratios`** (mono.skip_metadata + HIP radix filter, srt 8th file; c1 -1..-3% TPOT, GSM8K / NIAH OK).
Local only, not pushed; worktree clean. A and A2 dropped (patches /shared_nfs/kk/rfc_A_guard_check_every_1014.patch,
rfc_A2_guard_peer_every_1014.patch); neither recovers the guard's cost.
**Guard cost left (same-run, s4 -> h = HEAD):** TPOT 8k c1 +1.0%, 128k c1 +2.6%, 8k c4 +0.9%, 128k c4 +3.3%.
**TASK ITEM A3 (user 21:55: discuss tomorrow before doing anything):** ATOM is fast because its production kernels
have NO in-kernel fail-stop: `_spin` only; `_spin_bounded` (10 s, records diag per CTA, kernel runs on, host raises
after the step, eager only) exists only in ATOM_MONO_DEBUG builds (atom_454d0e3/atom/mono/device/sync.py:202,
runtime/debug.py). Our cost is likely the inlined stop path at every wait (K2 atomics 1 -> 133-139, code +20%; K1 +18%).
Options: (1) slim stop path: own ERROR / SITE + store to 0 only, no peer flags, no STOP poll (a fault tears the process
tree down; peers have their own deadline) -- deviates from RFC "flag every peer"; (2) keep full RFC behaviour, move
the stop path to one shared out-of-line block (FlyDSL support unknown); (3) ATOM-style: guard only in a debug build
(needs RFC discussion). Also open: the one-off 60 s mid-GSM8K fail-stop on A (not reproduced in 4 runs).
**Night 10-14 DONE 10-15 11:58 +08 (12 / 12 rc=0): h within noise of s4 and fwfinal at every TP2 / TP4 point; numbers
in the TOP TABLE (MONO_KERNEL_mi355-4.md Done rows 10-15).** Was (restarted 21:57 +08): user dropped c (known slower) and added
TP2. scripts/night_1014_rfc.sh: AgentX TP4 (GPUs 4-7) then TP2 (GPUs 4,5), c1 / c4 / c8, s4 vs h (snap_rfc_h_1014 =
020f4c2584) interleaved per point, tags mi355-4_n1014{s4,h}_tp{4,2}_c*, log /shared_nfs/kk/night_1014_rfc.log,
PGID 1696549 (stop `kill -- -1696549`, then kill GPU 4-7 sglang by HIP_VISIBLE_DEVICES). Table:
`python3 -I scripts/agentx_table.py mi355-4_n1014s4 mi355-4_n1014h`.
**Other open decisions:** experimental/monokernels move (wait for user), rebase (after the split), handle_monokernel
capture sizes (proposal: fill bs 1..8 when the user did not set them, never override; check whether the resolution
hook may write server_args), then step 6.

**Branch:** `perf/v41-mono-rfc` in /sgl-workspace/sglang_v41mono (local, NOT pushed), on b4822bae03:
- 24d4379e6e `Monokernel plugin and DeepSeek-V4.1 decode layers (RFC #43949)` -- framework + common_hip (incl. step 5
  fail-stop) + DSV4.1 (K1 / K2 on 40 layers, K2i in the plugin); srt = 7 files +45/-1 (exec_ flag, pipeline,
  resolution_hooks whitelist, environ SGLANG_TEST_MONO_*, deepseek_v4 load_tensor / attach / route, deepseek_v4_hip
  passes mono_plan, fused_mhc calls mono.kernel) = kevin-mii's shape
- d9b33f55f1 `DSpark: no per-step host sync in decode` (6 files, independent of monokernels)
- 2d998c2496 `DSpark draft stages on monokernels` (dspark.py hooks + DRAFT_SPEC)
- worktree, uncommitted: A = `common_hip/device/guard.py` guarded_spin as nested loops, clock / STOP read once every
  CHECK_EVERY = 32 polls (snapshot /sgl-workspace/snap_rfc_a_1014). Old history: branch perf/v41-mono-rfc-hist.
  WIP of tasks S3 / 22 / 27a / 27b: `stash@{0}` (old env names SGLANG_MONO_* and paths gfx950/*: rename on re-apply).
**Plugin layout:** kernels/ops/mono/{__init__ (re-exports the 5 hooks), spec (+ registry), guards
(+ handle_monokernel), runtime (MonoRuntime, routing, attach / route / kernel, tp_agree), check}.py,
common_hip/{guard.py (deadline, words, fault env), peer.py, device/ (guard.py, sync.py, ...), plan/},
deepseek_v41_gfx950/ (spec, glue, indexer_hook, runner, layer, ...), minimax_m3_gfx950/spec.py (skeleton).
**A chain DONE 18:10 +08: A FAILS** (real fail-stop mid-GSM8K on a, c1 TPOT +2.4% vs c; Session log "A chain
result"). A not to be committed as is. Was: `scripts/rfc_a_validate_1014.sh` started 17:06 +08, PGID 1566808, log
/shared_nfs/kk/rfc_a_validate_1014.log, ~60 min, stop `kill -- -1566808` + GPU 4-7 sglang processes: fault injection
on snap_rfc_a_1014 (expect 4 x "Memory access fault ... (nil)"), then A/B s4 (no guard, snap_rfc_s4_1014) / c (tip
2d998c2496, snap_rfc_c_1014) / a: GSM8K 1319 + NIAH 60 + TPOT x3 8k / 128k c1 / c4, tags mi355-4_rfca_{s4,c,a}_*.
**Next:**
1. Read the A chain (`rg 'memory faults|Accuracy|niah\] total|tracebacks' /shared_nfs/kk/rfc_a_validate_1014.log`;
   TPOT table: the python snippet pattern on `\[mi355-4_rfca_(s4|c|a)_ab isl(\d+)_r\d\] \[clean c(\d+)\].*TPOT ms mean`).
   Pass: faults on all 4 ranks, GSM8K >= 0.895, NIAH all, a vs c TPOT not worse; report a vs s4 too.
   If it passes: amend A into 24d4379e6e (rebuild the 3 commits: `python3 scripts/rfc_restack.py <tree>` needs
   BASE / HEAD adjusted -- simplest: commit A as a fixup on top, then `git rebase -i --autosquash b4822bae03`, author
   wunhuang via -c, no AI trailer, check `git log -3 --format=%B`).
2. Open decisions for the user: (a) c1 TPOT +2% after C (SGLang now builds aiter's decode schedule for the
   K2i-scored ratios; skipping it needs a 1-line srt hook beyond kevin-mii's 7 files); (b) experimental/monokernels
   layout (python/sglang/experimental/monokernels vs repo root; what moves; `_gfx950` vs kevin-mii's `_hip`; flatten
   common_hip/{device,plan} like kevin-mii's common_hip/*.py) -- user: do not move until confirmed; (c) rebase onto
   the PR base ea13359da9 or main (user: after the split); (d) handle_monokernel setting decode capture sizes (not done).
3. Still to do from the plan: K2i launch itself (indexer/k2i.py) has no deadline guard; step 6 (unit / kernels / e2e
   test layout, switch-point e2e tests, validate.py, skill, PR split framework / common_hip / DSV4.1 / DSpark).
**Pass criteria / tooling:** CPU `HIP_VISIBLE_DEVICES=4 PYTHONPATH=<tree>/python python3 -m pytest -q test/registered/mono
test/registered/unit/server_args/test_resolution_hook_registry.py` (needs a visible GPU; 62+ pass); ISA
`PYTHONPATH=<tree>/python HIP_VISIBLE_DEVICES=4 python3 scripts/mono_isa_dump.py OUT [--quick]`, diff -r vs
isa_task26 (pre-guard) / isa_rfc_s5 (guarded); GPU A/B `VARIANTS="x:<snap>:MONO_ARG=--enable-monokernel ..."
PHASES="gsm ab" NQ=1319 REPS=3 TAGP=<tag> bash scripts/fw_ab.sh` (snapshots: rsync python/ to /sgl-workspace/snap_*).

## Session log 10-14 (steps 0-5, B, C, A -- history, newest at the bottom)

**Status:** user 10-14 12:01: pause tasks 27b / S3 / 22 / 13 / 27a; first rebuild our mono framework in the RFC #43949
shape ourselves (do not wait for kevin-mii's prototype; it is not public). Queued GPU chains for S3 / 22+13 / 27a
were stopped (never started); task 25 (TP2 sweep) runs to its end.
**Base:** branch perf/v41-mono-fw @ 53d0945468 (task 26, local, not pushed) in /sgl-workspace/sglang_v41mono. The
uncommitted S3 / task 22 / 27a work is saved as
/shared_nfs/kk/results/DeepSeek-V4.1-Flash/patches/wip_s3_t22_t27a_1014.diff (+ wip_untracked_1014.tgz for
attention/index_query.py) and frozen in /sgl-workspace/snap_t27a_1014. Decide in the new session: refactor on a new
branch from 53d0945468 and re-apply the WIP after step 4, or keep the WIP in the tree.
**Step 0 decisions (user, 2026-10-14 12:20 +08):** switch = `--enable-monokernel` exactly as RFC (`--layer-backend`
removed, no alias); MonoSpec field names exactly as RFC, fields the RFC lacks (`state_bindings`, ...) kept and proposed
on the RFC; M3 skeleton kept, moved to `kernels/ops/mono/minimax_m3_gfx950/spec.py`; WIP: new branch from 53d0945468,
worktree WIP stashed, re-applied after step 4.
**Branch:** perf/v41-mono-rfc from 53d0945468 in /sgl-workspace/sglang_v41mono (local). WIP = `stash@{0}` there
("wip S3/22/27a/27b 1014 (pre RFC refactor)") + /shared_nfs/kk/results/DeepSeek-V4.1-Flash/patches/wip_full_1014_1220.diff
(+ _untracked.tgz: index_query.py, peer_layout.py, token_par.py).
**Step 1 done (mi355-4, 2026-10-14 12:50 +08), local commit 62f0532237 on perf/v41-mono-rfc (not pushed):** `scripts/rfc_step1_move.py`
(git mv + import rewrite, 56 files) + sources.py digest paths + build_key comment + shared-helper guard test now walks
`kernels/ops/mono/*_gfx950` + isort of 3 srt files + kernels/README.md exception note. Layout: `kernels/ops/mono/`
{spec, registry, guards, routing, check, failstop, layer}.py + runtime/; common_hip/; deepseek_v41_gfx950/ (incl.
spec.py, indexer_hook.py); minimax_m3_gfx950/spec.py. The package `__init__` stays empty (the old srt re-export of
attach / MonoRuntime dropped; callers import `kernels.ops.mono.layer`). Glue in deepseek_v4_fused_mhc.py NOT moved yet
-> step 3 (it becomes the span callable behind `mono.kernel()`). Checks: CPU tests 43 passed (HIP_VISIBLE_DEVICES=5;
collection needs a visible GPU), ruff F401/F821/F823/UP037 + format + isort clean, ISA: isa_rfc_s1 vs isa_task26
`diff -r` empty (78 builds, 0 errors).
**Step 2 (mi355-4, 2026-10-14 13:50 +08), uncommitted:** spec.py in RFC shape: `Point` (IntEnum, 8 points) + `Span`
(name, start, end) + `check_spans` / `span_label` replace LayerMode (DSV4.1: K1 layer_in..core_in + K2 core_in..layer_out,
startup log still `{'whole': 40}`; M3: one span); `DeviceLimits(widths, tp_sizes, resident_blocks)` in
`devices["gfx950"]` (DSV4.1 resident_blocks = BLOCKS = 256); `Requirements(dtypes, kv_cache, attention_backends,
moe_backends)` declared only -- NOT enforced yet (step 4; the env checks stay in config_disable_reasons); `Numerics`
(BITWISE / TOLERANCE, keys `<span>.<output>.<max_abs|max_rel>`; DSV4.1 `k2.hidden.max_rel = 0.2` from task 22's 15%
peak -- only measured bound; KV-bitwise bounds wait for check-mode stats); `config_disable_reasons` /
`model_disable_reasons`; `pack` + `build` -> `build(model, ctx) -> MonoModel` (`.spans`, `.kernel(rows)`; width
still built lazily at first step -> step 3); `load_tensor` (None). Kept (not in RFC, propose): `state_bindings`;
`eligible` until step 3's graph-key plan. Dropped (no reader): caches/CacheReq, bind, weight_source, options,
architectures. ISA check: `guards.current_isa()` in `spec.devices` (was is_gfx95_supported); MonoContext carries
`limits`; routing widths from `ctx.limits`. CPU 47 passed (+4: span contiguity, numerics keys, unknown ISA, spec
spans vs numerics). ISA: isa_rfc_s2 vs isa_task26 `diff -r` empty (0 build errors). Lint clean. GPU smoke of the attach path deferred to step 3's GPU check.
Step 2 committed 9e05c19f86 (local). NOTE: step 2 missed glue fused_mhc `mono_route.mode` (CPU tests do not reach it);
fixed by step 3 (glue moved) -- 9e05c19f86 alone does not serve.
**Step 3 (mi355-4, 2026-10-14 14:40 +08), uncommitted, snapshot /sgl-workspace/snap_rfc_s3_1014:**
`kernels/ops/mono/hooks.py`: `handle_monokernel` (body of the deleted srt/arg_groups/mono_hook.py; pipeline calls it),
`load_tensor(model, name, t) -> t` (wraps the raw weights iterable in DSV4 / DSpark load_weights, before renames and the
wo_a dequant) + `loaded(model)` (no-op: DSV4.1 has no load_tensor), `attach(loop_module, architecture)` from
`precompile_kernels_after_loading()` (DSV4: `self.model`, "DeepseekV4ForCausalLM"; DSpark: itself,
"DeepseekV4ForCausalLMDSpark" -> registry -> DRAFT_SPEC), `route(loop, fb) -> Plan|None` once a forward,
`kernel(plan, i, "layer_in") -> span|None`. attach builds every width a capture reaches
(decode capture bs x `MonoModel.rows_per_request`: DSV4.1 verify = speculative_num_draft_tokens, draft = gamma; eager:
none up front, lazy as before), refuses (MonoRefused) if any rank fails a width, all_gather-compares the ranks' plans
(spec, layer spans, widths), logs "attached in X s; built widths [...]; scratch MiB per width {...}". Glue
`_mono_whole` / `_mono_checked` / `_mono_indexer_between` -> `deepseek_v41_gfx950/glue.py` (`run_layer(route, <the
fused boundary's args>)`, same returns); `Packed.span(route)` (draft: `checked=False`, as before). SGLang side: fused_mhc
-189 lines (mono branches gone), deepseek_v4_hip.py `boundary = mono.kernel(...) or forward_hc_pre_from_prev_fused_boundary`,
deepseek_v4.py `self.mono_plan = mono.route(...)` (attach / mono_pending / post_load_weights hook removed), DSpark
route + kernel. srt diff of this step: +34 / -287. K2i indexer hooks in the attention backends unchanged (later step).
CPU 49 passed (+2: route/kernel, build_widths refuses). Lint clean.
GPU chain `scripts/rfc_s3_validate_1014.sh` (started 14:42 +08 on mi355-4, PGID 1454150, log
/shared_nfs/kk/rfc_s3_validate_1014.log; stop `kill -- -1454150` + GPU 4-7 sglang processes): smoke rfcs3smoke (GSM8K
200, NIAH 3x4, attach line) then A/B base snap_fw_f6_1013 vs rfc (GSM8K 1319 + NIAH + TPOT x3 8k/128k c1/c4), tags
mi355-4_rfcs3_{base,rfc}_{gsm,ab}.
Smoke (mi355-4_rfcs3smoke_rfc_gsm, 10-14 13:53 +08): GSM8K 200 0.910, NIAH 12/12, 0 Tracebacks; startup: target
"attached in 3.9 s; built widths [6, 12, .., 48]; scratch MiB {6: 1.6 .. 48: 10.6}", draft "0.1 s; [5, 10, .., 40];
{5: 1.2 .. 40: 8.4}" = exactly the capture widths of the 10-13 run. Full A/B running (started 13:53 +08).
**Step 3 A/B PASSED (mi355-4, done 2026-10-14 14:25 +08; tags mi355-4_rfcs3_{base,rfc}_{gsm,ab}):** GSM8K 1319 base
0.916 / rfc 0.905; NIAH 60/60 both; 0 Tracebacks; TPOT mean of 3 (ms, sim-AL) base -> rfc: 8k c1 1.431 -> 1.431
(+0.0%), 8k c4 2.160 -> 2.171 (+0.5%), 128k c1 1.466 -> 1.436 (-2.0%), 128k c4 2.176 -> 2.182 (+0.3%).
**Step 4 (mi355-4, 2026-10-14 15:30 +08), code done, uncommitted, in the same worktree on top of step 3 (step 3 and 4
will be one commit unless split by hand; step 3's GPU result = snapshot snap_rfc_s3_1014):** switch `--enable-monokernel`
(bool, off; `layer_backend` field + LAYER_BACKEND_CHOICES removed; all messages say `--enable-monokernel`); env renamed
SGLANG_MONO_CHECK(_STEPS) / SGLANG_MONO_DSV41_K2A(_CHECK) / SGLANG_MONO_UNSAFE_WARN -> SGLANG_TEST_MONO_* (WIP stash and
old scripts still use the old names -- rename on re-apply; new runs: MONO_ARG=--enable-monokernel); Feature = RFC
section 6 list (PP, multi-node TP, DP attn, EP, EPLB, CP (attn_cp/dcp), spec decoding, uneven verify (eagle topk > 1),
TBO, LoRA, PD, routed-expert / indexer top-k capture, EAGLE3 hidden states, KV offload (hicache), memory saver, CPU
offload, torch compile, deterministic; eager warns), a probe that cannot read its setting = in use; `requires`
ENFORCED on resolved args (DSV4.1: dtype {auto, bfloat16}, kv_cache_dtype {fp8_e4m3}, attention_backend {dsv4},
moe_runner_backend {auto, aiter} -- values read from the running 10-14 server's /get_server_info); partitioned GPU =
CUs < resident_blocks refused at attach; no lazy width builds / no fallback: every step width built at attach (eager:
all 48), an unbuilt width takes the normal path ("width not built"); weight updates refused via the updater's existing
`_derived_weight_cache_error` (set on the loop module at attach, no srt change); `SGLANG_READS` in hooks.py,
deepseek_v41_gfx950/{spec,glue}.py + test_mono_contract.py (declared == imported; each name exists in SGLang's source,
AST only). CPU 53/54 before the last test fix (health of all built widths); rerun after the A/B (CPU tests need a
visible GPU; one run was on GPU 6 during the A/B). Not done here: handle_monokernel setting the decode capture sizes
(capture sizes already line up), K2i hooks as a span (later).
Step 4 CPU 55 passed (14:31 +08). GPU chain `scripts/rfc_s4_validate_1014.sh` (snapshot /sgl-workspace/snap_rfc_s4_1014;
ISA dump isa_rfc_s4, smoke rfcs4smoke, refusal rfcs4refuse with --enable-two-batch-overlap), started 14:31 +08 on
mi355-4, PGID 1482282, log /shared_nfs/kk/rfc_s4_validate_1014.log, stop `kill -- -1482282`.
Step 4 chain result (done 14:44 +08): ISA identical (0 errors, diff empty); smoke GSM8K 200 0.885 (step 3 smoke 0.910;
200-q sigma ~2%), NIAH 12/12, 0 Tracebacks, attach lines as step 3; refusal check INVALID: --enable-two-batch-overlap
is refused earlier by DSV4.1 itself ("does not support two-batch overlap yet"). Recheck chain (ONLY_RECHECK=1, log
/shared_nfs/kk/rfc_s4_recheck_1014.log, started 14:46 +08): refusal with --enable-return-routed-experts, then GSM8K
1319 + NIAH 60 (tag mi355-4_rfcs4full_rfc4_gsm). RESULT (14:55 +08): refusal OK ("--enable-monokernel refused
(deepseek_v41, config): routed-expert capture is not supported by the deepseek_v41 monokernels", at config stage);
GSM8K 1319 0.901, NIAH 60/60, 0 Tracebacks -> step 4 PASSED (TPOT not rerun: no kernel / routing change, ISA identical).
Steps 3 + 4 committed b8728f891f (local, 10-14 15:00 +08).
**Step 5 (mi355-4, 2026-10-14 15:40 +08), uncommitted, snapshot /sgl-workspace/snap_rfc_s5_1014:** host
`common_hip/guard.py` (deadline = min(watchdog / 2, 60 s) in 100 MHz ticks, < 5 s refused in config_refusals; guard
words layout STOP / ERROR / SITE / ZERO, GUARD_BYTES 64 past the two peer halves of each rank's PeerBuffer;
SGLANG_TEST_MONO_FAULT_INJECT=rank:site[:after] -> MonoBuild.fault_site / fault_after (keyed), after = launch pairs);
device `common_hip/device/guard.py` (`guarded_spin`: clock + STOP read only once a poll came back stale; on deadline /
peer STOP: atomic_or ERROR, cas SITE, xchg STOP=1 into every rank's words, system fence, volatile store to the address
read from ZERO (= 0) -> GPU fault); Mailbox(guard=) -> poll uses guarded_spin; K1 / K2 (+ draft) get args `guard,
guards, deadline` and guard both mailboxes + back.epoch_end's mark wait; runner passes them (deadline from
server_args.watchdog_timeout); MonoKernel.health() = the guard words. NOT guarded yet: K2i (its own Mailbox / epoch),
host step_fence (triton). Quick ISA: 24 of 26 builds change (k2i same); K1 sgpr +6..13, K2 vgpr 250 -> 252 (t1),
246 -> 242 (t6), no scratch spill; code ~+18%. CPU 57 passed (+2: deadline, fault spec).
GPU chain `scripts/rfc_s5_validate_1014.sh` (started 15:42 +08, PGID 1502475, log /shared_nfs/kk/rfc_s5_validate_1014.log,
stop `kill -- -1502475`): full ISA dump, fault injection (rank 1, site attn, after 0, watchdog 20 s), A/B s4 vs s5.
Results so far: full ISA 78 builds, 0 errors. Fault injection (mi355-4_rfcs5fault_fault_gsm) PASSED: all 4 GPUs
"Memory access fault ... on address (nil)" (node-6..9 = GPUs 4-7), schedulers aborted, SGLang killed the tree
(07:18:23 UTC, ~18 s after the last capture log = reaching a mono width + 10 s deadline + teardown), no hang, 0
leftover processes. A/B s4 vs s5 (done 15:53 +08; tags mi355-4_rfcs5_{s4,s5}_{gsm,ab}): GSM8K 0.903 / 0.911, NIAH
60/60 both, 0 Tracebacks; TPOT (ms, mean of 3) s4 -> s5: 8k c1 1.431 -> 1.440 (+0.7%), 8k c4 2.171 -> 2.228 (+2.6%,
one 2.305 run), 128k c1 1.451 -> 1.484 (+2.3%, all 3 s5 runs above s4's max), 128k c4 2.202 -> 2.231 (+1.3%). Two
cells past +-2%: likely the per-iteration clock + STOP load inside waits that do spin (cross-rank hand-offs usually
spin a few rounds); option: check the deadline / STOP every N rounds only.
**PR #43984 kevin-mii review (10-14 14:19 +08) -> gaps:** (1) srt diff vs PR base ea13359da9 is 13 files +201/-34,
target 7 files / 39 lines (kevin-mii/sglang amd/monokernel-dsv41): move the K2i hooks out of
low_ratio_backend_hip.py / deepseek_v4_backend_hip_radix.py into the plugin; DSpark host-sync, simulated-accept fold
(managers, speculative/dspark_*, SGLANG_DSPARK_SIMULATE_FOLD) and the draft stages (deepseek_v4_dspark.py) to their own
PRs; serving_base.py diff is base drift (rebase); register SGLANG_TEST_MONO_* in environ.py; plan passed from
deepseek_v4_hip.py into the fused boundary, mono.kernel called there. (2) handle_monokernel was missing from
resolution_hooks._OVERRIDABLE_HOOKS (test_resolution_hook_registry failed) -- FIXED 10-14 16:00 (uncommitted, step 5
tree). (3) drop unused ATOM code: runtime/ except consensus + peer_memory (abi, compare, compile, debug, lifecycle,
mailboxes, route, step_fence, timeline, widths), common_hip/device/select.py, common_hip/plan/check.py, and the host
health() / failstop path. (4) module shape: hooks.py -> guards.py (refusals, capture sizes, deadlines) + runtime.py
(load_tensor, attach, route, kernel; our layer.py folds in); package re-exports the five calls (`from
sglang.kernels.ops import mono`). (5) handle_monokernel should set the decode capture sizes. Done already: every width
at attach + refuse on failure, rank plan comparison at attach.
User order (16:05 +08): B (align structure) -> C (shrink srt) -> A (step 5 TPOT cost).
**B done (mi355-4, 16:30 +08), staged with step 5, uncommitted, snapshot /sgl-workspace/snap_rfc_b_1014:** framework =
`__init__.py` (re-exports handle_monokernel, load_tensor, loaded, attach, route, kernel), `guards.py` (+ handle_monokernel),
`runtime.py` (old hooks.py + layer.py + routing.py + consensus.tp_agree / MonoUnsupported), `spec.py` (+ the registry),
`check.py`; peer memory -> `common_hip/peer.py` (step fence, debug, kernel_args dropped); deleted runtime/{abi,
compare, compile, consensus, debug, lifecycle, mailboxes, route, step_fence, timeline, widths}, common_hip/device/select.py,
common_hip/plan/check.py, failstop.py, health() (LayerKernel, DSV4.1 MonoKernel, runner.guard_words), CacheBinding;
srt imports `from sglang.kernels.ops import mono`; `handle_monokernel` whitelisted in resolution_hooks. Plugin
+763 / -1423. CPU 62 passed (mono + test_resolution_hook_registry); quick ISA identical to step 5's. Not done (wait for
the experimental/monokernels move): flattening common_hip/{device,plan} into kevin-mii's common_hip/*.py, `_hip` suffix.
GPU smoke rfcbsmoke (16:08-16:19 +08): GSM8K 200 0.870 (earlier 200-q smokes 0.910 / 0.885; sigma ~2%), NIAH 12/12,
0 Tracebacks -> rechecked with 1319 in the C chain.
**C (mi355-4, 16:50 +08), committed:** env vars SGLANG_TEST_MONO_* registered in environ.py (plugin reads envs);
deepseek_v4_hip passes `mono_plan` into forward_hc_pre_from_prev_fused_boundary, which calls mono.kernel (kevin-mii's
shape); K2i fully in the plugin (user): glue `_indexer_between` runs SGLang's forward_low_ratio_sources with
run_indexer=False (compressor), then `indexer_hook.index_topk_decode` (SGLang's low_ratio_index_topk_hip_decode with
K2i scoring; reads _indexer_inputs / low_ratio_decode_rows_are_identity / _fit_candidate_span, pinned in SGLANG_READS),
walk plans once a step in the route cache (`step_plans(backend, forward_batch)`); low_ratio_backend_hip.py and
deepseek_v4_backend_hip_radix.py back to the PR base (cost: SGLang still builds aiter's decode schedule for K2i-scored
ratios). Restack (user: stack) with scripts/rfc_restack.py, old history on perf/v41-mono-rfc-hist (b8728f891f + the
uncommitted step 5 / B / C): b4822bae03 -> 24d4379e6e framework + DSV4.1 (srt 7 files +45/-1: exec_, pipeline,
resolution_hooks, environ, deepseek_v4, deepseek_v4_hip, fused_mhc) -> d9b33f55f1 DSpark host-sync / fold (6 files
+164/-5) -> 2d998c2496 DSpark draft stages (dspark.py + registry + DRAFT_SPEC / draft_build / Packed.checked). Commit 1
alone: lint clean, CPU 48 passed. Rebase onto the PR base / main later (user). Step 5's commit is inside 24d4379e6e.
GPU chain `scripts/rfc_c_validate_1014.sh` (snapshot /sgl-workspace/snap_rfc_c_1014 = 2d998c2496; log
/shared_nfs/kk/rfc_c_validate_1014.log): K2i GPU test, then A/B snap_rfc_s5_1014 vs snap_rfc_c_1014 (tags mi355-4_rfcc_*).
C chain result (done 16:58 +08): K2i GPU test 14 passed; GSM8K 1319 s5 0.908 / c 0.904 (B's 0.870 smoke = noise);
NIAH 60/60 both; 0 Tracebacks; TPOT s5 -> c: 8k c1 1.451 -> 1.481 (+2.1%, no overlap), 8k c4 2.158 -> 2.157,
128k c1 1.499 -> 1.529 (+2.0%), 128k c4 2.235 -> 2.194 (-1.8%). Likely cause of c1 +2%: SGLang now builds aiter's
decode schedule for the K2i-scored ratios too (the srt hook that skipped it is gone); removing it needs one srt hook
(user to decide). A quick ISA: compiles, 0 errors, K1 vgpr 140 -> 152 (t1), 149 -> 173 (t6), K2 unchanged, no spill.
A chain `scripts/rfc_a_validate_1014.sh` (snapshot /sgl-workspace/snap_rfc_a_1014, started 17:06 +08, log
/shared_nfs/kk/rfc_a_validate_1014.log): fault injection, then A/B s4 / c / a (tags mi355-4_rfca_*), ~60 min.
A chain result (mi355-4, done 18:10 +08) -- A FAILS: fault injection OK (4 faults, all (nil), 0 leftover). GSM8K 1319 /
NIAH: s4 0.905 / 60/60, c 0.914 / 60/60, a: server fail-stopped mid-GSM8K WITHOUT fault injection (10:00:55 UTC last
prefill, 3 of 4 GPUs (node 6/7/9) "Memory access fault ... (nil)", scheduler died 10:01:57 = ~60 s deadline; node-8 no
fault line) -> no GSM8K / NIAH number. TPOT median of 3 (ms) s4 / c / a: 8k c1 1.440 / 1.469 / 1.505, 8k c4 2.163 /
2.216 / 2.194, 128k c1 1.450 / 1.490 / 1.525, 128k c4 2.229 / 2.252 / 2.227 -> a is +2.4% worse than c at c1, -1% at c4.
Open: is the mid-run fail-stop A-specific (nested scf.while carrying v?) or a rare hang in c too (one occurrence).
Repro (user chose, 18:16 +08, mi355-4): fw_ab.sh PHASES=gsm, variants a1 / c1 / a2 / c2 (snap_rfc_a / snap_rfc_c),
tags mi355-4_rfcarep_*, log /shared_nfs/kk/rfc_a_repro_1014.log, PGID 1602618 (stop: `kill -- -1602618`), ~40-60 min.
Repro result (done 18:54 +08): no repro. GSM8K a1 0.899 / c1 0.901 / a2 0.904 / c2 0.904, NIAH 60/60 all, 0 faults,
0 Tracebacks. Mid-run fail-stop seen once in 3 a runs, 0 in 4 c runs (C chain, A chain, 2 repro): cannot pin on A.
A gives no TPOT win (c1 worse) -> recommend dropping A; the one-off 60 s wait stays open for step 5's guard.
19:26 +08 (user: start): A saved to /shared_nfs/kk/rfc_A_guard_check_every_1014.patch, guard.py reverted (worktree
clean). Likely cause of A's c1 loss: K1 VGPR +12..24 from the nested loop. Follow-up "A2": one loop, scalar counter,
clock every round, CM_SYS STOP load every 32 rounds; gate: VGPR = s5, then TPOT A/B. c1 profile running:
scripts/rfc_c1_profile_1014.sh (s5 vs c, 8k c1, 40 decode steps, GPU-only torch trace), log
/shared_nfs/kk/rfc_c1_profile_1014.log, PGID 1627696, traces /shared_nfs/kk/results/DeepSeek-V4.1-Flash/profile_rfc_c1/.
Profile is blind here: s5 trace GPU busy 15%, verify-graph kernels missing (torch profiler after nosync; rocprofv3
broken). So test the fix directly: snap_rfc_ch_1014 = snap_rfc_c_1014 + radix filter `and not
k2i_hook.skips_schedule(forward_batch, ratio, indexer_metadata)` (indexer_hook.skips_schedule = step_plans' test, off
under K2A check; one-time "PROTO skips_schedule" log line). Chain scripts/rfc_hook_ab_1014.sh (waits for the profile
chain), c vs ch GSM8K + NIAH + TPOT x3, tags mi355-4_rfch_*, log /shared_nfs/kk/rfc_hook_ab_1014.log, PGID 1634451.
Hook A/B result (mi355-4, done 20:19 +08): skip fired (4 "PROTO" lines = 4 ranks, gsm and ab). GSM8K c 0.897 / ch
0.914, NIAH 60/60 both, 0 Tracebacks. TPOT median of 3 (ms) c -> ch: 8k c1 1.460 -> 1.445 (-1.0%), 128k c1 1.513 ->
1.467 (-3.0%; every ch run below every c run), 8k c4 2.195 -> 2.215 (+0.9%), 128k c4 2.196 -> 2.176 (-0.9%). ch c1 is
at / below s5 (1.451 / 1.499 in the C chain) -> the c1 cost of C is the unused aiter schedule; the hook recovers it.
20:45 +08 user: write the hook for real (generic `mono` entry, separate commit) + try A2. Worktree, uncommitted:
(1) hook = `mono.skip_metadata(name, forward_batch, key, source)` (runtime.py; spec field `skip_metadata`, default
False; attach registers the spec's), DSV4.1 MONO_SPEC.skip_metadata -> indexer_hook.skips_schedule for
"low_ratio_decode_schedule"; radix +7 lines (import + filter; srt 8th file, HIP only); test_skip_metadata. CPU 63
passed, pre-commit clean. (2) A2 = guard.py one loop, clock every poll, CM_SYS STOP load in `if (k & 31) == 0`
(PEER_EVERY = 32). Full ISA isa_rfc_a2 vs isa_rfc_s5: 72 of 78 builds change, VGPR max +1 (K1 t6/t12 149 -> 150),
no scratch. Snapshots snap_rfc_h_1014 (hook, guard as HEAD) / snap_rfc_a2_1014 (hook + A2). Chain
scripts/rfc_a2_validate_1014.sh (h / a2 gsm + ab, then s4 ab), tags mi355-4_rfca2_*, log
/shared_nfs/kk/rfc_a2_validate_1014.log, PGID 1666472, ~50 min (started 20:50 +08).
A2 chain result (mi355-4, done 21:39 +08): GSM8K h 0.898 / a2 0.904, NIAH 60/60 both, 0 Tracebacks, 0 faults (h, a2,
s4). TPOT median of 3 (ms) s4 / h / a2: 8k c1 1.447 / 1.461 / 1.456, 128k c1 1.446 / 1.484 / 1.514, 8k c4 2.143 /
2.162 / 2.190, 128k c4 2.168 / 2.240 / 2.247. h vs s4 (= the guard's cost after the hook): +1.0% / +2.6% / +0.9% /
+3.3%. A2 does NOT recover it (vs h: -0.3% / +2.0% / +1.3% / +0.3%) -> drop A2 (VGPR +1 only, so the polling reads
are not the cost). Next idea: find where the guard's cost really is (s4 vs h ISA diff per wait site; s_sleep(1)?).
**Next after the chain:** A (step 5 TPOT: check deadline / STOP every N rounds), amend into 24d4379e6e. A's code is in
the worktree (uncommitted, 17:00 +08): guarded_spin = nested loops, the inner one polls up to CHECK_EVERY = 32 rounds
without the clock / STOP reads; not compiled yet (GPUs busy with the C chain).
**Pending layout decision (user, 2026-10-14 13:49 +08: do not move until the user confirms):** RFC #43949 comment by
HaiShaw (10-14): "place kernel code in-tree under `experimental/monokernels` per model basis with acknowledgement".
Open: python/sglang/experimental/monokernels vs repo-root experimental/ (has only sgl-router today); what moves
(model kernels / common_hip / framework); spec+glue with the kernels; suffix `_gfx950` vs kevin-mii's `_hip`
(kevin-mii/sglang f1ca7e1cda, 10-14 12:12 +08, still kernels/ops/mono/{common_hip,deepseek_v41_hip}). Will be a
move-only step like step 1 (ISA identical).
**Next:** read the chain; pass = GSM8K >= 0.895, NIAH all, TPOT within +-2% of base, 0 Tracebacks; then commit step 3.
**Pass criteria (every step):** CPU tests pass; ISA of every DSV4.1 build identical to the pre-refactor dump
(isa_task26/ for the committed tree) except where a step changes kernels on purpose (step 5); GSM8K TP4 >= 0.895 and
TPOT within +-2% once the hook wiring changes (steps 3-4).

## RFC #43949 (as of 10-13, after kevin-mii's reply) -- what it asks

- One switch, off by default (RFC: `--enable-monokernel`); no model env vars; test hooks `SGLANG_TEST_MONO_*`.
- Everything under `kernels/ops/mono/`: `hooks.py` (the calls SGLang makes), `spec.py` (what a model declares),
  `common_hip/` (shared device runtime incl. `guard.py` deadlines), `<model>_<vendor>/` (kernels, runner, spec, glue).
  `kernels/README.md` notes the exception. SGLang changes only at the hook calls (~43 lines in the prototype).
- Hooks: config pipeline `handle_monokernel` (find spec, refuse with every reason, set decode capture sizes);
  `mono.load_tensor()` per checkpoint tensor (before renames / quantization) + `mono.loaded()` at the end of
  load_weights; `mono.attach()` in `precompile_kernels_after_loading()` (check weights, all ranks refuse together,
  allocate all memory before KV sizing, build every width, compare routing plans); layer loop `mono.route()` once per
  forward + `mono.kernel(plan, i, point)` at each span start.
- MonoSpec: name, devices (per ISA: widths, TP sizes, resident blocks), features, requires (dtypes, KV cache,
  attention / MoE backends), spans(config) -> {layer: (Span, ...)}, numerics (BITWISE or per-output bounds
  `<span>.<output>.<metric>`), config_disable_reasons, model_disable_reasons, build -> MonoModel, load_tensor.
- Span points: layer_in -> attn_in -> core_in -> core_out -> attn_partial -> mlp_in -> mlp_partial -> layer_out;
  layer_in / layer_out carry whatever the loop passes (DSV4.1: the pending mHC post). Whole layer by default; a span
  may run SGLang modules between its launches (our indexer layers). K1 = layer_in -> core_in, K2 = core_in -> layer_out.
- Routing from the graph key (step type, batch, tokens per request); widths count rows; plans compared across ranks
  at startup; pad rows write only the dummy slot. Refuse, never fall back (incl. never partway through a step).
- Fail-stop: every cross-WG / cross-GPU wait has a deadline (half the watchdog, <= 60 s, startup refuses < 5 s); on
  timeout: sticky error word, flag every peer, fence, stop the launch with a store to address 0; one error slot,
  copied before the step's copy-done event; fault injection `SGLANG_TEST_MONO_FAULT_INJECT=rank:site[:after]`.
- Weights read in place, checked; copies logged with sizes; all memory before KV sizing; weight updates refused.
- Tests: test/registered/unit/mono/ (CPU, incl. an `SGLANG_READS` contract test), test/registered/kernels/ops/mono/
  (kernel vs normal-path ops, graph replay across widths), test/registered/e2e/mono/ (switch points: one row past the
  largest width, decode <-> verify, covered next to uncovered layer; fault injection); test/manual/kernels/mono/
  validate.py; a skill file.

## Ours today (53d0945468) -> target

| ours | target | note |
|---|---|---|
| `srt/arg_groups/mono_hook.py` (`handle_layer_backend`) + fields/choices | `kernels/ops/mono/hooks.py: handle_monokernel` called from arg_groups | switch name: step 0 |
| `srt/layers/mono/spec.py` (124) | `kernels/ops/mono/spec.py` | fields remapped (below) |
| `srt/layers/mono/registry.py`, `guards.py`, `routing.py` | `kernels/ops/mono/{registry,guards,routing}.py` | guards: feature list as RFC section 6 |
| `srt/layers/mono/layer.py` (MonoRuntime: attach / begin_step / routed) | `kernels/ops/mono/hooks.py` (attach, route, kernel) + `runtime.py` | `route()` returns a plan; `kernel(plan, i, point)` |
| `srt/layers/mono/check.py` | `kernels/ops/mono/check.py`, env `SGLANG_TEST_MONO_CHECK` | propose to RFC |
| `srt/layers/mono/failstop.py`, `runtime/*` (peer_memory, mailboxes, step_fence, consensus, widths, compile, ...) | `kernels/ops/mono/runtime/` (host) + `common_hip/guard.py` (device) | 705 unreachable lines (task 23) decide here |
| `kernels/ops/mono/gfx950/common/` | `kernels/ops/mono/common_hip/` | license notices move with it |
| `kernels/ops/mono/gfx950/deepseek_v41/` + `srt/models/deepseek_common/amd/mono/{spec,indexer_hook}.py` + glue in `deepseek_v4_fused_mhc.py` (`_mono_whole`, `_mono_checked`, `_mono_indexer_between`) | `kernels/ops/mono/deepseek_v41_gfx950/` (kernels, runner, spec, glue) | srt keeps only the hook calls |
| `srt/models/minimax_m3_mono/spec.py` | `kernels/ops/mono/minimax_m3_gfx950/spec.py` (skeleton) | or drop until zijie's PR |
| test/registered/mono/ | test/registered/unit/mono/ (+ kernels/, e2e/ later) | |

MonoSpec field map: `layers` -> `spans` (WHOLE = layer_in..layer_out as K1 layer_in..core_in + K2 core_in..layer_out;
FFN mode gone in DSV4.1); `supports` -> `features`; `supported_tp` + `widths` -> `devices["gfx950"]`;
`refuse_config` / `refuse_model` -> `config_disable_reasons` / `model_disable_reasons`; `pack` + `build` -> `build`
returning a MonoModel (all widths); `eligible` -> graph-key plan (spec sees the whole key); `state_bindings` keep and
propose; new: `requires`, `numerics`, `load_tensor`.

SGLang touch points in our PR today (non-mono files, +607 lines): arg_groups (choices, fields/exec_, mono_hook,
pipeline), environ.py, deepseek_v4_backend_hip_radix.py, dsv4/low_ratio_backend_hip.py (K2i hook), managers
(overlap_utils, scheduler), deepseek_v4_fused_mhc.py (+223: the glue), deepseek_v4_hip.py, deepseek_v4.py,
deepseek_v4_dspark.py, speculative/dspark_* + kernels/ops/speculative/dspark/simulated_bonus.py (host-sync / fold:
NOT framework -> separate PR). Target: hook calls only; the indexer hook in low_ratio_backend_hip becomes a span with
SGLang modules between launches (the glue calls SGLang's compressor / indexer from the plugin side).

## Steps

0. Decisions with the user: switch name (`--enable-monokernel` per RFC vs our proposed `--layer-backend`; implement
   one name in one place either way); keep or drop the M3 skeleton; base branch handling of the WIP; MonoSpec field
   names exactly as RFC (prefer exact, to make #43497 / kevin-mii's code drop in).
1. Move only (no behaviour change): `srt/layers/mono/*` -> `kernels/ops/mono/` (framework), `gfx950/common` ->
   `common_hip`, `gfx950/deepseek_v41` + srt spec/glue -> `deepseek_v41_gfx950/`; imports fixed; JIT key / sources
   updated. Check: ISA identical, CPU tests, the shared-helper guard test.
2. Spec in RFC shape: field renames, `devices`, `requires`, `numerics` (DSV4.1: TOLERANCE per output with bounds from
   check mode; KV record bitwise where it is), `spans` with points; registry unchanged in spirit. CPU tests updated.
3. Hooks in RFC shape: `hooks.py` with handle_monokernel / load_tensor / loaded / attach / route / kernel; attach moves
   to `precompile_kernels_after_loading()` with all memory before KV sizing and every width built (cache: report
   startup time + scratch per width); rank plan comparison at startup; SGLang side reduced to hook calls. GPU check:
   GSM8K + TPOT A/B vs snap_fw_f6_1013.
4. Guards + switch + env: RFC section 6 feature list, `SGLANG_TEST_MONO_*` (K2A A/B, check), refusals, no fallback on
   width build failure (refuse instead), `SGLANG_READS` contract test.
5. Fail-stop in the kernels: `common_hip/guard.py` deadline; every mailbox / flag wait bounded (sync.py `_spin_bounded`
   exists from ATOM: wire it with the deadline, sticky error word, peer flag, fence, store to 0); host error slot
   read after the step event; fault-injection test. This changes the ISA on purpose: TPOT A/B (RFC open question 4:
   ~1 us at 1 row).
6. Tests + report: unit / kernels / e2e layout, switch-point e2e tests, validate.py, skill; then split into PRs:
   framework, common_hip runtime, DSV4.1 kernels (whole layer incl. indexer layers), DSpark draft + host sync.

Afterwards re-apply the WIP (S3 / 22 / 27a) and resume tasks 27b, 13, 22 on the new layout.
