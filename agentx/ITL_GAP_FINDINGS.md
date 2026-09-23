# ITL gap vs ATOM — findings (sessions of 2026-09-02 / 09-03)

> **2026-09-04: session handoff.** Start from
> `RESULTS_SUMMARY_20260904.md` (self-contained state of the whole
> investigation) and `NEXT_SESSION_PROMPT_20260904.md`. Headline: **ITL is
> solved** — we beat ATOM at c128 (57.52 vs 61.5 ms) and c256 (91.11 vs
> 97.6 ms); throughput is null at c128 and −5.05 % at c256. **The entire
> remaining deficit is TTFT**, 0.42 s at c128 and 6.7 s at c256, and it is
> policy-imposed prefill deferral. Next: `HICACHE_RATIO` 5–6 at c256 (CPU tier
> exactly 100 % full), then interval 15 at c128.
>
> **Check the tree first.** This container is at the pre-swap `33979a814b` and
> the three commits the recent arms ran on are ABSENT. Patches:
> `/shared_nfs/kk/tree-backup/20260903-postswap/` (0001 = #37660, 0002 = fp8
> buffer + preload, 0003 = runtime `Wc`, the OOR root cause) with `REBUILD.md`.

> **2026-09-03 07:15Z: the image swap is DONE and the tree is REBUILT.** See
> §16 for what the rebuild actually took — it was not the recipe in
> `IMAGE_SWAP_HANDOFF_20260903.md`, because the new image's sglang base is
> later than that document assumed. Current state: sglang `1e41776161` on base
> `2641e427be`, aiter `c16d44b93` + 9 local files, both import clean.
> **Every number below was measured on the old image** — re-run one arm as a
> bridge before quoting any cross-swap comparison.
>
> **Artifact locations changed.** `/workspace/handoff-20260903-image-swap/` is
> gone (consumed). The durable rebuild kit is
> `/shared_nfs/kk/tree-backup/20260903-postswap/` with a `REBUILD.md`. New
> logs, traces and scratch files go under **`/shared_nfs/kk/`**
> (`logs/`, `traces/`, `tmp/`) — xfs, 53 TB free, survives an image swap.
> Arm artifacts stay in `/workspace/results/`.
>
> **STATE 2026-09-04 02:10Z: nothing in flight, node idle (2.38 GB, no
> processes). Tree VERIFIED HERE: sglang `efaeb6f664`, aiter `c16d44b93` + 15
> local files.** The fusion series finished all three arms (§19 c128, §20 c192,
> §21 c256) at mem-frac 0.90 — the 0.90→0.87→0.85 ladder has never fired.
>
> **Cross-session hazards.** A concurrent session wrote
> `RESULTS_SUMMARY_20260904.md` and §21 here. Its §5 says the tree is back at
> pre-swap `33979a814b` — true of *its* container, NOT this one; a correction is
> appended to that file. **Always `git log -1` before rebuilding.** Also, most
> of `claude-skills/agentx/` shares one mtime (02:05Z), i.e. a directory-level
> restore, not per-file edits: assume these docs can be overwritten and keep
> anything load-bearing in `/shared_nfs/kk/` too.
>
> `hicache-fp4-int20-c256-fuse` (§21): 42,462 tok/s/GPU, **ITL p90
> 91.11 ms which BEATS ATOM's 97.6 ms** — first time at a matched concurrency —
> throughput now within **−5.05 %** of ATOM's 44,722, but TTFT 20.14 s against
> 13.4 s. The whole remaining deficit is TTFT, and it is policy-imposed deferral
> (queue p90 17, miss rate unchanged at 4.96 %), so the lever is the interval.
> The CPU tier is **100 % full** again, so P2 (`HICACHE_RATIO` 5–6 at c256) is
> still the untested lever with a measured reason to exist.
>
> **Reading the three fusion arms:** they are a CURVE, not a set of deltas.
> `arm_report.py` rejects a cross-concurrency headline
> (`conc-and-trace-mix.md` 19.6) and ISL moves ~10 % between them, so quote the
> tok/s points (30,373 → 38,822 → 42,462) and never the % delta.
>
> **§19 is the state before that.** Shared-experts fusion at c128 ran at mem-frac
> 0.90 first try, no OOR. Throughput and ITL are **null** (+0.07 %, −0.40 %),
> but it **saves memory**: weights −3.45 GB/rank (the shared experts are
> requantised FP8→FP4, not replicated), KV pool +3.13 %, and free VRAM goes from
> ~0 to **~15 GB** — this arm is no longer a coin flip. The TTFT −6.95 % is
> **not attributable to fusion** (two variables moved, and TTFT is drifting
> across arms). **The P0(b) gate reading was RETRACTED** — the watchdog is
> silent above 1 GiB free, so 0 late loads at 15 GB free proves nothing; arm
> scripts now set `SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=1000`.
>
> **§18: the bridge arm.** The image swap
> moved nothing measurable** (tok/s +2.11 %, ITL −0.40 %, both inside the
> 5.67 % noise band, KV pool and ISL matched) — so §1's table may be quoted
> again. P0(b)'s first attempt **failed** that arm with 74 late Triton loads;
> root-caused to `Wc` being a `tl.constexpr` that tracks context length, and
> fixed in `efaeb6f664`. Tree is sglang `efaeb6f664`. **The P0 gate has not yet
> passed a real arm** — until one shows `device-loaded after serving started`
> = 0, treat every arm as a coin flip. P0(a) (bounded fp8 buffer) is still
> unexercised and needs a non-FP4 arm.

## CONTINUE HERE (written 2026-09-02 14:15Z, node left clean and idle)

**HEADLINE: the ITL gap is a tuning choice, not a deficit. One flag closed it.**
`--prefill-decode-interval 10 → 20` at c128, single variable, both `GATES PASS`,
ISL matched 1.81 %, cache 0.942 on both, FP4 off, TBO on, mem-frac 0.90:

| metric | interval 20 | interval 10 (baseline) | ATOM c128 | delta |
|---|---|---|---|---|
| **ITL p90** | **58.78 ms** | 78.39 ms | 61.5 ms | **−25.0 %, now BEATS ATOM** |
| tok/s/GPU | 29,130.5 | 27,894.8 | 30,709 | +4.43 % (inside noise) |
| TTFT avg | 13.22 s | 8.50 s | 10.9 s | **+55.5 %, now WORSE than ATOM** |
| intvty p90 | 17.01 | 12.76 | 16.3 | +33.4 % |

Against the pre-registered criteria: ITL **passed** (58.78 ≤ 61.5) and TTFT
**failed** (13.22 > 10.9). We overshot on ITL and overspent the TTFT budget, so
**the optimum is between 10 and 20** — one flag moved ITL 20 ms, which is more
than the entire gap. There is **no need to profile a decode step**: SGLang's
decode is not intrinsically slower, it was being interrupted more often.
Throughput moved the *same* direction as ITL (+4.4 %, with OSL +4.5 % and succ
+2.6 % agreeing), so this is not a pure latency-for-throughput trade.

**UPDATE 2026-09-03 — THE c192 TRACK IS DONE AND IT IS THE BIGGEST RESULT ON THE
BOARD.** `hicache-fp4-c192` vs `fp4-dptbo-c192`, single variable (hicache CPU
tier at ratio 3.0), both `GATES PASS`, ISL matched 1.31 %: **tok/s/GPU +58.36 %**
(22,994 → 36,414), **TTFT avg −84.74 %** (59.71 → 9.11 s), ITL p90 flat
(−1.71 %). Full write-up in §13; §11 is the smoke, §12 fixes the pass criterion.

Consequences: §9's prefill-saturation mechanism is **confirmed** (queue p90
78 → 7, decode batch 15 → 23, miss rate 8.15 → 5.01 %); the "knee at c160" was an
artefact of hicache being off in every arm; §4b's outstanding OOR verification
**passes** (free VRAM median 6.77 GB, min 2.71 GB, 0 late Triton loads).

**c256 is now done too (§14): 39,284 tok/s (+7.9 % over c192), but we LOSE to
ATOM's c256 on all three axes** — throughput −12.2 %, ITL p90 +19.7 %, TTFT
+17.3 %. Not the engine's ceiling though: the **CPU tier hit 99.98 % full** and a
fifth of all reuse demoted to it, so `HICACHE_RATIO=3.0` is the binding
constraint at c256 (it was not at c192, 52.8 %). Overall cache hit is FLAT at
0.951 — only the tier split moved.

**c128 + interval 20 + hicache is also done (§15) and it FAILED the pair:** ITL
p90 57.98 ms passes, TTFT 13.77 s does not. hicache turned out to be a **no-op at
c128** — CPU-tier hit 0.5 pp, device pool only 73 % allocated with KV occupancy
median 0.27, so there is no eviction pressure and nothing to recover. It also
proved the two levers are **orthogonal**: interval 20's TTFT is policy-imposed
deferral (`#queue-req` p90 = 9, trivial), not prefill demand, so cheaper prefill
cannot pay it off. **Rule: hicache only pays where GPU KV pool occupancy is high
(96 % at c192, 100 % at c256, 73 % at c128 = nothing).**

**Next, in order:**
(a) **The fp8 bounded-buffer fix, FIRST and before any further arm.** The c128
    arm ran with free VRAM p10 **0.08 GB**, min **0.01 GB** and **10 late Triton
    device loads** — the exact §3 abort signature, survived on luck. Mirror
    `b6e3728`'s bounded buffer into `_aiter_fp8_paged_mqa_logits`
    (`indexer.py:160`) and pre-load the Triton specialisations at init.
(b) **Task 1's interval sweep, unchanged: 15 then 12 at c128, hicache OFF.**
    ATOM's (61.5 ms, 10.9 s) sits strictly inside the box bounded by int 10
    (78.39 ms, 8.50 s) and int 20 (57.98 ms, 13.77 s), so a middle value can
    satisfy both axes. Leaving hicache off keeps it a clean single-variable
    series and costs nothing, per §15.
(c) c256 at `HICACHE_RATIO=5`–6, where the tier IS the constraint (99.98 % full)
    — host DRAM is 3,023 GB with 1,442.6 GB pinned, so it fits.
Node left clean and idle.

Trap that nearly cost 95 minutes: the arm exits `ARM_EXIT=1` *after* a fully
successful benchmark because the aggregation step needs an undocumented third
variable, `KV_OFFLOAD_BACKEND_METADATA`. No re-run needed — §13 has the
one-command fix. Add it to every future hicache script.

**Next action, in priority order:**

1. **`--prefill-decode-interval 15` at c128** — same script, one number changed:
   `cp interval20_c128.sh interval15_c128.sh`, edit `EXTRA_SERVER_ARGS` and
   `RESULT_DIR`. Pass criterion: ITL ≤ 61.5 ms **and** TTFT ≤ 10.9 s, i.e. match
   ATOM on both axes at once. That would be the reportable result.
2. **c192 with hicache** (user's call, taken up 2026-09-03 — see §11 for the
   smoke result; the CPU tier **works**, 743.6 GB allocated, and the rust
   pool-name worry was unfounded).

   **`KV_OFFLOADING=hicache` is an invalid value** and would have aborted before
   the model loaded. `benchmark_lib.sh:44-67` accepts only `none` and `dram`; the
   working combination is `KV_OFFLOADING=dram` **plus**
   `KV_OFFLOAD_BACKEND=hicache` plus a positive integer `TOTAL_CPU_DRAM_GB`.

   This is the c192 TTFT question, which is a *different* problem from the ITL
   gap — see §9/§10 of `DATA_AND_ANALYSIS_20260902.md`. Concrete pass criterion:
   the miss rate `Σnew / Σ(new+cached)` from `Prefill batch` lines should fall
   from **11.70 %** back below **8.6 %** (c160's level), and TTFT should follow.
   If the miss rate falls but TTFT does not, prefill demand is not the driver.
   - The launcher supports it: `KV_OFFLOADING=hicache`, `HICACHE_RATIO` 1.5,
     `write_through`, `direct`, `page_first_direct` (b200align_mtp.sh:113-124).
   - **Run a `DURATION=300` smoke first.** hicache has been off in every arm on
     this node because #37353's rust `DeepseekV4C4IndexerScale` pool-name change
     was skipped. That pool is FP4-specific, so FP4-off *should* avoid needing the
     rust rebuild — inference, not verified. 30 min to find out vs 1.5 h to lose.
3. Do **not** re-run the chunk-size diagnostic: the previous node's c256 pair
   already measured it (16,384 → 8,192 gives ITL −25 %, TTFT +147 %).

**Node state:** clean and idle, 0 processes, 2.2 GB across 8 GPUs. `sglang` on
`33979a814b` with 19 other-session files still uncommitted. Scripts:
`interval20_c128.sh` (copy this one), `vram_sampler.sh`, `tbo_debug_probe.sh`.
Arm artifacts in `/workspace/results/interval20-c128/`.

**Still open:** the fp8 path's unbounded `torch.empty(total_tokens, max_seq_len)`
at `dsv4/indexer.py:160` — this arm's min free VRAM was **0.98 GB** with 6 late
Triton device loads, so the OOR cliff is still there for non-FP4 arms (§3 below).

Separate file on purpose: `AGENTX_20260901.md` is being edited by a concurrent
session and my earlier edits to it were overwritten. Merge from here, do not
assume that file carries any of this.

## 1. Experiment 1 is done: TBO is NOT the ITL gap

`dptbo-notbo-c128` vs `dptbo-c128`. Both `GATES PASS`, 3629.5 / 3629.4 s, ISL
matched to 1.12 %, cache hit 0.942 on both, FP4 indexer **off** on both
(verified in each `sglang_command.txt`), mem-frac 0.90 on both. Single variable:
`ENABLE_TBO`.

| metric | TBO off | TBO on | delta |
|---|---|---|---|
| tok/s/GPU | 26,994.1 | 27,894.8 | −3.23 % (inside 5.67 % noise → null) |
| **ITL p90** | **83.93 ms** | **78.39 ms** | **+7.07 %, WORSE** |
| TTFT | 10.13 s | 8.50 s | +19.18 % worse |
| intvty p90 | 11.92 | 12.76 | −6.60 % |

Turning TBO off moved ITL p90 **away** from ATOM's 61.5 ms (gap +27 % → +36 %).
Per the pre-registered reading in `ITL_GAP_PROMPT.md`, the scheduler is
exonerated and **the next step is profiling a decode step, not another arm**.

Side result worth quoting: prefill-only TBO **improves both ITL and TTFT** on
AgentX — the opposite sign to ATOM's fixed-seq-len crossover (−14 % / −10 % at
c64/c128, runs 30257759947 vs 30238071409). AgentX is prefill-heavy (ISL ~99 k
mean, p90 209 k), so prefill overlap has far more to work with. **Do not port
ATOM's "TBO only at c256+" threshold to agentic.**

## 2. Both engines run prefill-only TBO — there is no decode-side asymmetry

This corrects a wrong intermediate conclusion of mine from the same session.

- **ATOM:** bare `--enable-tbo` is `argparse const=prefill` →
  `enable_tbo_decode=False`; decode-TBO would drop MTP's `spec_decode_metadata`
  in `UBatchWrapper`. Source: AMD's own launcher,
  `InferenceX/benchmarks/single_node/fixed_seq_len/dsv4_fp4_mi355x_atom_mtp.sh:44`.
- **SGLang:** `DeepseekV4ForCausalLM._can_run_tbo`
  (`models/deepseek_v4.py:2840-2876`) requires
  `global_forward_mode.is_extend_without_speculative()`, whose own comment says
  "MTP target-verify also reports is_extend(); only real prefill should enter the
  prefill TBO strategy". Independently,
  `batch_overlap/operations_strategy.py:182` **raises** `NotImplementedError`
  ("DeepseekV4 TBO only supports prefill (EXTEND)") for any other mode — our
  arms ran a full hour without it, which alone proves decode never entered TBO.

Trap to not repeat: `SGLANG_TBO_DEBUG=1` shows TARGET_VERIFY batches being
*prepared* at graph-capture time (128 of them, `idx` 1..16 × 8 ranks). That is
model-agnostic batch prep which DSV4 then declines to use. It is **not** evidence
of decode TBO. Probe artifacts: `/workspace/results/tbo-debug-probe/`, driver
script `tbo_debug_probe.sh` (~8 min, replays a recorded `sglang_command.txt`
verbatim + gsm8k 64-way, no aiperf — much cheaper than a short agentic arm,
because server startup is only ~3 min of a 92 min arm).

Consequence for finding A: whether ATOM's DP table had `--enable-tbo` (the open
`run 33074134043` question, ROCm CI, unreachable — org blocks **all** classic
PATs, verified with a second token) now only decides which number to quote,
+27 % (both TBO-on) or +36 % (both TBO-off). It no longer changes the next step.

## 3. OOR abort: the trigger is a lazily device-loaded Triton kernel

The line immediately before the abort in `fp4-dptbo-c64-reclaim0/server.log`:

```
[2026-09-02 02:28:22 DP7 TP7] Triton kernel '_prefill_lengths_kernel'
device-loaded after serving started (free device mem: 0.00 GiB).
Pre-load it during engine init to avoid CUDA OOM.
:0:rocdevice.cpp:3582: Callback: Queue 0x714b24600000 Aborting with error :
HSA_STATUS_ERROR_OUT_OF_RESOURCES
```

So the failing allocation is a **code-object load** (`hipModuleLoad`), not a
tensor. SGLang has a watchdog for exactly this, `srt/utils/triton_load_watch.py:115`.
The kernels are `_prefill_lengths_kernel` / `_build_prefill_indices_kernel` in
`kernels/ops/attention/dsv4/unified_kv_kernels/runtime.py:298,332` — the
unified-KV path this launcher selects with
`SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`.

They load mid-run because their launch passes
`BLOCK=min(1024, next_power_of_2(max(win, Wc, 1)))`, `HAS_COMPRESS` and `Wc` as
**constexpr** specialisations, and `Wc = page_idx.shape[1]` grows with context
length. Triton loads one code object per constexpr combination, so every time a
session crosses into a new power-of-2 bucket a fresh object is loaded — onto a
device whose torch pool already holds everything.

Every arm walks this cliff; falling off is luck:

| arm | late loads | free at load | outcome |
|---|---|---|---|
| `dptbo-c128` | 2 | 0.17 GiB | survived |
| `fp4-dptbo-c128` | — | 0.09 GiB | survived |
| `dptbo-notbo-c128` | **101** | 0.84 → **0.00** GiB | survived |
| `fp4-dptbo-c64-reclaim0` | 2 | **0.00** GiB | **aborted** |

`dptbo-notbo-c128/vram.csv` (15 s cadence, 326 samples): min free **0.21 GB**,
median **0.95 GB**, max 3.93 GB across the whole hour. Flat at zero, no decay —
which matches the concurrent session's c96 observation of 1.0 GB flat.

### How this reconciles with the concurrent session's scratch hypothesis

That session concluded the transient logits tensor cannot be the cause "because
it goes through torch, so it would either fit or raise a torch OOM, and no torch
OOM has ever appeared". Both halves are right, and the conclusion still needs one
step added: **the transient never fails — it sets torch's high-water mark.** The
caching allocator does not return segments, so a 13.7 GB transient permanently
converts 13.7 GB of driver-visible memory into torch-reserved cache. That is
*why* driver-visible free sits at ~1 GB flat. Whether the thing that then dies is
a Triton code object (observed here) or HSA kernel scratch (their mechanism,
plausible, not yet directly evidenced) is downstream of the same cause.

This matters practically: capping the transient restores headroom **without**
spending KV pool, which mem-frac 0.85 cannot do.

## 4. Two candidate fixes, compared

**(A) upstream commit `b6e3728143031c26c3b0d94d4db85d50b8ff9002`** — "use a
bounded prefill logits buffer and process oversized batches in row chunks",
3 files, +249/−81, with `test_fp4_indexer_hip.py`. Pools the prefill logits into
a budgeted block (`SGLANG_DSV4_FP4_LOGITS_BUDGET_MB`, default 2 GiB — "2 GiB
covers 4096 rows over ~512K tokens of context") and loops
`rows_per_chunk = logits_rows_per_chunk(page_table)`, reducing each chunk before
the next overwrites it, so the transient is capped **independently of context
length and chunked-prefill size**.

Not in our tree (`logits_rows_per_chunk` absent). Its row-chunk loop is inside
`if use_aiter_fp4:`, so it covers the **FP4** path only. The fp8 path keeps the
unbounded allocation at `dsv4/indexer.py:160` in `_aiter_fp8_paged_mqa_logits`:
`torch.empty(total_tokens, max_seq_len, dtype=torch.float32)` — the 13.7 GB.

**(B) pre-load the Triton specialisations at engine init** — what the watchdog
message asks for. `BLOCK` has ≤ ~11 power-of-2 values, times `HAS_COMPRESS` and
`compress_ratio` ∈ {0, 4, 128}.

| | A: bounded logits buffer | B: Triton pre-load |
|---|---|---|
| attacks | the **cause** (pool high-water mark) | the **trigger** (one late allocation) |
| protects | every non-torch allocation, incl. HSA scratch | only these Triton loads |
| path coverage | FP4 only | path-independent |
| KV-pool cost | none | none |
| already written | yes, upstream, with a test | no |
| serving-path change | yes (chunked scoring) → FP4 numbers need re-measuring | no, init only |
| local conflict | **high**: `dsv4/indexer.py` is tracked-**modified** by our hand-applied #37353 and the commit rewrites the same region (upstream ~822 ≈ our 794); `fp4_indexer_hip.py` is **untracked**, created by hand | **none**: `unified_kv_kernels/runtime.py` is clean |
| verification | driver free should rise ~11 GB in `vram.csv` | `device-loaded after serving started` count → 0 |

### Recommendation

1. **Take (A) for the FP4 track**, and prefer it over the concurrent session's
   planned mem-frac 0.85 arm — or at least run it first. Reason beyond memory:
   **ATOM runs `--gpu-memory-utilization 0.9`**, so dropping to 0.85 breaks the
   memory matching that the entire ITL comparison rests on, and it invalidates
   every 0.90 arm on the board. (A) restores headroom at 0.90 and only costs a
   re-measure of the FP4 side; the `dptbo-c128` baseline stays valid.
2. **Also do (B)** — cheap, on a clean file, init-only, path-independent. It is
   belt-and-braces, not a substitute: it narrows the failure surface without
   restoring headroom.
3. **Highest-value extra:** mirror (A)'s pattern into
   `_aiter_fp8_paged_mqa_logits` (`indexer.py:160`) so the non-FP4 arms — the
   ones task 1 uses — get the same headroom. Upstream-shaped.

## 4b. DONE (2026-09-02 07:2x): FP4 committed, then `b6e3728` cherry-picked

`/sgl-workspace/sglang` is on branch `main` with two new local commits:

```
33979a814b  use a bounded prefill logits buffer and process oversized batches in row chunks
83310485e1  DSV4 FP4 C4 indexer (#37353) — hand-applied working-tree integration
52e1c24744  (was HEAD)
```

**Backup taken first, independent of git:** `/workspace/tree-backup-20260902-072044/`
holds `sglang-dirty.tgz`, `aiter-dirty.tgz`, both repos' `git status --porcelain`
and both `HEAD` SHAs, captured before anything was staged.

**What was committed in `83310485e1`:** the 15 modified DSV4/FP4 files plus the
by-hand `fp4_indexer_hip.py`. **Deliberately left uncommitted** — other sessions'
work, still 19 entries and byte-unchanged: `python/pyproject.toml`,
`kernels/aot/pyproject.toml`, the deletion of `kernels/aot/pyproject_rocm.toml`,
and 16 untracked `kernels/aot/csrc` `.hip`/`.h`/`.cuh` files. `aiter` was not
touched at all (its FP4 bits, #5126 + `pa_mqa_logits_fp4_prefill.py`, remain
uncommitted; the fix does not need them).

**The cherry-pick had two conflicts, both resolved:**

1. `test/registered/kernels/ops/attention/test_fp4_indexer_hip.py` —
   modify/delete, because our tree never had the PR's test. Took upstream's file
   whole, which is why the commit reads +1027 rather than the +249 of the
   original diff.
2. `dsv4/indexer.py`, one hunk. Upstream **hoists** the `raw_indices`
   resolution above the scoring loop ("Resolved before the scores because the FP4
   path consumes each row chunk's logits before the next chunk overwrites them"),
   and that hoist applied cleanly at lines 795-817 — so the HEAD side of the
   conflict was a verbatim duplicate of it and was dropped. Took upstream's
   structure, the `if not use_aiter_fp4:` wrapper, with **one deviation**:
   upstream calls `self.flashinfer_topk_transform(...)`, which **does not exist
   in this tree** (0 definitions), so the flashinfer branch keeps our
   `topk_transform_512_flashinfer_unfused(...)`. There is an inline comment
   saying so. `self.dsa_topk_backend.should_use_topk_v2()` **does** exist
   (`dsa/dsa_topk_backend.py:52`) so upstream's condition was taken as-is; note
   its semantics differ from the `envs.SGLANG_OPT_USE_TOPK_V2.get()` we had
   (`is_sgl_kernel() and env` vs env alone).

**Verified:** `logits_rows_per_chunk` present in both files,
`SGLANG_DSV4_FP4_LOGITS_BUDGET_MB` defaults to `"2048"` MB, and both
`dsv4.indexer` and `fp4_indexer_hip` import cleanly with aiter loading.

**Consequences for the numbers on the board.** The FP4 scoring path changed, so
**every FP4 arm's numbers are now stale**, including the +2.57 % c128 pair — that
pair has to be re-measured on this tree regardless of the replicate it was
already owed. The non-FP4 baseline `dptbo-c128` is untouched by the fix in
behaviour, but it too now runs different source, so state the commit SHA with any
new comparison. Arm scripts' `TREE_CHECKSUMS_AT_START.txt` md5s from earlier arms
no longer match; that is expected, not tampering.

**Revert path, if needed:** `git reset --hard 52e1c24744` would also destroy the
other sessions' uncommitted files — do **not** use it. Use
`git revert 33979a814b` to undo only the fix, or `git reset --soft 52e1c24744 &&
git reset` to return both commits to the previous uncommitted-working-tree state.

**Next verification, on the next FP4 arm:** driver-visible free VRAM in
`vram.csv` should rise by roughly 11 GB (from ~1 GB toward ~12 GB) if the 13.7 GB
transient was indeed what set torch's high-water mark, and
`rg -c 'device-loaded after serving started' server.log` should stop coinciding
with `free device mem: 0.00 GiB`.

## 5. Operational: this node is shared, and I broke another arm

My `dptbo-notbo-c128` finished 04:56:29Z. The concurrent session's
`fp4-dptbo-c96` attempt 1 auto-started 04:57Z. At 05:12–05:15Z I ran two kill
rounds on `[s]glang::|[s]glang\.launch_server|[s]glang_router` after
misreading "3 leftover processes, 38-50 GB/GPU" as my own arm's residue — it was
their server starting. Their server.log stops at **05:12:26Z**.

The inverse of the rule in the report: *processes being present is not evidence
they are yours*. Before any kill, check `ps -eo pid,lstart,args` and compare
against your own arm's end time. Also: `pkill -9 -f 'sglang::'` matched my own
shell's command line and killed the shell running it — use a bracketed-pattern
PID loop, which the arm scripts already do.

## 11. HiCache CPU tier: the smoke passed, and two premises were wrong

`hicache-smoke-c192` (`hicache_smoke_c192.sh`, `DURATION=300`, c192, FP4 off,
TBO on, mem-frac 0.90, launcher's own `--prefill-decode-interval 10`). Run as a
smoke deliberately: it answers go/no-go in ~40 min instead of spending 92.

**Two premises carried from 2026-09-02 were both wrong, in our favour:**

1. **`KV_OFFLOADING=hicache` does not exist.** `benchmark_lib.sh:44-67` accepts
   only `none` and `dram` and exits 1 on anything else, *before* the model
   loads. The working set is `KV_OFFLOADING=dram` +
   `KV_OFFLOAD_BACKEND=hicache` + positive integer `TOTAL_CPU_DRAM_GB`. Anyone
   copying the old instruction would have lost a slot to a 5-second failure.
2. **The rust `DeepseekV4C4IndexerScale` pool-name worry was unfounded**, and
   not for the reason we guessed. The guess was "FP4 off should avoid needing
   the rust rebuild". The real situation is stronger: there are **zero `.rs`
   references** to that name anywhere in the tree, `PoolName` already carries
   `DEEPSEEK_V4_C4_INDEXER_SCALE` (`mem_cache/hicache_storage.py:71`), and the
   launcher sets no `--hicache-storage-backend`, so **no rust storage tier is in
   play at all** — only the host-DRAM L2 tier. Verified empirically too: the
   `..._indexer_scale` pool is simply absent from the allocation log with FP4
   off, exactly as the FP4-specific reading predicted.

**Host pinned pools actually allocated** (`memory_pool_host.py:258`, per rank ×8,
`HICACHE_RATIO=1.5`, `write_through`, `direct`, `page_first_direct`):

| pool | per rank | × TP8 |
|---|---|---|
| `deepseek_v4_c4` | 80.05 GB | 640.4 GB |
| `deepseek_v4_c4_indexer` | 10.32 GB | 82.6 GB |
| `deepseek_v4_c128` | 2.58 GB | 20.6 GB |
| **total** | **92.95 GB** | **743.6 GB** |

Against 2,960 GB available host DRAM, so ratio 1.5 is not near any ceiling —
there is room to raise it later if the tier turns out to help but be too small.
`TOTAL_CPU_DRAM_GB` is **only echoed** by this launcher (`:118`); it does not
size the pool, so it is a declared budget for validation, not a knob.

**Confound to state with any number from this track:** the c192 baseline
`fp4-dptbo-c192` ran FP4 **on**, this runs FP4 **off**, so hicache is not the
only variable. Defensible for the cache metrics (the FP4 indexer changes
attention scoring, not what the radix tree stores or hits) but **not** for tok/s.
A clean headline pair needs the baseline re-run FP4-off — which §4b already owes
anyway, since `33979a814b` made every FP4 number stale.

### The smoke's own results (Q1-Q3 all green)

- **Q1 the flags landed:** `'enable_hierarchical_cache': True`,
  `'hicache_ratio': 1.5`, `'hicache_mem_layout': 'page_first_direct'` in
  `server_args`; all five `CACHE_ARGS` present once in `sglang_command.txt`.
- **Q2 the host tier is real and it is being used.** From aiperf's own
  `server_metrics_export.json`, summed over 8 series:
  `sglang:hicache_host_total_tokens` = **83.38 M** against a device pool of
  6,948,608 × 8 = 55.59 M, i.e. ratio **1.49** as configured, and
  `sglang:hicache_host_used_tokens` avg **62.35 M = 74.8 % full**, max 76.4 %.
  `hicache_dropped_tokens` = 0. So the CPU tier is not decorative — it filled to
  three quarters within one warmup pass.
- **Q3 it served cleanly:** 727 decode batches, 0 `Traceback` / 0
  `HSA_STATUS_ERROR`, FP4 off (`fp4=0`), TBO on, one server start (8 "fired up"
  lines = 8 DP ranks, a single Uvicorn bind), tree md5s unchanged.
  aiperf's "counter reset(s) ... indicates server restart(s)" WARNINGs are a
  red herring here — they are per-DP-rank counter resets, not a restart.
- **Device headroom got better, not worse**, which was the opposite of my worry
  that `page_first_direct` + `direct` IO would map host memory at device cost.
  Free-VRAM distribution over the run: median **8.36 GB**, p10 5.69 GB, min 0.10 GB
  — versus `interval20-c128`'s median 2.91 GB, p10 2.64 GB, min 0.98 GB. The
  single 0.10 GB sample is a mid-warmup transient on one GPU and there were **0**
  late Triton device loads, so the OOR trigger never armed.
- **Host RSS rose only 48 -> 130 GB** while 743.6 GB was requested, so the pinned
  pool is demand-paged rather than resident up front. Do not size future runs
  off `free`; use `hicache_host_total_tokens`.

## 12. The `11.70 % -> 8.6 %` pass criterion was invalid, and here is the fixed one

Both numbers in the criterion were computed over the **whole** `server.log`,
which on every arm includes a ~37-minute cold-cache aiperf warmup pass. That
warmup contributes roughly half of all `Prefill batch` lines and its miss rate is
far above steady state, so the figures measure the warmup as much as the arm.

Restricting to each arm's **measurement window** (the last 3600 s before the
log's final timestamp) changes every number:

| arm | miss, whole log | miss, **measurement window** | batches |
|---|---|---|---|
| `fp4-dptbo-c96` | 7.72 % | **5.59 %** | 4,942 |
| `fp4-dptbo-c128` | 8.14 % | **5.81 %** | 5,247 |
| `fp4-dptbo-c160` | 8.60 % | **5.96 %** | 5,594 |
| `fp4-dptbo-c192` | 11.70 % | **8.15 %** | 4,413 |

The **mechanism in §9 survives** — and its headline number survives almost
exactly. c192 over c160 is 8.15/5.96 = **+36.7 %**, against the +36 % claimed
from the contaminated figures. That agreement is luck (the contamination happens
to scale similarly across arms), not validation of the old method.

**Corrected pass criterion for the full hicache arm:** the measurement-window
miss rate must fall from **8.15 %** toward **5.96 %** (c160's level). Quoting
"below 8.6 %" would have declared victory on a number the baseline *already*
beats by half a point.

### Direction from the smoke, on matched windows

300 s of measurement cannot settle this, so compare like with like — the first
300 s of each arm's measurement window, ~530 prefill batches each:

| | miss |
|---|---|
| `fp4-dptbo-c192`, hicache off | 8.12 % |
| `hicache-smoke-c192`, hicache on | **7.61 %** |

−6.3 % relative, the right direction, but only about a quarter of the distance to
c160's 5.96 %. Two reasons to expect the full arm to do better than this, and one
lever:

- the host tier had been warmed by a single dataset pass when measurement began;
- it was already **74.8 % full** at ratio 1.5, so it is plausibly the binding
  constraint rather than the mechanism being wrong.
- **Lever: raise `HICACHE_RATIO`.** 1.5 costs 743.6 GB of a node with 2,955 GB
  available, so 3.0 (~1.49 TB) is comfortable and doubles the L2 window. Worth
  spending the full arm at 3.0 rather than re-confirming 1.5.

**Still confounded:** FP4 on in the baseline, off here. Fine for miss rate,
not for tok/s. See §11.

## 13. RESULT: hicache fixes the c192 collapse outright. +58 % tok/s, −85 % TTFT

`hicache-fp4-c192` vs `fp4-dptbo-c192`. **Single variable** (hicache on/off):
FP4 indexer **on** in both, TBO on, mem-frac 0.90, chunk 16,384/rank,
`--prefill-decode-interval 10` in both, c192, `HICACHE_RATIO=3.0`. Both
`GATES PASS`, duration 3629.9 / 3627.6 s, ISL matched **1.31 %**.

| metric | hicache on | baseline (off) | delta | for reference: c160 |
|---|---|---|---|---|
| **tok/s/GPU** | **36,413.9** | 22,994.1 | **+58.36 %** | 31,549 |
| **TTFT avg** | **9.11 s** | 59.71 s | **−84.74 %** | 9.56 s |
| succ records | 9,495 | 5,914 | +60.55 % | 8,651 |
| ITL p90 | 99.33 ms | 101.06 ms | −1.71 % (flat) | 87.8 ms |
| cache hit | 0.937 | 0.920 | +1.7 pp | 0.944 |
| ISL mean | 110,515 | 111,977 | −1.31 % | 105,049 |

`+58 %` is an order of magnitude outside the 5.67 % replicate spread. **36,414
tok/s/GPU is the best number on the whole board**, above the previous c160 peak,
so the c192 knee is not merely repaired — the curve now keeps climbing past it.
ITL p90 is flat, so none of this was bought with decode latency.

### §9's mechanism is confirmed, prediction by prediction

§9 said c192 was prefill-capacity saturation: prefill demand exceeded admission
capacity, the queue diverged, TTFT became queue wait, and the throughput loss
came from a *smaller* decode batch rather than slower decode. Every one of those
reverses when prefill demand is cut, measured over each arm's 3600 s measurement
window:

| | c160 | c192 baseline | **c192 + hicache** |
|---|---|---|---|
| miss rate `Σnew/Σ(new+cached)` | 5.96 % | 8.15 % | **5.01 %** |
| `#queue-req` med / p90 / max | 1 / 7 / 20 | **8 / 78 / 87** | **1 / 7 / 21** |
| decode `#running-req` med / p90 | 20 / 26 | **15 / 23** | **23 / 31** |
| `full token usage` med / p90 | 0.41 / 0.54 | 0.36 / 0.54 | **0.53 / 0.74** |

- **Pass criterion met with room to spare.** The corrected target was 8.15 % →
  5.96 % (c160's level); we got **5.01 %**, i.e. −38.5 % relative and *below*
  c160. Under the old, warmup-contaminated criterion this would have read
  7.93 % vs "below 8.6 %" and looked like a marginal pass instead of a rout.
- **TTFT followed**, which was the falsification condition: §9 said if the miss
  rate fell but TTFT did not, prefill demand was not the driver. TTFT fell 85 %,
  to 9.11 s — essentially c160's 9.56 s.
- **The queue is gone.** `#queue-req` p90 went 78 → 7, exactly c160's value. TTFT
  was queue wait, as claimed.
- **The decode batch grew past c160's**, 15 → 23 running requests, with KV
  occupancy 0.36 → 0.53. §9 said the 27 % throughput regression came from a
  smaller, less efficient decode batch, not from slower decode; ITL p90 staying
  flat while throughput rose 58 % is that statement's direct confirmation.

### Where the recovered hits actually come from — not where we predicted

The old rationale was the previous node's c256 run, where GPU tier was 66.3 % of
a 94.4 % total, i.e. "a third of all prefix reuse came from CPU". **That is not
what happened here.** aiperf's tier split:

| | baseline | hicache |
|---|---|---|
| GPU tier hit | 92.0 % | **93.7 %** (+1.7 pp) |
| CPU/offload tier hit | n/a (no tier) | **1.4 pp** |
| total | 92.0 % | ~95.1 % |

The two contributions sum to +3.1 pp, and the measured miss rate fell by
3.14 pp (8.15 → 5.01) — the arithmetic closes. But only **45 %** of the gain is
the CPU tier serving reads directly; the larger **55 %** is the *GPU* tier
hitting more often. The plausible reading, not yet directly evidenced: with a
host backing store the device pool can evict without losing a prefix
permanently, so device-tier evictions stop turning into cold prefill. Do not
report the "a third of reuse comes from CPU" framing — at ratio 3.0 on this node
the direct CPU read share is 1.4 pp.

Tier utilisation: `hicache_host_total_tokens` 171.7 M, `hicache_host_used_tokens`
avg 90.6 M = **52.8 % full** (max 119.1 M = 69 %), `dropped_tokens` 0. So ratio
3.0 is **not** the binding constraint any more — 1.5 was (74.8 % full in the
smoke). No reason to raise it further; if anything 2.0–2.5 would likely do.

### Two operational results that come free

1. **§4b's outstanding verification passes.** This is the first FP4 arm since
   `33979a814b`'s bounded prefill logits buffer. Free VRAM over the run: median
   **6.77 GB**, p10 4.74 GB, **min 2.71 GB** — against the old FP4 arms' 0.09 GB
   and `interval20-c128`'s 0.98 GB. **0** late Triton device loads. The OOR
   cliff described in §3 is no longer being walked on this path. The fp8-path
   fix (mirroring the bounded buffer into `_aiter_fp8_paged_mqa_logits`,
   `indexer.py:160`) is still outstanding and still worth doing, but it is no
   longer blocking FP4 arms.
2. **The FP4 scale pool has full host-tier support.** With FP4 on, hicache
   allocated `deepseek_v4_c4_indexer_scale` (0.64 GB/rank) alongside
   `deepseek_v4_c4` (164.86), `deepseek_v4_c4_indexer` (10.30) and
   `deepseek_v4_c128` (5.32) — 1,449.0 GB total across 8 ranks, host RSS 50 →
   170 GB (demand-paged). `hybrid_pool_assembler.py:530` builds its
   `_IndexerRegion` and `:1343` registers its hit-policy pair. The rust
   pool-name concern is dead, empirically as well as by inspection.
   Note `hicache_ratio` scales `deepseek_v4_c4` and `deepseek_v4_c128` (80.05 →
   164.86, 2.58 → 5.32 from 1.5 to 3.0) but **not** the indexer pool
   (10.32 → 10.30).

### TRAP: the arm exits 1 *after* a fully successful benchmark

`ARM_EXIT=1` with `replay_rc=0`, `Benchmark Duration: 3629.92 sec` and every
aiperf export written. The failure is in post-processing:

```
KV_OFFLOAD_BACKEND is required when KV_OFFLOADING is enabled
```

`utils/agentic/aggregation/process_agentic_result.py:89` requires a **third**
variable nobody documents: `KV_OFFLOAD_BACKEND_METADATA`, a JSON object whose
`name` must equal `KV_OFFLOAD_BACKEND`. Without it the arm produces no result
JSON and looks like a 95-minute loss. **It is not** — re-run the aggregation
alone, no re-benchmark needed. Only `KV_OFFLOADING` is `required_env`; every
other field defaults to empty, exactly as in prior arms' JSONs:

```bash
cd /workspace/InferenceX && RESULT_DIR=/workspace/results/<arm> \
 AGENTIC_OUTPUT_DIR=/workspace/results/<arm> \
 RESULT_FILENAME="dsv4_fp4_sglang_tp8-pp1-dcp1-pcp1-ep1-dpatrue_disagg-false_spec-mtp_agentic_c192" \
 MODEL="deepseek-ai/DeepSeek-V4-Pro" MODEL_PREFIX=dsv4 TP=8 EP_SIZE=1 DP_ATTENTION=true \
 CONC=192 DURATION=3600 KV_OFFLOADING=dram KV_OFFLOAD_BACKEND=hicache \
 KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}' TOTAL_CPU_DRAM_GB=2048 \
 /workspace/agentx-runtime/venv/bin/python -m utils.agentic.aggregation.process_agentic_result
```

**Add `KV_OFFLOAD_BACKEND_METADATA` to every future hicache arm script** so the
launcher's own aggregation step succeeds in-line.

### What this changes about the investigation

hicache has been off in **every** arm on this node and the previous one, so
**every concurrency-curve number above c128 is an artefact of a missing CPU
tier**. In particular:

- The "knee at c160" in `DATA_AND_ANALYSIS_20260902.md` §2 does not exist with
  hicache on. c192 now beats it by 15 %.
- **c224 and c256 should be re-run with hicache**, and ATOM's c256 (44,722
  tok/s/chip) is now a plausible target rather than a distant one — we are at
  36,414 at c192 with the queue empty (`#queue-req` p90 = 7) and KV occupancy
  only 0.53, i.e. with headroom on both axes.
- This does **not** touch the ITL/interval track (§CONTINUE HERE, task 1). ITL
  p90 is flat here, and the interval trade-off is a separate axis.

## 14. c256: throughput keeps climbing, but ratio 3.0 becomes the constraint

`hicache-fp4-c256` vs `hicache-fp4-c192`, **CONC the only difference** (FP4 on,
TBO on, DPA, mem-frac 0.90, chunk 16,384/rank, interval 10, `HICACHE_RATIO=3.0`
in both). Both `GATES PASS`, 3629.6 / 3629.9 s, ISL matched 0.32 %. This arm
exited **`ARM_EXIT=0`** — the `KV_OFFLOAD_BACKEND_METADATA` fix from §13 works,
and the aggregation now succeeds in-line.

| metric | c256 | c192 | delta | **ATOM c256** |
|---|---|---|---|---|
| tok/s/GPU | 39,284.1 | 36,413.9 | +7.88 % | **44,722** |
| ITL p90 | 116.82 ms | 99.33 ms | +17.61 % | **97.6 ms** |
| TTFT avg | 15.72 s | 9.11 s | +72.50 % | **13.4 s** |
| TTFT p50 | 9.64 s | 6.33 s | +52.3 % | — |
| succ | 10,279 | 9,495 | +8.26 % | — |
| OSL | 806.3 | 852.0 | −5.37 % | — |
| overall cache hit | **0.9508** | **0.9509** | **flat** | 93.4–94.5 % |

**Honest bottom line vs ATOM: we lose at c256 on all three axes** — throughput
−12.2 %, ITL p90 +19.7 %, TTFT +17.3 %. That is a real gap, and it is the first
directly comparable point we have against ATOM's best published number. But the
evidence below says it is not the engine's ceiling.

### The cache did NOT collapse — the tiers shifted, and the CPU tier filled up

| | c192 | c256 |
|---|---|---|
| **overall** hit | 0.9509 | **0.9508** |
| GPU-tier hit | 0.9366 | **0.7655** |
| CPU-tier hit | 0.0144 | **0.1853** |
| GPU KV pool | 96 % | **100 %** |
| **CPU KV pool** | 79.4 % | **99.98 %** |
| dropped tokens | 0 | 0 |

Total cache effectiveness is identical to three decimal places. What changed is
that the device pool saturated and **a fifth of all prefix reuse demoted to the
CPU tier**, which then hit **99.98 % full**. Measurement-window miss rate is
**4.99 %**, statistically the same as c192's 5.01 %, so prefill demand per token
did not rise — consistent with the tier still absorbing the working set, but
with no capacity left.

**This is the previous node's c256 pattern reproduced** (gpu 66.3 % of a 94.4 %
total), and it retroactively explains why that run looked the way it did. It also
means `HICACHE_RATIO=3.0` **is** the binding constraint at c256, whereas at c192
it was not (52.8 % full). The obvious next arm is c256 at a higher ratio: host
DRAM is 3,023 GB total with 1,442.6 GB currently pinned, so ratio 5–6 fits.
Whether that closes the 12 % throughput gap to ATOM is the open question.

### TRAP FIXED: `arm_report.py` was printing the device tier as "cache"

`arm_report.py:79` read `gpu_cache_hit_rate`, so this arm first reported
`cache 0.765` and read as a catastrophic cache collapse. It was not: overall was
flat at 0.951. `summary_table.py` already had this right and its comment even
warns that using the device-tier number "is what produced the bogus 'cache
collapse' read". `arm_report.py` now uses `overall_cache_hit_rate` and prints the
tier split plus both pool utilisations, with an explicit
`<!> CPU TIER FULL -- raise HICACHE_RATIO` when the CPU pool exceeds 98 %.
Any conclusion drawn from an `arm_report` "cache" figure on a **hicache** arm
before this fix should be re-read.

### Free VRAM is tightening again

median **2.97 GB**, p10 1.71 GB, **min 1.20 GB**, 0 late Triton device loads —
versus c192's median 6.77 / min 2.71 GB. c256 spends the headroom that
`33979a814b` recovered. Still no abort, but the fp8-path fix (§3/§4b: mirror the
bounded buffer into `_aiter_fp8_paged_mqa_logits`, `indexer.py:160`) is now the
thing standing between us and an OOR at higher concurrency.

### Reading the ITL and TTFT rises

Both are what a fuller machine looks like, not a regression: `intvty p90` fell
15 %, OSL fell 5.4 %, and the device pool is pinned at 100 %. The interval track
(task 1, §CONTINUE HERE) is the lever for ITL and it is untouched here —
`--prefill-decode-interval` is still 10 in both arms, and at c128 moving it to 20
bought 25 % of ITL p90. That lever has not been tried in combination with
hicache at all.

### Node state after c256 — 116 GB held with no visible owner

After the arm's own cleanup: **0** matching processes, nothing holding `/dev/kfd`
(checked via `/proc/*/fd`), yet `rocm-smi` reports **116.2 GB still allocated,
~14.5 GB evenly across all 8 GPUs**, and it does not drain. Even distribution
across exactly 8 devices is the signature of a TP8 job.

**Do not assume this is our leftover.** This node is shared and §5's trap was
precisely that a shared-node giveaway got misread as "my own leftover
processes", which then destroyed another session's arm. A process in another
container's PID namespace is invisible here but its GPU allocation is not.
Before the next arm, re-check; if 116 GB is still held, mem-frac 0.90 will be
computing against less memory than every arm on the board used, which breaks the
memory matching the whole ATOM comparison rests on.

## 15. c128 + interval 20 + hicache: the pair still FAILS, and hicache is a no-op here

`hicache-fp4-int20-c128`. `GATES PASS`, 3629.8 s, `ARM_EXIT=0`, all three
settings verified in the arm's own output: `'prefill_decode_interval': 20` (the
launcher's hardcoded 10 appears first, ours second, argparse keeps the last),
`'enable_hierarchical_cache': True`, `'hicache_ratio': 3.0`, `fp4=1 tbo=1`.
KV pool 7,187,200 tokens — *larger* than `interval20-c128`'s 6,979,584, so the
112 GB of stale VRAM seen before launch did not shrink the pool (the new VRAM
gate held it off until the node read 2 GB).

**Scored as the pair, this is a FAIL, and the hypothesis behind the arm is
falsified:**

| | this arm | ATOM c128 | verdict |
|---|---|---|---|
| ITL p90 | **57.98 ms** | 61.5 ms | **PASS** (best on the board) |
| TTFT avg | **13.77 s** | 10.9 s | **FAIL** |

The bet in the script header was that hicache would pay off interval 20's TTFT
bill. It did not: TTFT went 13.22 → 13.77 s, i.e. *slightly worse* than
`interval20-c128` and nowhere near 10.9 s. Throughput +2.04 % is inside the
5.67 % replicate spread — **null**. ITL improved 1.36 %, also null. So against
`interval20-c128` this arm is, on every axis, the same arm.

### Why: at c128 there is nothing for a CPU tier to do

Same 3600 s measurement window for all three c128 arms:

| | dptbo-c128 (int 10) | interval20-c128 | **+hicache+FP4 (int 20)** |
|---|---|---|---|
| miss rate | 5.85 % | 5.89 % | **5.35 %** |
| `#queue-req` med / p90 | 1 / 5 | 1 / 9 | 1 / 9 |
| decode `#running-req` med / p90 | 16 / 22 | 14 / 20 | 14 / 20 |
| `full token usage` median | 0.31 | 0.28 | **0.27** |
| GPU KV pool | — | 65 % | **73 %** |
| CPU-tier hit | n/a | n/a | **0.5 pp** |
| CPU tier full | n/a | n/a | 38.3 % avg |

Two facts kill the hypothesis:

1. **hicache recovered 0.54 pp of miss rate at c128** (5.89 → 5.35) against
   **3.14 pp at c192** (8.15 → 5.01). The CPU-tier hit rate is **0.5 pp**, versus
   1.4 at c192 and 18.5 at c256. Device KV occupancy median is **0.27** and the
   pool is 73 % allocated — there is no eviction pressure, so there is nothing to
   demote and nothing to recover. c128's 5.85 % miss rate is already at the
   trace's irreducible cold-prefix floor.
2. **TTFT at interval 20 is not queue wait.** `#queue-req` p90 is 9 — trivial,
   and the same with and without hicache. Compare the broken c192 baseline, where
   TTFT of 59.71 s came with a queue p90 of 78. Interval 20's TTFT is the
   scheduler *deliberately deferring* prefill, which is the flag working as
   designed. No amount of cheaper prefill fixes a policy-imposed delay.

**Therefore hicache and `--prefill-decode-interval` do not interact — they are
orthogonal, and hicache is a no-op below the concurrency where the device pool
saturates.** Generalising: hicache pays off only when GPU KV pool occupancy is
high (96 % at c192, 100 % at c256, 73 % at c128). Do not add it to low-concurrency
arms expecting anything.

**Task 1's plan therefore stands unchanged: sweep `--prefill-decode-interval`
between 10 and 20 at c128.** The two endpoints are (int 10: ITL 78.39, TTFT
8.50) and (int 20: ITL 57.98–58.78, TTFT 13.22–13.77). ATOM sits at 61.5 / 10.9,
strictly inside that box, so a value between them can satisfy both — 15 first,
then 12. hicache can be left off for that sweep, which also keeps it a clean
single-variable series.

### URGENT operational: this arm nearly aborted

Free VRAM: median 3.23 GB, p10 **0.08 GB**, **min 0.01 GB**, with **10 late
Triton device loads**. That is precisely the §3 abort signature — a code-object
load onto a device with nothing left — and `fp4-dptbo-c64-reclaim0` died on
exactly that with 2 late loads at 0.00 GiB. This arm survived on luck.

It is also *worse* than both hicache arms (c192 min 2.71 GB, c256 min 1.20 GB)
despite c128 being the smallest workload, and this arm has the largest KV pool
of the three (7,187,200 tokens). The fp8-path fix (§3/§4b: mirror `b6e3728`'s
bounded buffer into `_aiter_fp8_paged_mqa_logits`, `indexer.py:160`) plus
pre-loading the Triton specialisations at init should now be done **before** the
next arm, not after. Every arm from here is a coin flip otherwise.

## 16. Post-swap rebuild (2026-09-03 07:15Z) — done, and the recipe had drifted

`IMAGE_SWAP_HANDOFF_20260903.md` assumed the new image's sglang base would be
`f8cbf000f4a5` (#37353), the same commit PR #37660 sits on. It is not: the
image ships **`2641e427be`**, which contains `f8cbf000f4a5` as an ancestor plus
a few weeks of `main`. #37660 is still unmerged. Three consequences.

**a) The cherry-pick conflicts, and the conflict is a real refactor.** `main`
has since rewritten `_forward_indexer_core` to dispatch top-k through a local
`run_topk_transform(rows, logits)` helper and to run the SM120 paged indexer in
row chunks. #37660 predates that and carries its own copy of the backend
dispatch plus a trailing `if not use_aiter_fp4:` block. Resolved by keeping the
new structure and adding the FP4 chunk loop on top of it: the FP4 branch is now
the first arm of the chain and calls `run_topk_transform` per chunk.
`run_topk_transform` gained an optional `topk_plan` argument because the cached
`indexer_metadata.topk_metadata` routes rows by their index in the **full**
range, so a chunk needs `plan_topk_v2(lens_rows)` built over its own rows.

Result: sglang **`1e41776161`** on `2641e427be`. `logits_rows_per_chunk` = 2
references in each of `indexer.py` and `fp4_indexer_hip.py`; budget env var
`SGLANG_DSV4_FP4_LOGITS_BUDGET_MB` defaults to 2048 MB; imports clean.

**One deviation from the PR: dropped `assert indexer_metadata.page_table is
core_metadata.page_table`.** `PagedIndexerMetadata.copy_()` lists `page_table`
in `copy_fields`, i.e. it copies *into* a persistent buffer on the CUDA-graph
path, so object identity between the two metadata views is not obviously
guaranteed on HIP graph replay. It is a non-functional invariant check whose
failure mode is a crash during an arm's warmup. Restore it if this resolution is
offered back upstream.

**b) The old `flashinfer_topk_transform` deviation is no longer needed.** It is
now defined — assigned in `C4IndexerBackendMixin.__init__` at `indexer.py:447`,
picking `topk_transform_flashinfer_fused` or `_unfused` off
`SGLANG_DSA_FUSE_TOPK`. Nothing to substitute.

**c) The image's aiter already carries three of our ten changes**, byte-for-byte
(only an explanatory comment differed): the `@cache` fix in
`pa_mqa_logits_fp4_prefill.py`, the `torch.Stream` handle fix in
`csrc/cpp_itfs/torch_utils.py`, and the staged `col_offset` fix in
`csrc/kernels/dsv4_rotate_quant.cu`. **`git apply` is all-or-nothing**, so the
saved patch failed as a whole with `does not match index` on those two tracked
files even though the other six applied cleanly. Excluding the two let the rest
through. aiter HEAD is unchanged at `c16d44b93`.

That AMD shipped those three fixes in the image is worth noting: they were ours,
and the FP4 path in the new image is already partly hardened.

### Where things live now

- `/shared_nfs/kk/` — new home for logs, traces and scratch (`logs/`,
  `traces/`, `tmp/`). xfs, 53 TB free, survives an image swap.
- `/shared_nfs/kk/tree-backup/20260903-postswap/` — the durable rebuild kit for
  the *current* tree, with `REBUILD.md`, the resolved `0001-*.patch` (15 KB now,
  down from 50 KB, because it is against the right base), `aiter-dirty.patch`
  and `aiter-untracked.tgz`.
- `/shared_nfs/kk/archive/pre-swap-20260902/` — the 09-02 dirty tarballs and the
  other sessions' 19 uncommitted sglang files, moved off `/workspace` rather
  than deleted because they are not ours.
- `/workspace/handoff-20260903-image-swap/` — **deleted**, consumed. Its seven
  script copies were byte-identical to `claude-skills/agentx/`.
- `/workspace/results/` — unchanged, still the arm artifacts.

### Next action

Node is idle and clean: no sglang/aiperf processes, all 8 GPUs at 0.28 GB.
Nothing has been run on the new image yet, so **task 2 (the bridge arm) is the
next thing**, unless the P0 fp8-path OOR fix (§3/§4b, code only, no node time)
is done first. Every number in `DATA_AND_ANALYSIS_20260902.md` §1 is pre-swap.

## 17. P0 DONE (2026-09-03): the fp8-path OOR fix, both halves, code only

Uncommitted in `/sgl-workspace/sglang` on top of `1e41776161`; the patch is
`/shared_nfs/kk/tree-backup/20260903-postswap/0002-fp8-path-oor-fix.patch`
(4 files, +162/−18). Nothing here cost node time.

### a) The fp8 logits rectangle is now bounded

`_aiter_fp8_paged_mqa_logits` (`dsv4/indexer.py`) allocated
`torch.empty(total_tokens, max_seq_len)` per layer — the 13.7 GB transient, and
the one PR #37660 does **not** cover because #37660 bounds the FP4 path only.
Two changes, mirroring what #37660 does for FP4:

1. The wrapper now serves that rectangle from the **same pooled block** the FP4
   path uses (`alloc_logits_buffer` in `fp4_indexer_hip.py`, a public entry onto
   the existing `_alloc_logits`). One block, not two: only one of the two
   indexer paths runs in a given server, so reserving a second would double the
   footprint the pooling exists to bound. The FP4-named
   `SGLANG_DSV4_FP4_LOGITS_BUDGET_MB` (default 2048) therefore now bounds both.
2. The **caller** row-chunks so the pooled path is actually taken. This had to
   go in the caller, not the wrapper: bounding inside the wrapper alone would
   still materialise the whole rectangle. The chunk loop sits in the `else`
   branch of `_forward_indexer_core`'s paged dispatch, gated on
   `fn is _aiter_fp8_paged_mqa_logits and not is_decode`.

Sizing: `logits_rows_per_chunk_for_width(max_c4_seq_len * next_n)`. **The
`next_n` factor matters for us** — the wrapper scores `[rows * next_n, context]`
and we run MTP, so ignoring it would overshoot the budget by `next_n` and fall
back to a plain `torch.empty`, i.e. silently no fix. At 2 GiB and ~210K tokens
of context that is ~2,568 rows per chunk, so a 16,384-token prefill chunk
becomes 7 passes of 2 GiB each instead of one 13.7 GB rectangle.

**Correctness detail that is easy to get wrong:** the top-k v2 plan is
row-indexed (`plan_topk_v2` returns `(bs+1, 2)` whose rows are
`{batch_id, seq_len}`), so the cached `indexer_metadata.topk_metadata` is
**invalid for a row subset**. Every chunked call now passes
`plan_topk_v2(c4_seq_lens[rows])` instead. `run_topk_transform` gained an
optional `topk_plan` argument for this. Note this also repairs the same latent
mismatch on upstream's SM120 chunk path, which was passing the full-range plan.

Not done, deliberately: the pooled block is **not** keyed by stream. TBO here
interleaves operations on one stream (`srt/batch_overlap/` creates no side
stream), and every caller scores and reduces back to back inside one operation,
so nothing overlaps on the block. If a future backend runs two indexer calls
concurrently on different streams, key `_LOGITS_POOL` by `(device, stream)`.

### b) The late Triton device loads are pre-loaded at init

The 10 late loads were **not** a mystery — `/workspace/results/*/server.log`
names them, and it is exactly two kernels, `_prefill_lengths_kernel` and
`_build_prefill_indices_kernel` (`unified_kv_kernels/runtime.py`), with one pair
recorded loading at **0.00 GiB free**. Prefill is not graph-captured, so they
first load on a real request.

The specialisation space is small and, importantly, **fixed per engine**: both
kernels take `win`, `Wc`, `HAS_COMPRESS` and `BLOCK` as `tl.constexpr`; `win`
comes from the pool and the page-index buffers are allocated once at full width
(`page_idx.shape[1]` never varies — the call sites only ever slice rows), so the
only free axis is `compress_ratio ∈ {0, 4, 128}`. At most three specialisations
per kernel.

`runtime.preload_prefill_index_kernels()` therefore does one `T=1`
`build_prefill_indices` call per available ratio, reusing the real `win` /
`ring_stride` / `swa_pages` / page-index buffers so the specialisation key
cannot drift from what serving will use. Safe against buffer contents: both
kernels only *store* computed index values and never dereference what they load.
It is called from `DeepseekV4HipRadixBackend.on_after_cuda_graph_warmup` via
`_preload_unified_kv_prefill_kernels`, once, and a failure warns rather than
taking the engine down.

**Verified on GPU, not just by inspection**, using the project's own watchdog:

- preload compiles and loads all three ratios in **0.8 s**; a second call is
  1 ms, i.e. resident.
- with `SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1`, preloading and then arming
  `mark_serving_started()` and running the real-shaped builders for all three
  ratios → **no late load**.
- **negative control:** the same armed run *without* the preload raises
  `Triton kernel '_prefill_lengths_kernel' device-loaded after serving started`.
  So the pass is not vacuous — worth checking, because `triton_load_watch.install()`
  silently no-ops on ROCm Triton builds that lack `knobs.runtime.kernel_load_start_hook`.
  On this image the hook is present (`_installed = True`).

### What the next arm should check

- `rg -c 'device-loaded after serving started' server.log` → **0**.
- **Do NOT expect free VRAM to rise on an FP4 arm.** Measured on
  `hicache-fp4-int20-c128-postswap`: min 0.01 / p10 0.05 GB, i.e. unchanged from
  the partner's 0.01 / 0.08. That is correct, not a failure. With FP4 on, half
  (a) never engages and the FP4 path was already bounded by #37660, so the
  near-zero headroom is the KV pool at mem-frac 0.90 plus the allocator's
  high-water mark — not the indexer rectangle. What (b) removes is the *failure
  mode* (a code-object load that needs memory outside the allocator), not the
  pressure. Free VRAM is only expected to move on a **non-FP4** arm, where (a)
  replaces a 13.7 GB transient.
- Consider `SGLANG_CRASH_ON_TRITON_LOAD_AFTER_READY=1` on one arm to assert
  coverage rather than merely observe it. Note the watchdog arms *before* the
  server warmup request, so any kernel that only the warmup reaches will trip
  it; use it on a diagnostic arm first, not a scoring one.
- The fp8 chunking only engages on the **fp8** path
  (`SGLANG_OPT_USE_AITER_INDEXER` with FP4 off), so an FP4 arm exercises (b)
  only. To exercise (a), run a non-FP4 arm — which is what the task-2 bridge
  arm (`interval20_c128.sh`, FP4 off) already is.

## 18. RESULT: the bridge arm. The image swap moved NOTHING, and P0(b) failed once

`hicache-fp4-int20-c128-postswap` vs its pre-swap partner `hicache-fp4-int20-c128`.
Same six settings (c128, DP attention, TBO, FP4 indexer, hicache ratio 3.0,
`--prefill-decode-interval 20`), different image. Both `GATES PASS`,
`errors=0`, `duration_s` 3628.6 / 3629.8, KV pool **7,187,200 on both** (so the
memory matching the ATOM comparison rests on is intact), ISL matched 0.61 %.

| metric | postswap | pre-swap partner | delta |
|---|---|---|---|
| tok/s/GPU | 30,352.2 | 29,725.8 | +2.11 % |
| ITL p90 | 57.75 ms | 57.98 ms | −0.40 % |
| TTFT avg | 12.16 s | 13.77 s | −11.64 % |
| overall cache | 0.949 | 0.948 | flat |
| GPU-tier / CPU-tier hit | 0.945 / 0.004 | 0.943 / 0.005 | flat |

**THE BOARD IS UNBLOCKED.** Throughput +2.11 % and ITL −0.40 % are both inside
the 5.67 % replicate spread, i.e. **null**. sglang, aiter, ROCm and torch all
moved and the two headline axes did not, so `DATA_AND_ANALYSIS_20260902.md` §1
may be quoted again — state the sglang SHA (`e485dc2436`) with any comparison.

**Do not over-read the TTFT −11.64 %.** It is larger than the throughput noise
band, but we have never measured a replicate spread for TTFT, and this is one
sample against one sample. Treat it as suggestive, not established. If it
matters, it needs a replicate.

Re-confirmed, unchanged by the swap:
- **The pair criterion still fails**, on TTFT only: ITL p90 57.75 ≤ 61.5 passes,
  TTFT 12.16 > 10.9 fails. The gap narrowed from 13.77 but is still ~12 % over.
- **hicache is still a no-op at c128** (§15): CPU-tier hit **0.004**, tier only
  38.5 % full, dropped tokens 0, GPU pool 75 %. Nothing to evict, nothing to
  recover. Host DRAM 49 → 172 GB.
- Windowed measurement miss rate 5.24 % (whole-log 7.54 % — still contaminated,
  still do not quote it).

### P0(b) FAILED THIS ARM: 74 late loads, worse than the partner's 10

Same two kernels, at 0.62–0.79 GiB free. **My §17 premise was wrong.** I claimed
`Wc` (the page-index row width) was fixed per engine because the call sites only
slice rows. It is not: `_pad_last_dim` aligns the width to
`PAGE_INDEX_ALIGNED_SIZE` (64) but the *base* width tracks the batch's context,
and this workload's ISL spans 66K–628K tokens. So `Wc`, a `tl.constexpr`, took
many values and my preload covered only the ones live at init.

The earlier session's note — "`BLOCK` has ≤ ~11 power-of-2 values ×
`HAS_COMPRESS` × `compress_ratio`" — was **right**, and I misread it as three
specialisations in total. There were two axes, one of them unbounded.

Fixed in `efaeb6f664`: `Wc` is now a **runtime** argument (it is only a loop
bound and an index stride), which closes the unbounded axis. What remains is
`win` (fixed) and `BLOCK` = `min(1024, next_power_of_2(max(win, Wc)))` — a
closed set of powers of two, `{128, 256, 512, 1024}` at `sliding_window` 128 —
which `preload_prefill_index_kernels` now *enumerates* via a new `block`
override. The old preload covered one quarter of that set, which is why it did
nothing.

Verified on GPU, three ways:
1. **Mechanism nailed before fixing:** preload `Wc=64`, arm the watchdog, then
   sweep `Wc` — 64 costs 0 loads, every other width costs exactly 2 (one per
   kernel). That is the axis, measured, not inferred.
2. **After the fix:** same sweep over `Wc` ∈ 64…16384 → **0 late loads**, with
   8 `(ratio, BLOCK)` pairs preloaded.
3. **Correctness:** written `indptr` and payloads match a pure-torch reference
   (extend rows, SWA ring slots, compressed tail), and all four `BLOCK` values
   produce identical written output. Note a full-tensor compare is meaningless
   here — the buffers are `torch.empty` at worst-case capacity and only each
   token's valid prefix is ever written; comparing the tails reports a false
   mismatch.

### Still open

- **P0(a), the bounded fp8 buffer, is still unexercised.** FP4 was on, so it
  never ran. It needs a non-FP4 arm.
- Free VRAM did **not** improve (min 0.01 / p10 0.04 GB vs the partner's
  0.01 / 0.08) and should not have — see the corrected expectation in §17.
- The next arm re-tests the P0 gate. Until one arm shows
  `device-loaded after serving started` = 0, treat every arm as a coin flip.

## 19. Shared-experts fusion at c128: perf null, memory the real result

`hicache-fp4-int20-c128-fuse-mf090` vs `hicache-fp4-int20-c128-postswap`.
Fusion enabled through the launcher's own `FUSE_SHARED_EXPERTS=1` (`:188`), not
`EXTRA_SERVER_ARGS` — `EXTRA_ARGS` expands at `:372`, *after*
`SHARED_EXPERTS_ARGS` at `:359`, so an override would put both
`--enforce-shared-experts-fusion` and `--disable-shared-experts-fusion` on the
command line and depend on argparse keeping the last. Confirmed on the arm:
`enforce_shared_experts_fusion: True`, enforce flag count 1, disable 0.

**It ran at `--mem-fraction-static` 0.90, first try, no OOR.** The 0.90 → 0.87 →
0.85 retry ladder never fired, so the arm is ATOM-matched and board-comparable.

### Fusion SAVES memory here — the opposite of what I predicted

I predicted fusion would replicate the shared experts, grow the weights and
shrink the KV pool. **Wrong.** The load-time message says what actually happens:

> Loading FP8 shared expert weights into FP4 fused MoE weights. The shared
> expert is quantized at load time and may differ slightly from a checkpoint
> that stores shared experts directly in FP4.

The shared experts are **requantised FP8 → FP4** as they are folded in, i.e.
halved, not replicated. Measured:

| | fusion | baseline | delta |
|---|---|---|---|
| weights (per rank) | 130.30 GB | 133.75 GB | **−3.45 GB** |
| `max_total_num_tokens` | 7,412,480 | 7,187,200 | **+3.13 %** |
| free VRAM med / p10 / min | 15.45 / 14.30 / 13.96 GB | 0.18 / 0.04 / 0.01 GB | **~+15 GB** |
| GPU KV pool utilisation | 65 % | 75 % | −10 pp |

**This is the operationally important result.** This exact arm is the one that
ran at min 0.01 GB free with 10 (then 74) late Triton loads and was called a
coin flip. It now has ~14 GB of headroom, and that came from fusion, not from
our P0 patches.

**Caveat on the mechanism:** −3.45 GB of weights does not explain ~15 GB of
serving-time headroom, and I have not isolated the rest. The likely candidate is
a smaller per-step transient — folding the shared-expert GEMM into the fused MoE
removes its separate workspace, lowering the allocator's high-water mark — but
that is a hypothesis, not a measurement. Do not quote a mechanism for the 15 GB.

**Numerics are NOT verified.** The load message says the requantised shared
expert "may differ slightly". This benchmark measures throughput and latency
only. Do not report fusion as free until an accuracy check exists.

### Throughput and ITL are null; the TTFT delta is NOT attributable to fusion

| metric | fusion | baseline | delta |
|---|---|---|---|
| tok/s/GPU | 30,373.4 | 30,352.2 | +0.07 % |
| ITL p90 | 57.52 ms | 57.75 ms | −0.40 % |
| TTFT avg | 11.32 s | 12.16 s | −6.95 % |
| ISL | 102,308 | 103,000 | −0.67 % |

Both `GATES PASS`, `errors=0`, `duration_s` 3628.9 / 3628.6. tok/s +0.07 % is as
null as a number gets.

**Three reasons not to credit fusion with the TTFT −6.95 %:**

1. **The arm is not single-variable, and that is my mistake.** `efaeb6f664` (the
   `Wc` runtime change) landed *between* the two arms — baseline ran
   `e485dc2436`, fusion ran `efaeb6f664`. The `Wc` fix removes mid-serving
   specialisation compiles and device loads, which is precisely a TTFT-shaped
   effect. Fusion and that change moved together.
2. **TTFT is drifting monotonically across three same-settings arms:**
   13.77 → 12.16 → 11.32 s. Three independent wins is the less likely reading;
   a systematic effect (warm Triton / flydsl JIT caches across runs) is more
   likely. An A/B/A replicate would settle it.
3. We have **never measured a replicate spread for TTFT**, so there is no noise
   band to score it against.

### The pair criterion: closest yet, still fails on TTFT

ITL p90 57.52 ≤ 61.5 **passes**; TTFT 11.32 > 10.9 **fails — by 3.9 %**.
The history at these settings: 26 % over (interval 20, pre-swap) → 11.6 %
(post-swap) → 3.9 % (fusion). Genuinely close, but per the caveats above most of
that closing may not be fusion.

hicache unchanged as a no-op at c128: CPU-tier hit 0.003, tier 37.4 % full.

### RETRACTED: the P0 gate did NOT pass on this arm

The script printed `P0 GATE PASS (first real arm to clear it)` on 0 late loads.
**That claim is void.** `triton_load_watch._on_kernel_load` returns early unless
free VRAM is below `SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB`, **default 1 GiB**.
This arm ran at 14–15 GB free, so a late load could not have been reported. The
count measures the headroom, not the fix.

Late *compiles* were 0 on both arms, but that signal is weak: Triton's on-disk
cache makes a repeat compile a `cache_hit`, which the listener skips.

So the evidence for the `Wc` fix is still only the standalone GPU sweep in §18
(Wc 64…16384 → 0 loads with the threshold forced to 1000 GB). **P0(b) remains
unproven on a real arm.** Fixed going forward: the arm scripts now export
`SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=1000`, which makes the count a gate
independent of headroom. Warn-only, not crash mode, so a stray load cannot kill
an arm. Copy that export into every new arm script.

### Still open

- **P0(a), the bounded fp8 buffer, is still unexercised** — FP4 was on again.
  Needs a non-FP4 arm.
- Whether the TTFT trend is real or cache drift. Cheapest test: re-run the
  post-swap baseline on the *current* tree (an A/B/A), which also gives the
  first TTFT replicate spread.
- An accuracy check for fusion's requantised shared experts.

### Node trap CORRECTED: the post-arm VRAM residue does drain, in ~13 min

After this arm the node showed **~119 GB held evenly across all 8 GPUs with zero
KFD processes** (`rocm-smi --showpids` → "No KFD PIDs currently running", no
zombies) — the exact signature §5/the handoff called out as possibly another
container's invisible job, and which an earlier session recorded as "~116 GB
held ... and it did not drain".

**It drains.** Measured: 118.4 → 115.9 GB over the first 2 min (~1.2 GB/min),
then 115.4 → 2.4 GB over the next 5 min (~22.5 GB/min). Fully clear about 13 min
after the arm ended. Reclamation starts slow and accelerates, which is why
sampling only the first minute or two looks like a stall.

Practical consequence: **after an arm, wait rather than escalate.** The VRAM
gate's 30-min window already covers it, so a back-to-back arm will pass the gate
on its own. Do not conclude "another container is holding VRAM" from a single
early sample, and still never kill blindly — but also do not abandon a slot over
this. Only treat it as foreign if it is still flat after ~20 min.

## 20. Fusion series: c192 lands, and the late-Triton-load problem is much bigger than P0(b) assumed

`hicache-fp4-int20-c192-fuse-mf090`. Same six settings as the c128 fusion arm
plus fusion, `CONC=192`. **mem-frac 0.90 first try, no OOR** — the ladder has
still never fired. `GATES PASS`, `errors=0`, `duration_s` 3629.2.

38,821.8 tok/s/GPU, ITL p90 75.57 ms, TTFT 14.11 s, cache 0.953, KV pool
7,380,992, CPU tier 51.7 % full, CPU-tier hit 0.021.

**Fusion's memory win holds at c192:** free VRAM min 10.07 / p10 10.73 /
med 14.09 GB, against non-fusion `hicache-fp4-c192`'s p10 4.74 GB.

### Two comparisons, neither of them a fusion result

**vs `hicache-fp4-c192` (+6.61 % tok/s, ISL matched 1.71 %): NOT fusion.**
ITL p90 −23.92 % (75.57 vs 99.33) with TTFT +54.87 % (14.11 vs 9.11) is the
`--prefill-decode-interval` 10→20 signature almost exactly as §1 measured it
(ITL −25 %, TTFT +55 %). That arm ran the launcher default interval 10, so the
delta is dominated by the interval, not fusion. §6 already says the two levers
are orthogonal. Context only.

**vs the c128 fusion arm: a CURVE POINT, not a delta.** My script labelled this
"CLEAN ... quote this one" — **wrong, and now fixed.** `arm_report.py` flags a
cross-concurrency headline as INVALID (`conc-and-trace-mix.md 19.6`) because the
trace mix and ISL move with concurrency; ISL moved 9.9 % here. Read the two
rows as two points on the fusion throughput curve (30,373 → 38,822 tok/s/GPU)
and do not quote the +27.82 %.

### The 516 late loads: first COMPLETE measurement, and P0(b) was mis-scoped

This is the first arm to export
`SGLANG_TRITON_LOAD_WARNING_THRESHOLD_GB=1000`, so every late load is reported
regardless of headroom. **516 is therefore NOT a regression from §18's 74** —
74 was measured through a 1 GiB gate and only ever saw low-memory windows. The
two numbers are not comparable.

What the full picture shows is that lazy Triton loading is **engine-wide**, not
confined to the two prefill index builders P0(b) targeted:

| kernel | loads |
|---|---|
| `alloc_extend_kernel` | 148 |
| `_prefill_cta_info_kernel` | 68 |
| `assign_req_to_token_pool` | 54 |
| `_prefill_lengths_kernel` + `_build_prefill_indices_kernel` (the two we fixed) | 64 |
| `_get_last_loc_safe_kernel` | 24 |
| `apply_rotary_emb_flat_kernel`, `_router_triton_kernel`, `_init_compressed_attn_metadata_kernel` | 16 each |
| `_fused_rms_fp8_group_quant_kernel` | 12 |
| plus `_gather_rows_kernel`, `_clear_unaccepted_c128_draft_states_kernel`, aiter gemm kernels, … | rest |

**Our two kernels are 64 of 516 — 12 %.** So even a perfect fix there could not
have cleared the gate. Worse, they still load 4× per rank per kernel despite the
`Wc` runtime change, so a residual specialisation axis remains. Most likely
Triton's automatic scalar specialisation (`equal_to_1` / `divisible_by_16`) now
applying to the runtime `Wc` — the ratio-0 path passes `Wc=1`, which is its own
variant — or a second backend instance (the MTP draft worker, whose
`unified_swa_ring_size` is `sliding_window + spec_extra`) preloading a different
key. Unverified.

**Reframe the whole item.** Per-kernel preloading does not converge; there are
a dozen-plus lazily-loaded kernels across sglang and aiter and the set will grow
with upstream. The two viable directions are (a) an engine-level warmup that
actually drives a prefill through the real code path at init, so every kernel on
it loads while headroom exists, or (b) accept the loads and guarantee headroom.

**And (b) is what already happened, by accident.** Every one of these 516 loads
had ≥10.07 GB free, because fusion freed ~10-14 GB. The original abort
(`fp4-dptbo-c64-reclaim0`, `HSA_STATUS_ERROR_OUT_OF_RESOURCES` at 0.00 GiB)
required a near-zero floor. So **the OOR risk at c128/c192 is currently
mitigated by fusion's headroom, not by our preload.** State it that way; do not
claim P0(b) fixed it.

P0(b)'s standalone result still stands on its own terms: for those two kernels,
sweeping `Wc` 64…16384 after a preload gives 0 loads (§18). It just is not the
thing that was making arms coin flips.

## 21. c256 fusion arm lands: ITL p90 now BEATS ATOM, throughput within 5 %

`hicache-fp4-int20-c256-fuse-mf090`. Same six settings as the c128/c192 fusion
arms, `CONC=256`, mem-frac 0.90 first try, no OOR. `GATES PASS`, `errors=0`,
`records_error_dropped=0`, `duration_s` 3629.5, ISL 112,351, KV pool 7,349,248.

42,462.0 tok/s/GPU, **ITL p90 91.11 ms**, TTFT avg 20.14 s (p50 10.92 s),
overall cache 0.951, GPU-tier 0.822, CPU-tier 12.9 pp.

### Against ATOM's best published point

| axis | this arm | ATOM c256 | delta |
|---|---|---|---|
| tok/s/chip | 42,462 | 44,722 | **−5.05 %** |
| **ITL p90** | **91.11 ms** | 97.6 ms | **−6.6 %, we WIN** |
| TTFT avg | 20.14 s | 13.4 s | **+50.3 %, we lose** |

**This is the first time we beat ATOM on ITL at a matched concurrency**, and
throughput is now inside 5 %, down from −12.2 % on the previous c256 arm. The
remaining deficit is concentrated entirely in TTFT.

Read that TTFT number with §10 in mind: the two engines sit at different points
on one prefill-versus-decode trade-off, and interval 20 deliberately buys ITL
with TTFT. We have now spent the whole TTFT budget and more. ATOM's TTFT is
famously flat across a 4× concurrency range (10.5 → 13.4 s); ours is not.

### The ITL/throughput movement is the interval, not fusion

vs `hicache-fp4-c256` (interval 10, no fusion, **old image**): tok/s +8.09 %,
ITL p90 −22.0 % (116.82 → 91.11), TTFT +28.1 % (15.72 → 20.14). That
ITL-down/TTFT-up shape is the `--prefill-decode-interval` 10→20 signature (§1
measured ITL −25 %, TTFT +55 % at c128), so as in §20 the delta is dominated by
the interval. It is also a cross-swap comparison with two variables. **Do not
attribute any of it to fusion.**

Fusion's own contribution remains what §19 found: memory, not throughput.
Weights 130.30 vs 133.75 GB/rank, and free VRAM p10 **10.38 GB** (min 10.13,
med 12.82) where the non-fusion c256 arm had p10 1.71 GB.

### Prefill demand is unchanged — the TTFT is not a cache problem

Measurement-window numbers for the two c256 arms:

| | interval 10 (old img) | **interval 20 + fusion** |
|---|---|---|
| miss rate | 4.99 % | **4.96 %** |
| decode `#running-req` med / p90 | 31 / 41 | 29 / 37 |
| `full token usage` median | 0.72 | 0.63 |
| `#queue-req` med / p90 | 3 / 12 | **3 / 17** |

The miss rate is identical to two decimal places, so prefill work per token did
not change. The queue p90 rose 12 → 17, which is small in absolute terms and is
the interval doing its job (deferring prefill), not the c192-style divergence
where p90 hit 78. So TTFT of 20.14 s at c256 is **policy-imposed deferral**, the
same mechanism §15 identified at c128 — consistent, and it means the lever for
TTFT here is the interval, not hicache and not the cache tier.

### The CPU tier is still 100 % full — P2 is still the open lever

`cpu_used_tokens` 22,047,744 of `cpu_total_tokens` 22,047,744, i.e. **exactly
saturated**, with GPU pool also at 100 %. Same as the non-fusion c256 arm. So
`HICACHE_RATIO=3.0` remains the binding constraint at c256 even after fusion
freed ~10 GB of device memory — the two are independent resources. **Raising the
ratio to 5–6 is still the untested lever at this concurrency**, and it is the
one with a measured reason to exist.

### Late loads: 503, same engine-wide pattern as c192

`alloc_extend_kernel` 152, `_prefill_cta_info_kernel` 65,
`assign_req_to_token_pool` 56, the two kernels P0(b) fixed 32 + 32,
`_get_last_loc_safe_kernel` 24, then the tail. Near-identical to c192's 516
breakdown, confirming §20's conclusion: lazy Triton loading is engine-wide and
P0(b) was mis-scoped. All 503 loads had ≥10.13 GB free, so again the OOR risk is
mitigated by **fusion's headroom, not by the preload**.
