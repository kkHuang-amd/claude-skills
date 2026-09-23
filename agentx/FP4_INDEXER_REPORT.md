# AMD FP4 indexer (sglang#37353) on DeepSeek-V4-Pro / MI355X — integration and benchmark report

**Date:** 2026-09-02 · **Node:** 8 × gfx950 (MI355X), TP8 · **Model:**
DeepSeek-V4-Pro FP4 · **Workload:** InferenceX AgentX agentic trace replay
(aiperf, `dsv4` corpus) · **Engine:** SGLang `52e1c24744` + PR #37353 applied to
the working tree

Raw artifacts, scripts and the running log of decisions are in
`/workspace/claude-skills/agentx/AGENTX_20260901.md`; every number below is
reproducible from `/workspace/results/<arm>/`.

---

## 1. Bottom line

1. **The integration works.** #37353 runs on this node with DP attention **and
   TBO**, contradicting the PR's first commit message. It needed two working-tree
   patches, not the four the PR implies, and **no rebuild**.
2. **The measured gain at c128 DP+TBO is +2.57 %**, on a well-matched pair
   (ISL within 0.43 %, cache hit flat). That is **not a resolved result**: it sits
   inside this node's 5.67 % replicate spread, and below the PR's own +5.8 % at
   conc 48. A replicate is required before quoting it anywhere.
3. **An unresolved GPU out-of-resources abort kills roughly half of all hour-long
   arms at ~45 minutes**, regardless of the settings we varied. This is the
   blocking issue, and it is not specific to FP4 — it also affects the workload
   shape generally. Diagnosis is incomplete because no arm has VRAM-over-time
   data.

---

## 2. What had to be integrated

The survey expected four pieces. Two were already satisfied:

| # | piece | actual state |
|---|---|---|
| 1 | sglang #37353 | **applied** to the working tree (python only) |
| 2 | sglang #36581 (gfx950 enablement) | **already in** `52e1c24744` by content — the HIP radix backend, `dsv4/indexer.py`, `is_gfx95_supported()` and the CLI flag were all present |
| 3 | aiter #5034 (`8578af1`) | **already applied and already compiled** — `dsv4_rotate_quant.cu` carried a byte-identical hunk and `module_dsv4_rotate_quant.so` was newer than the source |
| 4 | aiter #5126 (`cc1b717b3`) | **applied** — 3 lines of Python (`lru_cache(maxsize=32)` → `cache`) |

Plus one local change: the InferenceX launcher
`dsv4_fp4_mi355x_sglang_b200align_mtp.sh` had no pass-through for one-off server
flags, so `--enable-deepseek-v4-fp4-indexer` could not be set. An
`EXTRA_SERVER_ARGS` hook was added.

**The PR's `docker/rocm.Dockerfile` route is inert here.** We run an editable
aiter from `/sgl-workspace/aiter`, so the image-build `sed`s never execute; those
two `sed`s *are* aiter #5126, which is why it had to be applied by hand.

**Deliberately skipped:** the PR's 3-line `rust/sglang-radix-tree` change (a
`DeepseekV4C4IndexerScale` pool-name variant). It is only reached through the
hicache / unified-radix path and taking it would require a rust rebuild, so
**hicache was left off in every arm**.

### Conflicts: exactly one line, and it is not FP4

The PR's own change set is 21 files (+1727 / −76) against main `ed122ea984`,
53 commits ahead of our HEAD. `git apply --3way` of the python-only diff applies
14 of 15 files cleanly. The single conflict is at
`deepseek_v4_backend_hip_radix.py:492`: main `5edcd0a445` (#33237, flashinfer
fused top-k) added a `use_topk_v2=False` argument we do not have, and the PR's
last commit rewrites that same line. **Resolution: drop that hunk** — our
`PagedIndexerMetadata` has no such parameter and the backend has no
`dsa_topk_backend`.

---

## 3. Does it work with TBO? Yes.

The PR's first commit (`a627e90`) says "MTP, TBO, HiCache and PD disaggregation
are not supported yet". That note is **stale**. In the final tree TBO appears in
exactly one place, `_refresh_fp4_prefill_workspace()`, added *later* by the MTP
commit `37b73f5d04`:

```python
if (
    get_parallel().attn_cp_size != 1
    or getattr(forward_batch, "tbo_children", None)
    or getattr(forward_batch, "tbo_parent_token_range", None) is not None
):
    return
```

This is a skip, not a disable. `prepare_fp4_prefill_workspace()` only
pre-computes `cta_info` outside graph capture; its `row_to_batch` /
`local_starts` are the same `arange` / `zeros` the inline path builds, and
`aiter_fp4_paged_mqa_logits()` handles `workspace is None` by letting the kernel
compute its own schedule. Under TBO the FP4 prefill kernel still runs — it loses
only the pinned schedule, which exists for CUDA-graph safety, and DSV4 TBO is
eager-prefill-only anyway (`tbo_supports_cuda_graph = False`). Decode is
unaffected because DSV4 decode / target-verify graphs are non-TBO.

Those two lines are the only `tbo` / `two_batch_overlap` additions in the whole
PR. All FP4 state lives on the per-backend `DSV4Metadata`, so TBO's per-ubatch
child backends each own theirs and cannot race.

**Caveat that follows from this:** TBO forfeits the prefill-workspace
optimisation, so a TBO measurement can legitimately read lower than the PR's
non-TBO +5.8 %.

### aiter is far behind main, but not on this surface

aiter HEAD `c16d44b93` is 163 commits behind, which sounds fatal and is not:
`git log c16d44b93..origin/main -- aiter/ops/flydsl/kernels/mqa_logits/` is
**empty**. Every symbol the adapter imports was verified **by import, not grep** —
`aiter.rope_rotate_activation`,
`aiter.rmsnorm_rope_rotate_activation_fp4quant_kvcache`, `aiter.dtypes.fp4x2`,
`flydsl_pa_mqa_logits_fp4{,_prefill}`, `compute_varctx_schedule`,
`compute_prefill_schedule`, `CTA_INFO_WIDTH` (= 6) — and the signatures match the
call sites positionally and by keyword. **No aiter upgrade, no rebuild.**

---

## 4. Results

All arms: TP8, DP8 attention, EAGLE MTP, `CHUNK_PER_RANK=16384`,
`DURATION=3600`, hicache off. `tok/s/chip` is per-GPU output throughput.

| mode | conc | mem-frac | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | cache hit | ISL mean |
|---|---|---|---|---|---|---|---|---|
| DPA+TBO | 64 | 0.85 | 20,730 | 21.4 | 46.8 ms | 4.62 s | 95.5 % | 116,596 |
| DPA+EP8+MoRI mxfp8 | 64 | 0.85 | 8,710 | 7.8 | 128.9 ms | 12.91 s | 95.4 % | 114,463 |
| DPA+EP8+MoRI mxfp8 +recvbound | 64 | 0.85 | 16,395 | 15.3 | 65.4 ms | 6.79 s | 95.3 % | 118,030 |
| **DPA+TBO** | **128** | **0.90** | **27,895** | 12.8 | 78.4 ms | 8.50 s | 94.2 % | 99,954 |
| **DPA+TBO +FP4 indexer** | **128** | **0.90** | **28,612** | 13.0 | 76.7 ms | 8.33 s | 94.3 % | 100,389 |
| DPA+TBO +FP4 indexer | 64 | 0.90 | 20,898 | 21.9 | 45.6 ms | 4.44 s | 95.5 % | 117,114 |

### 4.1 The one valid comparison: c128 DP+TBO, FP4 vs baseline

Both arms `GATES PASS` — `errors=0`, `records_error_dropped=0`, duration
3628.1 s / 3629.4 s, aiperf coverage ok.

```
baseline  dptbo-c128       27,894.8 tok/s/GPU   ISL  99,954   cache 0.942
FP4       fp4-dptbo-c128   28,611.7 tok/s/GPU   ISL 100,389   cache 0.943
                                     +2.57 %
```

Pair quality is good: **ISL differs by 0.43 %** (the >3 % "different workload"
flag is nowhere near) and cache hit is flat. Every secondary metric agrees in
direction and magnitude — TTFT −2.1 %, ITL p90 −2.2 %, interactivity p90 +2.2 %.

**Why this is not yet an answer.** +2.57 % is inside the 5.67 % replicate spread
measured on the previous node, so a single unreplicated pair cannot distinguish
it from zero. This was stated before the arms ran; it is the plan, not a
surprise. Two candidate explanations for undershooting the PR's +5.8 % are worth
separating: c128 vs the PR's conc 48, and TBO forfeiting the prefill workspace
(§3). A non-TBO c128 pair would split them.

### 4.2 The c64 arm has no matched partner

`fp4-dptbo-c64-reclaim1` is certified (`errors=0`, 3629.1 s, 20,897.8 tok/s/GPU,
ISL 117,114, cache 0.955). It must **not** be read against the `dptbo-c64` row:
`arm_report.py` returns `VERDICT: INSIDE NOISE — report as null` at +0.81 %, and
the two arms differ in **both** FP4 and mem-frac (0.90 vs 0.85). Its intended
partner aborted (§5).

---

## 5. The blocking problem: intermittent GPU out-of-resources abort

Roughly half of the hour-long arms die mid-measurement with the same signature:

```
:0:rocdevice.cpp:3582: Callback: Queue ... Aborting with error :
HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary
resources. ... Available Free mem : 0 MB
Fatal Python error: Aborted
```

One DP rank aborts; the rest then die of gloo `Connection closed by peer`, which
is secondary. **The server keeps answering `GET /metrics` with HTTP 200 after it
is dead**, so aiperf waits forever and the arm must be killed by hand.

### What is ruled out

- **Not the KV pool.** `full token usage` was 0.21–0.39 at abort time.
- **Not a torch-level OOM.** No `OutOfMemoryError` anywhere in any log. The
  caching allocator succeeded; the failure is a non-torch allocation finding
  0 MB free.
- **Not a broken workload.** 3,623 decode batches, accept len ≈ 2.4, no
  `--chat-template` — unlike the 2026-08-28 abort recorded in the launcher's own
  comment, which was an artefact of collapsed generation.
- **Not a corrupted tree.** The last aborting arm carried an md5 snapshot of the
  four key source files; all four still matched afterwards.
- **Not `HSA_NO_SCRATCH_RECLAIM`.** See below.

### `HSA_NO_SCRATCH_RECLAIM` does not explain it

An earlier conclusion in the working doc claimed `=0` was the fix, based on one
c128 before/after pair. **That was wrong**, and the c64 A/B refutes it. All four
arms below are mem-frac 0.90, FP4 on, DP+TBO, chunk 16384:

| conc | reclaim | outcome |
|---|---|---|
| 128 | 1 | abort at **2680 s** of measurement |
| 128 | 0 | full 3600 s |
| 64 | 1 | full 3600 s |
| 64 | 0 | abort at **2731 s** of measurement |

The pattern is **crossed** — each setting has one survival and one abort — so the
original pair confounded the flag with run-to-run luck. This is the same
single-pair error the report warns about for the +5.8 % headline.

### The actual signal: timing

Both aborts land at **2680 s and 2731 s** of the measurement window — within 2 %
of each other, ~45 minutes in, at two different concurrencies. That points at
something accumulating with runtime until free VRAM reaches zero, with survival
to 3600 s depending on where the longest sessions happen to fall.

Context that makes this plausible: the trace's sessions accumulate context, with
**ISL p90 = 208,893 and p99 = 628,643 tokens**, and the indexer allocates a
transient logits buffer of `total_tokens × max_seq_len × 4 B` per call — present
in **both** the FP4 and the fp8 path. At `CHUNK_PER_RANK=16384` that is 13.7 GB
at ISL p90 against ~41 GB free per rank after startup. This is a hypothesis, not
a conclusion.

### What is needed to close it

**No arm has VRAM-over-time data.** `gpu_metrics.csv` is 41 MB of clocks, power
and activity with **no memory-usage column**, so none of the six arms can say
whether free VRAM decays monotonically or dies to a single spike. The next arm
should carry a sampler (`rocm-smi --showmeminfo vram` every 30 s to a file). It
costs nothing and is what decides between three very different actions: lower
mem-frac, lower `CHUNK_PER_RANK`, or report a genuine leak upstream.

---

## 6. What is not resolved

1. **Is +2.57 % real?** Needs a replicate of the c128 pair. Until then the
   honest statement is "small positive effect, inside noise".
2. **Why below the PR's +5.8 %?** Concurrency (128 vs 48) and TBO's forfeited
   prefill workspace are both untested as explanations.
3. **The OOR abort.** Root cause unknown; ~50 % of hour-long arms are lost to it.
4. **FP4 effect at c64.** Never measured with a matched partner.

## 7. Reproduction

```bash
# arms (each ~1.5 h wall clock incl. warmup)
bash /workspace/claude-skills/agentx/dptbo_c128.sh          # c128 baseline
bash /workspace/claude-skills/agentx/fp4_dptbo_c128.sh      # c128 + FP4

# report, with gates printed before the headline
python3 /workspace/claude-skills/agentx/arm_report.py fp4-dptbo-c128 dptbo-c128
python3 /workspace/claude-skills/agentx/summary_table.py
```

Every arm script carries a three-process kill preamble
(`sglang.launch_server`, `sglang::tokenizer_worker:*`, `sglang::router`) because
**the launcher never kills its own server**, and the workers hold ports without
holding VRAM — so a clean `rocm-smi` is not evidence the node is free.

Two operational traps worth inheriting: never `pkill -f` with a pattern that
matches the agent's own command line (`pkill -f aiperf` killed a cleanup shell
here), and **never touch `/sgl-workspace/{sglang,aiter,mori}` while an arm is
running** — a branch switch mid-run destroyed one arm, because the FP4 adapter's
aiter imports are lazy and read the tree throughout the run.
