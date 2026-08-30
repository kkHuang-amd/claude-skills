# ATOM → SGLang: DeepSeek-V4 optimization survey (2026-08-26)

## CONTINUE HERE

**Status:** Survey done. ATOM history mined (1326 commits, 2025-08-08 → 2026-08-26);
132 explicit DSV4 commits + 195 adjacent perf commits classified. SGLang main
(`a1f9508dd4`, 2026-08-25) compared feature-by-feature. No port started yet.
**Next:** pick from the P0 list below; each entry names the ATOM PR and the SGLang file to touch.
**Repro (regenerate the raw lists):**
```bash
git clone --filter=blob:none --no-checkout https://github.com/ROCm/ATOM /workspace/atom-survey
cd /workspace/atom-survey && git log --format='%ad|%h|%s' --date=short > /tmp/atom_log.txt
rg -i 'deepseek|dsv4|ds[-_]v4' /tmp/atom_log.txt          # 132 explicit
```
**Pass criteria for any port:** DSV4-Pro on 8xMI355X TP8+DPA, throughput delta measured
on a *fragmented* workload, gsm8k unchanged (`max_tokens>=8192`).

---

## 0. Method & the structural finding

`gh` API is blocked for the ROCm org (classic PAT → HTTP 403), and the local `gh` (2.4.0)
has no `search`. Git-over-HTTPS works, so the survey is built from **commit history of a
blobless clone**, not the PR API. ATOM squash-merges, so every subject carries its `(#NNNN)`.

**The finding that shapes everything below:** every candidate commit touches `atom/…`
— ATOM's own inference engine. The single exception is #876, which touches
`atom/plugin/sglang/attention_backend`. There is **no ATOM patch that can be
cherry-picked into SGLang**. Porting means reimplementing the *technique* against
SGLang's own abstractions. Cost per item is therefore "reimplement + A/B", not "apply patch".

## 1. Where ATOM's DSV4 effort actually went

132 explicit commits, concentrated 2026-05 → 2026-08 (109 of them). By volume the work is:
model bring-up and correctness (~55), CI/benchmark/recipe plumbing (~30),
vLLM/SGLang plugin glue (~20), and **genuine kernel/scheduler optimization (~27)**.
Only that last group is interesting for us.

## 2. Already in SGLang — do not port

| ATOM PR | Technique | SGLang equivalent |
|---|---|---|
| #1611, #930 | Prefill coalescer / PrefillDelayer | `srt/managers/prefill_delayer.py` — already upstreamed |
| #1709, #1971 | FP4 indexer + FP4 indexer cache | `kernels/ops/attention/dsv4/fp4_indexer.py` |
| #847, #1701, #1781 | MLA DCP | `kernels/ops/attention/dcp_kernels.py`, ~192 `dcp` refs |
| #766, #1275, #1142 | TBO prefill/decode + ubatch split | `srt/batch_overlap/two_batch_overlap.py` (1157 lines) |
| #1542 | Skip TBO below a min token count | `tbo_token_distribution_threshold` server arg |
| #1414 | DSpark speculative decoding | `srt/models/deepseek_v4_dspark.py` |
| #1541 | PIECEWISE cudagraph | SGLang piecewise capture |
| #1648 | Host-side TBO split scalars (cut D2H) | mostly done — only 3 sync sites remain (`two_batch_overlap.py:197,456,457`) and 456/457 already read a CPU tensor |

SGLang upstream is also shipping DSV4/AMD work weekly of its own (#36004 indexer top-k
1024-thread block, #32577 aiter fused mHC, #35919 FlashInfer MXFP4 MoE). Assume anything
generic lands upstream on its own; port only what is AMD/ATOM-specific.

## 3. Real gaps — ranked

### P0 — highest value/effort ratio

**#1917 `feat(moe): fuse shared experts into all-to-all MoE`** (+ #958 layerwise shared-expert fusion)
SGLang has `shared_experts_fusion`, but only in `srt/layers/moe/token_dispatcher/standard.py`
(5 refs; `num_fused_shared_experts`, `local_expert_mapping`). The **deepep / mori a2a
dispatchers have zero shared-expert references** — so on the EP/a2a path, which is exactly
the DSV4 8xMI355X config, the shared expert is still a separate GEMM outside the
dispatch/combine. Folding it into the a2a routing removes a full expert launch per layer.
*Touch:* `srt/layers/moe/token_dispatcher/{deepep,mori}*.py`, `moe_runner/aiter.py`.

**#1681 `perf(tbo): overlap pure-TP all_reduce on TBO + Delay TP`**
SGLang's `two_batch_overlap.py` has **no `all_reduce` reference at all** — the TBO schedule
does not currently hide the pure-TP all-reduce behind the other ubatch, and has no delayed-TP
concept. This is the classic TBO win and the one ATOM measured on DSV4 specifically.
*Touch:* `srt/batch_overlap/two_batch_overlap.py`, `srt/layers/attention/tbo_backend.py`.

**#1911 `perf(mla): take the decode's KV-split budget from the machine, not a hardcoded 16`**
SGLang derives per-token splits in `get_num_kv_splits_triton`, but the ceiling is the static
server arg `triton_attention_num_kv_splits` (`triton_backend.py:262`). ATOM's fix sizes the
budget from the actual CU count. Small, self-contained, low-risk — good first port.
*Touch:* `kernels/ops/attention/metadata.py`, `srt/layers/attention/triton_backend.py`.

### P1 — worth doing, more work

**#1895 `feat: support fp4 dispatch and fp8 combine`**
`moe_runner/aiter.py` has `fp4_dispatch`, but one branch literally reads
`dispatch: no kernel for the fp4`. ATOM has the working pair. Halves a2a payload on
dispatch and quarter-ish on combine for FP4 DSV4. Blocked on the aiter kernel being available.

**#1870 `feat(mla): enable persistent decode for all MLA models under DPA`**
SGLang's `aiter_backend.py` has exactly one `persistent` reference; the persistent-kernel
decode path is present for trtllm backends but not generalized to MLA-under-DPA on ROCm.
Relevant because DPA is the DSV4 serving config.

**#2001 `[DCP][Opt] Query replication, project-before-merge, all-to-all merge backend`**
SGLang DCP exists but `dcp_kernels.py` shows no query-replication / a2a-merge variant.
This is ATOM's newest DCP work (2026-08-24) — worth tracking, but it will likely churn.

**#1345 `Route prefix-cache-hit prefill through sink ASM MHA kernel`**
No sink/ASM-MHA routing found anywhere under `srt/layers/attention/`. On prefix-cache-heavy
traffic this changes which kernel serves the extend, not just its tuning.

### P2 — measure before believing

- **#1472 persistent ubatch worker threads.** SGLang TBO has *zero* threading — it is
  sequential-yield, not thread-per-ubatch. Adopting ATOM's model is an architectural change
  to TBO, not a tweak. Only justified if profiling shows launch-gap stalls between ubatches.
- **#1938 optimize MLA prefill chunk**, **#1464 adaptive BLOCK_K for `csa_translate_pack`**,
  **#1498 sparse-prefill triton kernel**, **#1270 einsum → Triton BMM**, **#704 fusions phase 1**.
  All live in `atom/model_ops/v4_kernels` and are tuned against ATOM's own tensor layouts.
  Port only if a profile points at that exact kernel.
- **#1627 "TBO: any rank 8k opens all"**, **#1715 eplb for pure prefill**.
  Policy heuristics; cheap to try, but they are workload-shaped and ATOM tuned them on
  their own traffic mix.

## 4. Not portable

Everything under vLLM plugin (#1060, #1166, #1372, #1451, #1454, #1595, #1664),
`sgl_atom` plugin glue (#1204, #1224, #1393, #1470), ATOM CI/recipes/dashboards, and
ATOM-internal model bring-up (#650, #705, #745, #746, #875). ~85 of the 132 explicit commits.

## 5. Suggested order

1. #1911 (kv-split budget) — smallest, proves the A/B harness.
2. #1681 (TBO all-reduce overlap) — biggest single TBO gap.
3. #1917 (shared expert into a2a) — biggest MoE gap on the actual serving config.
4. Then #1895 / #1870 depending on whether the aiter FP4 combine kernel is ready.

Related: `sglang-prefill-coalescer/` in this repo is the worked example of an ATOM→SGLang
port (#1611) — reuse its A/B methodology, especially the "gain only shows on fragmented
DP workloads" lesson.

---

## 6. Pending (open) PRs — added 2026-08-26

### How this list was obtained, and what it is worth

The PR API is 403 for this token, so open PRs were enumerated over plain git:

```bash
git ls-remote https://github.com/ROCm/ATOM 'refs/pull/*' > /tmp/atom_pull_refs.txt
# refs/pull/N/merge exists only while a PR is open and mergeable
# subtract PR numbers already squash-merged into main (from "(#NNNN)" in subjects)
```

1792 PRs total, 1168 merged into main, **207 with a live `merge` ref = open**.

Two caveats, both real:
- `refs/pull/N/merge` can linger for a closed-unmerged PR. Anything older than ~2026-06
  in this list should be assumed stale until confirmed. 81 of the 207 are from 2026-08 and
  are almost certainly genuinely open.
- The text below is each branch's **head-commit subject**, not the PR title. For a
  multi-commit branch (`AHEAD` > 1) it may describe only the last commit.

59 of the 207 touch DSV4-adjacent territory. The ones that matter:

### Directly relevant to our port list

| PR | Date | Ahead | Head subject | Why we care |
|---|---|---|---|---|
| #1974 | 08-24 | 9 | `perf(dsv4): derive FP4 scheduling from gfx950 topology` | Same family as merged #1911 (budget from machine, not constant) — and it is **gfx950**, our exact target. Watch this one. |
| #2011 | 08-24 | 1 | `feat: integrate the AITER MK1 persistent decoder` | The in-flight version of P1 item #1870 (persistent MLA decode). If it lands, port the merged form, not #1870. |
| #2018 | 08-24 | 1 | `feat(mori-v2): let the MoE dispatch wire be chosen, for the fp4 GEMM it feeds` | Directly on top of P1 #1895 (fp4 dispatch / fp8 combine) and mori — which is the a2a path SGLang also uses. |
| #1410 | 08-21 | 14 | `Switch pa decode from unified attention to pa_decode_sparse` | Sparse paged-decode kernel swap. Compare with `aiter-pa-decode-gluon-design` skill; long-running branch (14 commits). |
| #1765 | 08-26 | 20 | `prefill flydsl decode gluon` | Largest active branch, updated today. FlyDSL prefill + gluon decode. Scope unclear from the head subject alone. |
| #1666 | 07-22 | 1 | `feat(moe): FlyDSL MegaMoE fused EP-MoE integration` | Relates to `sglang-moega-moe` work already on disk (`feat/aiter-megamoe-v2`). |

### Smaller, cheap-to-steal ideas

- **#1952** (08-19) `perf(dsv4): let low-concurrency serving turn the side streams off` — multi-stream
  costs more than it saves at low concurrency. Cheap heuristic, likely applies to SGLang's DSV4 path too.
- **#1937** (08-18) `perf(mtp): defer draft proposal publication` — MTP/spec-decode latency.
- **#1926** (08-20) `refactor(moe): take the kernel mask from the placement`.
- **#2010** (08-24) `Apply the simulated-DP repeat to the MoE token capacity at dp_size == 1`.
- **#1946** (08-18) `feat(state-cache): add prefill-end checkpoint toggle`.

### Bearing on section 5's ordering

Nothing here displaces **#1911** as the first port — it is merged and self-contained.
But **#2011 and #2018 overlap our P1 items (#1870, #1895)**: both are one commit ahead of main
and dated 08-24, so they may merge within days. Do not start those two until they land or stall.
**#1974** is worth watching closely — gfx950-specific FP4 scheduling is our exact hardware.

**Refresh command:**
```bash
cd /workspace/atom-survey && git fetch -q origin
git ls-remote origin 'refs/pull/*/merge' | rg -o 'pull/(\d+)/' -r '$1' | sort > /tmp/now_open.txt
```

---

## 7. Deep-dive: the two branches worth tracking

Commit bodies on both branches are **empty** — subjects only — so everything below is read
off the code and the file layout, not off a PR description.

### PR #1765 — branch `shaoclee/ep_moe` (base `b8dcfe09`, 20 commits, 2026-08-01 → 08-26)

**The head subject is misleading.** `prefill flydsl decode gluon` is only the last commit;
the branch as a whole is an **EP-MoE / Triton-MoE overhaul**. Diffstat, `+1445/-237`:

```
atom/model_ops/fused_moe_triton.py                 +504
atom/model_ops/moe.py                              +417
atom/model_ops/fused_moe/modular_kernel.py         +232
atom/model_ops/fused_moe/mori_v2_prepare_finalize.py +118
atom/utils/envs.py                                  +31
tests/test_mxfp4_triton_moe_decode.py              +380  (new)
```

New env flags, which name the features better than the commits do:

| Flag | Meaning |
|---|---|
| `ATOM_USE_TRITON_MOE` / `_DECODE` / `_A4W4` | Triton MoE path, separately gated for decode and for a4w4 |
| `ATOM_USE_TRITON_GEMM` | Triton GEMM under the MoE |
| `ATOM_MOE_GU_ITLV` | gate/up interleaving in the MoE weight layout |
| `ATOM_EP_TRIM_PREFILL` | EP dispatch trim on the prefill path — same family as merged #1900 (`drop top-k from mori dispatch trim bound`) |

Development arc: `add triton support` → `add v1 v2 path` → `a4w4 support` +
`preshuffled a8w4 arg` → `block_m` tuning → `v3` → `refactor` → **`fuse ep`** → `prefill
flydsl decode gluon`. The `fuse ep` commit (`51cad582`) is the interesting one: 82 lines
across `modular_kernel.py`, `fused_moe_triton.py`, `moe.py`.

**Why it matters to us:** this is the *same territory* as P0 #1917 (shared expert into a2a)
and P1 #1895 (fp4 dispatch / fp8 combine), and it modifies `mori_v2_prepare_finalize.py` —
mori is the a2a backend SGLang uses too. It also overlaps the `sglang-moega-moe` work
already on disk (`feat/aiter-megamoe-v2`) and PR #1666 (FlyDSL MegaMoE).

**Do not port from this branch yet.** 20 commits with subjects `tmp`, `merge`, `v3`,
`refactor` — it is in-development, not review-ready, and rebases on main roughly weekly.
Track it; if it lands, it likely supersedes both #1917 and #1895 as the thing to port.

### PR #1974 — branch `dsv4-fp8-indexer-paged-prefill` (base `084385d6`, 9 commits, 08-20 → 08-24)

`+1473/-52` across 9 files, but **only 4 are source** — 1007 of those lines are tests
(including two GPU tests). Two independent changes rode in together:

**(a) Paged FP8 indexer prefill** (`ec2dce0f`, made default in `c54d89e4`)
Adds `paged_prefill_block_tables_per_token` and `paged_prefill_max_seq_len` metadata in
`deepseek_v4_attn.py`, plus `_score_topk_prefill_paged` and `_prefill_chunk_rows` in
`deepseek_v4.py`. The indexer's prefill scoring reads a **paged** FP8 KV layout instead of a
contiguous one. SGLang's counterpart lives in `srt/layers/attention/nsa/nsa_indexer.py` and
`srt/layers/attention/dsv4/indexer.py`.

**(b) FP4 MQA prefill launch-geometry model** (`b4cc7833` → `a20410a1` → `d3ae72c3`)
A new 251-line `atom/model_ops/v4_kernels/fp4_mqa_schedule.py`. This is **an analytical model,
not an autotuner** — the header comments carry the reasoning:

- A wave in the FlyDSL kernel always owns four 16-token MFMA N tiles, so *coarse*
  (256 K rows / CTA / 4 waves) and *fine* (64 K rows / CTA / 1 wave) do identical work per
  wave — only the hardware mapping differs.
- gfx950 exposes **256 CUs, ≤32 resident waves per CU**; for ragged causal work the target is
  ~1.5 full resident-wave sets, expressed as a **wave-task budget** (`12,288` tasks), not as
  a CTA count.
- `rocminfo` reports a **4 MiB L2 slice** on gfx950; the model reserves a quarter of it for Q.
- API: `FP4MQAPrefillConfig` (NamedTuple), `fp4_mqa_prefill_wave_tasks_per_row()`,
  `fp4_mqa_prefill_parallel_unit_num()` — the latter converts the wave-task budget into a
  persistent-grid CTA count, rounded to a four-wave-aligned quantum.
- Stated policy: express the schedule in **wave tasks per row, not CTAs per row**.

**Why it matters to us:** this is merged #1911 (`take the decode's KV-split budget from the
machine, not a hardcoded 16`) generalized into a reusable model — and it is tuned for
**gfx950, our exact hardware**. SGLang's equivalent ceiling is still the static server arg
`triton_attention_num_kv_splits` (`triton_backend.py:262`).

**The transferable artifact is the model, not the kernel.** `fp4_mqa_schedule.py` is pure
Python arithmetic over `(num_rows, CU count, resident waves, L2 slice)` with no ATOM
dependencies, so it is the one file in this whole survey that could be lifted almost
verbatim. The two GPU tests (`test_fp4_mqa_prefill_wave_granularity_gpu.py`,
`test_v4_indexer_paged_prefill_gpu.py`) are worth reading before writing our own validation.

### Effect on the ordering in section 5

Unchanged for #1911 — still the first port, and #1974 strengthens the case by showing where
that idea goes next. Revised guidance for the MoE items: **#1917 and #1895 should wait on
#1765** rather than only on #2018, because #1765 is the larger rework of the same code.
