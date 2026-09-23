# MI355X node — CONTINUE HERE

Counterpart to `agentx/b200/CONTINUE_HERE.md`. The two nodes share nothing but
this git repo; see `agentx/exchange/README.md`.

## CONTINUE HERE (2026-09-18 22:1x UTC+8) — MLA work moves to a FlyDSL kernel. Read `FLYDSL_MLA_HANDOFF.md`, not this block

**Node is idle:** 0 `sglang::`, no drivers, VRAM at the 0.28 GB baseline.

**The MLA line's state after a full day of arms and prototypes:** the Triton
kernel is at its ceiling. Three ways of growing the MFMA `M` dimension all
regressed (registers +28 %, forcing LDS 7-11x, D-tiling +148 %), and the root
cause is measured: `Accum_VGPR_Count = 0` — the fp32 accumulator never reaches
the AGPR half of gfx950's register file, and Triton on ROCm will not put it
there. **The next step is a hand-written FlyDSL kernel**, and everything needed
to start writing it — target numbers, the four required properties, shapes,
metadata plumbing, corpus entry points, gates, and the closed avenues — is in:

```
/workspace/claude-skills/agentx/mi355x/FLYDSL_MLA_HANDOFF.md
```

**Arms run today, all recorded below:** `block_k=16` (+0.36 ms, falsified —
production was already at 16 on the bf16 path), segment plan (−0.02 ms,
falsified — production straggler 0.373 against the 0.55-0.6 crossover), plus
the first GSM8K numbers this line has produced (0.937 baseline / 0.939 with the
plan). Two Triton prototypes and a full straggler sweep are in `/shared_nfs/kk/`.

**Still open, unrelated to MLA:** PR
[#39968](https://github.com/sgl-project/sglang/pull/39968) has 13 failing checks;
Phase 2's launch-elimination items (~1.5 ms ceiling); `fillBufferAligned`'s call
site (183 calls/step, 0.78 ms, needs a `--disable-cuda-graph` capture);
`prepare`'s own parallelism (≤4.6 ms of real work); and the 226-idle-CU overlap
question, which TBO cannot answer until `split_spec_info` learns
`DFlashVerifyInput`.

---

## Previous block (2026-09-18 10:3x) — the attention win does NOT help the first MoE kernel. `megamoe_prepare_compact` is a barrier and it ATE BACK ~2.7 ms/step

`/shared_nfs/kk/pr35619/trace_hcasplit4_steady_s24`, `num_steps=24`, **6/8 ranks**
(TP-3 and TP-6 never flushed — see the cascade note below). `TARGET_VERIFYx46`
at bs=11-17, i.e. inside the reference's bs 9-20, so this capture IS comparable
— the first one of this line that is.

### The matched cell: rank 1 at bs=14, in both traces

| per call (µs) | reference | split-K=4 | per step (ms, ×61 calls) |
|---|---|---|---|
| own `compute` | 38.91 | 27.39 | — |
| `mla_fused` | 330.7 | 138.3 (30 calls) | — |
| `mla_split` | — | 147.0 (31 calls) | — |
| **MLA total** | — | — | **20.17 → 8.71 (−11.46)** |
| **`prepare`** | **102.9** | **146.8** | **6.28 → 8.95 (+2.67)** |
| `ep_combine` | 83.8 | 76.3 | 5.11 → 4.65 (−0.46) |
| `stage1` | 238.9 | 219.4 | 14.57 → 13.38 (−1.19) |

**On this cell the barrier absorbed 2.67 ms of the 11.46 ms MLA saving.** Rank 1
was the *busiest* rank in the reference (compute 38.91, the top row); after
split-K it is mid-pack (27.39 of a 17.11-33.44 range), so it now arrives early
and waits longer.

**But do not generalise that to the whole trace — in aggregate the wait FELL.**
n-weighted mean `prepare` across all cells: **217.5 → 162.1 µs/call** (−25 %),
and on TP-7 specifically 16.24 → 10.48 ms/step. So the barrier did shrink; it
just shrank far less than the work behind it (MLA −57 % on the matched cell),
which is why MoE dispatch's share of the step barely moved (47.5 % → 45.0 %).

**The confound, stated plainly:** the two traces do not share a batch-size
distribution. The reference cells span bs 9-20; this one spans bs 4-17, with
three idle-rank cells at bs 4-6 that have no counterpart. `prepare` is a wait,
so its value is set by how far ahead of the slowest rank a rank is — that makes
it a function of the whole rank *distribution*, not of one rank's config. Any
single-cell or cross-trace mean of `prepare` therefore mixes the split-K effect
with a different DP load spread. What is NOT confounded: the sign of the
correlation (barrier in both traces), and the per-rank MLA spread below.

**Why the fast ranks still wait: MLA is still wildly unequal across ranks.**
`mla_split` runs 42.9 µs/call at bs=4 and 277.8 µs at bs=16 — a **6.5x spread
inside one trace**. split-K made every rank's MLA cheaper by a similar
*fraction*, so the *ratio* between ranks is untouched, and the absolute gap
that the barrier has to absorb narrows only in proportion. The underlying
driver is DP batch dispersion (bs 4 to 17 at the same instant), which split-K
does not address at all.

The barrier verdict is stronger than before, not weaker: `prepare` r = **−0.980**
(n=12 cells, spread 74.6-306.0 µs) against the reference's −0.942. `ep_combine`
is also still a wait (−0.597). `stage1` remains real work (+0.713), and
`mla_split` is real work with r = **+0.976** — the split path tracks own work
exactly as intended.

`calls/step: prepare=61 stage1=61 stage2=61 ep_combine=61 mla_split=31
mla_fused=30` — clean confirmation that the layer-aware gate hit the ~31 HCA
layers and left the other 30 on the fused path. The override is not leaking.

### The bucket table, split-K column added

Extends `exchange/FINDINGS.md:1126` (B200 bs=9 serial vs MI355X bs=10) with the
split-K=4 capture. TP-7 in every column, `TARGET_VERIFY full` step, ms/step.
**bs is 9 / 10 / 12 across the three columns** — this is a shape comparison, not
a matched-bs delta, so read the buckets and not the total.

| bucket | B200 serial (bs=9) | MI355X ref (bs=10) | MI355X split-K=4 (bs=12) | Δ vs ref | still vs B200 |
|---|---:|---:|---:|---:|---:|
| **moe** | 13.42 | **35.75** | **29.46** | **−6.29** | +16.04 |
| **attn** | 8.20 | **15.67** | **13.26** | **−2.41** | +5.06 |
| comm | 0.00 | 6.17 | 4.82 | −1.35 | +4.82 |
| copy | 0.13 | 3.98 | 4.10 | +0.12 | +3.97 |
| quant | 1.14 | 2.21 | 2.25 | +0.04 | +1.11 |
| other+norm_rope+sample | 1.47 | 2.11 | 2.18 | +0.07 | +0.71 |
| **gemm** | **9.81** | **9.36** | **9.43** | +0.07 | −0.38 |
| **total** | **34.17** | **75.24** | **65.50** | **−9.74** | **+31.33 (1.92x)** |

Of the `moe` −6.29, **5.76 is `megamoe_prepare_compact` alone** (16.24 → 10.48
ms/step on this rank) — i.e. the MoE bucket improved mostly because the barrier
waited less, not because MoE compute got faster. `stage1` is 13.50 ms/step here
against the reference's 14.10, roughly flat as expected.

`attn` −2.41 at a LARGER bs is the split-K win showing up in the bucket view;
`_paged_decode_split_kernel` is 7.97 ms/step of it.

The platform gap closes 2.20x → 1.92x, and `moe` is still half of what remains.

### `copy` and `comm` attributed AT DECODE (was only ever done for EXTEND)

`analysis/kernel_dump.py <dir> 12` — the bs is now an argument, because bs 10
was hard-coded and no capture is guaranteed to contain it. TP-7, bs=12.

**`comm` 4.82 ms/step is ONE kernel: `ep_combine_intranode_0`, 61 calls at
79.1 µs.** It is not a slow collective and it is not 17 things — and
`prepare_wait.py` classes it as a WAIT (r = −0.597). So the +4.82 vs B200's 0.00
is exposed idle, **not an independent target**; it belongs to the overlap story
above, and cutting it directly buys nothing on the critical path.

**`copy` 4.10 ms/step is 17 small kernels, none over 0.8 ms**, all 4-8 µs/call —
this bucket is call-count, not bandwidth:

| kernel | ms/step | calls/step | note |
|---|---:|---:|---|
| `__amd_rocclr_fillBufferAligned` | 0.780 | 183 | memset, 3 × 61 layers |
| `_fill_padded_rows_kernel` | 0.743 | 183 | **ROCm-only fallback** |
| `elementwise_kernel_manual_unroll` ×2 | 1.066 | 67+66 | aten copies |
| `index_elementwise_kernel` | 0.400 | 69 | |
| `bfloat16tofloat32_copy_kernel` | 0.398 | 92 | dtype conversion |
| `_swa_scatter_kernel` | 0.274 | 61 | SWA KV write, 1/layer |
| `_fill_compress_tail_kernel` | 0.143 | 31 | HCA layers only |
| `Memcpy DtoD` / `DtoH` / `copyBuffer` / +6 | 0.294 | — | |

**⚠ REVISED: flipping the `_is_cuda` gate would save approximately NOTHING.**
This doc previously called it "the cheapest item on the whole list". The claim
was wrong and here is the measurement that kills it.

`_fill_padded_rows` launches **one Triton program per row** (`grid=(n_rows,)`,
`kernels/ops/moe/fill_padded_rows.py:71`), and the same kernel appears at two
very different row counts:

| window | grid | µs/call |
|---|---:|---:|
| decode (bs=12) | **886** | 4.1 |
| EXTEND | **8192** | 5.26 |

**9.2x the rows costs 1.28x the time.** Fitting `t = a + b·rows` gives
**a = 3.96 µs fixed, b = 0.16 ns/row** — i.e. at decode sizes ~97 % of the call
is dispatch overhead and the actual filling is ~0.14 µs. Corroboration: every
trivial kernel in the trace lands on the same floor regardless of what it does
(`Memcpy DtoD` 4.0, `__amd_rocclr_copyBuffer` 4.0, `fillBufferAligned` 4.3,
`_swa_scatter_kernel` 4.5, `dynamic_per_group_scaled_quant` 4.5 µs/call). Note
cuda-graph replay is 100 % of steps, so this floor is GPU-side dispatch, not CPU
launch overhead that a graph could hide.

Therefore **replacing one launch with another launch saves ~0.8 µs/call at
best**, and `mask_topk_ids` is in any case a CUDA-JIT module
(`load_jit(..., cuda_files=["deepseek_v4/..."])`,
`kernels/ops/attention/dsv4/moe.py:108`) with no ROCm build — a HIP port would
buy nothing here. The `hash_topk` function right below it shows the house
pattern for branching to Triton on HIP, if a port is ever wanted for other
reasons.

**The real shape of the `copy`+`quant` gap: it is a kernel-COUNT problem.**
Those two buckets are 6.35 ms/step over roughly **1,200 launches**, averaging
~5 µs each against a ~4 µs dispatch floor — so about **4.8 ms/step is pure
dispatch overhead** and only ~1.5 ms is work. B200 issues none of it because its
MoE is one fused kernel. The lever is *eliminating launches*, not making any
individual kernel faster:

1. Fold the padded-row mask and the weight zeroing into the **topk kernel that
   already writes those tensors** — removes 2 of 3 launches per layer (122/step,
   ~0.5 ms).
2. `mega_moe_flydsl.py:259-263` does `.to(torch.int32)`, `.to(torch.float32)`
   and then `_fill_padded_rows` on the result — three launches per layer that
   could be one fused cast+fill (the `bfloat16tofloat32_copy_kernel`, 92 calls
   / 0.398 ms, is this cast).
3. `fillBufferAligned` at 183 calls/step (0.780 ms) is the other 3-per-layer
   pattern and is **still unattributed** — `copy_attrib.py` resolves 81 % of
   EXTEND copies and this kernel is not among them. Pin its call site before
   planning anything around it.

Ceiling for the whole line of work: ~1.5 ms/step (2.3 % of a 65.5 ms step) and
only if the launches actually disappear into neighbouring kernels.

### DECISION — attack the SLOWEST RANK's absolute cost, not the dispatch wait

**⚠ THE TRAP: barrier wait is not cashable.** A barrier's duration is idle time
that already belongs to a rank which is *ahead*. The step is set by the LAST
rank to arrive, so removing 200 µs/call of `prepare` spin on a fast rank buys
**zero** wall time — it only makes the idleness less visible. Any plan of the
form "`prepare` is 10-19 ms/step of waiting, so attack `prepare`" is counting
money that is not there. Only two things shorten the step: making the
**critical-path rank** (the slowest) do less work, or removing work that every
rank does.

**This trap has already produced one wrong prediction and one null arm on this
node.** The dispersion reading says balancing should pay; the `total_tokens`
A/B pushed `#full token` skew 2.08x → 1.69x, verified with `kv_skew.py` on the
new `server.log`, and matched-bs step time moved 0 to +2.4 % — nothing. It is
consistent, not anomalous: balancing shortens the *fast* ranks' wait, which was
never on the critical path. (Keep the flag anyway; TTFT 11.07 → 9.61 s.)

**Corollary — the "how much of `prepare` is real work" question needs no new
experiment.** The busiest rank waits least, so its `prepare` is an upper bound
on the real dispatch cost: **74.6 µs/call ≈ 4.6 ms/step**. Against the
n-weighted mean of 162.1 µs/call, **≥54 % of the average `prepare` is pure
spin**, and the attackable part of the 10.48 ms/step on TP-7 is at most ~4.6 ms.
Matching B200's absolute MLA time is NOT a prerequisite for reading this — the
min across ranks already bounds it.

**Why "make MLA as fast as B200" does not convert the wait into work.** The
wait is set by the rank-to-rank *spread*, not the absolute level. Split-K cut
MLA 57 % on the matched cell and `prepare` fell only 25 %, with MoE dispatch's
share of the step essentially unchanged (47.5 % → 45.0 %) — because a uniform
proportional speedup leaves the *ratios* between ranks intact (`mla_split` still
spans 42.9-277.8 µs, 6.5x, inside one trace). Absolute MLA speed is still worth
chasing, but for its own sake: `attn` is 13.26 vs B200's 8.20, i.e. +5.06
ms/step on the critical path.

**So, in priority order:**

1. **The slowest rank's MoE compute.** `stage1` is 13.50 ms/step and is real
   work (r = +0.713) on every rank — it is paid on the critical path, and it is
   the largest single non-barrier gap to B200.
2. **The slowest rank's MLA.** +5.06 ms/step vs B200 and directly on the
   critical path. splits=8 in situ is the cheap untested knob.
3. **DP batch dispersion (bs 4 to 17 at one instant), but only as a
   *max*-reduction.** Worth it if it lowers the peak per-rank load; worthless if
   it merely equalises waits, which is what `total_tokens` did.
4. **`prepare`'s ~4.6 ms/step of real dispatch work** — a genuine target, and
   the only part of that kernel that is.

### PLAN (2026-09-18) — closing the remaining +31.33 ms/step

Ordered by expected ms/step per unit of effort, not by size of the bucket.
Every phase carries a pre-registered prediction and a falsification threshold,
because three items on this list already died of "plausible mechanism, null
result" (`total_tokens`, the `_is_cuda` gate, uniform split-K).

**Phase 0 — measurement debt. Cheap, and Phase 1 cannot be read without it.**

- **0a. Pin `fillBufferAligned`'s call site.** 183 calls/step, 0.780 ms,
  unattributed; `copy_attrib.py` resolves 81 % of EXTEND copies and misses this
  one. Trace + `rg`, no GPU.
- **0b. Turn the ~4 µs dispatch floor into a measured number.** Right now it is
  a two-point fit (`a = 3.96 µs`, from grid 886 vs 8192). Launch an empty
  Triton kernel at grid 886 inside and outside a graph. **Pass:** floor within
  ±20 % of 3.96 µs. If it comes back at ~1 µs, Phase 2's premise is wrong and
  the copy cost is real work after all.
- **0c. One 8/8-rank decode capture.** Needed for any cross-rank claim. Fix the
  EXPORT window, not `num_steps`: 16 steps, or raise the gloo/`PrefillDelayer`
  timeout for the capture run.

#### Phase 0 RESULTS (2026-09-18 11:2x)

**0a — BLOCKED, and the blocker is structural.** `copy_attrib.py` states it in
its own header: attribution only works in **EXTEND**, because a graph-replayed
window emits no per-kernel launch events to walk back to a `cpu_op`. Decode is
100 % graph-replayed here, and `__amd_rocclr_fillBufferAligned` **does not
appear in EXTEND at all** — the only fill kernel there is
`_fill_padded_rows_kernel`. So no existing capture can attribute it. It is a HIP
memset (`hipMemsetAsync`), and the fact that it exists only under graph replay
is itself a lead: buffers zeroed during capture become memset NODES replayed
every step, where eager would run them once. Cheapest way to pin it: one short
`--disable-cuda-graph` decode capture, where the launches become eager and
`copy_attrib.py` works. Source-side reading is hampered by this checkout's
token mangling (`torch.zeros` renders as `n(`, `overlap` as `ln`).

**0b — DONE, and it FAILED its own pass criterion.** Pre-registered: floor
within ±20 % of 3.96 µs. Measured (`/tmp/floor.py`, `/tmp/floor2.py`, logs in
`/shared_nfs/kk/floor_0b*.log`):

| case | grid 886 | grid 8192 |
|---|---:|---:|
| fill kernel, in graph | **1.87 µs** | 3.65 µs |
| no-op kernel, in graph | **1.62 µs** | 2.93 µs |
| fill kernel, eager | 9.56 µs | 9.43 µs |

Bench fit: **fixed 1.66 µs, 0.24 ns/row** — the floor is ~1.6 µs, **not 4 µs**.
Two follow-up hypotheses for the gap to production's 4.1 µs were both
falsified: per-node cost is flat at **1.62 / 1.55 / 1.55 µs** for graphs of
100 / 1000 / 3000 nodes, and **1.63 / 1.56 / 1.55 µs** when 8 *distinct*
kernels alternate (so neither graph size nor kernel heterogeneity explains it).

What this changes:

- **RETRACTED:** "~4.8 ms/step of `copy`+`quant` is pure dispatch overhead."
  With a 1.6 µs floor it is **~2.0 ms** (1,263 launches × 1.6 µs).
- **Unchanged:** the `_is_cuda` gate verdict. An in-situ launch costs 4.1 µs
  whatever its composition, so swapping one launch for another still buys ~0.
- **Unchanged:** Phase 2's ~1.5 ms estimate, because fusion removes *whole*
  launches, not just their dispatch component.
- **NEW, and cheap:** the unexplained ~2.3 µs/call premium in production is
  worth 1,263 × 2.3 ≈ **2.9 ms/step** if it has a cause we control. Leading
  suspect is clock/power under sustained MoE load (a small kernel's duration
  stretches when sclk drops). Probe with `rocm-smi` sclk sampling during the
  next decode arm — no extra run needed.

**Phase 1 — TBO × MegaMoE. The one structural lever, and the cheapest possible
test of it.**

**⚠ PRE-FLIGHT BLOCKER (found 2026-09-18, before spending the arm): setting
`ENABLE_TBO=1` on this arm would do NOTHING.** `ENABLE_TBO` is read only by
`dsv4_fp4_mi355x_sglang_b200align_mtp.sh:157`, which is the launcher the
`fp4_dptbo_*` arms used. Every MegaMoE arm — including this whole split-K line —
runs `dsv4_fp4_mi355x_sglang_mtp.sh`, which has **no TBO wiring and no
pass-through array** for one-off server flags (the b200align launcher has
`EXTRA_SERVER_ARGS`; ours does not). sglang has no env override either; the only
switch is the CLI flag `--enable-two-batch-overlap`. Had this not been caught,
the arm would have produced a clean null and "TBO cannot exploit the window"
would have entered the record as a measured result.

Good news on feasibility: `check_two_batch_overlap`
(`arg_groups/validation_hook.py:481`) only rejects TBO when
`moe_a2a_backend == "none"` *and* DP attention is off. With MegaMoE (an a2a
backend) plus DP attention, **validation passes** — so the flag is accepted, and
whether the MegaMoE path actually splits ubatches is the open question
(`agentx_mori.sh:19`: "TBO x a2a is untested here").

**RESOLVED 2026-09-18:** the `EXTRA_SERVER_ARGS` pass-through was added to
`dsv4_fp4_mi355x_sglang_mtp.sh` (2 lines next to `PARALLEL_ARGS`, plus
`"${EXTRA_ARGS[@]}"` in `SGLANG_CMD` after `CACHE_ARGS`), mirroring the
b200align launcher. Inert when unset. That file already carried our own
uncommitted MegaMoE block, so this is additive to the same line of work.
`agentx_c128_hcasplit_tbo.sh` uses it, and the flag was verified to reach
`sglang_command.txt`.

#### Phase 1 RESULT — TBO is BLOCKED BY A CODE GAP, not by config. The arm never served

Launched 03:28 UTC, died during **cuda-graph capture**:

```
two_batch_overlap.py:246 in split_spec_info
AttributeError: 'DFlashVerifyInput' object has no attribute 'retrieve_index'
  <- filter_batch (743) <- prepare_raw (566) <- capture_one_batch_size (366)
  <- decode_cuda_graph_runner.capture_one_shape (1136)
```

**`split_spec_info` is written for EAGLE's tree-shaped verify input** — it slices
`custom_mask`, `positions`, `retrieve_index`, `retrieve_next_token`. This stack
runs **DFlash MTP**, whose `DFlashVerifyInput` (`speculative/dflash_info.py`) is
a *linear chain* and carries only:

```
draft_token: torch.Tensor      positions: torch.Tensor
draft_token_num: int           topk: int = 1
custom_mask: torch.Tensor | None = None
num_tokens_per_req: int = -1
```

So TBO × MegaMoE is not "untested and slow", it is **not wired for the spec
decoder this whole line of work uses**. The flag is accepted by validation and
then explodes at capture.

**The fix looks small and is the next Phase 1 action:** teach `split_spec_info`
to handle `DFlashVerifyInput` — slice `draft_token` and `positions` by the
ubatch token range, slice `custom_mask` when present, carry `draft_token_num`,
`topk` and `num_tokens_per_req` through. There is no tree to split because
`topk == 1`, which is why none of the EAGLE fields exist. It is a change in
`/sgl-workspace/sglang-MegaMoE`, i.e. outside `claude-skills/`, so it needs a
go-ahead; it is also plausibly upstreamable on its own.

Fallback if that is not wanted: run TBO with MTP **off** as a pure mechanism
probe. It answers "can TBO put work into the `prepare` window on the a2a path?"
but needs its own no-MTP baseline (2 arms), and a no-MTP step has a different
shape, so it cannot be compared to anything in the table above.

The whole decode step is a serial chain: `ovl = ksum/wall = 1.00` on every rank,
3,065 launches/step, and 226 of 256 CUs sit unclaimed for the 10-16 ms that
`megamoe_prepare_compact` spins. TBO already exists as `ENABLE_TBO=1` in the
launcher and every `fp4_dptbo_*` arm used it — but **`agentx_mori.sh:19` records
"TBO x a2a is untested here"**, and MegaMoE is an a2a backend. So this may not
even launch; that is worth one arm to find out.

- **Action:** `ENABLE_TBO=1` on `agentx_c128_hcasplit.sh`, nothing else changed.
- **Pre-registered:** if micro-batch B's `attn`/`gemm` lands in A's `prepare`
  window, **−5 to −10 ms/step** at matched bs. **Falsification: better than
  −2 ms ⇒ TBO cannot exploit this window.**
- **Read the outcome from `ovl`, not from the step time.** The mechanism claim
  is "kernels now overlap", so the check is `ksum/wall > 1.0` in a trace. A
  step-time win with `ovl` still 1.00 means something else moved.

**Phase 2 — launch elimination in the MoE housekeeping. ~1.5 ms realistic,
~5.1 ms ceiling.**

- **2a.** Fold the padded-row mask and the weight zeroing into the topk kernel
  that already writes those tensors (`topk.py:1578` / `1591`) — removes 122
  launches/step.
- **2b.** Fuse the `.to(int32)` + `.to(float32)` + `_fill_padded_rows` triple at
  `mega_moe_flydsl.py:259-263` into one cast+fill.
- **Pre-registered:** **−0.4 to −0.6 ms on the `copy` bucket each**;
  falsification: `copy` unchanged within 0.1 ms ⇒ the per-launch model is wrong
  (see 0b). Ceiling for the whole line is `copy` 4.10 → 0.13 and `quant`
  2.25 → 1.14, i.e. −5.08 ms, and only via fusion.
- **DO NOT flip the `_is_cuda` gate at `topk.py:1573`** — measured to buy ~0.8
  µs/call at best, and `mask_topk_ids` has no ROCm build.

**Phase 3 — MLA absolute cost. +5.06 ms/step vs B200, squarely on the critical
path. REDESIGN, not a knob.**

**3a (`SGLANG_MLA_HCA_KV_SPLITS=8`) was CANCELLED on 2026-09-18** after launch,
before it served: the split count is a knob, and no value of it removes the
per-rank dispersion (`mla_split` 42.9-277.8 µs, 6.5x, inside one trace) — the
microbench already put 4 at ~93 % of 8 at steady-state kv_len, so the whole
prize was ≤1.5 ms. Node was returned to baseline.

**Why the kernel is slow is now a roofline statement, not an impression.**
Rank 7, bs=12, `_paged_decode_split_kernel` at 130.7 µs/call × 61 calls =
7.97 ms/step. Shapes from the DSv4-Pro config: `head_dim` 512 + `qk_rope_head_dim`
64 = 576 elements/token/layer, fp8 KV cache, `num_attention_heads` 128, HCA
kv_len ≈ ctx/128 ≈ 1,170 at this operating point (median 1,300, cap 5,000).

| axis | achieved | MI355X peak | utilisation |
|---|---:|---:|---:|
| HBM read (12 × 1,170 × 576 B in 130.7 µs) | **~62 GB/s** | ~8 TB/s | **~0.8 %** |
| FLOPs (QK+PV, 4 draft tokens × 128 heads) | **~112 TFLOP/s** | ~1.25 PF bf16 | **~9 %** |

Both axes are an order of magnitude from the roofline, and the conclusion holds
even if kv_len is off by 3x. Corroborated independently by
`mla_microbench.py`'s own header: clocks flat at 2372-2393 MHz with throttle 0
and `umc_activity` 19.5 %, so **not DVFS and not bandwidth**. The kernel is
**latency / occupancy / imbalance-bound**, which is the same thing the original
split-K finding said locally (base grid 196 CTAs against a 384 target, cost set
by one CTA walking ~5,000 KV entries beside CTAs walking 200).

**So the design lever is the DECOMPOSITION, not the math.** Work is currently
tiled per (request, layer) with a fixed split count, on a grid that underfills
the device before imbalance is even considered. The direction to think through:

1. **Tile over the batch's TOTAL KV entries, not per request.** A persistent
   grid sized to the device (CU count × occupancy) pulling equal-sized KV tiles
   from a work queue, then a fixed-cost reduction per (request, head-group),
   makes cost ∝ Σkv_len / device width and **insensitive to per-request
   dispersion**. It removes the intra-call straggler and the 6.5x rank spread at
   once, and it retires the 4-vs-8 question permanently.
2. **Reuse each KV tile across all 128 q heads × ~4 draft tokens** (512 q rows)
   before evicting it — that is the MFMA-friendly shape and it raises arithmetic
   intensity, which is what a 9 %-of-peak compute number is asking for. Check
   whether the current kernel re-reads KV per head-group or per split.
3. **Keep the fp8 path fp8 end to end.** If the KV dequants to bf16 before the
   MFMA, that alone is a 2x peak-compute handicap on gfx950.

**Prize, for sequencing:** B200 does the same work in 40.2 µs/call versus our
130.7 (3.25x). Matching it takes `attn` 13.26 → ~4 ms, i.e. **−9 ms/step** —
larger than every other item in this plan combined. Even halving the gap beats
Phase 2's entire ceiling.

**Iterate without arms.** `analysis/mla_microbench.py` (kv_len ≤ 2048) and
`analysis/mla_tail_bench.py` (the real distribution out to ~5,000) already run
the kernel standalone on one GPU with config-derived shapes. Any redesign should
be developed and falsified there first; an arm is only for confirming that the
win transfers, exactly as the split-K line did (microbench −8 predicted,
−7.42/−7.50 measured).

#### Phase 3 MEASUREMENT DONE (2026-09-18 11:4x) — it is UNDER-OCCUPANCY first, straggler second. MFMA is idle

Standalone, no server: `analysis/mla_tail_bench.py --bs 12 14` and
`rocprofv3 --pmc OccupancyPercent MfmaUtil` driven one config at a time by
`/tmp/mla_one.py` (the tail bench mixes heuristic + flat + a split sweep in one
process, so counter means over its 114 dispatches are meaningless — drive one
case). Logs: `/shared_nfs/kk/mla_tail_20260918.log`,
`/shared_nfs/kk/prof_mla_sweep/`.

**Imbalance, priced first.** Same total KV, only the distribution changed
(`real_lens`: p50 1,090, p99/max 5,000, mean 1,203, total 101,098):

| bs | ragged (heuristic) | flat at the mean | a perfect straggler fix is worth |
|---|---:|---:|---:|
| 12 | 1,152.7 µs | 547.9 µs | **52.5 %** |
| 14 | 1,883.3 µs | 539.2 µs | **71.4 %** |

split-K captures part of that and then stops: at bs=12, splits 1/2/4/8 give
1,873 / 1,152 / 789 / 688 µs — so **8 does beat 4 on the tail** (−40 % vs the
heuristic) but never approaches the 548 µs balanced floor. This is also why
Phase 3a was pointless: the knob's whole remaining range is inside the gap that
balance alone would close.

**Then the counters, and they reorder the diagnosis.** Profiled times are
inflated by serialisation (921 vs 789 µs unprofiled); read the ratios:

| config | time | occupancy | MfmaUtil | CTAs (grid/wg) |
|---|---:|---:|---:|---:|
| ragged, splits=4 | 921 µs | **12.3 %** | **3.4 %** | 672 |
| ragged, splits=8 | 811 µs | 15.9 % | 4.0 % | 1,344 |
| ragged, splits=16 | 882 µs | 17.5 % | 3.7 % | 2,688 |
| flat, splits=4 | 603 µs | **20.9 %** | 5.7 % | 672 |

Dispatch shape is identical in every row: **workgroup 512 threads (8 waves),
VGPR 128, Accum_VGPR 0, LDS 0.**

Four conclusions that a redesign has to answer to:

1. **The MFMA pipe is idle — 3.4 to 5.7 %.** Whatever the kernel is bound by, it
   is not arithmetic. With HBM at ~1 % of peak too (see the roofline above), the
   waves are stalling on memory *latency*, and there are too few of them to
   hide it. That is the textbook signature and it is now measured, not inferred.
2. **`LDS = 0` and `Accum_VGPR = 0`.** The kernel stages no KV tile in LDS and
   uses no AGPRs, i.e. it is not built the way a CDNA flash-decode kernel has to
   be built to keep MFMA fed. This is the single most actionable structural
   finding of the whole session.
3. **More CTAs is not the axis.** splits 4→16 quadruples CTAs and lifts
   occupancy 12.3 → 17.5 %, but time gets *worse* past 8 (811 → 882 µs) as
   partial-buffer traffic and the reduce kernel eat the gain. Occupancy has to
   come from work per CTA and latency hiding, not from more splits.
4. **Balance is worth ~8.6 points of occupancy on its own** (12.3 → 20.9 % at an
   identical grid) — the CTA duration spread leaves CUs idle at the tail, so
   the straggler and the occupancy problem are partly the same problem.

Ceiling check: even the balanced case sits at 20.9 % occupancy and 5.7 % MFMA,
so the redesign's headroom is much larger than the 3.25x B200 gap. Conversely
B200 is not obviously more *efficient* — some of its 40.2 µs/call advantage is
peak FLOPs — so "match B200" is the wrong target; "keep MFMA fed" is the right
one.

#### ⭐ Phase 3 FOUND A ONE-LINE 2x BEFORE ANY REWRITE — `block_k` is tuned at the wrong operating point

`paged_decode.py` resolves the K-tile like this:

```python
_bk, num_warps, num_stages = _kernel_config(block_h)   # _bk is always 16
if block_k is None:
    # "fp8 dequant inflates per-tile ALU work ~4x; ... Empirically BLOCK_K=32
    #  wins ~20% over BLOCK_K=16 on fp8 (bs=512 ctx=4096: 3000us -> 2300us)"
    block_k = 32 if quant_kv else _bk
```

Production runs `--kv-cache-dtype fp8_e4m3`, so **every MLA decode call takes
`block_k=32`**, and `_kernel_config`'s own documented choice of 16 is never used
on the quantised path. That 32 was justified at **bs=512, ctx=4096**. Measured
now, one config per fresh process (`/shared_nfs/kk/mla_cfg.py`,
`/shared_nfs/kk/mla_bk_check.py`, `block_h=64`, µs/call):

| shape | splits=4, bk16 | splits=4, bk32 | splits=8, bk16 | splits=8, bk32 |
|---|---:|---:|---:|---:|
| bs=12, ragged tail (production) | **396.4** | 789.3 | **338.5** | 686.9 |
| bs=24, ragged | **622.0** | 1,269.6 | 614.0 | 1,270.7 |
| bs=48, ragged | **1,021.4** | 2,166.8 | 1,157.4 | 2,377.0 |
| bs=96, ragged | **2,054.4** | 4,347.9 | 2,317.9 | 4,775.4 |
| bs=128, flat kv_len=32 | **477.6** | 520.2 | 540.7 | 587.3 |
| bs=512, flat kv_len=32 | **1,821.0** | 2,025.9 | 2,164.9 | 2,358.9 |

**`block_k=16` wins everywhere, including the comment's own regime** (bs=512
short context: −10 %). At the production shape it is **2.0x**. The "+20 % for 32"
claim does not reproduce on this kernel at all — it almost certainly predates
the split-K + exp2 rewrite.

**Numerics check:** bk16 vs bk32 relL2 **1.76e-03**, max_abs 1.22e-04, all
finite — same family as the 2.46e-03 the split-K arm shipped with (the
difference is reduce ordering, not accuracy).

**Also visible in the same table:** `kv_splits=8` beats 4 only while CTAs are
scarce (bs=12: 338 vs 396) and *loses* once they are not (bs=48: 1,157 vs
1,021). So the splits heuristic should be shape-aware, but that is second-order
next to `block_k`.

**Why bigger K tiles lose here, mechanically:** `block_k=64` does not even
launch — `OutOfResources: shared memory, Required: 327680`. At `BLOCK_D=512`
the K tile is what sets LDS/register pressure, so 32 already trades occupancy
(measured 12 %) for dequant amortisation that a latency-bound kernel cannot
use. That is the same trade the roofline numbers say we are on the wrong side
of.

**Expected production effect:** `_paged_decode_split_kernel` is 7.97 ms/step on
rank 7 at bs=12, so a 2x takes `attn` 13.26 → ~9.3 ms, i.e. **about −4 ms/step**
— larger than Phase 2's entire ceiling, from one line. Pre-registered for the
arm: **−3 to −5 ms/step** at matched bs vs `megamoe-eplb-c128-hcasplit4`;
**falsification: better than −1 ms ⇒ the microbench does not transfer** (and
then the next question is what the graph-captured path does differently).

**SHIPPED (env-gated) AND AN ARM IS RUNNING.** `paged_decode.py` now reads
`_FP8_BLOCK_K = int(os.environ.get("SGLANG_MLA_FP8_BLOCK_K", "32"))` at import —
module level, like every other knob there, because the value must be identical
between CUDAGraph capture and replay — and the quant path uses it instead of the
literal 32. **Default is unchanged (32), so absent the env nothing moves.**
Verified end to end through the wrapper: `env=32 → 790.8 µs`,
`env=16 → 399.0 µs` at bs=12 ragged splits=4.

Arm: `agentx_c128_hcasplit_bk16.sh` → `megamoe-eplb-c128-hcasplit4-bk16`,
launched 2026-09-18 04:05 UTC via `run_chain.sh` (runner PID 1405599, detached),
log `/shared_nfs/kk/chain_bk16.log`, results appended to
`/shared_nfs/kk/chain_summary.md`. Only `SGLANG_MLA_FP8_BLOCK_K=16` differs from
`agentx_c128_hcasplit.sh`; splits stay 4.

**When it lands:** matched-bs weighted Δstep against
`megamoe-eplb-c128-hcasplit4` with `/shared_nfs/kk/matched_bs.py`, not against
b200aligned — the question is the K tile, not split-K. Check `accept len`
(~3.77) and `cmd_diff.py` for flag identity before reading the delta.

#### Phase 3 REWRITE STARTED — #39172's segment plan is ported to dsv4 `paged_decode`, env-gated, NOT yet validated

PR [#39172](https://github.com/sgl-project/sglang/pull/39172) (merged, and
already in this tree as `3eeb7d37f9`) fixes exactly this problem for the
**gfx950 assembly** attention kernel: instead of giving every sequence the same
number of segments, a Triton plan kernel picks one segment **length** from the
batch's lengths, writes a compact work list, and the kernel runs as a 1-D grid
over it. It does **not** touch dsv4's `_paged_decode_split_kernel`
(`--attention-backend dsv4`), which still does
`tiles_per_segment = cdiv(kv_len, KV_SPLITS * BLOCK_K)`.

Its own numbers confirm our diagnosis independently: uniform batches do not move
(verify 16×119k: 176 → 176 µs) while ragged ones do (bs16 agentic 266 → 179 µs,
bs20 482 → 254 µs), and end to end it is TPOT p90 −3.6 to −4.6 % with
throughput +0 to +0.9 % — i.e. read the step/tail, not the throughput.

**Ported into `dsv4/unified_kv_kernels/paged_decode.py`** (all new code, shipped
path untouched, `SGLANG_MLA_SEG_PLAN=1` selects it, default off):

- `_seg_plan_kernel` — one program per token, recomputes the batch-wide plan
  (same structure as the reference), 16-iteration binary search for the smallest
  tiles-per-segment `T` with `Σ cdiv(nt,T) ≤ target_wgs` and
  `max cdiv(nt,T) ≤ SEG_MAX`, writes `plan[0]=T`, the work list
  `plan[1+w] = (token<<16)|seg` with a `-1` tail, and `seg_start[N]`/`seg_cnt[N]`.
- `_paged_decode_planned_kernel` — grid `(W_MAX, n_head_blocks)`; decodes its
  work item, exits on `-1` before touching memory; inner loop is a faithful copy
  of the split kernel's.
- `_paged_decode_planned_reduce_kernel` — same 2D-tile reduce, but a token's
  partials are at flat work-item slots `seg_start[t] .. +seg_cnt[t]`.

Adaptations forced by this kernel's shape, and worth knowing: the unit is a
**token** not a sequence; lengths come from `kv_indptr` differences; the
work-group target is divided by `n_head_blocks` (the head axis is a separate
grid dimension here) rather than by `kv_heads`; and partials are indexed by
**work item** — `[W_MAX, H, D]` with `W_MAX = max(N, target_wgs)` — which is
what makes a variable segment count possible at a static shape and is also
**4x less memory** than the current `[N, KV_SPLITS, H, D]`.

Knobs: `SGLANG_MLA_SEG_MAX` (default 16) and `SGLANG_MLA_SEG_WG_MULT`
(default 1). The multiplier exists because the reference targets one work-group
per CU on a kernel that saturates a CU with one, whereas this kernel measures
12-21 % occupancy — more, smaller segments may pay twice (balance *and*
parallelism). With `mult=1` at bs=12 the plan can only hand out ~128 work items
against the split path's 336, so it trades parallelism for balance; that is
exactly what the sweep has to settle.

#### bk16 ARM: FALSIFIED (+0.36 ms), and the root cause kills the hypothesis outright

`megamoe-eplb-c128-hcasplit4-bk16`, gates clean (`accept len` 3.77 both sides,
cuda-graph replay 100 %): matched-bs weighted **+0.36 ms** (19 cells, w=2970),
aggregate p50 114.12 → 114.24. Pre-registered −3 to −5 with falsification at
−1 ms, so this is a clean falsification, not a weak result.

**Why, and it is not "the microbench doesn't transfer".** The backend's own
docstring (`deepseek_v4_backend_hip_radix.py:1375`): *"q_rope present means q is
a packed fp8 row and the pool is the two-pool fp8 one, so decode goes to the asm
reader; absent means both are plain bf16 and it goes to Triton."* And
`runtime.decode` calls `sparse_attn_v4_paged_decode` **without `kv_scales`**. So
production's Triton MLA decode runs with **`QUANT_KV=False`, where the wrapper
already picks `block_k=16`** — the `32` only ever applied to the quantised
Triton path, which production takes to aiter's ASM reader instead. The env knob
changed nothing because there was nothing to change.

Measured both configurations to close it (bs=12, ragged, total KV 101,098):

| path | default | bk16 | bk32 | ns per KV entry |
|---|---:|---:|---:|---:|
| fp8 + scales (what the harness fed) | 792.1 | **397.5** | 789.4 | 7.83 |
| **bf16 (what production runs)** | **244.3** | 237.5 | 450.5 | **2.42** |

So the 2x was real but for a configuration we do not run, and the bf16 path is
3.2x cheaper per KV entry than the harness suggested. Index layout was ruled out
separately (random / paged-256 / sequential all give bk32/bk16 = 1.98x).

**⚠ METHODOLOGY TRAP, now fixed — `import sglang` does not resolve to the arm's
tree.** Bare `import sglang` gives `/sgl-workspace/sglang` (933-line
`paged_decode.py`); the arms pin `/sgl-workspace/sglang-MegaMoE` (1,535 lines).
`analysis/mla_microbench.py:34` inserts the MegaMoE path, so scripts that import
it *first* are fine — but two of today's scripts imported `sglang` before it and
silently measured the wrong tree (the tell was `relL2 = 0.000e+00` and total
insensitivity to the plan's knobs, because the branch did not exist there).
**Always run microbenches with `PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python`.**

#### ⭐ THE SEGMENT PLAN WORKS — within 7 % of the perfect-balance floor, and an arm is running

Re-measured on the production path (bf16, no scales), PYTHONPATH pinned,
`/shared_nfs/kk/mla_bf16_balance.py`, ragged `real_lens`:

| bs | split-K=4 (shipped) | split-K=8 | **plan (best)** | balance floor (flat) | plan vs split-4 |
|---|---:|---:|---:|---:|---:|
| 12 | 237.4 | 229.6 | **153.4** (seg16, x2) | 143.0 | **−35 %** |
| 24 | 382.6 | 413.4 | **299.0** (seg16, x4) | 253.3 | **−22 %** |
| 48 | 658.0 | 808.1 | **559.0** (seg16, x2) | 434.5 | **−15 %** |

**Numerics pass:** relL2 **1.74e-03 / 1.89e-03 / 2.04e-03**, max_abs 1.95e-03,
all finite — the same family as split-K's shipped 2.46e-03.

At bs=12 the plan captures **89 % of the whole imbalance gap** (237.4 → 153.4
against a 143.0 floor). `seg_max=16` beats 32 and 64 everywhere (more segments
cost more reduce than they buy in balance), which is the opposite of what the
tile arithmetic alone predicted — worth remembering that the reduce is not free.

#### The first segplan arm FAULTED — my port wrote past the work list. Fixed, and the bound is now structural

`megamoe-eplb-c128-segplan`, 05:47 UTC: **`Memory access fault by GPU node-N ...
Reason: Unknown` on all 8 ranks**, during startup, before serving.

**Root cause, and it is a bug the reference has too.** The reference's binary
search looks for the smallest `T` with `Σ ceil(nt/T) ≤ target_wgs`. When **no**
`T` satisfies that — which happens as soon as `num_tokens > target_wgs`, routine
in production where `N` is the padded graph batch — the search returns a `T`
whose segment count can be several times `target_wgs`. I had sized the pool at
`max(N, target_wgs)`, so the work-list store ran off the end.

**Fixed by construction, not by searching.** Since
`Σ ceil(nt/T) ≤ total/T + N`, any `T ≥ total/target_wgs` gives
`Σ ≤ N + target_wgs`. So the pool is now `w_max = N + target_wgs` and the plan
takes `T = max(ceil(total/target_wgs), ceil(mx/SEG_MAX), 1)` directly — the
binary search is deleted, because that `T` is already the finest safe
granularity. Still ~3x less memory than the split path's `N × KV_SPLITS`.

**Second latent fault found while fixing the first:** the reference's idle-tail
store (`tot + pid + w * num_seqs`, masked to `num_work`) only covers
`num_tokens × BLOCK_W` slots, which is short of the pool whenever
`target_wgs > (BLOCK_W - 1) × num_tokens` — a small batch with a wide
work-group target. Unclaimed slots would be read as live work items. Replaced
with a host-side `torch.full(-1)` of the pool, which cannot get it wrong.

Re-validated after the fix (bf16, PYTHONPATH pinned): **relL2 1.718e-03**, no
fault, and the best plan config is now `seg_max=16, wg_mult=4` at **156.5 µs**
against split-K=4's 241.3 — the same −35 % as before, so the safety fix cost
nothing.

#### CORRECTNESS FIRST — GSM8K A/B running before any performance arm

After a GPU memory fault, an accuracy gate comes before a timing number.
`gsm8k_segplan.sh` (driver PID 1433397, detached, started 2026-09-18 05:5x UTC)
runs two arms back to back — `SGLANG_MLA_SEG_PLAN=0` then `=1`, everything else
identical — and records gsm8k 1319 for each. Status:
`/shared_nfs/kk/gsm8k_segplan_status.txt`, per-arm logs
`/shared_nfs/kk/gsm8k_{planoff,planon}.log`.

Two things this run finally gets right, both of which broke `gsm8k_ab.sh`
before:

- **It serves through the arm script**, so the MegaMoE env block
  (`SGLANG_AMD_USE_FLYDSL_MEGA_MOE`, `..._MEGA_QUANT=a8w4`, `SGLANG_USE_AITER`,
  `SGLANG_MOE_PADDING`) is inherited rather than reconstructed from
  `sglang_command.txt`, which is what previously dropped MegaMoE into the Triton
  MoE path and died at `fused_moe_triton_kernels.py:863`.
- **`EVAL_ONLY=true`** (`dsv4_fp4_mi355x_sglang_mtp.sh:313`) — the launcher's own
  accuracy switch. It skips the aiperf replay *and* leaves
  `SGLANG_SIMULATE_ACC_LEN` unset; with that pin at the golden 3.77 the MTP
  acceptance is faked and the accuracy number means nothing.

#### ✅ GSM8K PASSED — the port is correct on the real server, not just in relL2

| arm | accuracy | invalid |
|---|---:|---:|
| `planoff` (shipped split-K path) | **0.937** | 0.000 |
| `planon` (segment plan) | **0.939** | 0.000 |

1319 questions, `--max-new-tokens 8192`, `EVAL_ONLY=true` so MTP acceptance is
real rather than pinned to the golden 3.77. +0.002 is ~3 questions — inside
noise, and exactly what a different fp32 reduce order across new segment
boundaries should look like. `Invalid 0.000` on both sides also means nothing
truncated, so the 8192 trap was avoided.

This is the first accuracy number this line has produced at all (the item was
open since 2026-09-17), and `gsm8k_segplan.sh` is now the working recipe: serve
through the arm script so the MegaMoE env is inherited, plus `EVAL_ONLY=true`.

#### Performance arm running (after the accuracy gate, not before)

`agentx_c128_segplan.sh` → `megamoe-eplb-c128-segplan`, relaunched 2026-09-18
06:34 UTC, runner PID 1457135, log `/shared_nfs/kk/chain_segplan2.log`, results
appended to `/shared_nfs/kk/chain_summary.md`. `seg_max=16`, `wg_mult=4` — the
same pair GSM8K was run with, so accuracy and timing refer to one configuration.

#### RESULT: FALSIFIED (−0.02 ms), and production's own kv_len histogram says why. CLOSED.

`megamoe-eplb-c128-segplan`, gates clean (`accept len` 3.77 both sides,
cuda-graph replay 100 %, `0/10,648` errors), env verified in the launch log
(`SGLANG_MLA_SEG_PLAN=1`, `SEG_MAX=16`, `SEG_WG_MULT=4`, `HCA_KV_SPLITS=4`):
matched-bs weighted **−0.02 ms** (19 cells, w=2961), aggregate 114.12 → 114.28.
Pre-registered −1 to −3 with falsification at −0.5, so: falsified, and the plan
was definitely active.

**The reason is a number we already had and I failed to look up first.** The
`SGLANG_MLA_KVLEN_STATS` probe arm (`megamoe-eplb-c128-kvlenprobe`) recorded
production's actual per-batch kv_len distribution:

```
kvlen mean/p50/p99/max/min: 1192/1148/1492/1901/128
kvlen straggler: 0.373            # (max - mean) / max
```

The microbench's `real_lens` models **p50 1,090 / p99 5,000 / max 5,000, mean
1,203, straggler 0.759**. Same mean, but production's longest token is **1.6x
the mean** where the harness's is **4.2x**. The plan's entire value is closing
the max-vs-mean gap per CTA, so it was tuned against a spread production does
not have.

And at this CTA count the schedule already absorbs 1.6x: ~96 tokens × 2 head
tiles × 4 splits ≈ 768 CTAs on 256 CUs, i.e. three CTAs deep per CU, so a
1.6x-longer CTA is hidden by the other waves instead of setting the finish
time. That is also why split-K=4 was enough.

**Corrections this forces to the Phase 3 write-up above:**

- "A perfect straggler fix is worth 52.5 % (bs=12) / 71.4 % (bs=14)" and the
  bf16 "plan −35 %" numbers are properties of **`real_lens`, not of
  production**. They stand as measurements of the kernel's behaviour under a
  4.2x spread and nothing more.
- The occupancy figures (12.3 % ragged vs 20.9 % flat) were taken on the same
  synthetic spread, so the *gap* between them is likewise not a production
  number. The absolute low occupancy is still real — it is visible in the
  production trace too.
- **`analysis/mla_tail_bench.py`'s distribution should be re-derived from the
  probe** (mean 1,192, max 1,901) before it is used to price anything else. Its
  header already warns it models "the tail"; the tail it models is ~2.6x wider
  than the node's.

**Disposition:** the port stays, env-gated and **off by default**. It is
correct (GSM8K 0.939 vs 0.937), costs nothing when enabled (−0.02 ms), uses 3x
less partial memory than the split path, and is the right mechanism for a
workload whose dispersion actually is wide — which this one, measured, is not.
Do not spend another arm on it without first showing a straggler ratio well
above 0.373.

**Method lesson, and it is the second one today:** both falsified arms
(`block_k`, the segment plan) came from a microbench whose *configuration* did
not match production — first the fp8-vs-bf16 path, then the length
distribution. The trace and the probe had the production numbers all along.
**Check the harness against a production artefact before pre-registering a
prediction from it.** Pre-registered **−1 to −3 ms/step** at
matched bs vs `megamoe-eplb-c128-hcasplit4`; **falsification: better than
−0.5 ms**, and the next step then is a kv_len histogram from the arm's own
`server.log` (the microbench's p50 1,090 / max 5,000 may be harsher than
production), **not** more tuning.

**STATUS OF THE EARLIER (pre-fix) NUMBERS — DO NOT QUOTE THE FIRST RUN.** It compiles and runs
(one call returned 373.6 µs against 399.0 µs for split-K=4 at `block_k=16`), but
that run happened **while the bk16 arm was serving on all 8 GPUs**, so it is
contaminated by contention, and **numerics have not been checked at all**.
`/shared_nfs/kk/mla_plan_check.py` does both — same tensors, both paths in one
process (the module flag is read per call), then a `SEG_MAX × wg_mult` sweep
against the split path and against the flat-at-the-mean floor. It OOM'd on the
busy node; rerun it the moment the arm finishes:

```bash
SGLANG_MLA_FP8_BLOCK_K=16 HIP_VISIBLE_DEVICES=0 \
  python3 /shared_nfs/kk/mla_plan_check.py --bs 12
```

Gate before anything else: **relL2 vs the split path must be ~1e-3**, the same
family as the split-K and bk16 checks. A faster wrong kernel is the failure mode
this line has already avoided twice (the fake-kvlen arm, the `-DECODE` trace).

#### Q-FOLD PROTOTYPE (2026-09-18) — KV reuse alone buys NOTHING. The lever is M, and M needs LDS

`/shared_nfs/kk/mla_qfold_proto.py`: a second Triton split kernel whose CTA owns
**one request's 7 draft tokens × BLOCK_H heads** instead of one token × 64 heads,
so the request's KV is read once for all 7 (measured reuse factor **6.98x**) and
the MFMA M dimension is composed as `Q_PAD × BLOCK_H`. Production shape
(bs=12, q_len=7, lognormal σ=0.113 per the probe, bf16, no scales).

**Correct:** merged-output relL2 **2.21e-04**, max_abs 2.41e-04, finite. (Partials
are NOT comparable between the two — they segment the KV differently, per token
vs per request — so the gate is the FlashAttention-merged output. Comparing
partials first showed a bogus 1.2e-01 and cost a debug cycle.)

| variant | M | µs | vs shipped |
|---|---:|---:|---:|
| shipped, 1 token/CTA | 64 (1×64) | 145.8 | — |
| q-folded, BLOCK_H=4 | 32 (8×4) | 223.1 | +53.0 % |
| **q-folded, BLOCK_H=8** | **64 (8×8)** | **141.5** | **−3.0 %** |
| q-folded, BLOCK_H=16 | 128 (8×16) | 187.3 | +28.4 % |

Counters at matched M=64 (`rocprofv3 --pmc OccupancyPercent MfmaUtil`):
occupancy **17.8 → 19.3 %**, MfmaUtil **11.8 → 13.1 %**. The mechanism moved in
the right direction, by about as much as the time did.

**So the sub-hypothesis "the redundant KV reads cost us 7x traffic" is
FALSIFIED as a *time* claim.** Removing 6.98x of the reads bought 3 %. That is
consistent with everything else measured: HBM sits at ~1 % of peak, and 96
tokens × 1,192 entries × 1,024 B ≈ 117 MB fits inside MI355X's 256 MB
Infinity Cache, so the duplicate reads were already being served without
touching HBM.

**What the table actually says:** the only knob that matters is M, and M cannot
grow in registers — at M=128 the q tile is 128 × 512 bf16 = 128 KB and the
kernel loses 28 %. This is the same wall `BLOCK_H=128` hit earlier (909 vs
398 µs). **The next step is therefore LDS staging of the KV tile** (`LDS = 0`
today) so the KV lives once per CTA and the q rows can grow to M=256+ without
VGPR pressure — that is the standard CDNA flash-decode layout and the only path
left to the 3.3x B200 per-call gap. Q-folding is a *prerequisite* for it (it is
what supplies the extra M rows), not a win on its own.

Integration note if this ever ships: the kernel needs per-request index slices
plus per-token lengths. Today `kv_indptr` is per token and each token owns its
own slice of `kv_indices`; `runtime.decode_qo_indptr` is deliberately an
`arange` ("one q token per sequence"), and its own comment points at
`cu_seqlens_q` as the per-request grouping that exists but is not passed down.
The prototype builds both layouts so the two kernels see equivalent data.

Reference material for the rewrite:
`flydsl/01-playbooks/FLYDSL_KERNEL_OPT_PLAYBOOK.md`,
`flydsl/00-foundations/03_mfma_layout.md` (MFMA layouts),
`flydsl/00-foundations/02_memory_layout.md` (LDS staging),
`flydsl/02-profiling/trace_profiling.md`.

**Phase 4 — `prepare`'s own parallelism. Bounded, and sequenced last on
purpose.**

grid 30 of 256 CUs, but the real dispatch work is **≤4.6 ms/step** (the busiest
rank's 74.6 µs/call bounds it). Do this *after* Phase 1: if overlap works, those
226 CUs are no longer free, and widening `prepare` then competes with the
overlapped work instead of adding to it.

**Explicitly NOT on the plan, with reasons:**

- Load balancing / `total_tokens` — skew 2.08x → 1.69x bought 0 to +2.4 %.
  Barrier wait is not cashable.
- `comm` +4.82 ms — one kernel, `ep_combine_intranode_0`, classed a WAIT
  (r = −0.597). It is exposed idle, not a slow collective; it belongs to Phase 1.
- `gemm` — equal to B200 (−0.38 ms). There is no dense-GEMM gap.
- A HIP port of `mask_topk_ids` — buys ~0 for this purpose.

**Parallel track, not part of the gap work:** PR
[#39968](https://github.com/sgl-project/sglang/pull/39968) is out of draft but
has 13 failing checks (`call-gate / pr-gate`, `stage-a-unit-test-mlx`,
`stage-a-test-1-gpu-xpu`, `base-b-test-2-npu-a3`,
`stage-c-test-large-8-gpu-amd`, plus `*-finish` aggregators). The change is ~20
lines and env-gated, so mlx/xpu/npu failures are unlikely to be ours — triage
before touching code.

### Why TP-3 and TP-6 are missing — a NEW trap, not the old one

Not the `num_steps=40` rank death. Both ranks logged `Stop profiling-DECODE...`
at 02:11:52 and were still inside `_stop_profile` writing the trace when, 3 s
later at 02:11:55, four peers died together in
`prefill_delayer.py:373` → `distributed_c10d.py:4056` with
`gloo ... Connection closed by peer`. The trace EXPORT on a rank outlives the
peers' tolerance in the DP-attention prefill-coalescer collective, so the
exporting ranks get killed by the cascade before they finish writing.

Consequence: 6 ranks is enough to correlate (n=12 rank/bs cells) but the
capture is NOT a cross-rank picture. If a full 8/8 decode capture is needed,
the thing to fix is the export window, not `num_steps` — e.g. drop to 16 steps
so the export is smaller, or raise the gloo/`PrefillDelayer` timeout for the
capture run. Note `sglang-prefill-coalescer` is the skill covering that code.

### Previous block (2026-09-18 09:4x) — the num_steps=12 attempt captured PREFILL

**Status:** `run_trace_steady.sh` with `STEPS=24` (PID 1362174, started
2026-09-18 01:42 UTC) is driving the node with no operator attached.

```bash
cat /shared_nfs/kk/trace_summary_s24.md        # verdict lands here
tail -20 /shared_nfs/kk/trace_run_s24.log      # runner progress / gates
ps -eo pid,args | rg 'run_trace_stead[y]'      # still alive?
```

**Read the `TARGET_VERIFY steps captured` line at the top of the summary
FIRST.** If it is 0 the capture landed on prefill again and nothing else in
that file addresses decode.

### What attempt 1 (num_steps=12) established, and what it did not

`/shared_nfs/kk/pr35619/trace_hcasplit4_steady`, 8/8 ranks, gate stopped at
bs=13 / tok-per-req 151,738 — **both earlier failures fixed**: every rank
flushed, and the operating point is inside the reference's bs 9-20.

**But the trace holds `EXTENDx1` and zero `TARGET_VERIFY` steps.** One 1,076 ms
chunked-prefill step filled the whole 12-step window, so `prepare_wait.py`
returned `no full TARGET_VERIFY steps found` and the MoE question is still
open. This is exactly the trap `analysis/MI355X_CAPTURE_PROMPT.md:94` records:
the file is named `-DECODE` regardless, so **the step annotation is the only
evidence**. `run_trace_steady.sh` now extracts that count itself.

`STEPS` is two-sided: 40 kills a rank mid-capture (4/8 flushed), 12 is too
small to contain a decode step. 24 is the first value tried between them; a
full `TARGET_VERIFY` step is 74-120 ms, so it should span decode easily.

Attempt 1 is still useful for the prefill side: moe 38.8 %, attn 21.1 %,
gemm 17.6 %, comm 15.9 %, with `megamoe_prepare_compact` at 75.39 ms/step over
61 calls.

### The question, still unanswered

MLA got ~7.4 ms/step cheaper. Does `megamoe_prepare_compact` — the first MoE
kernel — shrink with it, or is it a cross-rank barrier that absorbs the slack?
**On the reference side it is unambiguously a WAIT:** r = −0.942, spread
102.9-325.0 µs, alongside `ep_combine` (r = −0.665), while `stage1` (+0.672),
`stage2` (+0.943) and `mla_fused` (+0.961) all track own work. The split-K side
has no comparable number yet. If `prepare` absorbs the win, MoE dispatch is
the next straggler and that is the next line of work.

Result dirs from these runs (`...-trace2`, `...-trace-s24`) are **not valid
throughput arms** — the script kills the benchmark as soon as the ranks flush.
Keep them out of `summary_table.py`.

**Previous block (2026-09-18 06:5x):** the overnight chain (PID 1311610)
completed 2026-09-17 18:37 UTC, all three arms clean. Node was idle before the
trace run above started.

**Results, matched-bs n-weighted (weight = min(n_ref, n_test) per common bs):**

| arm | result dir | Δstep vs b200aligned | agg p50 |
|---|---|---|---|
| c128 split4 **replicate** | `megamoe-eplb-c128-hcasplit4-rep2` | **−7.42** (was −7.50) | 121.76 → 114.55 |
| c256 split4, `total_requests` (clean) | `megamoe-eplb-c256-hcasplit4-totalreq` | **−4.29** | 156.19 → 151.87 |
| c128 split4 + `total_tokens` (2x2 cell) | `megamoe-eplb-c128-hcasplit4-totaltokens` | **−6.11** | 121.76 → 115.50 |

**The three things these arms were run to decide, now decided:**

1. **Replicate spread on matched-bs step time is 0.07 ms** (direct rep2-vs-orig
   comparison over bs 2-20, w=3033; aggregate p50 differs by 0.43 ms). The
   −7.50 ms split-K claim is reproducible to ~1 %. This is the first
   step-time replicate number we have; the 5.67 % figure was throughput-only
   and does not apply to Δstep.
2. **Clean c256 split-K is −4.29 ms, not −5.37.** With `total_requests` held
   fixed so split-K is the only variable, the gain is −4.29 (−2.7 % of a 156 ms
   step). The earlier −5.37 was split-K *plus* balancer; the direct pair
   (`c256-hcasplit4-totalreq` → `c256-hcasplit4-totaltokens`) puts the balancer
   at −1.09 ms **in the presence of split-K**. So c256 really does gain less
   than c128 (−4.29 vs −7.42), consistent with the grid-filling explanation in
   the bs-sweep section below.
3. **`total_tokens` costs +1.12 ms on top of split-K at c128** (direct pair,
   w=2870), i.e. it is null-to-slightly-negative, matching its standalone
   +0.78 ms. Note the sign flip vs c256, where the same balancer helped by
   −1.09 ms; the interaction is concurrency-dependent and neither cell is
   outside ~1 ms, so **do not quote the balancer as a win**.

Recompute/verify with:

```bash
python3 /workspace/claude-skills/agentx/analysis/decode_stats.py \
  /workspace/results/megamoe-eplb-c128-b200aligned/server.log \
  /workspace/results/megamoe-eplb-c256-b200aligned/server.log > /shared_nfs/kk/ref_stats_20260918.txt
python3 /shared_nfs/kk/matched_bs.py /shared_nfs/kk/ref_stats_20260918.txt /shared_nfs/kk/chain_summary.md \
  'c128-rep2=c128-b200aligned/,c128-hcasplit4-rep2/'
```

`matched_bs.py` (new, written 2026-09-18) parses `decode_stats.py` blocks and
prints the n-weighted matched-bs delta. Validated by reproducing the doc's
−5.37 for `c256-hcasplit4-totaltokens` exactly.

**Pass criteria met:** pre-registered −8.00 ms with falsification at −2; c128
lands −7.42/−7.50 across two runs, c256 −4.29. Nothing falsified.

### Three traps the chain script encodes — keep them if you rewrite it

1. **The agentx launcher never exits.** It leaves the server running after the
   benchmark ends, so completion is detected from the **log marker**
   (`Validated aiperf request error rate`), not from process exit.
2. **`pgrep '^sglang::'` misses the parent.** `python3 -m sglang.launch_server`
   gets reparented to init and respawns tokenizer workers, holding the port for
   hours. Kill it FIRST — and match it as `launch_serve[r]` so the pattern does
   not match your own command line, which self-killed a shell today.
3. **VRAM sits on a multi-GB plateau for 20-35 min** after the processes die.
   Gate on the 0.28 GB baseline confirmed twice a minute apart; launching on the
   plateau OOMs at cuda-graph capture.

### Still open — this is the next action list

1. **Mark PR [#39968](https://github.com/sgl-project/sglang/pull/39968) ready
   for review** — it is the only thing blocking CI, and the measurement backing
   it is now replicated.
2. **GSM8K A/B** (see env trap below).
3. **Kernel trace** at the right operating point.


- **GSM8K is NOT done.** `gsm8k_ab.sh` failed: hand-rolling the server from
  `sglang_command.txt` is not enough, because MegaMoE is turned on by launcher
  **env**, not by `--moe-a2a-backend megamoe` alone. The missing set includes
  `SGLANG_AMD_USE_FLYDSL_MEGA_MOE=1`, `SGLANG_AMD_FLYDSL_MEGA_QUANT=a8w4`,
  `SGLANG_USE_AITER=1`, `SGLANG_MOE_PADDING=1`; without them the MoE falls into
  the Triton path and dies on
  `fused_moe_triton_kernels.py:863 assert triton.cdiv(...) == B_scale.shape[-2]`.
  Copy the env block out of a launch log. Use `--max-new-tokens 8192` (DSv4
  reasoning CoT truncates at 2048 and scores 0) and do **not** set
  `SGLANG_SIMULATE_ACC_LEN`, which fakes MTP acceptance.
- **The kernel trace** still has no usable capture; see the trace section below.
- **PR [#39968](https://github.com/sgl-project/sglang/pull/39968)** is filled in
  but is a **draft**, and every `pr-gate` check fails on the step literally named
  `Block draft PR`. Marking it ready for review is what unblocks CI.

---

## EVERY ARM RUN ON 2026-09-17, ONE TABLE

All at mem-fraction 0.85, chunk/rank 8192, MegaMoE+EPLB EP8, DP8, MTP, tp8.
`Δstep` is the **n-weighted matched-bs** log-implied step delta against the
same-concurrency b200aligned reference — that is the claim. The aggregate
columns are context, and at these sizes the throughput differences sit inside
the 5.67 % replicate spread.

| mode | conc | step p50 | **Δstep** | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU-tier | GPU pool | ISL mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref (b200aligned) | 128 | 121.76 | — | 36,995 | 29.5 | 33.9 ms | 11.07 s | 4.01 s | 96.00 % | 95.61 % | 91 % | 108,371 |
| total_tokens | 128 | 122.32 | **+0.78** | 39,029 | 29.2 | 34.3 ms | 9.61 s | 4.03 s | 96.10 % | 95.68 % | 100 % | 109,612 |
| fake-kernel **[INVALID]** | 128 | 103.59 | **−17.07** | 43,291 | 35.0 | 28.6 ms | 8.03 s | 3.44 s | 96.16 % | 95.73 % | 91 % | 110,409 |
| **HCA split-K=4** | 128 | 114.12 | **−7.50** | 40,346 | 31.4 | 31.8 ms | 9.31 s | 3.75 s | 96.12 % | 95.70 % | 84 % | 110,700 |
| ref (b200aligned) | 256 | 156.19 | — | 57,516 | 22.8 | 43.9 ms | 26.22 s | 15.69 s | 96.41 % | 95.52 % | 78 % | 118,262 |
| **split-K=4 + total_tokens** | 256 | 150.51 | **−5.37** | 58,508 | 23.2 | 43.1 ms | 26.30 s | 15.34 s | 96.39 % | 95.49 % | 69 % | 118,195 |

Three more arms landed overnight 2026-09-17 (chain, see top block). All six
split-K arms are now rows in `summary_table.py`, so the house-format aggregate
columns regenerate with `python3 /workspace/claude-skills/agentx/summary_table.py`
(full output kept at `/shared_nfs/kk/summary_20260918.txt`). `Δstep` remains the
claim; every throughput move below is inside the 5.67 % replicate spread.

| mode | conc | step p50 | **Δstep** | tok/s/chip | P90 intvty | ITL p90 | TTFT avg | TTFT p50 | cache hit | GPU-tier | GPU pool | ISL mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **split-K=4, replicate** | 128 | 114.55 | **−7.42** | 39,793 | 31.5 | 31.8 ms | 9.59 s | 3.76 s | 95.6 % | 95.6 % | 88 % | 109,138 |
| **split-K=4 + total_tokens** | 128 | 115.50 | **−6.11** | 40,397 | 31.5 | 31.8 ms | 9.26 s | 3.73 s | 95.7 % | 95.7 % | 97 % | 110,392 |
| **split-K=4, total_requests (clean)** | 256 | 151.87 | **−4.29** | 58,866 | 23.5 | 42.6 ms | 25.28 s | 15.01 s | 95.9 % | 95.6 % | 77 % | 118,817 |

Gates for all three: `errors=0`, `records_error_dropped=0`, cuda-graph replay
100 %, `accept len` 3.77, ISL within 1 % of the matching reference, cache hit
flat. `cache hit` here is `overall_cache_hit_rate` (the c256 rows also carry a
0.3-0.4 pp CPU tier), and the earlier table's 96.0/96.1 % figures are the same
runs read off a different key — do not mix the two columns across tables.

Reading notes, in order of how easy they are to get wrong:

- **The fake-kernel row is marked INVALID for every column except `Δstep`.**
  It clamps kv_len to 128, so the model emits garbage; OSL, queueing, TTFT,
  throughput and cache hit are all meaningless. It exists only as the
  **ceiling**: −17.07 ms is what deleting ~95 % of MLA buys, and it is the
  denominator the split-K result should be scored against.
- **split-K captures 44 % of that ceiling at c128** (−7.50 of −17.07) for ~20
  lines of code.
- **`total_tokens` is +0.78 ms, i.e. null**, confirming the earlier c128
  finding at full weighting. Its tok/s/chip looks +5.5 % better, which is
  exactly why throughput is not the metric here.
- **c256 gains less** (−5.37 ms on a 156 ms step, −3.4 %, versus −6.2 % at
  c128) and none of it reaches end-to-end throughput (+1.7 %, inside noise).
  **Now explained and measured — see the bs sweep below.**
- The c256 arm changed **two** variables against its reference (balancer and
  split-K), so it cannot attribute between them on its own. Given the c128
  `total_tokens` null, the −5.37 is presumed mostly split-K.
- Earlier prose quoted −7.67 ms for c128 split-K from bs 8-20; the −7.50 here
  is the same computation over every common bs cell and supersedes it.

### WHY c256 GAINS LESS — measured, and it is the grid filling up, not dispersion

`/shared_nfs/kk/hca_bs_sweep.log`, HCA shape (median 1,300, cap 5,000), % vs the
heuristic's splits=1:

| bs | T | base CTAs | dispersion | split 2 | split 4 | split 8 |
|---|---|---|---|---|---|---|
| 14 | 98 | 196 | 0.699 | −35.1 | −50.3 | **−54.2** |
| 18 | 126 | 252 | 0.707 | −25.6 | −41.8 | **−44.5** |
| 21 | 147 | 294 | 0.701 | −24.3 | −34.1 | **−36.8** |
| 24 | 168 | 336 | 0.706 | −13.8 | **−24.4** | −22.7 |
| 28 | 196 | 392 | 0.713 | −16.8 | **−24.3** | −21.9 |

**Dispersion is flat at ~0.70 in every row, so this is not a change in the
straggler — it is the base grid filling the device.** The target is
1.5 × 256 = 384 CTAs; the benefit decays as `T × 2` approaches it and flattens
once past it (bs≈28). Split-K helps by giving idle CUs something to do, and
past saturation there are none.

This predicts the c256 result quantitatively. c256 runs at bs p50 21 against
c128's 12-14, i.e. a per-call win of 34.1 % instead of ~50.3 %. Scaling the
c128 step delta by that ratio gives −7.50 × 34.1/50.3 = **−5.08 ms predicted
against −5.37 measured** — within 6 %, from a microbenchmark to an end-to-end
arm. (The step composition also differs between the two, so treat the agreement
as strong support rather than proof.)

**It also re-confirms splits=4 as the right static choice:** at bs≥24, 8 is
WORSE than 4 (−22.7 vs −24.4, and −21.9 vs −24.3). 4 degrades gracefully across
the whole batch range; 8 only wins at small batch.

---

## CONTINUE HERE (2026-09-17 16:3x UTC+8) — layer-aware split-K WORKS: −7.50 ms/step for a ~20-line change. Ship it, then trace it

**Node state:** arm complete, server and the orphaned launcher both killed,
zero `sglang::`, ports clear, VRAM draining from 32 GB/GPU.

### THE RESULT — predicted −8 ms, measured −7.67 ms

`megamoe-eplb-c128-hcasplit4` vs `megamoe-eplb-c128-b200aligned`, log-implied
step p50 at matched `bs`. The fake-kernel column is the ceiling (MLA ~deleted):

| bs | ref ms | split4 ms | Δ | ceiling Δ | captured |
|---|---|---|---|---|---|
| 10 | 116.45 | 111.55 | −4.90 | −13.72 | 36 % |
| 12 | 119.90 | 112.96 | −6.94 | −15.15 | 46 % |
| 14 | 123.83 | 117.27 | −6.56 | −18.19 | 36 % |
| 16 | 130.40 | 116.82 | −13.58 | −20.03 | 68 % |
| 18 | 134.65 | 121.83 | −12.82 | −22.87 | 56 % |

**n-weighted mean over bs 8-20: −7.67 ms/step** against a pre-registered −8.00,
with the falsification threshold at −2. The synthetic microbench transfers.

Aggregate: implied step p50 121.76 → 114.12 ms, ITL p50 32.28 → 30.24 ms,
`accept len` 3.78 → 3.77, cuda-graph replay 100 % both, 46/46 flags identical,
`0/10,488` errors. Output is CORRECT here (unlike the fake arm), and gen
tput/rank moved 389.36 → 393.73 tok/s — **inside the 5.67 % replicate spread,
so do not quote it**; the step-time result is the claim.

Per-cell Δ is noisy (bs=19 only −2.20 with n=41, bs=16 −13.58 with n=173),
which is why the weighted mean is the headline and single cells are not.

### THE TRACE ATTEMPT — partial, and at the WRONG operating point. Do not quote it

`/shared_nfs/kk/pr35619/trace_hcasplit4`, triggered 17:04 by
`trace_trigger.sh`. **Two independent problems, both fixable, neither fatal to
the arm result above.**

1. **Only 4 of 8 ranks flushed** (TP-1..4). At `num_steps=40` a rank died
   mid-capture and the rest cascaded through NCCL heartbeat / TCPStore reset —
   the exact risk `MI355X_CAPTURE_PROMPT.md` records for 40 steps, except here
   it landed before all ranks wrote.
2. **The window caught bs=5-7, while the reference trace is bs=9-20.** A
   120 s settle after warmup is not steady state: the profiling phase had only
   just begun and the batch was still ramping. Nothing at bs=5-7 can be
   compared against the reference at matched bs, and matched bs is the whole
   method.

What the partial capture still says, qualitatively:

- **`prepare` is STILL a wait, and if anything more so: r = −0.969** (reference
  −0.942). Its spread narrowed to 98.3-183.2 µs from the reference's
  102.9-325.0, which is the direction a smaller straggler predicts — but bs
  differs, so treat it as a hint, not a measurement.
- **`ep_combine` is also a wait** (r = −0.959), as before.
- **`mla_split` = 61 calls/step and `mla_fused` = 0.** Not a bug and not
  evidence the override leaked to CSA: at bs=5-7 the occupancy heuristic
  already picks splits=4 on its own (T=35, 70 base CTAs against a 384 target),
  so every stream takes the split path at that size. It does confirm the split
  path runs and is captured cleanly under cuda graph.

**Re-capture recipe (the two fixes):** settle to真 steady state — wait for
tok/req to reach ~150k rather than a fixed 120 s, i.e. 10-15 min into the
profiling phase — and drop to `num_steps=8-16` so all 8 ranks flush before any
instability. `trace_trigger.sh` takes `SETTLE` as an env var and `num_steps` as
`$3`.

### NEXT — in this order

1. **Re-capture the trace** with the two fixes above, then `prepare_wait.py`
   against the reference at MATCHED bs. The question is unchanged: how much of
   the −7.67 ms came from MLA per-call time versus the `prepare` wait
   collapsing, and whether `prepare` stays anti-correlated once the straggler
   is smaller. If it does, the imbalance has a second source and that is the
   next line of work.
2. **Then decide 4 vs 8 on real shapes.** The microbench said 4 is the robust
   pick and it was right about the magnitude, but the arm never tested 8 in
   situ. One arm with `SGLANG_MLA_HCA_KV_SPLITS=8` answers it, and the
   partial-buffer cost (205 MB vs 103 MB at bs=14) is the thing to watch.
3. **Upstream it.** The change is ~20 lines across three files and is
   env-gated; it is the first shippable win of this line.

### The change, for the record

`_kv_splits_for_stream(compress_ratio)` in `paged_decode.py` → threaded through
`runtime.decode(kv_splits=...)` → set at the one call site that knows the
stream (`deepseek_v4_backend_hip_radix.py`). HCA (ratio 128) gets splits=4;
SWA and CSA keep the occupancy heuristic. `SGLANG_MLA_HCA_KV_SPLITS=0` restores
the old behaviour with no code edit.

**Node state during the run (kept for reuse):** launched 15:11 UTC+8, PID
1203690, launch log `/shared_nfs/kk/hcasplit4_c128.log`, results
`/workspace/results/megamoe-eplb-c128-hcasplit4/`, `DURATION=3600`.

**The change:** `_kv_splits_for_stream(compress_ratio)` in `paged_decode.py`,
threaded through `runtime.decode(kv_splits=...)` from the one call site that
knows the stream (`deepseek_v4_backend_hip_radix.py`). HCA (ratio 128) gets
splits=4; SWA and CSA keep the occupancy heuristic. Env-tunable, 0 restores the
old behaviour, so the A/B needs no code edit.

Pre-flight passed (`analysis/layer_split_validate.py`, GPU0, ~8 s,
`/shared_nfs/kk/layer_split_validate.log`): stream gating `[None, None, 4]`,
the override reaching the kernel (heuristic would have picked 1), **numerics
against the fused path relL2 2.46e-03**, and CUDAGraph capture + replay on the
split path — which is a different code path (partial buffers + reduce kernel)
from the fused one production has been capturing.

**Prediction, written before the run:** HCA is ~80 % of MLA time, rank 1's MLA
is 20.17 ms/step, splits=4 takes ~50 % off the HCA part ⇒ **bs=14 should go
123.83 → ~116 ms (−8 ms)**, about 44 % of the fake-kernel arm's −18.19 ms
ceiling. **Under −2 ms means the synthetic microbench does not transfer to the
real per-layer shapes** — then investigate why, do not tune the split count.

Unlike the fake-kernel arm the output is CORRECT here, so throughput, TTFT and
cache hit are all readable; still lead with matched-bs step time for
comparability.

### Also fixed in this session (rides along with this arm)

`metrics_reporter.py`'s `kvlen straggler` now reports per-layer
`(max−mean)/max` averaged over layers, instead of dividing a layer-averaged
mean by the across-layer max. That was recorded debt #1. Inert while
`SGLANG_MLA_KVLEN_STATS=0`.

### Cleanup trap that cost a launch gate today — `pgrep '^sglang::'` IS NOT ENOUGH

After the fake-kernel arm, VRAM returned to the 0.28 GB baseline and
`pgrep '^sglang::'` returned zero, yet port 8889 was still held. The parent
**`python3 -m sglang.launch_server`** had been orphaned to init and was
respawning tokenizer workers for two hours; its command line does not start
with `sglang::`, so every check in the existing checklist missed it. Kill the
parent FIRST:

```bash
pgrep -af 'sglang.launch_server'     # list before killing
kill -9 <launch_server pid>; sleep 5
for p in $(pgrep '^sglang::'); do kill -9 "$p"; done
```

---

## Previous block (2026-09-17 14:3x UTC+8) — the fake-kernel arm PASSED. The MLA line is confirmed end to end; next is the tail microbench

**Node state:** arm complete, server killed, VRAM draining from a 32 GB/GPU
plateau at the time of writing. Wait for the 0.28 GB cliff before any GPU work.

### THE RESULT — "MLA faster ⇒ step wall drops" is now tested under intervention, and it holds

`megamoe-eplb-c128-fakekvlen` vs reference `megamoe-eplb-c128-b200aligned`,
log-implied step p50 at matched `bs` (`analysis/decode_stats.py`):

| bs | ref ms | fake ms | Δ |
|---|---|---|---|
| 8 | 111.11 | 97.58 | −13.53 |
| 10 | 116.45 | 102.73 | −13.72 |
| 12 | 119.90 | 104.75 | −15.15 |
| **14** | **123.83** | **105.64** | **−18.19 (−14.7 %)** |
| 16 | 130.40 | 110.37 | −20.03 |
| 18 | 134.65 | 111.78 | −22.87 |

**The prediction was written before the run and it landed: bs=14 was predicted
at ~104 ms from −19.5 ms of "MLA free", and measured 105.64.** The
falsification threshold was a drop of only 3-5 ms; the measured drop is 18.19.
So the floor does **not** grow and no new serialisation appears when MLA
shrinks — the component-level model of the step composes, and the trace
inversion's −19.54 ms for "MLA free" is trustworthy as an upper bound.

Δ grows monotonically with `bs` (−13.5 at 8 to −22.9 at 18), which is what MLA
cost scaling with batch predicts and is a second, independent consistency check
on the attribution.

**Not confounded:** `accept len` 3.78 vs 3.78, cuda-graph replay 100 % of steps
in both, KV pool identical at 12,077,312, 46/46 flags identical, aiperf
`0/11,284` errors. The `bs` mix did shift (running-req p50 11 vs 12) exactly as
expected from garbage output, which is why only matched-`bs` rows are quoted.
Aggregate ITL p90 34.54 vs 39.60 ms moves the same way but is part composition,
so do not quote it as the effect size.

**What this does and does not license.** It licenses spending on MLA: the
ceiling for a perfect straggler fix is 59 % of this 18 ms, i.e. ~11 ms/step at
bs=14. It does **not** say any realisable kernel reaches that — the clamp
removed 95 % of the work, which no straggler-aware design can.

### THE TAIL MICROBENCH IS DONE TOO, AND IT OVERTURNS "split-K is the wrong tool"

`analysis/mla_tail_bench.py`, GPU0, logs `/shared_nfs/kk/mla_tail_bench.log`
and `..._csa.log`. Ragged vectors from LogNormal(median 1,000, p99 3,096); the
`--cap` flag switches between the two layer families.

**The two families are different kernels in all but name.** At bs=14:

| | CSA layers (`--cap 1152`) | HCA layers (`--cap 5000`) |
|---|---|---|
| dispersion (max−mean)/max | 0.19 | 0.77 |
| ragged, heuristic splits | 504.1 µs (splits=1) | 1884.2 µs (splits=1) |
| flat at the mean | 441.5 µs | 539.3 µs |
| straggler-fix ceiling | **12.4 %** | **71.4 %** |
| split-K=8 vs heuristic | **+5.1 % (loses)** | **−59.4 % (wins)** |

**`_kv_splits_heuristic` picks splits=1 for both, which is right for CSA and
catastrophic for HCA.** Plain uniform split-K captures −59.4 % of the HCA call
— most of the 71.4 % dispersion ceiling — with no new kernel at all.

**This retires the "uniform split-K is the wrong tool" verdict, and the reason
is a one-line correction to the premise.** That verdict rested on "the
heuristic reads only capture-time scalars, so it cannot tell ragged from
uniform shapes". True of the batch — but the thing that decides which regime a
layer is in is **`compress_ratio`, a static per-layer constant from
`config.json`, which IS known at capture time.** CSA is clamped to
`index_topk`+128 = 1152 and is nearly uniform; HCA is unclamped and is where
all the dispersion lives. So the discrimination the heuristic was said to be
incapable of is available for free.

Also: at bs=10 the heuristic picks splits=2 and split-K=8 still wins −33.6 %;
at bs=18, −51.6 %. The mis-selection is not a bs=14 artefact.

**Read the ratios, not the absolute µs.** The synthetic applies one aggregate
distribution to *every* call, whereas real layers alternate between the two
regimes, so 1884 µs/call is far above the in-situ 330.7 µs/call at bs=14 (and
the probe's own stats are layer-averaged). The dispersion here, 0.77, is also
above the 0.61 measured in production.

### THE kv_len x splits SWEEP IS DONE — pick 4, not 8

`/shared_nfs/kk/hca_split_sweep.log`, bs=14, HCA shape (cap 5,000), median
swept over the run's range (HCA kv_len ~ context/128: a few hundred early,
~1,300 at the 165-170k tok/req steady state). % is vs the heuristic's splits=1:

| median | p50 | dispersion | split 2 | split 4 | split 8 | split 16 |
|---|---|---|---|---|---|---|
| 200 | 211 | 0.78 | −30.0 | **−41.8** | −35.8 | −21.8 |
| 500 | 529 | 0.78 | −35.7 | −51.4 | **−51.9** | −45.1 |
| 1300 | 1377 | 0.70 | −34.8 | −50.1 | **−54.0** | −49.7 |
| 3000 | 3179 | 0.36 | −13.5 | −23.8 | **−27.0** | −22.4 |

Three things the sweep settles:

1. **Split-K never loses anywhere on the HCA shape** — the worst cell is still
   −21.8 %. The concern that a static choice must survive the low-kv_len early
   run does not bite. The losing case is the CSA shape (+5.1 %), which
   `compress_ratio` gates out.
2. **16 is always worse than 8.** The sweep's earlier upper bound was not a
   ceiling artefact; the optimum is interior.
3. **Take 4.** It wins outright at the low end (−41.8 vs −35.8) and gives ~93 %
   of 8's benefit at the steady-state operating point (−50.1 vs −54.0), for
   **half the partial-buffer memory**: `acc_partial` is
   `T x splits x h_padded x D x 4 B` = 103 MB at splits=4 against 205 MB at
   splits=8 for bs=14, and splits=1 allocates none at all (fused path). That
   memory is the real reason the heuristic is conservative, and it is charged
   inside the graph pool at `mem_fraction_static` 0.85.

Even at dispersion 0.36 split-K wins 27 %, so the lever is not narrowly tuned
to the high-dispersion assumption.

### NEXT ACTION — make `_kv_splits_heuristic` layer-aware, then re-measure

Cheapest first, in this order:

1. **Confirm the premise in code:** find where `compress_ratio` is available at
   the `_sparse_attn_v4_paged_decode_triton` call site
   (`deepseek_v4_backend_hip_radix.py:333` is where CSA gets clamped to
   `index_topk`) and thread it, or the resulting kv_len cap, into
   `_kv_splits_heuristic` in `paged_decode.py`.
2. **Pick splits from the cap, not from `T`/`H` alone:** unclamped (HCA) ⇒ 4
   (see the sweep above); clamped (CSA) ⇒ leave at the current choice, which
   the table above shows is already right.
3. **Arm it exactly like the fake-kernel arm** — same launcher template, same
   gates, same matched-`bs` read. Budget the expectation against the fake-kernel
   arm's −18.19 ms at bs=14: that is MLA reduced ~95 %, so a split-K fix on half
   the layers should be scored as a fraction of it, not against zero.
4. Only if that disappoints: per-sequence split or a persistent-CTA work queue.

### OPEN QUESTION — does the MegaMoE `prepare` wait shrink too? The arm does NOT answer it

Asked of `megamoe_prepare_compact_m32_dcu32_pcu1_pc384_qcu28qcap256_fov_runtime_dyn_tss12488_v13`.
The fake-kernel arm produced **server-log step times only, no trace**, so this
is not measured and must not be asserted.

What IS established, from `analysis/prepare_wait.py` on the reference trace:
`prepare` is **r = −0.942 ANTI-correlated** with the rank's own compute, spread
102.9-325.0 µs across 61 calls/step — i.e. it is a **wait**, not work. Rank 1 at
bs=14 has the slowest MLA (330.7 µs) and the shortest prepare (102.9); rank 3 at
bs=9 has the fastest MLA (117.5) and the longest prepare (325.0). The straggler
does not wait; everyone else waits for it.

Two consequences, and the second is the trap:

1. **Expect the spread to collapse, not merely shrink.** The clamp removes MLA
   on every rank, so MLA stops contributing to the cross-rank difference that
   `prepare` absorbs. Whether `prepare` then goes to ~0 or simply re-forms
   around the next-largest rank-varying kernel is exactly what is unmeasured —
   and "a new straggler appears" is the same failure mode the arm was built to
   test at step level.
2. **Do not add it to the 18.19 ms.** `prepare` is a wait that already sits
   inside the step wall. Rank 1's MLA is 330.7 µs x 61 = **20.17 ms/step**, and
   the measured step drop is 18.19 ms — i.e. the drop is already ~90 % of
   "delete the straggler rank's entire MLA". Counting a prepare reduction on
   top would double-count the same time.

**Cheap way to settle it:** a short fake-kernel arm with tracing on (recipe in
`analysis/MI355X_CAPTURE_PROMPT.md`), then `prepare_wait.py` on the new trace
against the reference. It also re-tests the anti-correlation, which is the real
claim: if `prepare` stays large and stays anti-correlated after MLA is gone,
the imbalance has a second source that no MLA work can fix.

### Housekeeping before the next timing arm

`SGLANG_MLA_FAKE_KVLEN` defaults to 0, so leaving it unset disables the clamp —
but the code sits in the **dirty, uncommitted** working tree of
`/sgl-workspace/sglang-MegaMoE` next to `mla_kvlen_stats.patch`. Re-run
`cmd_diff.py` and check the env echo for any future arm regardless.

### How the arm was built and validated (for reuse)

The intervention is `SGLANG_MLA_FAKE_KVLEN=128`: `_fake_clamp_indptr` in
`paged_decode.py` rebuilds the indptr device-side as a compacted cumsum of
`clamp(len, 128)`, so every token reads at most 128 entries. It is the upper
bound of any MLA work, straggler-aware or otherwise. Offsets only ever shrink,
so `kv_indices` is never read out of bounds; the output is garbage.

Pre-flight passed before launch (`analysis/fake_kvlen_validate.py`, GPU0, ~7 s,
log `/shared_nfs/kk/fake_kvlen_validate.log`): indptr arithmetic exact on a
hand-computed case, eager cost at the production distribution
**1891.7 → 90.9 µs/call (0.05x)**, and a local `torch.cuda.CUDAGraph` capture +
replay around the real kernel succeeds with finite output. That third check is
the one that matters — launch #4 died with `hipErrorStreamCaptureUnsupported`.

**Verdict rule, fixed before the run.** Compare **log-implied step ms at matched
bs** and ITL p90 only — never throughput, TTFT, OSL or cache hit, all of which
the garbage output invalidates. Reference `megamoe-eplb-c128-b200aligned`:
**bs=14 p50 = 123.83 ms, n=247** (`analysis/decode_stats.py`). MLA free is
−19.5 ms on the pure-decode scale, so **bs=14 should land at ~104 ms**. A drop
of only 3-5 ms falsifies the whole MLA line, and then neither the microbenchmark
nor any kernel work should proceed.

**Read it with:**

```bash
cd /workspace/claude-skills/agentx
python3 analysis/decode_stats.py /workspace/results/megamoe-eplb-c128-fakekvlen/server.log
python3 arm_report.py /workspace/results/megamoe-eplb-c128-b200aligned \
                      /workspace/results/megamoe-eplb-c128-fakekvlen
```

### After it finishes, in this order

1. Read the verdict above and write it into `exchange/FINDINGS.md`.
2. Only if the step moved: the microbench at the real distribution (SECOND
   ACTION below), which is still unrun — `mla_microbench.py` sweeps kv_len
   ≤2048 while the real max is ~5,000. Needs 1 GPU, 10 min, so it must wait for
   this arm to release the node.
3. Revert the fake clamp before any timing arm that is not this one:
   `SGLANG_MLA_FAKE_KVLEN` defaults to 0, so unsetting it is enough — but the
   code lives in the dirty working tree of `/sgl-workspace/sglang-MegaMoE`
   alongside `mla_kvlen_stats.patch`, uncommitted.

---

## Previous block (2026-09-17 13:0x UTC+8) — fresh-session handoff. MLA straggler confirmed; next action is the fake-kernel arm

**Node state:** nothing running, GPUs released, VRAM draining from the probe arm
(28 GB/GPU plateau at handoff — wait for the 0.28 GB cliff before any arm).

### What is settled today, and must not be re-litigated

| line | verdict | evidence |
|---|---|---|
| **A `total_tokens`** | **closed — keep the flag, not an ITL lever** | skew 2.08x→1.69x but step time moved 0 to +2.4 % at matched bs. TTFT 11.07→9.61 s, cache unchanged, throughput +5.5 % is inside the 5.67 % replicate spread = null |
| **cross-rank balancing** | **falsified as a route** | the −7.35 ms "balanced ranks" counterfactual row is withdrawn |
| **MLA straggler** | **confirmed, within-batch** | 2,772 samples: at 150-200k tok/req the batch's p50 token reads ~1,000 entries, p99 reads ~3,100. Dispersion grows with KV (0.20→0.61) |
| **MLA roofline** | **1.2-3.1 % of both** | so the 4.06x gap to B200 is a design gap, not silicon |
| **uniform split-K** | **wrong tool** | wins ragged (419→355 µs), loses uniform (247→317); `_kv_splits_heuristic` reads only capture-time scalars so it cannot tell them apart |
| **C, copy kernels** | **attributed, low ceiling** | whole bucket 3.85 ms; largest single kernel 0.757 ms; the dead-write fix is ~0.25 ms (0.34 % of the step). Cleanup, not a main line |

### The structural fact that explains everything (found late, easy to miss)

**Two of the three index streams have different caps.** CSA (`compress_ratio 4`)
is clamped to `index_topk`=1024 (`deepseek_v4_backend_hip_radix.py:333`), so its
kv_len tops out at 1024+128. **HCA (`compress_ratio 128`) has no such clamp** —
it covers the whole context at 1/128 resolution, so its kv_len ≈ context/128 and
grows without bound (ISL p99 634,941 ÷ 128 = 4,960, matching the measured
per-layer maxima of 3,958-5,288). `config.json`'s `compress_ratios` alternates
128/4 per layer.

Consequences: `#full token` is **not** fully decoupled from per-step cost (it
drives the HCA layers); the within-batch straggler **is** the long-context
request on HCA layers; and a fix should target those layers, not all 61.

### NEXT ACTION — the fake-kernel arm. It is the decisive experiment

Everything above is component-level. The chain "MLA faster ⇒ step wall drops"
has never been tested under intervention, and the chain is what has been wrong
twice today. `floor` = 5.19 ms was measured with MLA unchanged; whether it grows
or another serialisation appears when MLA shrinks, no model can say.

**Design (single variable, capture-safe):** clamp per-token kv_len to a small
constant (128) inside `_sparse_attn_v4_paged_decode_triton`. Build the indptr
device-side (`cumsum` of a filled tensor) — **never write a host scalar into a
device tensor**, that is what aborted a launch today with
`hipErrorStreamCaptureUnsupported`. Validate under a local
`torch.cuda.CUDAGraph` capture+replay **before** launching, as the kv_len probe
now is.

**Quote only step-level numbers**: log-implied step time at matched `bs`, and
ITL p90. Not throughput, not TTFT — garbage output changes OSL and queueing.
`accept len` is **not** a confound: AgentX pins it
(`SGLANG_SIMULATE_ACC_LEN=3.77`, launcher :302-311).

**Prediction to falsify, written before the run:** MLA free takes the pure
decode step 74.02 → 54.48 ms, i.e. −19.5 ms. On the log-implied scale bs=14
should go **123.83 → ~104 ms**. A drop of only 3-5 ms kills the whole MLA line
and neither the microbenchmark nor the kernel work should proceed.

### SECOND ACTION — microbench at the real distribution (10 min, 1 GPU, no cliff needed)

Retires two debts and fills the wait for the VRAM cliff. `mla_microbench.py`
currently sweeps kv_len ≤2048 while the real max is ~5,000 — **the tail is
exactly where the win is and it has never been measured.** Three cases, ragged
vectors matching the measured distribution (p50 ~1,000, p99 ~3,100, max ~5,000):
ragged as-is; clamp-to-mean (identical total work, zero dispersion — the floor a
perfect straggler fix reaches); uniform split-K 1/2/4/8. Difference between the
first two is the **measured** achievable win per call.

### Two debts of mine, both recorded so they are not inherited silently

1. **The logged `kvlen straggler` field is the wrong statistic.** It mixes
   within-batch dispersion with across-layer dispersion (its `max` is across
   layers, its `mean` is layer-averaged). Use `(p99−mean)/p99`. Fixing it is one
   line in `metrics_reporter.py` (per-layer `(max−mean)/max`, then average) plus
   one arm.
2. **`mla_microbench.py`'s kv_len ceiling is 2048**, below the real maximum, so
   its absolute per-call figures understate the tail. The inversion's "implied
   kv_len 244-708" are 61-layer averages, not per-token lengths.

### Repro / tools (all no-GPU unless noted)

```bash
cd /workspace/claude-skills/agentx
python3 analysis/mla_counterfactual.py /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/copy_attrib.py        /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/kv_skew.py     /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens/server.log
python3 analysis/decode_stats.py /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens/server.log
python3 arm_report.py /workspace/results/megamoe-eplb-c128-b200aligned \
                      /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens
HIP_VISIBLE_DEVICES=0 python3 analysis/mla_microbench.py --quick   # needs 1 GPU
```

### Launching any arm — the checklist that cost five failures today

1. VRAM at the **0.28 GB baseline**, confirmed twice a minute apart. Never on a
   plateau.
2. Zero `sglang::*` processes (`pgrep '^sglang::'`) — **list PIDs before
   killing**, and note a finished arm leaves its server running.
3. Ports 8888/8889 clear.
4. Use `agentx/agentx_c128_totaltokens.sh` as the template: it pins
   `PYTHONPATH=/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori`,
   `MEM_FRACTION_STATIC_DP_MEGAMOE=0.85`, `MORI_SHMEM_HEAP_SIZE=16G`. **The
   launcher's own defaults (0.65, 40G, and a bare import resolving to
   `/sgl-workspace/sglang`) do not reproduce any published MegaMoE number.**
5. 90 s after launch run `analysis/cmd_diff.py` against the reference arm. It is
   necessary and **not sufficient** — it compares CLI flags only, and all five
   of today's failures had identical flags. Also check the tree in `server.log`
   and the environment echo in the launch log.

---

## Earlier today (2026-09-17 11:0x UTC+8) — `total_tokens` A/B: balancing is NOT an ITL lever

**Status:** arm complete and clean (3,628 s, `errors=0`, gates pass, 46 flags
with only `--load-balance-method` differing, KV pool identical at 12,077,312).
Nothing running. Full block in `exchange/FINDINGS.md`, last section.

**The intervention worked and the step did not care.** `#full token` skew
**2.08x → 1.69x** (verified with `kv_skew.py` on the new `server.log`, not from
the flag), and at matched `bs` the decode step moved **0 to +2.4 %** — nothing,
or marginally slower. The new arm even carries *more* KV at the same `bs`
(tok/req 140.8-232.6k vs 114.3-218.3k).

**So the balancing route is closed, by two independent methods agreeing.** The
microbenchmark said the kernel's cost follows the batch's **longest** sequence;
`total_tokens` equalises the **total**; the A/B confirms removing 19 % of the
skew buys zero step time. **The −7.35 ms "balanced ranks" row in this morning's
counterfactual is withdrawn.**

**End to end:** TTFT **11.07 → 9.61 s (−13.2 %)**, ITL p90 flat (+1.0 %),
throughput +5.5 % which is **inside the 5.67 % replicate spread, so a null**,
cache hit unchanged (0.956 → 0.957). **The predicted TTFT regression did not
happen** — `total_tokens` does not disturb the router's `cache_aware` prefix
reuse. Keep the flag (free, better TTFT, less skew) but **do not count it
against the MLA gap**.

**Next: MLA is the only remaining line, and it is unambiguous.** The kernel runs
at **1.2-3.1 % of both rooflines**, its cost is set by one straggler CTA, and
uniform split-K cannot fix that (it wins on ragged shapes, loses on uniform, and
the heuristic cannot tell them apart — it reads only capture-time scalars).
Two things gate the design, in this order:
1. **Size it.** `(max − mean)/max` of the per-token kv_len in production. Not in
   any artefact we have. The probe for it (`mla_kvlen_stats.patch`) is written
   but **not capture-safe** — see below. Cheapest correct route: a short
   `--disable-cuda-graph` run, where the distribution is identical and Python
   runs every step, so no device buffer is needed.
2. **Then design** per-sequence split or a persistent-CTA work queue in
   `paged_decode.py`, under the constraint that kv_len is unknowable at capture
   time.

**The kv_len probe is now capture-safe and validated locally.**
`agentx_c128_kvlenprobe.sh` + `mla_kvlen_stats.patch`. Two fixes to the bug that
killed launch #4: allocate the buffer only when
`torch.cuda.is_current_stream_capturing()` is false (sglang's warmup forwards
run eager, so it lands there), and never write a host scalar into the device
buffer. **Verified before use, not on the node**: a local `torch.cuda.CUDAGraph`
capture + replay around the real kernel succeeds and returns correct stats, and
the overhead is noise (345.6 vs 347.5 µs). It logs
`kvlen mean/p50/p99/max/min` and `kvlen straggler = (max−mean)/max` per decode
line.

**Why `(max−mean)/max` decides the next step.** The kernel's cost follows the
batch's longest sequence, so that ratio *is* the headroom a straggler-aware
kernel can win. It also discriminates two worlds: the 2.9x spread we know about
(per-rank implied kv_len 244-708) is **between** ranks, and nobody has measured
the spread **within** a batch. Within-batch dispersion ⇒ the straggler fix
works. Ranks internally uniform but at different levels ⇒ it does nothing, and
the only route left is making MLA faster outright.

Priced beforehand so the probe has something to falsify
(`mla_counterfactual.py`, same trace): equalising MLA across ranks to what the
**cheapest rank already achieves** takes the step wall **74.02 → 62.48 ms
(−11.54, −15.6 %)**; to the mean, −6.48; MLA free, −19.54. Dispersion removal
alone is 59 % of the total MLA opportunity and needs no work moved between
ranks — which is why it survives the `total_tokens` falsification.

**Read the probe against tok/req, never against the clock.** tok/req reaches
~140k by minute 15 and 147-172k by 25-40 (steady state 165-170k), so discard
early samples. `DURATION=1800`.

### C — the copy kernels are attributed. The biggest one is a ROCm-only fallback

`analysis/copy_attrib.py` on the existing trace, no GPU. 27.58 ms of `copy`-role
kernels across 8 ranks' EXTEND windows, 80 % resolved.

**`_fill_padded_rows_kernel` — 7.41 ms in EXTEND, grid 3254x256, and exactly
183 calls/step = 3 x 61 layers.** Three call sites, all MoE top-k padding
housekeeping: `topk.py:1578` (`_mask_topk_ids_padded_region`),
`topk.py:1591` (`_zero_topk_weights_padded_region`) and
`mega_moe_flydsl.py:263`. **B200 has no counterpart because the CUDA path never
reaches this kernel** — `topk.py:1573` is
`if _is_cuda and topk_ids.dtype == torch.int32 and fill_value == -1:
mask_topk_ids(...)`, and ROCm falls through to `_fill_padded_rows`. This is a
gated fast path, not a missing optimisation, which makes it the cheapest item on
the whole list to attack.

**`_swa_scatter_kernel` — 3.42 ms, grid 3254x512, 61 calls/step = one per
layer**, from `store_swa_into_unified` (`unified_kv_kernels/runtime.py:85`).
The SWA KV write.

Everything else resolves to ordinary aten ops (`aten::copy_`, `aten::cat`,
`aten::scatter_add_`, `aten::index`) and `Memcpy DtoD`.

**Two method notes worth keeping:**
- **`External id` does not attribute Triton kernels.** A kernel event carries
  `correlation`, not `External id`; the id lives on the `cuda_runtime` launch
  event, and only aten's `hipLaunchKernel` has one —
  `hipModuleLaunchKernel` (Triton) does not. The tool therefore walks
  `correlation -> launch event -> timestamp` and finds the innermost enclosing
  `cpu_op`/annotation. Triton launches sit inside no aten op, so the trace can
  only place them in the step; the call site came from `rg` on the kernel name
  plus the calls/step count as the cross-check (183 = 3x61 pins all three sites).
- **⚠ CORRECTION: grid IS in the ROCm trace.** Every kernel event carries
  `args.grid` and `args.block` (e.g. `_fill_padded_rows_kernel` grid
  `[3254,1,1]`, block `[256,1,1]`). This file previously said it was not, and
  that `megamoe_prepare_compact`'s grid 30 had to be inferred from aiter's
  kernel-name encoding. It can be read directly, and `copy_attrib.py` prints it.

---

## Earlier (2026-09-17 09:3x UTC+8) — reference environment recovered

**Status:** five launches, five failures, all root-caused, **all from launcher
and environment drift, none from the change under test**. Nothing running, GPUs
at the 0.28 GB baseline. The arm is now configured from the reference arm's own
launch log and is ready to go on approval.

**The reference environment, recovered verbatim** from
`/shared_nfs/kk/pr35619/b200aligned_c128.log` (the arm's own launch log — it
echoes the environment, which `sglang_command.txt` does not):

```
MEM_FRACTION_STATIC=0.85
MORI_SHMEM_HEAP_SIZE=17179869184        # 16 GiB, NOT the launcher's current 40G
PYTHONPATH=/workspace/InferenceX:/sgl-workspace/sglang-MegaMoE/python:/sgl-workspace/mori:
```

**Provenance of the 16 GiB, since no wrapper script sets it:** the launch log is
a `set -x` trace, and line 453 is `+ export MORI_SHMEM_HEAP_SIZE=17179869184`
immediately after `SGLANG_AMD_FLYDSL_MEGA_QUANT=a8w4` — the *launcher's own*
line, at the exact position where the working copy now reads `40G`. The
committed launcher has no such export at all, so that whole MegaMoE block has
always been uncommitted, and **the line was edited in place from 16 GiB to 40G
after the reference arm ran**. `git diff` shows the block as a pure addition and
therefore hides the edit; only the execution trace reveals it. Setting 16G is
restoring the reference configuration, not deviating from it.

**Why 40G cannot work here, arithmetically** — the heap is charged *outside*
`mem-fraction-static`:

```
236.93 (PyTorch static at 0.85) + 40 (heap) + 20.99 (target-verify capture)
  = 297.9 GiB  >  287.98 GiB      before the ~5 GiB driver context
```

Four launches OOMed on exactly this, on GPUs 1/2/7, each with <1 GiB free. The
reference log confirms it independently: at capture end it had `avail mem=20.85`
with 257.92 GiB PyTorch-allocated, leaving **9.23 GiB** non-PyTorch — a 40 GiB
heap cannot be resident in that.

**The five failures, so none is repeated:**

| # | cause | mine? |
|---|---|---|
| 1 | `mem-fraction-static` 0.65 (launcher's new MegaMoE default) vs 0.85 | no |
| 2 | wrong tree: bare `import sglang` → `/sgl-workspace/sglang`, 27 dirty files, live vim | no |
| 3 | same, plus 40G heap | no |
| 4 | **my kv_len probe is not capture-safe** — `hipErrorStreamCaptureUnsupported` | **yes** |
| 5 | 40G heap, probe off, correct tree — clean proof that 0.85+40G cannot fit | no |

Failure 4 is a real bug in `mla_kvlen_stats.patch`: writing the host-side
`d.numel()` into the device buffer is a pageable H2D copy, which aborts cuda
graph capture, and lazily allocating the buffer inside capture is a second
problem. **The probe is now default-OFF.** Fix both (drop the host scalar,
allocate from backend init) or sample the distribution from a
`--disable-cuda-graph` run instead — the distribution is workload-driven and
identical there, and Python runs every step so no device buffer is needed.

**Standing lesson, stronger than before:** `cmd_diff.py` reported "46 flags,
only the expected difference" on **every one of these failures**. A CLI-flag
diff is necessary and nowhere near sufficient. The launch log's environment
echo is the artefact that actually settles reproduction — **capture it for every
arm, and diff it too.**

**Ready to launch (needs approval — ~1 h GPU):**
```bash
nohup bash /shared_nfs/kk/pr35619/agentx_c128_totaltokens.sh \
      > /shared_nfs/kk/pr35619/tt_arm.log 2>&1 &
sleep 90 && python3 /workspace/claude-skills/agentx/analysis/cmd_diff.py \
  /workspace/results/megamoe-eplb-c128-b200aligned \
  /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens \
  --expect load-balance-method
```
Only `--load-balance-method` differs from the reference. "16G + probe off" has
never actually been run: failure 4 was the probe, not memory.

---

## Earlier (2026-09-17 09:1x UTC+8) — root cause of the launch failures

**Status:** three launches, three failures, all the same OOM at cuda-graph
capture (`236.93 GiB` PyTorch-allocated, <1 GiB free of 288, on a different GPU
each time). Root-caused. **Nothing is running; GPUs released.**

**Root cause: the launcher has UNCOMMITTED changes made after the reference arm
ran on 2026-09-15/16, and two of them move memory.** From
`git diff` of `InferenceX/benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh`
(97 insertions, uncommitted):

```
+    export MORI_SHMEM_HEAP_SIZE="${MORI_SHMEM_HEAP_SIZE:-40G}"
+        MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC_DP_MEGAMOE:-0.65}"
```

The mori symmetric heap is charged **outside** `mem-fraction-static`. Memory
accounting at the same point in both runs:

| | reference (2026-09-15) | now |
|---|---:|---:|
| PyTorch allocated | 236.93 GiB | 236.93 GiB |
| free at target-verify capture begin | **41.84** | **~1.0** |
| implied non-PyTorch | **~9.2 GiB** | **~50.1 GiB** |

The 40.9 GiB difference is the new 40G heap. Target-verify capture needs
20.99 GiB, which fits in the reference's 41.8 and not in our ~1.

**Two further traps found on the way, both already fixed in the arm script:**

1. **The reference arm ran from `/sgl-workspace/sglang-MegaMoE/python`** (36
   occurrences in its `server.log`), but a bare `import sglang` now resolves to
   **`/sgl-workspace/sglang`** — a different checkout with **27 dirty files**
   (including `dp_attn.py`, `forward_batch_info.py`, `eplb/*`) and a **live vim
   session**. Two launches went there. The arm script now pins
   `PYTHONPATH=/sgl-workspace/sglang-MegaMoE/python`, and the accidental patch
   to the other owner's tree has been reverted. *(The microbenchmark results are
   unaffected: the two trees' `paged_decode.py` are byte-identical apart from
   the instrumentation.)*
2. `mem-fraction-static` 0.65 vs the 0.85 that **all** published MegaMoE numbers
   used — fixed with `MEM_FRACTION_STATIC_DP_MEGAMOE=0.85`.

**`cmd_diff.py` cannot catch any of this.** It compares CLI flags, and all three
confounds were environment or code. Flags matched exactly (46, one expected
difference) in every failed launch. **A flag diff is necessary and not
sufficient; the tree and the memory-affecting env have to be checked too.**

**Next — a decision is needed, it is not mine to make:**
- `MORI_SHMEM_HEAP_SIZE=16G` is the value this file's trap notes record
  ("mem-frac 0.85 plus the 16 GiB mori heap needs 261 of 288 GiB") and would
  leave ~26 GiB for a 21 GiB capture. But the reference ran with ~9 GiB
  non-PyTorch, i.e. effectively **no** mori heap — the a2a backend here is
  `megamoe`, not mori, so the heap may simply be unused.
- Setting it to 16G reproduces the *documented* configuration, not the
  *reference* one. If the reference is the comparison target, the cleanest route
  is to **re-baseline**: run `total_requests` and `total_tokens` back to back on
  today's launcher, and compare those two to each other rather than to
  2026-09-15.

---

## Earlier (2026-09-17 08:3x UTC+8) — direction A arm staged

**Status:** the `total_tokens` A/B arm is written and was launched once, then
**killed 90 s in** because `cmd_diff` caught a confound (below). Waiting on the
VRAM cliff before relaunching — 50 GB/GPU plateau, KFD entries stale, processes
all dead. Do not launch on the plateau.

**Arm:** `agentx/agentx_c128_totaltokens.sh` (also at
`/shared_nfs/kk/pr35619/`). It answers two things at once:
- Does `total_tokens` help end to end? Report ITL, TTFT and cache hit together;
  expect a TTFT regression since it fights `cache_aware` prefix reuse.
- **It is a falsification test of the straggler finding.** The kernel's cost
  follows the batch's longest sequence, and `total_tokens` equalises the total,
  so the prediction is that **MLA µs/call barely moves**. If it moves a lot, the
  straggler result is wrong.
Verify the outcome with `analysis/kv_skew.py` on the new `server.log`, not by
the flag being present.

**Instrumentation, `agentx/mla_kvlen_stats.patch`** (applied in
`/sgl-workspace/sglang-MegaMoE`, env `SGLANG_MLA_KVLEN_STATS=1`): adds
`kvlen mean/max/min` and `kvlen straggler` to each decode log line. This is the
number that **sizes a straggler-aware rewrite** — `(max − mean)/max` — and
nothing already in the trace or the log carries it. `kv_indptr` is built *inside*
the cuda graph, so the reductions are device-only (capture-safe) and the
scheduler reads the buffer from outside at the existing log interval. Recording
in all 61 layers cost 14 % per call (345.6 → 394.4 µs); emitting in **one layer
per forward** puts it at noise (344.1 vs 346.2).

**⚠ New trap, cost an aborted launch: the launcher's MegaMoE+DP branch defaults
`mem-fraction-static` to 0.65** (`dsv4_fp4_mi355x_sglang_mtp.sh:216`,
`MEM_FRACTION_STATIC_DP_MEGAMOE`) while the reference arm ran **0.85**. Left
alone it changes the KV pool, and with it cache hit and batch composition.

**This is not specific to this arm. Every MegaMoE number we have published was
taken at 0.85**, so the 0.65 default is off-matrix and any MegaMoE arm launched
without `MEM_FRACTION_STATIC_DP_MEGAMOE=0.85` is not comparable to any of them.
Export it in every MegaMoE arm script, or change the launcher default. New
tool **`analysis/cmd_diff.py`** diffs a new arm's `sglang_command.txt` against
the reference and exits non-zero on any unexpected flag — **run it ~60 s after
every launch**. With the fix the two commands differ in exactly one flag.

`--load-balance-method` is now `${LOAD_BALANCE_METHOD:-total_requests}` at
launcher line 241, so the default is unchanged for every other arm.

**Relaunch, once VRAM has cliffed to the 284 MB baseline:**
```bash
nohup bash /shared_nfs/kk/pr35619/agentx_c128_totaltokens.sh \
      > /shared_nfs/kk/pr35619/tt_arm.log 2>&1 &
sleep 90 && python3 /workspace/claude-skills/agentx/analysis/cmd_diff.py \
  /workspace/results/megamoe-eplb-c128-b200aligned \
  /workspace/results/megamoe-eplb-c128-b200aligned-totaltokens \
  --expect load-balance-method
```

---

## Earlier (2026-09-17 08:0x UTC+8) — MLA is straggler-bound; A is the wrong knob for it

**Status:** microbenchmark done on an idle GPU 0, `analysis/mla_microbench.py`.
Full block in `exchange/FINDINGS.md` (last section). Four results:

1. **No absorbed stall.** Inverting the sweep gives implied kv_len 244-708 for
   the four in-situ points, all **below** the `index_topk = 1024` cap. The rank
   spread is a kv_len spread and the counterfactual is not circular.
2. **Cost follows the longest sequence, not the total.** At fixed mean kv_len,
   raising the max 500→1000 costs **+71 %**; halving the batch's *total* KV
   costs **−4 %**. One straggler CTA holds the grid. **So `total_tokens`
   balancing equalises something this kernel barely feels — expect direction A
   to do little for MLA.**
3. **Flat in bs within a wave, then a step.** kv_len=1024: bs 14-18 all
   469-480 µs, bs 19 **851 µs**. Marginal batch is free, then catastrophic.
4. **1.2-3.1 % of both rooflines**, corroborated in situ by `umc_activity`
   19.5 %. The 4.06x gap to B200 is a design gap, not a silicon gap.

**The fix, and the shape of it:** straggler-aware split-K. Uniform split-K wins
on ragged shapes (419 → 355 µs) and loses on uniform ones (247 → 317), and
`_kv_splits_heuristic` cannot tell them apart — by construction it reads only
`(T, H, block_h)` at capture time, never kv_len. It is correctly tuned for
uniform and wrong for ragged. Per-sequence split or a persistent-CTA work queue
closes the rest of the 419→247 gap.

**Next:** design that, in
`sglang-MegaMoE/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py`.
The CUDA-graph constraint is the hard part: kv_len is not knowable at capture
time, so the split factor cannot depend on it — a persistent-CTA work queue
sidesteps that, since the grid is then capture-time constant and the *work
assignment* is what varies.

**Repro (GPU 0, ~7 s):**
```bash
HIP_VISIBLE_DEVICES=0 python3 /workspace/claude-skills/agentx/analysis/mla_microbench.py --quick
```

---

## Earlier (2026-09-17 07:4x UTC+8) — MLA is the whole imbalance; step 0 done

**Status:** offline critical-path counterfactual finished, no GPU used. The
component table below **double-counts**: the MLA row (+7.47) and the cross-rank
idle row (+9.96) are *the same slack*. Rank order by own work is exactly rank
order by MLA time; remove MLA and the cross-rank spread collapses from 13.50 ms
to 1.96 ms. Multiplier on an MLA speed-up is **1.5x** (the step wall responds to
the critical rank's 20.18 ms, a kernel table credits the 13.29 ms mean).
Full block and the joint pricing table: `exchange/FINDINGS.md`, last section.

Re-priced, from `analysis/mla_counterfactual.py`: MLA at B200 speed **−11.26 ms**,
perfect rank balancing alone (direction A, upper bound) **−7.35**, both
**−14.69**, MLA free **−19.54**. A is sub-additive with MLA and its value falls
as MLA improves, so price A at ~7 ms, not ~10.

The barrier model is self-validating: predicted slack sits below each rank's
observed `prepare` by a constant 5.08-5.25 ms on all five ranks, equal to the
independently measured 5.19 ms floor. So
`prepare = cross-rank slack + 5.19 ms irreducible protocol`.

**Next:** two GPU arms to validate, in this order, neither started — ask first.
1. **k=2 duplication arm.** Run the real MLA kernel twice per call, discard the
   extra. Outputs bit-identical, so accept len / OSL / ISL / KV are untouched:
   a true single-variable test with every metric quotable. Predicted step wall
   **94.16 ms** (+20.14 against a naive +13.29).
2. **k=0 fake arm.** Zeros, never uninitialised memory (NaN → the sampling
   `ASSERT_TRAP`). Predicted **54.48 ms** and per-rank `prepare` collapsing to
   the 5.19 ms floor. **Only trace-derived per-step numbers are quotable from
   this arm:** DSPARK is on (accept len 3.65, rate 0.44 of 7 draft tokens) and
   OSL is EOS-driven, so garbage logits move both.

Injection point for both: `sparse_attn_v4_paged_decode`,
`sglang-MegaMoE/python/sglang/kernels/ops/attention/dsv4/unified_kv_kernels/paged_decode.py:895`
— single Triton file, env-gated, no aiter JIT rebuild.

**Repro (no GPU, ~7 s):**
```bash
python3 /workspace/claude-skills/agentx/analysis/mla_counterfactual.py \
        /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
```

Still open: **C**, the 3.85 ms of unfused copies. **B is started and is blocked
on one measurement — see below.**

### B — the desk roofline is UNDER-DETERMINED. Do not publish a number from it.

The prompt's recipe (derive KV bytes from the DSv4 config plus per-rank
`#full token`) **does not work on this model**, for a reason worth recording:

- **DSv4 decode attention is sparse.** `index_topk = 1024`, `sliding_window =
  128`, `head_dim = 512`, and the decode path runs three ragged index streams
  (SWA / CSA / HCA, `unified_kv_kernels/runtime.py:403 build_decode_streams`)
  whose per-token length is a *prefix sum of real valid entries*, not the
  context length. So `#full token` is an upper bound on what the kernel reads,
  and on this workload it is ~100x too large.
- KV is **fp8_e4m3** with `page-size 256`, plus 1x64 block scales
  (`NUM_GROUPS = D/64 = 8` per token), so a KV token costs 512 + 32 = 544 B.
- Grid is `(N_tokens, ceil(H/BLOCK_H))`, one CTA per (token, head-tile), and
  each CTA re-reads that token's whole KV. With `H=128`, `BLOCK_H=16` that is
  **8 re-reads**, absorbed by L2 or not — which the desk calculation cannot
  decide either.

Bracketing it gives **2.3 % (topk-capped) to 32 % (whole working set) of the
8 TB/s peak**. That spread spans "nowhere near the wall" and "half way to it",
so it answers nothing.

**And the per-call times falsify every simple work model.** Capture-window KV
(server.log 02:55-03:01, matching the 03:00 trace):

| rank | trace bs | `#full token` | tok/req | kernel | us/call |
|---:|---:|---:|---:|---|---:|
| 0 | 17-18 | 892,928 | 52,525 | fused | 165-166 |
| 0 | 19-20 | 892,928 | 52,525 | fused | 223-232 |
| 1 | 14 | 1,978,240 | 152,172 | fused | **330.7** |
| 6 | 16 | 2,398,848 | 171,346 | fused | 266.1 |
| 3 | 9 | 1,834,368 | 141,105 | split | 117.5 |
| 7 | 10 | 2,395,776 | 171,127 | split | 163.5 |

Rank 6 has **more** context *and* **more** bs than rank 1 and is **20 %
faster**. So the cost is not `bs`, not `#full token`, and not tok/req. Within
rank 0 it *is* superlinear in bs (bs 17→20 costs +40 %), which smells like a
tiling or occupancy step rather than a data volume.

#### B, part 2: at a FIXED kv_len the kernel is nowhere near any wall — and that makes the rank spread unexplainable by work

Taking `kv_len = topk = 1024` (the sparse cap, so this is the honest upper
bound on a decode query's KV), per call, against MI355X peaks of 8 TB/s and
~2.5 PFLOP/s bf16, with `Nq = bs x 7` draft tokens:

| rank | bs | us/call | KV MB | achieved BW | of peak | x8 re-read | achieved | of peak |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 18 | 166.3 | 70.2 | 534 GB/s | 6.7 % | 43.6 % | 216 TF/s | **8.6 %** |
| 6 | 16 | 266.1 | 62.4 | 297 GB/s | 3.7 % | 24.2 % | 120 TF/s | 4.8 % |
| 1 | 14 | 330.7 | 54.6 | 209 GB/s | 2.6 % | 17.1 % | 85 TF/s | **3.4 %** |

**Two independent measurements confirm it is not bandwidth.** From the run's own
`gpu_metrics.csv` in the capture window (epoch 1789527600 ±60 s): `umc_activity`
**19.5 %** median across all 8 GPUs, and `gfx_0_clk` flat at **2372-2393 MHz**
(0.9 % spread) with `throttle_status = 0` everywhere. So the memory controller
is ~80 % idle during decode, and clock/power skew is **ruled out** as the
explanation for the rank spread.

**What is left is a 2.5x efficiency difference between two ranks running the
identical kernel on identical hardware at the same clock** (rank 0 at 216 TF/s,
rank 1 at 85). No work model produces that. Either `kv_len` is *not* constant
across ranks, or the kernel's measured duration is absorbing something that is
not its own work.

**⚠ A flaw in the evidence we have been leaning on.** `prepare_wait.py` reports
`mla_fused r=+0.961 -> real work`, but its `compute` proxy is
`attn+gemm+quant+norm_rope+sample` and **MLA is inside `attn`** — the kernel is
correlated against a bucket containing itself. B200's `flash_fwd_splitkv_mla
r=+0.917` row has the same defect. So **neither node has clean evidence that
the MLA per-call spread is work rather than absorbed stall**, and if it is
partly stall, the counterfactual above is circular: it would "remove the
imbalance" by removing the kernel that happens to be holding it. Fixing the
proxy to exclude the kernel under test is a small change to `prepare_wait.py`
and should be done before the next publication from either node.

**Next, and it is the cheapest decisive test:** a standalone microbenchmark of
`_paged_decode_fused_kernel` with swept `(N, kv_len)`. It gives achieved
bandwidth directly instead of by derivation, and by *inversion* recovers the
`kv_len` that reproduces 330.7 µs — which is the quantity the desk route could
not pin. GPUs are idle (297 MB/GPU baseline, no KFD PIDs). Minutes, not an arm.

---

## Earlier (2026-09-17 07:0x UTC+8) — diagnosis closed, optimisation open

**The cross-platform investigation is finished.** Both nodes are serial, both
per-role tables are elapsed and subtract cleanly, and the 41.07 ms decode-step
gap is fully attributed and agreed (FINDINGS, B200 `e9dce10`). Nothing is
running on this node; no measurement is blocked on B200.

| component | ms | share | nature |
|---|---:|---:|---|
| MoE pipeline work — 3 stages vs 1 fused | **+12.4** | 30 % | kernel work |
| Cross-rank idle, parked in `prepare` | **+9.96** | 24 % | load balancing |
| MLA decode kernel | **+7.47** | 18 % | kernel work |
| `ep_combine` exposed vs fused a2a | **+6.17** | 15 % | structure |
| `copy` — fill kernels, no B200 counterpart | **+3.85** | 9 % | kernel work |
| quant + misc | +1.7 | 4 % | |
| `gemm` | −0.45 | −1 % | equal, settled |

### Directions, ordered by payoff ÷ effort

**A. Switch `--load-balance-method` to `total_tokens`** — one launcher flag,
worth up to ~10 ms (24 %). Both nodes' logs now justify it: `running-req` is
level (MI355X 1.38x, B200 1.29x) while `#full token` is not (2.27x / 2.45x),
because `cache_aware` concentrates long conversations. `total_tokens` exists
already (`data_parallel_controller.py:92,125-130`, fed by
`LoadSnapshot.num_total_tokens`). **A/B it** — it fights prefix reuse, so expect
a TTFT cost; same family as the pdi knob. EPLB is irrelevant here (this is
attention/KV, not expert routing).

**B. The MLA decode kernel** — 7.47 ms direct, *and* it is the multiplier on A,
so it is the only item that pays twice. 117.5-330.7 µs/call against B200's
18.7-47.5. **Do not start by rewriting it: first get achieved bandwidth**, which
nobody has on either node. `record_shapes` is a dead end here (all dim-carrying
events are `aten::*` cpu_ops), so derive the KV bytes per call from the model
config plus per-rank `#full token` and compare against MI355X's HBM roofline.
That says whether 4x is closeable or whether the kernel is already at the wall.
Two variants: `_paged_decode_split_kernel` (bs 9-10) and
`_paged_decode_fused_kernel` (bs 14-20); the 4.06x figure is the split one.

**C. `copy`, 3.85 ms of unfused fills** — the most ordinary win on the list and
entirely local. Seven kernels where B200 has 0.03 ms: `_fill_padded_rows` 0.757
(183 calls), `__amd_rocclr_fillBufferAligned` 0.533 (122 = 2/layer, a hipMemset
that smells like a buffer that could be persistent), two `direct_copy`
elementwise 1.428 total, bf16→fp32 copy 0.392, `index_elementwise` 0.363,
`_swa_scatter` 0.271, `_fill_compress_tail` 0.143.

**D. MoE pipeline structure, +12.4 ms** — the largest single work item, but it
is "fuse three stages into one", i.e. real aiter/FlyDSL work, not a knob. The
cheap adjacent piece is **pipelining `prepare(n+1)` against `stage1/2(n)`**,
which attacks the 6.28 ms protocol floor rather than the 12.4.

**E. `ep_combine` exposed, +6.17 ms** — B200 pays ~0 because its a2a is fused
inside `mega_moe_impl`. Same family as D: can the combine overlap stage2 or the
next layer? Structural, not tuning.

### Two measurements that gate the above

1. **TP-only / single-DP-rank run** (`npes = 1`, nobody to wait for). If
   `prepare`'s 102.9 µs/call floor collapses it is synchronisation and D's
   pipelining is the fix; if it holds, it is real plan emission and `pcu1`
   (one CTA) is worth raising after all. Also gives the first uncontaminated
   compute number on either node.
2. **Achieved bandwidth on the MLA decode kernel** — gates B (see above).

### Ruled out, with evidence — do not revisit

- **`prepare`'s CU count / `pcu1`+`qcu28` tuning.** A `wait_i32_until_equals`
  spin does not parallelise; the 226 idle CUs are idle *because the rank is
  waiting*. Withdrawn by both nodes.
- **Stream overlap / co-scheduling.** B200 measured true full serialisation at
  **3 % end to end** (8 % on the step wall, diluted by pdi=24). MI355X has no
  room in the wait anyway. Noise against a 2.20x kernel-work gap.
- **`gemm`.** Equal once B200's starvation artefact was removed (9.81 vs 9.36).

---

## ⚠ EARLIER: this session's shells were wedged — resolved by a terminal restart

**State at handoff (2026-09-16 21:1x UTC+8):** I ran `kill -9 <pid>` on the PID
the tool reported for a backgrounded `sed`, and that PID was the Cursor **shell
bootstrap**. Every new shell now hangs — even `echo alive` did not complete in
85 s. This is the exact trap already documented further down this file
("never `pkill -P` / kill a process whose children you have not listed"), and
hitting it again means the warning needs to be read as: *do not kill any PID the
tool hands you unless you have listed its children first.*

Nothing was broken on the node — no GPU work was running. A terminal restart
fixed it and the verification then ran normally. **Kept as a warning, because
this is the second time the same trap has been hit: do not `kill` a PID the tool
hands you for a backgrounded shell without listing its children first.**

---

**Newest, and it reframes the overlap question (2026-09-16 17:2x):** answered
B200's grid-vs-CU request. MI355X is gfx950 **SPX, 256 CUs**, and
`megamoe_prepare_compact` — the largest kernel in the step, 16.244 ms,
21.6 % of it — launches **grid 30**, so it cannot occupy more than **30 of
256 CUs**. Nothing co-resides with it: total co-resident time in the step is
**0.024 ms** against B200's 15.92 ms. **This is the opposite of B200's case**
(their MoE claims 146 of 148 SMs, so they have nothing to overlap into and
measured only 5-13 % upside for themselves). Their bound does not transfer.
Grid is *not* in the ROCm trace; it came from aiter encoding the launch config
in the kernel name (`pcu1` + `qcu28` + 1) plus the local generators.

**Next:**

1. **Co-schedule real work against `megamoe_prepare_compact` and watch the step
   wall.** This is the one experiment that settles the overlap upside, and it is
   local. Careful: grid 30 is an occupancy *ceiling*, not a utilisation
   measurement — the prepare stage is a producer/consumer ticket protocol and
   part of its 266 µs/call may be irreducible. If the wall does not move, this
   line closes. Cheaper probe first: are `pcu1` / `qcu28` simply mistuned for a
   256-CU part? That is a config in
   `aiter/ops/flydsl/kernels/mega_moe/mega_moe_prepare.py`, not a rewrite.
2. **Own the MLA decode kernel.** The narrowest target on either node:
   163.5 µs/call against B200's 40.2, same call count, same layer count, 13.3 %
   of the step. Start at aiter's `_paged_decode_split_kernel` /
   `_paged_decode_reduce_kernel` and the KV layout they read. No B200 needed.
4. **TP-only / single-DP-rank capture** is now the cheapest uncontaminated
   compute number for either node, since `record_shapes` is a dead end here.
5. **Waiting on B200 for one thing only:** their `busy_ms.py` `credited` column.
   Until it lands, no per-role delta is quotable in either direction.
6. **Not blocking any more:** B200 has published steady-state class-split
   numbers, so the comparison is live. The `compute` *ratio* stays withdrawn —
   B200's side is pace-pinned, mine is not, and one usable side is not a
   comparison. Quote step wall (30.0 vs 74.0, 2.47x) or the kernel pairs above.

**Repro — re-run the analysis on the existing traces (no GPU needed):**
```bash
cd /workspace/claude-skills/agentx
python3 analysis/trace_summary.py /shared_nfs/kk/pr35619/trace_c128_pdi24_steady/*TP-7-*.gz
python3 analysis/trace_ranks.py   /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/kernel_dump.py   /shared_nfs/kk/pr35619/trace_c128_pdi24_steady
python3 analysis/busy_ms.py       /shared_nfs/kk/pr35619/trace_c128_pdi24_steady 10
python3 analysis/decode_stats.py  /workspace/results/megamoe-eplb-c128-b200aligned/server.log
```

**Earlier result, still standing:** full-model `TARGET_VERIFY` with no
time-overlap against any of the 8 `EXTEND` annotations — 5 ranks, 16-17 steps
each, p50 73.9-74.1 ms (max/min 1.002x), `n_hit=0`. The 74 ms is not waiting on
a prefill. Ranks 2/4/5 have no verify in this window and 0 kernels during it.

## Where things are

| what | path |
|---|---|
| steady-state traces (use these) | `/shared_nfs/kk/pr35619/trace_c128_pdi24_steady/` |
| mid-ramp traces (do not conclude from) | `/shared_nfs/kk/pr35619/trace_c128_pdi24/` |
| trace run's server.log | `/workspace/results/megamoe-eplb-c128-b200aligned-trace/server.log` |
| complete c128 run (agg metrics) | `/workspace/results/megamoe-eplb-c128-b200aligned/` |
| capture orchestrator (reusable) | `/shared_nfs/kk/pr35619/trace_c128_pdi24.sh` |
| idle-wait + launch (reusable) | `/shared_nfs/kk/pr35619/wait_and_launch_c256.sh` |
| findings, pushed | `agentx/exchange/mi355x-decode-trace.md`, `agentx/exchange/FINDINGS.md` |

Benchmark result rows and the pdi/router/load-balance alignment history live in
`dsv4/megamoe/C256_REGRESSION_HANDOFF.md`.

**Uncommitted on purpose:** `agentx/summary_table.py` carries three new `ROWS`
entries (c256 post-merge, c256 B200-aligned, c128 B200-aligned). They point at
MI355X-local result dirs. Commit them if the other node should see the registry;
they are not needed for the trace work.

## Node-specific traps, all hit at least once on 2026-09-15/16

- **Barrier wait is not cashable, and a big one invites a wrong plan.** Idle
  time inside a wait kernel (`megamoe_prepare_compact` 4.6-18.7 ms/step,
  `ep_combine`) belongs to a rank that is already ahead; the step is set by the
  last rank to arrive, so shortening a fast rank's wait buys nothing. Classify
  with `analysis/prepare_wait.py` (anti-correlation = wait) before budgeting any
  kernel, and bound the real-work part by the BUSIEST rank's value, which waits
  least. The `total_tokens` arm is the worked example: skew 2.08x → 1.69x, step
  time 0 to +2.4 %. See the decision section at the top.
- **VRAM reclaim is plateau-then-cliff, never linear.** After killing a server it
  sits flat at 29-53 GB/GPU with **zero** KFD holders for 10-20 min, then drops
  to the 284 MB baseline in one step. A creep-rate extrapolation once predicted
  7 hours; it cliffed within the minute. Do not launch on top of a plateau —
  mem-frac 0.85 (245 GiB) plus the 16 GiB mori heap needs 261 of 288 GiB.
  Per-GPU reset is **unsupported on this node**; do not reach for it.
- **The launcher orphans everything.** Every run so far left `launch_server`,
  `sglang::*`, tokenizer workers and aiperf alive holding ~290 GB/GPU *and* the
  dist-init port, which makes the next launch die with `port_base ... is not
  available`. Verify all three after cleanup: process count, VRAM, port listeners.
- **Intermittent EPLB rebalance deadlock, ~1 run in 3.** `returned=` frozen,
  `errors=0`, `/metrics` still 200, schedulers alive, VRAM full. Last server line
  is `Resetting ExpertDistributionRecorder...` from all 8 ranks, then every rank
  hangs in a collective and the NCCL watchdog kills it 600 s later with
  `c10::DistBackendError`. Nothing reaches launcher stdout. Judge liveness only
  by `returned=`/`done=` moving. A straight retry cleared it.
- **`pgrep -f` / `rg` match your own shell command.** Reported phantom aiperf and
  watcher processes three times. Worse: a broad `pkill -9 -P` plus loose patterns
  killed the Cursor shell bootstrap (`bash -O extglob -c snap=$(command cat <&3)`)
  and **wedged every new shell in the session** — `echo` would not complete,
  which looks exactly like a dead node but is not. Recovery is restarting the
  terminal. Match on `ps -eo comm` or bracket the first character, and never
  `pkill -P` a process whose children you have not listed.
- **A fixed settle before `/start_profile` captures mid-ramp.** Context length is
  still climbing minutes into the profiling phase. Trigger on per-request
  `#full token` / `#running-req` plateauing (>= ~130k here), not a constant.
- **`SGLANG_TORCH_PROFILER_DIR` did not appear in any scheduler's
  `/proc/*/environ`.** Inconclusive, but pass `"output_dir"` in the
  `/start_profile` body instead — `profile_utils.py:117` prefers it and it costs
  nothing in comparability.

## Config state of the launcher

`dsv4_fp4_mi355x_sglang_mtp.sh` is aligned to the B200 sibling as of 2026-09-15:
`--prefill-decode-interval` 24 (20 plus `--balance-abs-threshold 32` when
`CONC >= 160`), `--load-balance-method total_requests`, router `--policy
cache_aware`. Pre-edit backup: `/shared_nfs/kk/pr35619/mi355x_mtp.sh.bak.1228`.
The file was already dirty before that edit (+81 lines of MegaMoE support), so
**do not `git checkout` it**.

`CONC` counts AgentX *session trees*, not requests: `MAX_RUNNING_REQUESTS=2*CONC`
= 256, so with dp8 a rank can legitimately run up to 32 concurrent requests.
Observed distribution at c128 peaks at 11-12 and tails to 29 — batches above
16/rank are expected, not a bug.
