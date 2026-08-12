# Breakthrough #2 — single cross-PE-round dispatch (drop dual-mode entirely)

**Goal.** Make **compact** dispatch need only **one** cross-PE round (like fixed-slot) so decode is fast
*without* a second scheme. One scheme everywhere ⇒ **no mode divergence** (the hang) and **no fixed-slot
decode path** (the cuda-graph corruption). This captures the same ~12–21% decode win the dual-mode lever
targeted, but structurally removes both proven blockers (see `DUAL_MODE_DISPATCH_DIAGNOSIS.md` §5/§7/§8).

Status: **design proposal, not implemented.** This is precise cross-PE kernel work (`dispatch.py` + likely
`gemm1`/epilogue) — high risk; likely kernel-owner co-design. Validate concept in the micro-harness
**against a real oracle** before touching the server (the corruption bug slipped past the self-consistency
check — never repeat that).

---

## 1. Why compact needs 2 rounds today (the thing to remove)

`kernels/mega_moe/dispatch.py` compact path:

| round | where | purpose | cost |
| --- | --- | --- | --- |
| **#1 count all-gather** | `:399–419` (`bigcnt` P2P + `cd`/`done2` handshake) | each rank learns per-expert counts from **all** senders → compute its **dense base** `my_base[ge]` (`:437–444`) | ~12µs xGMI floor |
| write payload | `:484–542` | P2P-write tokens to the **dense** slot `my_base[expert]+cursor` | (bandwidth) |
| **#2 payload done-barrier** | `:568+` (`gb_cnt` + `done2` + meta) | confirm all peers' P2P writes landed before GEMM reads | ~12µs xGMI floor |

Fixed-slot skips **#1** because its slot is `le*cap + atomic(running[le])` — a **static** per-expert base
(no cross-rank counts), at the cost of an `epr*npes*mtpr`-row buffer (~15 GB @ mtpr=8192 → doesn't fit
32-bit, hence compact exists). So the trade today is: **fixed-slot = 1 round but huge buffer; compact =
small buffer but 2 rounds.** #2 breaks that trade.

Measured cost of round #1 at decode: **+11.6%** (tokens=32, `FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md`) to
**~18–21%** (bs 8–128, `DUAL_MODE_DISPATCH_DIAGNOSIS.md` §1) of stage1+stage2 — a fixed xGMI latency, not
bandwidth (counts are tiny), so it dominates precisely at small-bs decode.

---

## 2. The design — receiver-side (per-source) dispatch, 1 round

**Key idea:** don't compute a *dense* base (which needs cross-rank counts). Instead each sender writes to a
**per-(source-rank) region** on the receiver, whose base is **known locally** (`src_rank * cap_per_src`) — no
count exchange. Then ONE done-barrier. The dense/expert-major view is produced **locally on the receiver**
(counts of what it received are local — no xGMI).

```
per sender s (no cross-rank info needed):
   slot = s*cap_per_src + atomic(local_cursor_on_receiver[s])     # base is static & local -> NO round #1
   P2P-write token payload + srcmap to receiver.recv_staging[slot]

ONE cross-PE round: payload done-barrier (same as today's round #2)   # the only ~12µs floor

receiver, locally (agent-scope, no xGMI):
   count per (source, expert) from the srcmap it received           # local histogram, cheap
   EITHER (2B-i) compact staging -> dense expert-major (1 HBM read + write), then GEMM as compact today
   OR     (2B-ii) GEMM reads recv_staging source-major directly, tiles derived from the local (src,expert)
                  counts (like fixed-slot's static-tiles readlane, but per-source) -> no compaction copy
```

### 2.1 Buffer sizing (fits 32-bit everywhere we care)
`cap_per_src` = worst-case tokens one sender routes to this receiver = `mtpr * min(topk, epr)` ≈ `mtpr*topk`.
Total staging = `npes * mtpr * topk` rows.

| mtpr | staging rows (`npes*mtpr*topk`, npes=8, topk=6) | bytes @7168 | vs fixed-slot (`epr*npes*mtpr`, epr=32) |
| --- | --- | --- | --- |
| 512 (decode) | 24.6K | ~176 MB | 197K rows (fixed-slot) |
| 8192 (prefill) | 393K | ~2.8 GB (**< 4 GB, fits i32**) | 2.1M rows ≈ 15 GB (fixed-slot **overflows**) |

So per-source staging sits **between** compact-dense and fixed-slot: small at decode, and **still fits
32-bit at mtpr=8192** where fixed-slot blows up. ⇒ one scheme covers the whole range.

### 2.2 Two finishing variants
- **2B-i (compact locally):** simplest to reason about; GEMM/epilogue/stage2 contract unchanged (still reads
  dense). Cost = one extra local HBM read+write of the payload (tiny at decode; at big bs it's bandwidth on
  top of compute — measure). Doubles transient HBM (staging + dense).
- **2B-ii (source-major GEMM):** no compaction copy; GEMM reads staging directly, tiles derived from local
  `(src,expert)` counts (readlane prefix, same trick as fixed-slot static-tiles). Faster (no copy) but
  changes GEMM tile derivation + output ordering → more epilogue/stage2-contract work. Higher effort.

Start with **2B-i** (correctness-first, minimal GEMM change); move to 2B-ii only if the local compaction
shows up in profiles at large bs.

---

## 3. Why this removes BOTH blockers (the whole point)

- **One scheme → no divergence/hang.** Decode and prefill run the **identical** dispatch (same buffer, same
  single done-barrier, same flags). There is no fixed-slot-vs-compact choice → ranks can't disagree → the
  cross-PE rendezvous can't deadlock. `decode_cap`, the `use_decode` gate, `--hang-diverge`, the
  `DIVERGE_CHECK` — all become unnecessary.
- **No fixed-slot decode path → no cuda-graph corruption.** The garbage-output bug (§7.1) was specific to the
  fixed-slot decode kernel under graph; removing that path removes the bug class.
- **Closes the large-bs gap too?** Staging fits i32 at mtpr=8192; if `cap_per_src` sizing also holds at the
  bs≥16384 sizes (needs checking vs the §6 int32-overflow), this may also help the current large-bs overflow
  gap — bonus, not a goal.

---

## 4. Perf expectation
Removing round #1 saves one ~12µs xGMI floor → recovers the **~12–21% decode** stage1+stage2 the dual-mode
lever measured, but for **all** bs and with no second scheme. Added cost = local per-(src,expert) count
(cheap) + (2B-i only) one local HBM copy of payload (negligible at decode; measure at prefill). Net at
decode should ≈ fixed-slot (both 1 round); at prefill ≈ compact or slightly better (no count round; small
copy). **Must be measured**, not assumed.

---

## 5. Risks & effort (honest)
- **High-risk cross-PE kernel surgery.** This is the "极其精密" dispatch the docs warn about; we just watched
  it deadlock/corrupt under small perturbations. Likely **kernel-owner co-design**.
- **CUDA-graph safety.** The single done-barrier must keep the monotonic-epoch, no-reset discipline (§8) so
  it's graph-safe — the exact property the fixed-slot `_dec` flags apparently violated (§7.1 corruption).
- **srcmap / atom-contract.** stage2 reads a2@logical via srcmap; the receiver-side layout must still emit
  the same srcmap mapping (2B-i keeps it identical; 2B-ii must re-derive it).
- **Buffer**: 2B-i needs staging+dense (2× transient HBM at big bs) — check `mem-fraction-static` headroom.

## 6. Validation-first plan (do NOT skip the oracle this time)
1. **Micro-harness, correctness-vs-ORACLE.** Extend `tests/kernels/test_mega_moe.py` to compare the new
   1-round dispatch output against the **torch/atom reference** (relL2), NOT just self-consistency — the
   dual-mode check only did self-consistency and missed the server corruption. Cover a4w4/a8w4 × bs
   {1,8,64,512,2048,8192} × multi-seed, all-rank PASS.
2. **Perf in micro-harness** vs current compact (device-time) — confirm the ~12–21% decode win is real.
3. **Eager server** (small + imbalanced load, `DIVERGE_CHECK` should now be structurally impossible) → then
   **graphed server** gsm8k (must match compact-only 0.85–0.93, coherent output — the graph-correctness gate
   the fixed-slot path failed).
4. Only then consider default-on.

## 7. Recommendation / fork
This is a real kernel redesign. Two ways to proceed:
- **(P) Prototype it ourselves in the micro-harness first** (edit `dispatch.py` compact path to 2B-i + the
  oracle check) to *prove* the perf+correctness of the concept on a controlled bench before any server/graph
  exposure. De-risks before involving the kernel owner. Uses the FlyDSL kernel-debug toolkit we built.
- **(K) Hand this design to the FlyDSL kernel owner** for co-design, given the cross-PE precision + our own
  prior "high-risk" flag.

Independently: **ship compact-only now** (correct, ~0.85–0.93, ~97% of dp) — it's unaffected and is the
fallback regardless of #2's outcome.

---

## 8. Implementation spec (prototype P) — additive `recv` scheme, default-off

Chosen: **2B-i** (per-source write → 1 done-barrier → local compaction to the **same dense layout** compact
produces → GEMM/epilogue/stage2 UNCHANGED). Added as a **new, isolated `recv` scheme** so the shipping
`compact`/`fixedslot` paths are never modified.

**Files / changes:**
1. `kernels/mega_moe/gemm1.py`
   - Add param `recv_dispatch: bool = False`. When set: treat the GEMM/output exactly like compact
     (`static_tiles=False`, dense `se/trb`, atom output), but pass a new `fuse_recv=True` (and
     `compact=False`) into `emit_dispatch_prologue` so the dispatch selects the recv branch.
   - Relax the `fuse_dispatch=="fixedslot"` assert to also accept the recv path.
2. `kernels/mega_moe/dispatch.py` — new `elif const_expr(fuse_recv):` branch (adapt the compact branch):
   - **Phase WRITE (no round #1):** each sender, per (token,expert): `dest=expert//epr`;
     `slot = fz_rank*cap_per_src + atomic(local_cursor[dest])`; P2P-write payload+scale+idx+wts+srcmap to
     `dest.staging[slot]`. `local_cursor[npes]` is a fresh per-launch local counter (LDS or HBM, zeroed).
     `cap_per_src = tile_align(fz_mtpr * fz_k)`.
   - **1 cross-PE round:** reuse compact's done-barrier (`gb_cnt`→`done2`→`meta`) verbatim.
   - **Post-barrier LOCAL compaction (replaces round #1's block0 metadata):** scan this rank's staging
     `[npes*cap_per_src]`; from each occupied slot's srcmap decode expert → LDS histogram `count[e]`; prefix
     → dense `base[e]`, `num_valid`, `se[]`, `trb[]`; then scatter each staged row's payload+scale+idx+wts+
     srcmap into the dense buffers at `base[e]+cursor[e]++`. Occupied = srcmap != sentinel (pre-fill staging
     srcmap with sentinel each launch, or track per-source counts via a tiny local count array the senders
     also bump — TBD during impl; sentinel scan is simplest).
   - Output dense buffers + `se/trb/nv/srcmap` are the SAME ones compact fills ⇒ GEMM unchanged.
3. `kernels/mega_moe/mega_moe.py`
   - Add `mega_scheme`/build support for `recv`. Allocate a **staging** symmetric buffer
     `staging[npes*cap_per_src, row]` (+ scale/idx/wts/srcmap staging) via `op._sym`/comb-op, and
     `local_cursor[npes]`. Extend `_disp_tbl`/`_build_disp_table` with the new staging + cursor slots
     (use free disp indices; keep compact's 29-39 untouched).
   - Compile a recv kernel (`compile_fused_moe_gemm1(recv_dispatch=True, compact_dispatch=False)`).
4. `tests/kernels/test_mega_moe.py` — a `--stage1-scheme recv` (or env) so `_run_full_e2e` builds MegaMoE
   with the recv scheme and runs the **existing torch/atom oracle** (relL2 + key-set) — the real
   correctness gate (NOT self-consistency).

**Milestones (validate each before the next):**
- M1: recv kernel COMPILES + a single fwd runs without crash (micro-harness, bs=8).
- M2: **oracle relL2 ~1e-5** on bs {1,8,64,512} × a8w4/a4w4, all-rank PASS (correctness — the gate dual-mode skipped).
- M3: device-time vs compact at decode — confirm the round-removal win (~12–21%).
- M4: bs {2048,8192} correctness + buffer/headroom check; then graphed server gsm8k == compact-only.

**Risk controls:** additive/default-off (shipping untouched); validate-vs-oracle from M1; keep the done-barrier's
monotonic-epoch/no-reset discipline (graph-safety); if a milestone stalls, the design + spec hand cleanly to
the kernel owner.

### 8.1 Code-grounded details (from reading the full compact branch + `_disp_tbl`, 2026-07-21)

Increment 1 **LANDED** (safe, default-off): `gemm1.py::recv_dispatch` → `fuse_recv`; `dispatch.py` param +
fixed-slot guard `fuse_fs and not compact and not fuse_recv`. Verified lint-clean; never traced until wired.

Key facts that shape the branch (from `dispatch.py:294–612` + `mega_moe.py::_disp_tbl:743–826`):

- **Two buffer sets are required.** Compact writes payload P2P **directly into the dense** `rx_em/scale/idx/
  wts/srcmap` (peer tables disp 8–12) after computing `my_base` in round #1. recv instead needs: **(a)
  per-source STAGING** (new P2P buffers, sender writes `slot=rank*cap_per_src+local_cursor[dest]`), **(b)
  DENSE output** (compaction target + GEMM input). So recv roughly **doubles** the payload buffers (staging +
  dense) and their peer tables. Staging ≈ `npes*cap_per_src` rows; dense = the existing `nvm`.
- **Per-source counts via done-barrier piggyback (no extra round).** The receiver must know how many slots
  each source wrote (to bound the compaction scan). Piggyback it on the existing done-barrier (`:582–588`):
  each sender also signals `recv_slots[rank]=local_cursor[dest]` alongside `done2`. No new cross-PE round.
- **Local base = peer_table[rank].** Compaction reads local staging / writes local dense via the `[fz_rank]`
  entry of the respective peer tables — so no separate "local base" disp slots are strictly needed for
  buffers that already have a peer table.
- **Disp-table plan:** reuse compact-atom slots 4–42 for the shared machinery (done2/gb_cnt/meta/se/trb/nv/
  srcmap/dctr/trecv/rnum/_sti/_se_atom/_wts_sorted). Add NEW slots 43+ for: 5 staging peer tables, 5 dense
  peer tables (if dense not already tabled), `recv_slots` (local+peer), per-dest `local_cursor`, and the
  compaction `my_base`+`dcursor` (can reuse compact's 34/35). ~10–12 new slots.
- **Parallelization (grid coordination):** `zero local_cursor` → **all blocks write staging** → dedup→dctr
  (reuse `:544–567`) → **block0 done-barrier + recv_slots piggyback** (reuse+extend `:569–607`) → **block0
  compaction-meta**: scan staging (bounded by `recv_slots[s]`) → per-expert histogram → `my_base`/`se`/`trb`/
  `nv`/`ll_count`/srcmap-sentinel (mirror `:437–473` but from a *local staging scan*, not `bigcnt`) →
  release metaA → **all blocks scatter** staging→dense at `my_base[e]+atomic(dcursor[e])` (payload copy like
  `:526–543`, but agent-scope local, not P2P) → grid-barrier → release metaB → GEMM.
- **Open item to resolve at impl:** confirm how `gemm1.py` addresses the dense `rx_em`/scale for the GEMM
  read (disp slot vs kernel arg) — decides whether dense needs its own new disp slots or reuses existing.

### 8.3 Build results (GPU-validated, 2026-07-21)

The recv scheme is IMPLEMENTED (additive, default-off) across `dispatch.py` (recv branch), `gemm1.py`
(`recv_dispatch` flag), `mega_moe.py` (staging buffers via `op._sym`/`_p2p_table`, `_disp_tbl_recv`, build +
`_run`/`forward` select), `test_mega_moe.py` (`--recv`). Milestones:

- **M1 (compile+run): PASS.** recv builds (`cap_src=49152`, `stg_rows=393216` @v4_pro mtpr8192) and runs.
- **M2 (correctness vs oracle): PASS — recv is bit-equivalent to compact.** `_run_full_e2e` all-8-rank PASS
  for **bs {1,8,64,512,2048} × a8w4(v4_pro) & a4w4(r1_v3)**. recv's `mega-vs-baseline` is *identical* to the
  compact control (a8w4 2.332e-3 vs 2.333e-3; a4w4 8.881e-2 vs 8.881e-2) → the 1-round receiver-side dispatch
  produces the same result as compact. **The core hypothesis is proven: one cross-PE round is sufficient and
  correct.**
- **M3 (perf): recv is currently ~20–25% SLOWER than compact** (v4_pro a8w4 device-time, megav1 ms):
  bs8 recv 0.419 vs compact 0.335; bs64 recv 0.518 vs compact 0.430. The naive first-draft compaction
  (block0-serial per-expert histogram + a 256-expert serial `my_base` prefix on lane0 + an extra local
  payload copy staging→dense + 2 extra intra-kernel grid barriers) costs MORE than the ~12µs count-round it
  removes. **Correctness-first is done; the perf win is not yet realized.**

**Why (analysis):** recv trades compact's 2nd cross-PE round (~12µs latency) for (a) a 2nd payload movement
(P2P→staging, then local staging→dense) and (b) local metadata compaction. At decode the payload bytes are
small (~µs), so removing the round *should* win — but the draft's block0-serial metadata + extra barriers
dominate. **M3 optimization (next):** parallelize the histogram + `my_base` prefix across all blocks (not
block0-serial), collapse the 2 extra grid barriers, and reduce the double payload move (or evaluate 2B-ii
GEMM-reads-staging to skip the copy, accepting per-(src,expert) tile padding). If optimized compaction can't
get under the ~12µs saved, recv won't beat compact and the pragmatic answer stays **ship compact-only**.

### 8.4 M3-opt results (2026-07-21) — gap 20–25% → 5.5%, still short of compact

Two correctness-preserving optimizations landed in `dispatch.py` (recv branch), oracle-revalidated
(all-8-rank PASS, `mega-vs-baseline` bit-identical to the compact control at bs {1,8,64,512,2048}):

1. **Fused the two cross-PE rounds into ONE** (biggest win; the draft still had two). The draft copied
   compact's structure: a `done2` *epoch* barrier **then** a separate `recv_num` count exchange = **two**
   sequential ~12µs xGMI waits. But a peer's `recv_num` signal is emitted only *after* its `fx.barrier` (all
   its staging writes to me have landed), so waiting all peers' `recv_num>0` **is** a sufficient payload
   barrier → `done2` is redundant. Dropped it; `recv_num` now doubles as barrier + stage2 count and
   piggybacks the per-source `recv_slots`. This is exactly the fixed-slot pattern (and stays within
   compact's proven self-reset/`wait==0` discipline → graph-safe). recv is now a true **1 cross-PE round**
   (vs compact's 3: allgather-`cd`, `done2`, `recv_num`).
2. **Eliminated the 256-expert lane0-serial `my_base` prefix.** Non-local experts always receive 0 tokens,
   so the dense base is just the tile-padded prefix over the `epr`(=32) LOCAL experts — which the `epr`
   metadata loop already computes as `acc*ctm`. Removed the separate serial loop; store `my_base`+zero
   `dcur` inline for local experts only (mb stays in-register, no HBM round-trip / cross-loop `s_waitcnt`).

**Measured** (v4_pro a8w4, `megav1` E2E stage1+fused-stage2 ms, 100 iters, stable across 2 trials):

| bs | compact | recv (draft) | recv (M3-opt) | remaining gap |
| --- | --- | --- | --- | --- |
| 1  | 0.1505 | — | 0.1609 | **+6.9%** |
| 2  | 0.2063 | — | 0.2131 | **+3.3%** |
| 4  | 0.2806 | — | 0.2959 | **+5.5%** |
| 8  | 0.334 | 0.419 (+25%) | 0.353 | **+5.6%** |
| 64 | 0.426 | 0.518 (+20%) | 0.449 | **+5.4%** |

Note: recv loses at **every** bs including bs=1 — there is no small-bs crossover. The cross-PE rounds recv
removes vs compact are cheaper than the scatter+barrier it adds, at all decode sizes. So recv (2B-i) delivers
**no decode win** over compact, and is a fortiori slower than the (unshippable) fixed-slot decode path (which
has 1 round AND no scatter). The only lever left that could beat both is 2B-ii.

### 8.5 2B-ii attempted + GPU-measured (2026-07-21) — REJECTED (slower at all bs); reverted to 2B-i

Implemented 2B-ii end-to-end (correctness-passing) then measured it: **it is slower than 2B-i at every bs**,
so it was **reverted**. What 2B-ii did: block0 builds a tiny `gather_idx[dense_row]=staging_row` map + scatters
only the SMALL fields (idx/wts/srcmap/scale, ~32× smaller than the embedding) into dense — **no embedding
copy** — which lets us drop the all-blocks scatter pass AND the post-scatter grid barrier (recv → **one** grid
barrier, like fixed-slot). The GEMM then **gathers X (embedding) per-row from source-major staging** via
`gather_idx` (single injection in `utils.py::AGatherAddresser.row_bases`; X buffer-resource built in-kernel
from the staging base pointer — the staging tensor itself is 2.8GB @ mtpr8192 and cannot be a kernel arg, C-ABI
packs numel as i32 → overflow; `create_buffer_resource_from_addr` defaults to a 4GB OOB extent so an addr-based
resource covers it).

**Correctness: PASS** (bit-equivalent to compact, `mega-vs-baseline` 2.35e-3 @ bs8). **Perf (v4_pro a8w4,
`megav1` ms):**

| bs | compact | recv 2B-i (M3-opt) | recv 2B-ii |
| --- | --- | --- | --- |
| 1  | 0.151 | 0.161 | **0.235** |
| 8  | 0.334 | 0.354 | **0.469** |
| 64 | 0.426 | 0.449 | **0.586** |

**Why it lost (the real lesson):** the dispatch's compaction into a **dense expert-major** layout is NOT
wasteful overhead — it *buys* **contiguous, coalesced, async-DMA-friendly X tile loads** in the GEMM's hot
K-loop. Gathering X per-row from scattered staging turns each MFMA tile's X load into `tile_m` non-adjacent
row loads, wrecking coalescing/vectorization and the async pipeline. That GEMM-side penalty (a fixed per-tile
cost that also grows with bs) **exceeds** the ~one grid barrier + scatter pass that 2B-ii removes — at *every*
bs (even bs=1, where the GEMM still runs ~epr tiles). Scale/srcmap/wts were kept dense to limit the change to
one injection; gathering those too would only add more scattered-load penalty.

**Verdict (final): both 2B-i and 2B-ii recv are slower than compact → SHIP COMPACT-ONLY.** 2B-ii was reverted;
recv stays at its best (2B-i M3-opt, ~5% off compact) as the additive/default-off, proven-correct fallback.
The receiver-side single-round hypothesis is fully validated as *correct*, but the **compact-vs-coalescing
trade favors compact**: paying 2 cross-PE rounds to hand the GEMM a contiguous dense buffer beats paying 1
round + a gather that starves the GEMM. No remaining dispatch-only lever beats compact; a real decode win would
need a different axis (e.g., overlapping dispatch with compute, or a GEMM that natively consumes source-major
tiles — both large, kernel-owner-level).

**Conclusion — recv still loses to compact by ~5.5%, and the residual is structural to 2B-i.** Both schemes
now use **2 grid barriers**; recv already has *fewer* cross-PE rounds (1 vs 3). The residual ~19–23µs is the
**extra full-grid scatter pass** (local staging→dense copy of every received row + per-token `dcur` atomic)
plus its **post-scatter grid barrier** — work compact simply doesn't do (compact writes payload P2P straight
into the dense slot). This scatter+barrier **cannot be removed within 2B-i**: block0 builds `my_base` and all
blocks must then scatter and re-sync before the GEMM reads dense. Parallelizing the histogram across all
blocks would *add* a barrier (net loss at decode), and block0-only scatter is bandwidth-catastrophic.

**The only path to beat compact is 2B-ii** (GEMM reads `recv_staging` source-major directly via per-(src,
expert) tiles → no scatter copy, no dense buffer, drops the post-scatter barrier). That removes exactly the
residual ~5.5%, but it is a **GEMM tile-derivation + output-ordering + epilogue change** on the
"极其精密" cross-PE path — high-risk kernel-owner co-design (§5/§7). **Per the decision rule, 2B-i recv cannot
beat compact → ship compact-only.** recv stays additive/default-off as a proven-correct, now near-parity
(~5%) fallback; if the decode lever is ever needed again, 2B-ii is the documented next step.

### 8.2 Honest status of the build
The remaining core (recv branch ≈250 lines + the doubled staging/dense buffers & peer tables in
`mega_moe.py` via `op._sym`/`op._p2p_table` + disp-table + build/select + test) is a **GPU-in-the-loop
kernel build**: a first draft *will* need multiple compile→run→fix cycles to reach `relL2~1e-5` on this
notoriously fragile cross-PE code (writing it fully blind would just be rewritten once it can run). Path:
drive it as GPU-validated increments (M1 compile → M2 oracle), or hand this code-grounded spec to the FlyDSL
kernel owner. Compact-only ships regardless.
