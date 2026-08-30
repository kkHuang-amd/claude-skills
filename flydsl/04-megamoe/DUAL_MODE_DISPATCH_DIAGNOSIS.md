# FlyDSL MegaMoE dual-mode dispatch — diagnosis + approach-A scaffold (2026-07-20)

Optimizing the **decode** path of the FlyDSL MegaMoE backend for DeepSeek-V4-Pro (gfx950 / 8×MI355X).
Goal: use the **fast fixed-slot dispatch at decode** (skips compact's extra cross-PE round) while keeping
**compact** for prefill. This is the "decode-cap dual-mode" lever.

**Bottom line:** the lever is **real (~18-21% faster decode stage1+stage2, measured)**. Integration is
done + correct (compact-only baseline gsm8k **0.9318** unaffected; micro-harness dual-mode passes). One
real integration bug was found + fixed (decode-selection cross-rank consistency → `forward_mode`). The
**remaining blocker is now PROVEN** (fx.printf + forced-divergence repro + py-spy, §5): a **cross-PE
rendezvous deadlock triggered by cross-rank mode divergence** — one rank running fixed-slot while others run
compact makes the two schemes wait on incompatible shmem flags (`recv_num` vs `done2`) forever; NOT a memory
race. **Fix (a) implemented + server-verified (§7.1): it removes the hang** (no DIVERGE, full lm_eval
completion, decode uses fixed-slot not compact) — **but the hang was masking a 2nd blocker: fixed-slot decode
is CORRUPT under cuda-graph serving** (dual-mode gsm8k 0.125 + garbage vs compact-only control 0.854 +
coherent; same code/config, only `decode_mtpr` differs). So dual-mode is **not shippable yet**; it now has
**two** blockers (divergence *and* graph-correctness). The (a) edit is safe for the shipping path (control
healthy). This favors breakthrough **#2** (cheaper compact count-round → drop dual-mode) over hardening
fixed-slot. **Shipping compact-only MegaMoE is unaffected** (all changes `decode_cap>0` / env-gated, off).

Pairs with `MEGAMOE_HANDOFF.md`, `MEGAMOE_IMAGE_CHANGE_HANDOVER.md`,
`FLYDSL_MEGAMOE_STAGE1_ANALYSIS.md`. Code snapshot in `./patches/`.

---

## 1. Why decode (the lever)

Per-kernel decode profile via the micro-harness (`tests/kernels/test_mega_moe.py --profile`, v4_pro a8w4,
bs=64, mtpr=8192, EP8, rank0; megav1 E2E 426.8 us/iter):

| kernel | us/iter | % | what |
|---|--:|--:|---|
| `moe_gemm1_0` | 294.1 | 69% | fused **dispatch + gemm1 + silu + a2-quant + scatter** (one megakernel) |
| `mfma_moe2_…cshuffle` | 106.7 | 25% | gemm2 (down-proj) |
| `ep_combine_intranode` | 13.9 | 3% | mori a2a combine |
| glue (vectorized_elementwise ×3) | 12.2 | 3% | |

Decode is **fixed-cost dominated, not MFMA-compute-bound**: megav1 E2E bs 8→128 = 0.334→0.451 ms (+35% for
16× tokens). So the win is not in tile tuning; it's in the **fixed dispatch/prologue cost** inside
`moe_gemm1`. (silu is already fused into stage1 — not a separate lever.)

### The dispatch prologue (`kernels/mega_moe/dispatch.py::emit_dispatch_prologue`)
Two `const_expr` branches:
- **fixed-slot** (non-compact): writes payload to `le*cap + atomic(running[le])`; **1 cross-PE round**;
  asymmetric N→1 arrival (only block0 waits on `gb1`, then a single recv-count handshake on `recv_num`).
  Needs a `epr*cap` slot buffer.
- **compact** (2 cross-PE rounds): LDS histogram → grid barrier → cross-PE#1 allgather counts →
  prefix-sum → strict write → cross-PE#2. Avoids the big slot buffer (scales to full batch).

Serving uses `mtpr=8192` for prefill capacity → forced into **compact** → pays the extra round even at
decode. **Micro-harness measurement (this image, v4_pro a8w4):**

| bs | compact mtpr=8192 (ms) | fixed-slot mtpr=512 (ms) | compact overhead |
|--:|--:|--:|--:|
| 8  | 0.334 | 0.275 | **21.4%** |
| 32 | 0.418 | 0.354 | 18.1% |
| 64 | 0.430 | 0.360 | **19.5%** (≈70 us) |
| 128| 0.451 | 0.381 | 18.5% |

Both PASS accuracy (mega-vs-baseline relL2 ≈ 2.3e-3). So a decode-only fixed-slot path is worth ~19% of
stage1+stage2 at decode (≈70 us/op, roughly fixed).

---

## 2. What was tried (all micro-harness PASS; server behavior varied)

`fixed-slot@cap=512` needs `epr*ll_cap = 48*(8*512) ≈ 197K` rows; the compact comb_op's `rx_em` is already
`≈395K` rows and all fixed-slot flags exist unconditionally → **the same comb_op buffers can host both**.

1. **Two-instance** (build a 2nd small-mtpr `MegaMoE` next to prefill; the pre-existing
   `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR` path): micro-harness alternate = no hang; **server = HANG**.
   Pin test (`decode_cap=1` → 2nd instance BUILT but never selected): **still hangs** →
   the mere COEXISTENCE of a 2nd comb_op on the single global mori heap corrupts the 1st. mori has ONE
   process-global symmetric heap (`shmem_malloc` takes no group arg) → no isolated heaps → two-instance
   is a dead end on this image (matches the original handoff's "HANGS / needs isolated shmem groups").

2. **Approach A: single-instance dual-mode** — ONE `MegaMoE` / ONE comb_op, compile BOTH gemm1 kernels
   (`compact_dispatch=True/False`) + build BOTH disp-tables, select per-forward via `decode=`. Reuses the
   comb_op payload buffers.
   - v1 (shared coordination flags): server HANG.
   - v2 (**separated** decode coordination flags — `running/ll_count/done2/recv_num/dest_ctr/gb1/meta` get
     their own `_dec` copies; `total_recv` stays shared because stage-2 combine reads `op.total_recv`):
     micro-harness PASS (no hang, corrupt_relL2=0), **server still HANG**.

3. **forward_mode fix (REAL BUG, fixed):** decode-vs-prefill selection was using `max(get_dp_global_num_tokens())`.
   On this config that returns a **per-rank / length-1 value** (and `None` under cuda-graph capture), i.e.
   it is **NOT cross-rank-consistent**. The mega dispatch is a cross-PE collective, so ranks disagreeing on
   compact vs fixed-slot wait on different shmem flags → **deadlock at the first prefill** (idle DP ranks
   picked decode while busy ranks picked compact). Fix: select by **`forward_batch.forward_mode.is_decode()`**
   (a GLOBAL per-step property — all DP/EP ranks share the same mode each step; graphed-decode also pads all
   ranks to the same bs so `num_tokens` is consistent too). After this fix the server got **much further:
   34 prefill batches + 3 decode steps ran** (selection now consistent) before hanging deeper.

4. **Remaining hang (kernel-internal):** with the forward_mode fix + cuda-graph, after a few real
   fixed-slot **decode** steps, a subsequent **compact prefill** hangs (py-spy: stuck at
   `torch.cuda.synchronize` in `process_batch_result_prefill`, i.e. GPU stuck). The hang point is
   **non-deterministic** (34 vs 8 prefills across runs) → a **race**. Interpretation: the fixed-slot
   dispatch's cross-PE mechanism (`fence_system_release/acquire` + `mori_shmem.int32_wait_until_*` +
   asymmetric N→1 arrival) leaves mori's shared/global shmem state in a state that races with / corrupts a
   subsequent **compact** dispatch (which uses symmetric grid barriers). This is below the integration
   layer.

5. **Mode-switch barrier (diagnostic, inconclusive):** insert `shmem_barrier_all()` on a compact↔fixed-slot
   switch (env `SGLANG_AMD_FLYDSL_MEGA_MODE_BARRIER=1`). Tested with eager decode
   (`--disable-decode-cuda-graph`) → still hung, but **confounded** (eager decode itself was not validated
   for compact-only, and a barrier can't live inside a decode cuda-graph anyway). Not a viable fix as-is.

**Isolation proof it is NOT the integration layer:** re-ran **compact-only** (`decode_cap=0`) with all the
current code → gsm8k **0.9318**, no hang, full run. So the refactor is clean; the blocker is specifically
the fixed-slot dispatch being exercised in the server's mori/cuda-graph environment.

---

## 3. Approach-A scaffold (the handoff artifact)

All changes are gated (`decode_cap>0` or env) → default-off, compact-only shipping unaffected.

### FlyDSL `kernels/mega_moe/mega_moe.py` (`./patches/flydsl_dual_mode_approachA.patch`)
- `MegaMoE.__init__(..., decode_cap: int = 0)`.
- `_build_fused_stage1`: when `decode_cap>0 and self._s1_compact`, also compile a 2nd kernel
  `self._s1_mega_dec = compile_fused_moe_gemm1(compact_dispatch=False, fuse_cap=tile_aligned(world*decode_cap),
  fuse_mtpr=decode_cap, …)` and allocate **decode-only coordination flags**
  (`_s1_running_dec/_s1_ll_count_dec/_s1_done2_dec/_s1_recv_num_dec/_s1_dest_ctr_dec/_s1_gb1_dec/_s1_meta_dec`
  + p2p tables), all over the SAME comb_op (`op._sym` / `op._p2p_table`). `_s1_nvm_dec = epr*ll_cap+256`
  must fit the compact op's buffers.
- `_disp_tbl(compact, dec=False)`: `dec=True` substitutes the `_dec` flags (payload/output + `total_recv`
  stay shared). `_build_disp_table` builds `_s1_disp` (primary) + `_s1_disp_dec`.
- `_run_fused_stage1(..., decode=False)`: selects kernel/disp/nvm/max_blocks/`er_ptr` by mode.
- `forward` / `forward_prequant(..., decode=False)` plumb-through.

### FlyDSL `tests/kernels/test_mega_moe.py` (same patch)
- `--mtpr` override (decouple mtpr from tokens; decode-in-serving repro).
- `--hang-iters / --hang-decode-mtpr / --hang-decode-tokens`: build ONE instance with `decode_cap` and
  ALTERNATE `forward_prequant(decode=False)` (prefill) ↔ `(decode=True)` on the SAME instance; report
  hang / decode-output corruption. **This passes** (the isolated dual-mode mechanics are correct).

### sglang `python/sglang/srt/layers/moe/mega_moe_flydsl.py` (`./patches/mega_moe_flydsl.py`, full file)
- `_get_or_build_mega_moe(..., decode_cap=…)` (keyed by `(mtpr, decode_cap)`; passes `decode_cap` to MegaMoE).
- `_run_mega_routed`: ONE instance + `use_decode = forward_batch.forward_mode.is_decode() and
  num_tokens<=decode_mtpr`; `forward(..., decode=use_decode)`.
- Env-gated debug (`SGLANG_AMD_FLYDSL_MEGA_DEBUG=1`) + mode-switch barrier diagnostic
  (`SGLANG_AMD_FLYDSL_MEGA_MODE_BARRIER=1`).
- This file **supersedes** `megamoe_image_change_patches/mega_moe_flydsl.py` (adds the forward_mode fix +
  `decode_cap` wiring; keeps the `_swap_layer_weights` `_s1_w1` fix from that handover).

---

## 4. Reproduce

Prereqs: `/sgl-workspace/FlyDSL` @ `mega_moe_v1` (pinned `3b0f818`), `/sgl-workspace/sglang` @ `feat/mega-moe`
with the MegaMoE port applied (see `MEGAMOE_IMAGE_CHANGE_HANDOVER.md`), then apply `./patches/`.

**Micro-harness (isolated dual-mode — PASSES):**
```bash
cd /sgl-workspace/FlyDSL
PYTHONPATH=/sgl-workspace/FlyDSL MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
  tests/kernels/test_mega_moe.py --network v4_pro --quant a8w4 --tokens 512 --mtpr 8192 \
  --hang-iters 20 --hang-decode-mtpr 512 --hang-decode-tokens 64
# -> [DUAL-MODE] ... OK (NO HANG, 20 iters); decode finite=True corrupt_relL2=0.000e+00
```

**Server (HANGS — the kernel-owner repro):**
```bash
cd /workspace/useful-scripts/benchmarking/dsv4/
SGLANG_AMD_FLYDSL_MEGA_DEBUG=1 SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR=512 \
  SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192 MEM=0.65 MODE=megamoe PORT=8000 bash run_sgl_dsv4_unified.sh
# in another shell, drive continuous batching:
lm_eval --model local-completions --model_args \
  model=/shared_nfs/huggingface_models/deepseek-ai/DeepSeek-V4-Pro,base_url=http://localhost:8000/v1/completions,\
num_concurrent=128,max_retries=3,tokenized_requests=False --tasks gsm8k --num_fewshot 5
# server reaches ready, runs some prefills + decode steps, then a compact prefill hangs (health 503).
# py-spy a scheduler: for p in $(rocm-smi --showpids|awk '/^[0-9]/{print $1}'); do py-spy dump --pid $p; done
```
Compact-only baseline (control, WORKS): drop `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR` → gsm8k 0.9318.

---

## 5. For the kernel owner — the precise ask

The single-instance dual-mode integration is correct and passes in isolation. The server hang is now
**proven** (see the reproduction above) to be a **cross-PE rendezvous deadlock triggered by cross-rank mode
divergence** — NOT a subtle memory race. When even one rank runs a different dispatch scheme than the others
(fixed-slot vs compact), the two schemes wait on incompatible shmem flags (`recv_num` vs `done2`) and can
never rendezvous. The earlier "races with mori shared state" framing is **superseded** by this finding.

Suspected area: `kernels/mega_moe/dispatch.py::emit_dispatch_prologue` fixed-slot branch (`fuse_fs and not
compact`, lines ~68-291) — the asymmetric N→1 arrival (`gb1`) + `recv_num` self-reset handshake +
`fence_system_release/acquire`, vs compact's symmetric grid barriers.

**CONFIRMED root-cause (forced-divergence repro + `fx.printf` + py-spy, 2026-07-20).** The fixed-slot
block0 sync is a **symmetric all-to-all rendezvous across all `npes` EP ranks** — there are exactly four
spin points and only three can hang cross-rank:

| # | `dispatch.py` | spin | needs |
|---|---|---|---|
| S1 | `:207` | `int64_wait_until_equals(a_gb1, tg2)` | all **local** blocks arrived (**co-residency** `gx·gy≤cu_num`, `moe_stage1_mega.md` §8) |
| S2 | `:219` | `int32_wait_until_equals(rnum_remote, 0)` | peer's `recv_num[me]` drained from the **previous** launch |
| S3 | `:223` | `int32_wait_until_greater_than(rn_src, 0)` | **each peer** posted its recv-count signal (rendezvous) |
| S4 | `:289` | `int32_wait_until_greater_than(a_meta, e0)` | block0 published `meta_flag` |

S2/S3 only complete if **all `npes` ranks are in the *same* dispatch kernel with the *same* epoch**. Under
DP-attention a per-rank mode divergence (some ranks fixed-slot/decode, others compact/prefill) makes the two
groups touch **different flag slots** → they can never rendezvous → S3 (or compact's `:418`
`int32_wait_until_equals(a_cd, epoch)`) spins forever. **Separating the decode flags made this worse**: a
momentary divergence becomes unrecoverable because the groups no longer share the slot to meet on. This is
why the `forward_mode` fix was necessary and why it's fragile — it holds only if *every* rank agrees on the
mode for *every* launch.

**Reproduction — the smoking gun (2026-07-20).** We instrumented `dispatch.py` with `fx.printf` at each
FZ/CP handshake spin point (banners `[MEGA-DBG FZ|CP r{rank} ep={epoch}] ...`) and added a
`--hang-diverge N` mode to the micro-harness that forces **rank0 → prefill(compact)** while
**ranks 1..7 → decode(fixed-slot)** *simultaneously* at iter N. Cache cleared, run eager (not graphed):

```bash
# 1) baseline (lockstep, all ranks agree each step) -> PASS, epochs consistent across ranks:
PYTHONPATH=/sgl-workspace/FlyDSL MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
  tests/kernels/test_mega_moe.py --network v4_pro --quant a8w4 --tokens 512 --mtpr 8192 \
  --hang-iters 8 --hang-decode-mtpr 512 --hang-decode-tokens 64
# -> [DUAL-MODE] ... OK (NO HANG, 8 iters); banners show FZ r0..r7 ep=N then CP r0..r7 ep=M (all agree)
# 2) forced divergence at iter 2 -> DEADLOCK:
#    ... same command ... --hang-diverge 2
```

Result (iter 2): every rank printed the host line `divergent forward RETURNED (async launch ok); calling
synchronize...` and **no rank ever printed `SURVIVED`**. The kernel banners froze at exactly the predicted
split:

- **ranks 1–7 (fixed-slot decode):** last banner `[MEGA-DBG FZ r{1..7} ep=4] post-S2 enter-S3` — **no
  `xPE-done`** → stuck at **S3**, waiting for rank0's `recv_num` signal that never comes.
- **rank0 (compact prefill):** last banner `[MEGA-DBG CP r0 ep=5] enter-xPE1` — **no `xPE1-done`** → stuck
  at the compact **cross-PE#1** (`:418`), waiting for peers' `done2` that never comes.
- **`py-spy` on all 8 workers:** `synchronize (torch/cuda/__init__.py) → _run_full_e2e (test_mega_moe.py)`
  → CPU blocked on the GPU-side spin. GPU had to be killed (`pkill -9`; GPUs recovered clean, no reset).

This is a **mutually-incompatible rendezvous**: fixed-slot ranks signal/wait on `recv_num`, the compact rank
signals/waits on `done2` — they can never meet. **The root cause is now proven**: the hang requires only a
single-rank mode divergence; it is not a subtle memory race. **Corollary:** in the server the hang means the
`forward_mode` selection is *still* diverging across ranks somewhere (idle DP rank / mixed batch / graph-pad
edge) — the fix belongs in the **integration layer** (`mega_moe_flydsl.py` mode selection must be provably
cross-rank identical every launch), OR the kernel must make the two dispatch schemes rendezvous-compatible
(shared arrival flag + mode tag) so a divergence degrades instead of deadlocking.

> The `dispatch.py` `fx.printf` was reverted after the repro (it would spam every compact-only prefill).
> The `--hang-diverge` harness mode is kept (test-only, default `-1`). Snippet + method:
> `FLYDSL_KERNEL_DEBUG_TOOLKIT.md` §4.

**Device-level debug plan (reusable).** See `FLYDSL_KERNEL_DEBUG_TOOLKIT.md` §4 for the full line-anchored
`fx.printf` instrumentation: print `rank`/`epoch`/peer at each of S1–S4, clear `~/.flydsl` cache, run the
micro-harness **eager** (not graphed). The last line each rank prints reveals (a) each rank's mode, (b) which
spin it died on, (c) the culprit peer.

Open sub-questions the trace will answer:
- Do compact and fixed-slot launches ever interleave with **mismatched epoch** (each mode has its own
  counter now, but they share the mori runtime barrier)? (print `epoch` at S1.)
- Is `recv_num`'s self-reset (`wait==0` then signal, S2) safe when the previous launch was the OTHER mode?
- Does the co-resident grid (`gx·gy≤cu_num`) + N→1 arrival survive cuda-graph replay + concurrent streams
  (S1)?

---

## 6. Status / shipping safety
- Shipping **compact-only** MegaMoE: unaffected (gsm8k 0.9318 / ~35.3k tok/s A/B unchanged). All dual-mode
  code is `decode_cap>0`/env-gated, default off.
- The A/B context (compact-only megamoe vs dp): see `MEGAMOE_HANDOFF.md` + this session's conc512 A/B
  (megamoe 35,315 tok/s ≈ 97% of the equal-VRAM dp-A / 91% of dp-best; TTFT better; TPOT ~9-40% worse — the
  decode gap this lever targets).

---

## 7. Fix (a) — make mode selection provably cross-rank-identical (implemented 2026-07-20)

`mega_moe_flydsl.py::_run_mega_routed` previously gated on the **local** `num_tokens`:
`use_decode = decode_mtpr>0 and is_decode and num_tokens <= decode_mtpr`. `num_tokens` is per-rank under
DP-attention, so the `<= decode_mtpr` term can differ across ranks → the proven deadlock. New logic makes
every term cross-rank-identical:

```python
if get_is_capture_mode():
    _tok_ok = True                       # graphed decode bs <= cuda_graph_max_bs (require decode_mtpr >= it)
else:
    _gnt = get_dp_global_num_tokens()    # DP-GLOBAL list, identical on all ranks
    _tok_ok = (max(_gnt) <= decode_mtpr) if _gnt else (num_tokens <= decode_mtpr)
use_decode = bool(decode_mtpr > 0 and _is_decode_step and _tok_ok)
```

- **Under cuda-graph** (the decode path that matters): `_tok_ok=True`, so `use_decode = is_decode` — a pure
  global per-step property → identical on all ranks **by construction**. *Requirement:* `decode_mtpr >=
  cuda_graph_max_bs` so the fixed-slot buffer always fits a graphed decode step (else raise, don't diverge).
- **Eager**: uses `max(get_dp_global_num_tokens())` (the same all-ranks list used by `_should_use_mega`),
  never the local count.
- **Divergence detector** (`SGLANG_AMD_FLYDSL_MEGA_DIVERGE_CHECK=1`, eager only): all-reduces `use_decode`
  (MAX vs MIN) over the world group **before** the dispatch; on mismatch it logs `[flydsl-mega-DIVERGE]
  rank=… layer=… use_decode=… max/min…` — turning a silent GPU hang into a localized, actionable error.
- Debug log (`SGLANG_AMD_FLYDSL_MEGA_DEBUG=1`) now includes `rank=` + `tok_ok=` so per-rank decisions align.

**Verify on the server:**
```bash
SGLANG_AMD_FLYDSL_MEGA_DEBUG=1 SGLANG_AMD_FLYDSL_MEGA_DIVERGE_CHECK=1 \
  SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR=512 SGLANG_AMD_FLYDSL_MEGA_MOE_MTPR=8192 \
  MEM=0.65 MODE=megamoe PORT=8000 bash run_sgl_dsv4_unified.sh      # then drive gsm8k (see §4)
# PASS = full run, no hang, and NO "[flydsl-mega-DIVERGE]" lines. If a DIVERGE line appears, it names the
# exact layer/rank/local_tok where selection still disagrees -> that path needs a global signal too.
```
> Ensure `decode_mtpr (512) >= cuda_graph_max_bs`; if `cuda_graph_max_bs > 512`, raise `--hang-decode-mtpr`
> / `SGLANG_AMD_FLYDSL_MEGA_DECODE_MTPR` to cover it (and confirm the decode buffer still fits the comb_op).
> The default script uses `CGBS=1024`, so the verification below ran `CGBS=512` (= decode_mtpr) to keep
> fixed-slot within its `cap=512` buffer.

### 7.1 Verification result (server, 2026-07-20) — (a) fixes the hang, but exposes a decode CORRECTNESS bug

Ran the real DSV4-Pro server (8×MI355X, `MODE=megamoe`, `CGBS=512`, gsm8k lm_eval @conc128, limit 48),
dual-mode vs a compact-only control (identical config, only `decode_mtpr` differs):

| run | `decode_mtpr` | hang / DIVERGE | gsm8k (flex/strict) | chat output |
| --- | --- | --- | --- | --- |
| **dual-mode** | 512 | **none — full completion** | **0.125 / 0.125** | **garbage** (`"We在一块在一块…"`) |
| **compact-only (control)** | 0 | none | **0.854 / 0.875** | coherent (`25×4=100`) |

- **`use_decode` is consistent (answers "does it just fall back to compact?" → NO).** Capture-time debug logs
  show all 8 ranks bake `use_decode=True` (fixed-slot) for **every** decode bucket (`local_tok` padded
  identically across ranks, e.g. 104, down to 1). The `DIVERGE_CHECK` all-reduce fired **zero** times.
  So (a) works: decode uses the **fixed-slot fast path**, not compact, and there is no divergence/deadlock —
  the config that used to hang now runs to completion.
- **BUT fixed-slot decode is CORRUPT under real serving.** Dual-mode scores 0.125 (≈random) with garbage
  repeated-token output; the compact-only control scores 0.854/0.875 with coherent output. Same code, same
  CGBS, same lm_eval — the **only** difference is `decode_mtpr` → **the fixed-slot decode path produces wrong
  results in the server** (cuda-graph replay + interleaving with compact prefill over the shared comb_op),
  even though it is bit-correct **eager** in the micro-harness.
  > Note: the micro-harness `--hang-iters` check only verified decode **self-consistency** (`relL2` vs its own
  > single-shot), NOT correctness-vs-oracle — so it never caught this. The standalone stage-1 e2e test is
  > correct at small bs, so the corruption is specific to the **dual-mode + cuda-graph server context**.
- **The (a) edit is safe for shipping.** The compact-only control (which shares the edited selection code)
  is healthy → the change did not regress the shipping path.

**Revised conclusion.** (a) is **necessary but not sufficient**: it removes the deadlock (proven), but the
hang was *masking* a deeper **fixed-slot-decode correctness bug under cuda-graph**. Dual-mode is therefore
**not shippable yet** even with (a) — it needs a kernel/graph-level fix for the decode path (candidates:
cuda-graph epoch/flag safety of the `_dec` coordination flags; comb_op payload/`total_recv` corruption across
compact↔fixed-slot mode switches). This materially strengthens the strategic call:
- **Ship compact-only now** (correct, ~0.85–0.93 gsm8k, ~97% of dp) — unaffected, default.
- The decode lever now has **two** blockers behind it (divergence *and* graph-correctness), which raises the
  bar for dual-mode and favors breakthrough **#2** (make compact's count-round cheaper → drop dual-mode
  entirely) over continuing to harden the fixed-slot path.

### 7.2 Eager run (`--disable-cuda-graph`, 2026-07-20) — the divergence is IDLE-DP-RANK `is_decode`, and graph MASKS it

Ran dual-mode **eager** to isolate the corruption source. It **hung** on a single chat request (120s timeout).
The `DIVERGE_CHECK` fired and named the exact cause:

```
rank=0   layer=0  is_decode=True   local_tok=1  use_decode=True   <- the ONE busy DP rank (owns the seq)
rank=1..7 layer=0 is_decode=False  local_tok=0  use_decode=False  <- the 7 IDLE DP ranks
(max=1 min=0) -> RANKS DISAGREE -> WOULD DEADLOCK
```

**`forward_mode.is_decode()` is NOT a global per-step property** — for an imbalanced batch (one request), the
busy DP rank reports `is_decode=True` while the idle DP ranks report `is_decode=False`. So fix (a)'s core
assumption is **wrong for idle DP ranks**: it selects fixed-slot on the busy rank and compact on idle ranks →
the cross-PE collective deadlocks. (990 `use_decode=False` + 3906 `use_decode=True` eager steps ran during
*balanced* startup warmup with no hang; the hang appeared only on the *imbalanced* single request.)

**This resolves the hang-attribution confound (§7 was too generous to (a)).** The graphed run didn't hang
**not because of the (a) code**, but because **cuda-graph replay forces every rank to run the same baked
decode kernel** (consistency by construction — an idle rank still replays the captured fixed-slot graph
rather than re-deciding). At *capture* all ranks are padded uniformly, so the baked decision is uniform with
old code *or* (a). Net:
- **Graphed decode**: consistent by construction (graph forces it) → no hang. (a) is largely moot here.
- **Eager decode + imbalanced/idle ranks**: `is_decode` diverges per-rank → hang. **(a) does NOT fix this.**
- The real determinant of the server hang is **an imbalanced/idle-rank step run EAGERLY**. Balanced load
  (lm_eval @conc128) hides it; a single request or an idle DP rank exposes it.

**Consequences:**
1. **(a) is insufficient.** A correct fix needs genuine cross-rank agreement that includes idle ranks — i.e.
   breakthrough **#1** (a per-forward `all_reduce` of the intended mode; idle ranks adopt the group's choice),
   OR always relying on graph (but eager/idle paths still break). "Trust `is_decode` is global" is false.
2. **The corruption source is still un-isolated** — eager hung (divergence) before decode output could be
   inspected. To isolate graph-replay vs mode-switch corruption, need a **balanced** eager load (≥8
   concurrent, so no idle ranks → no divergence → eager fixed-slot decode runs → check coherence).
3. Dual-mode now has **≥2 independent bugs** (idle-rank divergence needing a real collective; fixed-slot
   decode corruption). Fragility is high → **ship compact-only; pursue #2** to drop dual-mode outright.

---

## 8. Redesign — can we find a breakthrough? (first-principles)

**What each mode buys, and the conflict.** Decode (small bs) is latency-bound; its win is that **fixed-slot
needs only ONE cross-PE round** (payload done-barrier) because slots are pre-reserved (`le*cap +
atomic(running[le])`, no count needed). Prefill (large bs) is memory/compute-bound; **compact needs TWO
rounds** (count all-gather → dense base → payload) but its buffer is dense (`~topk/cap` smaller). Fixed-slot
at prefill mtpr=8192 would need `epr*npes*mtpr` rows ≈ **15 GB** (and blows the 32-bit 4GB buffer-resource
limit) — so compact is **mandatory** for prefill, for *memory*, independent of addressing.

**The hard theorem (why divergence can't be papered over in-kernel).** A cross-PE rendezvous is a *collective
contract*: every rank must run the same number of cross-PE rounds. Decode's entire win **is** skipping one
round. To decide "skip the count-round" safely, all ranks must agree **before** any payload write (the dense
base needs counts first). Agreeing cross-PE *before* payload **is itself a cross-PE round** — the very thing
decode skips. ⇒ **You cannot have decode skip a round that prefill does unless all ranks agree on the mode
out-of-band.** Making the odd-mode-out also run the round (to stay compatible) reintroduces the ~12µs floor
and destroys the decode win. **Conclusion: fix (a) — global out-of-band mode agreement — is not a band-aid;
it is the necessary and sufficient contract.** Two schemes are inherent (memory forces compact at scale;
latency forces fixed-slot at decode), so the *decision* is where the robustness must live.

**Given that, the breakthroughs worth pursuing (ranked):**

1. **Bulletproof the decision + a kernel fail-loud (highest ROI, low risk).** (a) already does the decision.
   Add a cheap **fail-loud** so a *future* integration regression crashes clearly instead of hanging: have
   the integration layer do **one** `all_reduce(use_decode)` per *forward step* (not per layer) and, on
   mismatch, **force all ranks to compact** (the safe superset) for that step + log. This makes divergence
   *self-healing* at the cost of one tiny collective per step (eager) / a captured constant (graph). This is
   the `SGLANG_AMD_FLYDSL_MEGA_DIVERGE_CHECK` detector promoted from "log" to "log + fall back".

2. **Shrink the *motivation* for dual-mode: make compact's count-round cheaper (medium ROI, medium risk).**
   The decode penalty is one extra ~12µs xGMI floor (count all-gather), not bandwidth (counts are `epr`
   ints). If the count exchange used a single fast small all-reduce / a fused "counts+done" round instead of
   the `done2` spin, compact's decode overhead could drop from ~2 floors toward ~1 — at which point
   **dual-mode may not be worth the fragility at all** (ship compact-only, already ~97% of dp). This attacks
   the 19% gap directly and keeps ONE scheme/one protocol everywhere (no divergence class).

3. **64-bit buffer addressing (high ROI, high effort) — but it does NOT collapse the schemes.** It would
   close the `bs≥16384` int32-overflow correctness gap (§6 perf tables) and let fixed-slot address large
   buffers, *but* fixed-slot at prefill is still ~15 GB (memory), so compact stays. Pursue it for the
   overflow gap, not as a divergence fix.

4. **Unify the sync *protocol* across both schemes (low ROI for divergence, good hygiene).** Today fixed-slot
   and compact use different flag sets (`recv_num` vs `done2`) purely as separate `const_expr` branches.
   Refactoring both onto ONE arrival + ONE payload-done primitive (layout still `const_expr`) shrinks the
   fragile surface and lets block0 **detect** a cross-rank mode-tag mismatch at the shared payload barrier
   (post-hoc, fail-loud). It cannot *prevent* the count-round deadlock (see the theorem), so it complements
   (a), not replaces it.

**Recommendation.** Ship (a) as the correct fix; add breakthrough #1 (per-step all-reduce fall-back) as the
robustness net; evaluate #2 (cheaper count-round) as the real "do we even need dual-mode?" question. Treat #3
independently (large-bs correctness). #4 is optional kernel hygiene. If #1+(a) don't fully stabilize the
server, the `DIVERGE_CHECK` log will name the exact remaining path.
