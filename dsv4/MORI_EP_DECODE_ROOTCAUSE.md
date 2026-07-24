# >>> NEXT ACTIONS (2026-07-15, resume here tomorrow) <<<

## ROLLBACK (2026-07-16): masked MoE feature REMOVED (prefix-slice KEPT)
The masked (deep-gemm) MoE feature was rolled back per request. See
MASKED_MOE_GFX950_CHANGES.md for the full code record + restore steps.
- aiter: grouped_moe_gfx950.py + test DELETED; the 4 tracked files
  (fused_moe.py, grouped_moe_gfx1250.py, mixed_moe_gemm_2stage.py,
  moe_route_maps.py) reverted to HEAD (git clean).
- sglang moe_runner/aiter.py: `_maybe_run_mori_masked` + its call + the
  `_masked_skip_upscale` pre_permute block REMOVED. **prefix-slice KEPT.**
- Backup for restore: masked_moe_rollback/ (patches + new-file copies).
- NOTE: prefix-slice debug env renamed SGLANG_MORI_MASKED_DEBUG ->
  SGLANG_MORI_DECODE_PREFIX_DEBUG. So the actions below about the MASKED feature
  (A6/C6) are on-hold/archived; A1-A5 (prefix-slice + comm) remain valid.


Current state: masked MoE CLOSED at 1.12x (canonical). Active thread = prefix-slice
(decode MoE auto-sizing to real tokens), opt-in SGLANG_MORI_DECODE_PREFIX_SLICE=1,
default safety=4, cuda-graph-safe, drop-check in DEBUG. Prefill already precise
(eager -> dynamic moe_sorting grid); imprecision is decode+cuda-graph only.
A1 DONE (2026-07-16): at a LARGE/untuned cap (its real use case) prefix-slice gives
+38%/+30%/+15% decode tput at C=64/128/256 (cap=2048, cuda-graph), gsm8k unchanged
(0.915 vs 0.945, ns). Mostly cap-independent (~8% residual = comm-side padding->B5).
A2 DONE: bs-dependent safety (floor+linear, SGLANG_MORI_DECODE_PREFIX_FLOOR) is
correct/drop-safe but NO e2e tput win (bottleneck off MoE at high bs) -> keep flat
safety=4 default, floor opt-in.
A3 DONE: full gsm8k no regr. at default (0.9212 vs 0.9287); under-sizing (safety=1)
CRASHES (GPU mem-fault, not silent drop) -> landed a safety guard (clamp <2 -> 4 +
warn), verified crash-free. Remaining: A4 (vs tuned fixed cap + default-on decision),
then B5 (comm-side padding = the real remaining perf lever). See sections below.

A. prefix-slice (decode speedup, no kernel change):
  A1. [DONE 2026-07-16] Validated under CUDA-GRAPH at large/untuned cap
      (cap=2048 -> M=16384): slice ON vs OFF = +38%/+30%/+15% decode tput at
      C=64/128/256 (TPOT -31%/-25%/-13%), gsm8k 0.915 vs 0.945 (NOT significant,
      z=1.18), 0 logged drops. Confirmed cap-INDEPENDENT (slice-ON @cap4096 ~=
      @cap2048). See "A1 VALIDATION" section below. => prefix-slice IS the env-free
      auto-cap: leave dispatch cap large, get near-optimal MoE size per bs.
  A2. [DONE 2026-07-16] Implemented min-floor + linear safety (SGLANG_MORI_DECODE_
      PREFIX_FLOOR; floor=128,safety=2). Correctness-equivalent + drop-safe (eager
      bs=0..128 = 0 drops, gsm8k 0.95), cuts M_bs 46% at bs=128, but NO e2e tput
      win at any concurrency (±2.4% noise) -- once slice right-sizes M, further
      MoE-side tightening is inert; at high bs bottleneck is comm-side. KEEP flat
      safety=4 default; ship floor+linear opt-in (robustness/anti-clamp only).
      => pivots the perf work to B5. See "A2" section below.
  A3. [DONE 2026-07-16] Full-set gsm8k (n=1319): default (flat safety=4) NO regr.
      (0.9212 vs base 0.9287, z=0.73). KEY: under-sizing (safety=1/floor=0) CRASHES
      (GPU memory-access fault @cuda-graph capture), not a silent drop -> safety is
      a CRASH-safety req. Landed a guard: floor<=0 & safety<2 -> clamp to safety=4 +
      warn once (verified it prevents the crash). Fallback under cuda-graph is
      impossible (Python not run on replay) => PREVENTION via conservative margin.
      See "A3" section below.
  A4. Compare vs a well-tuned fixed cap; decide default-on vs opt-in.

B. comm side (deeper EP-vs-DP lever):
  B5. Profile mori dispatch/combine a2a cost vs cap; check if comm moves padded
      buffers and whether comm volume can track real token count (MoE-side slice
      cannot touch this).

C. masked MoE (CLOSED):
  C6. combine-fusion e_vec=2 acc-layout bug -> hand to flydsl/aiter kernel owner
      (diagnosis + /tmp/dbg_combine.py). Else leave as documented dead-end.
      Before ANY new attempt on a flydsl MFMA kernel, follow the repeatable
      playbook FLYDSL_KERNEL_AUTHORING.md (observability + true reference +
      micro-harness + escalation criteria) -- it exists specifically to stop the
      blind-IR-iteration failure this C6 bug represents.

Key facts to remember:
- origin_bs (origin_topk_ids.shape[0]) = local tokens SENT (static, cuda-graph-ok),
  NOT received. total_recv (received) ~= origin_bs * ep_size ~= origin_bs*topk*1.33.
- CUDA DeepEP-LL uses the SAME estimate+device-mask pattern (expected_m static
  estimate + masked_m device real count); it also can't be dynamically precise
  under cuda-graph. Only DeepEP-normal compacts precisely (dynamic shape, not graph).
- M = cap * world (world=8 here). cap=32 -> M=256. Constraint M >= total_recv.

Files touched (all opt-in / guarded, default behavior unchanged):
- moe_runner/aiter.py: prefix-slice block (SGLANG_MORI_DECODE_PREFIX_SLICE/_SAFETY).
- grouped_moe_gfx950.py + mixed_moe_gemm_2stage.py + moe_route_maps.py: masked path
  (a4w4/route/a2-fusion landed; combine-fusion dead-end opt-in OFF).
- run_sgl_dsv4_masked.sh: SGLANG_MASKED_CUDA_GRAPH toggle, a2 dtype defaults.

---

# A1 VALIDATION (2026-07-16) — prefix-slice @ LARGE/untuned cap, cuda-graph

Setup: NON-masked (default) path, SGLANG_MASKED_CUDA_GRAPH=1, safety=4, decode
random 2000in/400out, concurrency sweep. Driver: a1_prefix_slice.sh (one config
per launch: launch->health->sweep->gsm8k->teardown). "Large/untuned cap" =
SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS large -> M_padded = cap*8. This is the
slice's REAL use case (at the small tuned cap=256 the safety=4 slice clamps).

cap=2048 (M_padded=16384), slice OFF vs ON (safety=4):
| C   | base tok/s | slice tok/s | Δtput  | base TPOT | slice TPOT | ΔTPOT  |
|-----|-----------:|------------:|-------:|----------:|-----------:|-------:|
| 64  |     1058.9 |      1462.1 | +38.1% |     54.9  |      38.0  | -30.8% |
| 128 |     1746.1 |      2265.3 | +29.7% |     65.4  |      49.3  | -24.7% |
| 256 |     1760.8 |      2022.0 | +14.8% |    111.4  |      97.4  | -12.6% |
gsm8k (limit=200): base 0.945/0.950, slice 0.915/0.910. Δ=0.03 flexible,
z=1.18 => NOT statistically significant (2-sample binomial, SE~0.026). => no
evidence of a drop-induced regression at safety=4. (CAVEAT: the A1 driver set the
OLD debug var SGLANG_MORI_MASKED_DEBUG, but the drop-log is now gated by
SGLANG_MORI_DECODE_PREFIX_DEBUG, so the "0 drop events" grep was vacuous -- and
the log only fires on eager/capture steps anyway, never on graph replay. gsm8k is
the real replay-time correctness signal; a1_prefix_slice.sh now sets the correct
var for future runs.)

Win is LARGER at lower concurrency (small bs -> more padding waste relative to
real tokens -> M_bs=round_up(bs*topk*4,32) is a much smaller fraction of M). At
C=256 (bigger bs) M_bs grows so the padding cut shrinks -> smaller (but still
+15%) win. Matches the theory exactly.

Cap-independence check (slice ON, safety=4, C=128):
| cap  | M_padded | slice tok/s | TPOT   |
|------|---------:|------------:|-------:|
| 2048 |   16384  |      2265.3 | 49.3ms |
| 4096 |   32768  |      2087.6 | 53.6ms |
=> MOSTLY cap-independent on the MoE side (slice sizes GEMM by bs, not cap), BUT
~8% residual cap-dependence remains (2265 -> 2088 as cap 2x). The MoE-side slice
CANNOT touch it: the mori dispatch/combine a2a still moves the FULL cap*world
padded buffer. This is direct motivation for B5 (comm-side padding is the
remaining EP-vs-DP lever). Even so, slice@cap4096 (2088) still >> base@cap2048
(1746), i.e. the slice recovers most of the loss from an untuned-large cap.

CONCLUSION (A1): prefix-slice is the env-free, per-bs, cuda-graph-safe auto-cap.
Leave SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS large and the slice auto-tightens
the MoE to origin_bs*topk*safety for every captured bs -> +15..38% decode tput,
no manual cap tuning, correctness intact. Residual cap cost is comm-side (B5).

Repro: a1_prefix_slice.sh; logs a1_{driver,srv,bench,gsm8k}_{base_cap2048,
slice4_cap2048,slice4_cap4096}*.

---

# A3 (2026-07-16) — correctness hardening: full gsm8k OK + under-size = CRASH + guard

Three full-set (n=1319) gsm8k runs, cuda-graph, cap=2048/M=16384, no bench:
| config                         | gsm8k (flex=strict) | note                          |
|--------------------------------|--------------------:|-------------------------------|
| baseline (slice OFF)           |      0.9287 ±0.0071 | reference                     |
| slice ON safety=4 (DEFAULT)    |      0.9212 ±0.0074 | Δ=0.0075, z=0.73 => NO regr.  |
| slice ON safety=1 floor=0 (NEG)|      CRASHED         | GPU memory-access fault @cap  |
Full-set confirms the shipping default (flat safety=4) has NO correctness
regression (2-sample z=0.73, not significant).

**KEY A3 FINDING**: the negative control (safety=1, floor=0) did NOT merely drop
tokens -- it triggered a HARD "Memory access fault by GPU" on all ranks DURING
cuda-graph capture (crash in the MoE fwd; the downstream mori combine/moe_sorting
indexes rows up to the real total_recv, so a recv sliced below total_recv -> OOB).
=> Under-sizing M_bs is CATASTROPHIC (crash), not a silent accuracy dip. A
conservative safety is therefore a CRASH-safety requirement. (safety=4 and
floor=128/safety=2 both capture & run fine; the crash threshold is between 1 and 2.)

HARDENING (landed in aiter.py, guarded): when slice is ON with floor<=0 and
safety<2, clamp effective safety up to the validated-safe default (4) and log a
one-time WARNING. Verified: relaunching the exact crashing config (safety=1) now
logs the warning, runs at effective safety=4 (M_bs=192@bs8), CAPTURES SUCCESSFULLY
(no fault), gsm8k 0.9 (limit=50). Default path (safety=4) and floor mode are
untouched by the guard.

FALLBACK decision (A3 sub-item "what to do when a drop would occur"): a per-step
host fallback is impossible under cuda-graph (Python is not executed on graph
replay; M_bs is baked per captured bs). So the correct design is PREVENTION via a
conservative static margin: safety=4 gives 3x headroom over the measured peak
ratio (~1.33 at real bs; small-bs spikes are tiny-absolute, covered by the tile
floor / additive floor), which the full gsm8k + the crash-free capture confirm.
The eager PREFIX_DEBUG drop-log remains for calibrating new workloads/topologies.
Repro/logs: a3_chain.sh; a1_*_{a3_base_full,a3_slice_s4_full,
a3_slice_s1_full_NEGCTRL,a3_guard_s1_check}*.

---

# A2 (2026-07-16) — bs-dependent safety: IMPLEMENTED, correct, but NO perf win

Impl: added SGLANG_MORI_DECODE_PREFIX_FLOOR (rows). When >0:
  M_bs = round_up(floor + local_bs*topk*safety, 32)
= additive min-floor + linear multiplier. floor absorbs small-bs variance + the
near-0-bs ratio artifact ABSOLUTELY (so a small `safety` suffices); the multiplier
tracks the asymptotic regime -> effective safety HIGH at small bs, LOW at large bs.
floor=0 => original flat-safety behavior (A1-validated), unchanged/back-compat.
(aiter.py AiterRunnerCore.run, guarded, opt-in.)

Calibration (EAGER, PREFIX_DEBUG=1 drop-log fires every step; truncation drops
tokens even in eager so it's a valid correctness harvester). floor=128, safety=2,
bs=0..39 swept: **0 DROP events**. Observed total_recv/(bs*topk): ~1.33 at bs>=8
(matches doc), <1 at the c=256 samples, and only the near-0-bs artifact spikes
(maxratio 16.5 at bs=0, total_recv=4 << floor=128). Worst real case bs=37 ->
total_recv=177 vs M_bs=576 (3.3x headroom). gsm8k(eager,limit=100)=0.95 (=baseline).

Perf A/B (CUDA-GRAPH, cap=2048/M=16384): A2(floor=128,safety=2) vs A1(flat safety=4):
| C    | A1 flat-s4 | A2 f128s2 | Δ      | note (M_bs @ that bs)          |
|------|-----------:|----------:|-------:|--------------------------------|
| 64   |     1462.1 |    1477.2 |  +1.0% | bs~8:  s4=192  a2=224          |
| 128  |     2265.3 |    2254.4 |  -0.5% | bs~16: s4=384  a2=320          |
| 256  |     2022.0 |    2070.2 |  +2.4% | bs~32: s4=768  a2=512          |
| 512  |     2331.4 |    2330.4 |  -0.0% | bs=128 seen: s4=3072 a2=1664   |
| 1024 |     2252.7 |    2272.5 |  +0.9% | bs=128: s4=3072 a2=1664 (½!)   |
gsm8k(cuda-graph,limit=200) A2=0.925 vs A1-slice 0.915 vs base 0.945 (all ns).

VERDICT (A2): bs-dependent safety is CORRECTNESS-equivalent, drop-safe, and cuts
M_bs a lot at large bs (46% at bs=128: 3072->1664), but gives **NO measurable e2e
tput change at ANY concurrency (all within ±2.4% = noise)**. KEY INSIGHT: once
prefix-slice right-sizes M to O(bs*topk*1.33), the residual safety headroom (2x vs
4x) is too small a fraction of step time to matter -- and at HIGH bs the bottleneck
has moved OFF MoE compute entirely (halving M_bs 3072->1664 did nothing e2e). This
points straight at B5 (comm-side a2a padding is the remaining lever at high load).

RECOMMENDATION: keep flat safety=4 as the simple default (A1-validated). A2's
floor+linear is a good ROBUSTNESS option (tighter M, drop-safe, and it avoids the
flat-safety clamp at a SMALL cap: bs=128 s4=3072 would clamp a 2048-cap but a2=1664
still slices) -- but it is NOT a perf lever. Ship opt-in (floor=0 default).
Repro: a1_prefix_slice.sh with SGLANG_MORI_DECODE_PREFIX_FLOOR; logs
a1_*_{a2cal_f128s2_eager,a2_f128s2_cap2048,a1s4_hc_cap2048,a2f128s2_hc_cap2048}*.

NEXT-LEVER NOTE (data-driven pivot): A1 showed prefix-slice recovers most of an
untuned-large-cap loss (MoE-side), A2 showed further MoE-side tightening is inert.
=> the remaining EP-vs-DP decode gap at load is COMM-side (B5): mori dispatch/
combine a2a still moves the full cap*world padded buffer (also explains A1's ~8%
residual cap-dependence). B5 is now the highest-value thread.

---

# PREFIX-SLICE CALIBRATION (2026-07-15) — formula + safety

Calibrated the M_bs formula with DEBUG logging of real total_recv (= num_local_tokens):
- WRONG idea (dropped tokens): M_bs = origin_bs*topk*(E_local/E_global). The
  *E_local/E_global (=1/ep) UNDER-sized by ep_size -> observed origin_bs=8 got
  total_recv=64 but M_bs=32 -> DROP. DO NOT use the ep fraction.
- Empirical: **total_recv ~= origin_bs * ep_size** (bs1->recv8, bs8->recv64 with
  ep=8), i.e. ~= origin_bs*topk * (dp/ep). For DSV4 (dp==ep=8) ~= origin_bs*topk*1.33.
- total_recv/(origin_bs*topk) has HIGH VARIANCE at small bs (peaks ~2.8; a near-0
  bs gives a ratio artifact up to ~9 but tiny absolute, covered by the 32-row floor).
- Correct formula: **M_bs = round_up(origin_bs * topk * safety, 32)**.
  safety=2 -> 6 drops (small bs); **safety=4 -> 0 drops**, gsm8k ~0.875 (limit40 noise,
  ~= baseline). Default set to 4.

CAVEAT (benefit window): with a SMALL cap (e.g. 256 -> M_padded=2048), safety=4 can
clamp at large bs (128*topk*4 > 2048 -> no slice). The slice's real value is with a
LARGE / untuned cap (the user's actual problem): it auto-tightens M to
origin_bs*topk*4 (drop-safe) for every captured bs, so you can leave the dispatch
cap large and still get near-optimal MoE size WITHOUT manual tuning.

# PREFIX-SLICE PROTOTYPE (2026-07-15) — WORKS under cuda-graph (+6.7%)

Goal: "process only the real decode tokens (like CUDA), env-free" on the DEFAULT
(non-masked) path. mori returns a STATICALLY padded recv [M_padded,K] (world*cap).

CRUCIAL mode caveat (I initially got this wrong):
- **EAGER**: `moe_sorting` sizes the GEMM grid DYNAMICALLY to `num_valid` (real),
  so M_padded is free -> an eager A/B (452 vs 456 tok/s) wrongly showed NO win.
- **CUDA-GRAPH** (real deployment): the grid is CAPTURED at M_padded/tile blocks,
  so MoE kernel time scales with M (matches user profiling: EP stage1 169us@M=2048
  vs DP 47us; smaller SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS -> near-DP).

v2 (cuda-graph-safe): slice recv to `M_bs = round_up(local_bs*topk*safety, 32)`
where `local_bs = origin_topk_ids.shape[0]` (STATIC per captured decode graph),
then pad the MoE output back to M_padded for mori combine. Tracks the real decode
size per captured bs -> auto per-bs equivalent of the dispatch cap, no manual tune.

A/B (CUDA-GRAPH, cap=256=M2048, C=128, 2000in/400out):
| | tok/s | TPOT | gsm8k |
|-|------|------|------|
| slice OFF (M=2048) | 2163 | 47.1ms | - |
| slice ON v2 (M_bs=1536, safety2) | **2308** | **43.7ms** | 0.9 |

=> **+6.7% tput, -7% TPOT**, correct. Only a 25% M cut here (bs=128, safety=2);
bigger win at smaller bs or safety=1 (M_bs tracks local_bs*topk tightly). This is
the env-free, cuda-graph-safe, per-bs auto version of the decode dispatch cap.
Opt-in SGLANG_MORI_DECODE_PREFIX_SLICE=1 (+ SGLANG_MORI_DECODE_PREFIX_SAFETY).
NOTE: safety factor must keep M_bs >= actual total_recv (else drops tokens).

---

# ✅✅ CLOSED (2026-07-15): MASKED MoE = canonical design perf, 1.12× default

Final status of the DSV4 mori-EP masked (deep-gemm-style) MoE effort:

- **Correctness**: gsm8k 0.92–0.93 (= baseline), eager + cuda-graph, unit 4e-6–6e-4.
- **Perf** (decode, cuda-graph, 2000in/500out, C=128): masked **2157 tok/s / TPOT
  50.5 ms = 1.12× default** (2426 / 43 ms). Up 4× from the 536 tok/s start.
- **cuda-graph-safe**: 0 capture faults.

Perf wins that landed: a4w4 direct fp4 (no round-trip) + on-device guarded route
kernel (replaced one_hot+cumsum) + a2 quant fused into stage1 epilogue
(masked-aware). Route-map + a2-fusion are the big ones.

**This 1.12× IS the canonical grouped-MoE design ceiling on mori normal dispatch.**
The shipping gfx1250 grouped MoE (`grouped_moe_gfx1250.py`) uses the SAME design:
token-major recv -> route/scatter into expert-major [E,max_m] -> masked GEMMs ->
SEPARATE `flydsl_moe_gather_reduce` combine. Our masked path mirrors it exactly.

Two dead-ends (ruled out, documented):
- **low-latency dispatch**: mori LL returns token-major recv (NOT expert-grouped
  like CUDA DeepEP-LL), so masked still routes/scatters -> NO structural win; LL
  only changes comm (equal for default & masked). Not pursued.
- **combine fusion into stage2**: non-canonical (gfx1250 also uses gather_reduce)
  AND hits a kernel bug (e_vec=2 masked accumulate epilogue misreads MFMA acc:
  out[r]=c[r%tile_m]*s2[r]). Kept opt-in OFF + diagnostics; needs kernel owner.

To actually BEAT default would require a genuinely expert-grouped dispatch (which
mori does not provide) or the contiguous-compaction option gfx1250 has (but
padding is not our bottleneck: max_m 128->64 was only +11%).

Env: SGLANG_MORI_MASKED_MOE=1, SGLANG_MORI_MASKED_A1_DTYPE=fp4,
SGLANG_MORI_MASKED_A2_DTYPE=fp4 (fused), SGLANG_MORI_MASKED_ROUTE=ondevice,
SGLANG_MASKED_CUDA_GRAPH=1. Repro of the dead-end combine bug: /tmp/dbg_combine.py.

---

# ✅ MASKED MoE e2e RESOLVED (2026-07-15) — gsm8k 0.93 = baseline

The masked (deep_gemm-style, padding-free) MoE GEMM now works **e2e-correct** on
real DSV4 mori-EP decode: **gsm8k flexible/strict = 0.93** (limit=200, eager),
matching the DP/default baseline; sniff output fully coherent.

**Root cause of the earlier garbage (gsm8k=0):** masked used the WRONG per-stage
config vs the shipping DSV4 decode default. The default fused_moe key is
`(gfx950, ..., Silu, bf16, torch.float8_e4m3fn, float4_e2m1fn_x2, per_1x32, ...)`
with kernel `flydsl_moe1_afp8_wfp4_..._gui` → i.e. **a8w4 (FP8 activation) +
gate_mode INTERLEAVE (GUGU, the `_gui` suffix)**. masked was doing **a4w4 (fp4
activation) + gate_mode SEPARATED (GGUU)**:
  1. wrong activation dtype (fp4 vs fp8) → coarse quant, ~15% diff; and
  2. wrong gate_mode → gate/up halves misinterpreted (GGUU vs GUGU) → garbage.

The in-branch "REALDIFF" ref was ALSO a4w4-separated (didn't pass gate_mode), so
it wrongly agreed with masked at 0.02 — masked matched a wrong oracle, not the
real default. **The unit test likewise never caught it: its torch reference uses
masked's OWN per-stage dtype/gate_mode, so it validates kernel mechanics, not
precision-path parity with the shipping default.**

**Fix (all three needed):**
  - `moe_runner/aiter.py` masked branch: `gate_mode="interleave"`.
  - `grouped_moe_gfx950.py` recv primitive: mxfp4 recv is DEQUANTED fp4→bf16 then
    re-quantized per-1x32 **FP8** (a8w4 stage1), matching the default's q_dtype_a=fp8.
  - stage2 a2 = **fp8** (`SGLANG_MORI_MASKED_A2_DTYPE=fp8`, now the script default).

Unit test `test_flydsl_masked_moe_stage1_gfx950.py --stage recv --recv-mode fp4`
still PASS (4.18e-6).

## ✅ CUDA-GRAPH-SAFE (2026-07-15) — masked captures, gsm8k 0.93 in graph mode

The masked path is now fully cuda-graph-capturable (0 capture-sync errors) and
gives the SAME gsm8k 0.93 under cuda-graph as eager. Removed every host sync /
data-dependent shape (each surfaced one at a time as
`HIP error: operation not permitted when stream is capturing`):
  1. branch `base`/`total_recv`: `.item()` -> keep as 0-d DEVICE tensors; guard
     the DEBUG print with `torch.cuda.is_current_stream_capturing()`.
  2. routing: boolean-index `x[keep]` (dynamic shape) -> full-size static
     `index_put_` into a `[E*max_m + 1]` buffer with a dedicated TRASH row
     (invalid slots -> row E*max_m, sliced off); `bincount(lc[keep])` ->
     `(one_hot(lc,E) * keep).sum(0)`.
  3. fp4->bf16 dequant: `fp4_utils.mxfp4_to_f32` does
     `torch.tensor(list, device=cuda)` EVERY call (host->device memcpy) and
     `e8m0_to_f32` uses boolean-mask assign -> wrote
     `_mxfp4_dequant_bf16_graphsafe` with a module-CACHED 16-entry LUT + bit-op
     e8m0 decode.
  4. `_quant_per1x32_fp8`: chained `f32_to_mx_e8m0_scale` (ceil impl has
     `exp[nan_case]=0xFF` bool-assign) + `e8m0_to_f32` -> rewrote self-contained
     with bit ops + `torch.where` (bit-exact: unit test still 4.18e-6).
  5. stage1/stage2 `num_valid_ids = torch.tensor([tokens_in], device=)`
     (host->device memcpy) -> `torch.full((1,), tokens_in, device=)` (device
     fill; scalar unused by the masked scheduler anyway).

Note: allocations (`torch.empty/zeros/full`) during capture ARE allowed (caching
allocator's graph pool) — only host<->device MEMCPY and data-dependent shapes /
`.item()` are illegal. Launch: `SGLANG_MASKED_CUDA_GRAPH=1` (script drops `--disable-cuda-graph`).

## ⚠️ PERF (2026-07-15): masked correct but ~4x SLOWER than default (needs fusion)

Decode A/B, cuda-graph, `bench_serving` random 2000in/500out, C=128, np=256:

| path | Output tok/s | Mean TPOT | duration | overflow |
|------|-------------|-----------|----------|----------|
| default (masked OFF) | **2426** | **43.0 ms** | 52.8 s | - |
| masked max_m=128 | 536 | 215.1 ms | 238.6 s | 0 |
| masked max_m=64  | 595 | 193.3 ms | 215.2 s | 0 |

masked adds ~150-170 ms/decode-step. **Padding is NOT the bottleneck**: halving
max_m (128->64) only bought ~11% (536->595). The cost is the many DISCRETE EAGER
TORCH glue ops in `flydsl_masked_moe_gfx950_recv` — fp4->bf16 dequant
(`_mxfp4_dequant_bf16_graphsafe`, M*7168), bf16->fp8 requant
(`_quant_per1x32_fp8`, 6144*3072), one_hot(8192,48).cumsum routing, the
`[E*max_m+1, K]` zero-init + scatter, 2x `e8m0_shuffle`, gather_reduce — vs the
default's SINGLE fused `fused_moe` (on-device route+GEMM+combine).

To make masked competitive (future work):
  1. Replace eager routing with the on-device kernels already in
     `grouped_moe_gfx1250.py` (`build_route_maps`, `flydsl_moe_scatter_copy_token`,
     `contiguous_psum`) instead of one_hot/cumsum/index_put_ in torch.
  2. Kill the fp4->bf16->fp8 round-trip: fuse dequant+requant into one kernel, or
     add a native "a8w4 activation from mxfp4 recv" quant path.
  3. Fuse the per-1x32 fp8 activation quant + e8m0_shuffle into the stage kernels.

Correctness is done (gsm8k 0.93, eager+graph); perf is the open item.

## COMBINE FUSION (in progress, opt-in OFF) — deep epilogue bug isolated

Goal: fuse the weighted un-permute (★combine) into masked stage2 (like default's
accumulate) to drop the gather_reduce + s2 HBM round-trip (~11%). Wiring done:
row_to_token (arg_expert_ids slot, new E*max_m resource), row_weight
(arg_sorted_weights slot, already E*max_m), accumulate+doweight under
grouped_masked_m, token-major precompute_row. Isolated debug: /tmp/dbg_combine.py.

Fixed: bf16 `llvm.AtomicRMWOp fadd` unsupported -> use buffer-atomic
(raw_ptr_buffer_atomic_fadd), and OOB-sentinel byte-offset for invalid rows.

REMAINING BUG (precisely isolated): un-permute PLACEMENT is correct (~340/344
tokens hit), but the written VALUES are wrong AND only 256/512 (one tile_n) of
model_dim columns are written per row. Present with cshuffle on AND off, weight=1.
=> the masked grouped acc is mis-consumed by the `c_shuffle_epilog`+`accumulate`
(e_vec=2) store path.

FINAL ISOLATION (identity map grouped-row r -> token r, weight=1, so out[r] should
== grouped s2[r]): out[r] = a PER-ROW-VARYING SCALE of s2[r] -- token0 ~0.03,
per-row ratio mean 0.11, range 0.00-0.33 (all 512 cols written). So it is NOT
placement, NOT weight, NOT column-coverage: the ACCUMULATE store (buffer-atomic,
e_vec=2) reads the masked MFMA acc registers with the wrong layout/stride vs how
the working grouped PLAIN store (e_vec=8) reads the same acc. Per-row-varying
scale => the acc fragment->store mapping is wrong per MFMA row under
accumulate+grouped_masked_m. The non-masked (sorted) accumulate path works because
its acc/row arrangement differs.

CONCLUSION: fixing needs flydsl/MFMA kernel expertise (trace acc register indexing
in c_shuffle_epilog write_row_to_lds/store_pair for e_vec=2 under grouped_masked_m)
or true device printf -- beyond reliable blind IR iteration. Combine fusion kept
opt-in OFF (SGLANG_MORI_MASKED_FUSE_COMBINE=0). Repro: /tmp/dbg_combine.py.
Default gather_reduce combine validated: 2157 tok/s, gsm8k 0.92, unit 5.9e-4.
Recommendation: park combine on normal dispatch (capped at parity anyway); the
big win is --deepep-mode low_latency (masked skips route/scatter). The a2/route
fusions already landed are reusable there.

## PERF OPT ROUND 2 (2026-07-15): fused a2 quant -> masked ~matches default

KEY REFRAME (user): the masked GEMM ALREADY only processes masked_m rows (correct
deep-gemm behaviour; stage1/stage2 were only 7/9% in the profile). The waste was
that my SURROUNDING torch glue (esp. the a2 quant) ran DENSE over all E*max_m
rows. Fix = make the a2 quant masked-aware by FUSING it into stage1's epilogue
(exactly what default does via `fuse_quant`):
  - `flydsl_masked_moe_stage1(out_dtype="fp4"/"fp8")` now emits the quantized a2 +
    tiled e8m0 scale directly (SGLANG_MORI_MASKED_FUSE_A2=1, default). Only
    masked_m live rows are written.
  - **kernel bug fixed** in `mixed_moe_gemm_2stage.py`: the fused-quant scale write
    used the LOCAL tile `row` instead of the masked GLOBAL row
    `expert_idx*max_m + row` (line ~2549). Under masked mode every expert collided
    on rows [0,max_m) -> a2 scale misaligned vs stage2 (unit 0.85). Guarded by
    `if const_expr(grouped_masked_m)`, so the non-masked/default path is unchanged.
    After fix: unit logits_diff 5.9e-4.

Decode A/B (cuda-graph, 2000in/500out, C=128):

| config | Output tok/s | TPOT | vs default |
|--------|-------------|------|-----------|
| masked start (a8w4, torch route) | 536 | 215 ms | 4.5x |
| + a4w4 + on-device route | 1204 | 94 ms | 2.0x |
| **+ fused a2 quant** | **2157** | **50.5 ms** | **1.12x** |
| default | 2426 | 43 ms | 1.0x |

gsm8k 0.92, cuda-graph 0 faults. From 536->2157 = 4x over starting masked; gap to
default 4.5x->1.12x. Remaining ~12%: route (19%) + combine/gather (11%) still
separate ops -- fuse combine into stage2 / use moe_sorting-style fused route next.

## PERF OPT ROUND 1 (2026-07-15): 2.2x faster (536 -> 1204 tok/s)

Profiled the masked MoE per-section (eager, SGLANG_MORI_MASKED_PROF=1) -> route
one_hot+cumsum = **57%**, a2 fp4 quant = **24%**, GEMMs only ~8%. So neither the
fp4->bf16->fp8 round-trip nor max_m padding was the bottleneck (both <11%). Two
changes:
  1. **a4w4 direct (SGLANG_MORI_MASKED_A1_DTYPE=fp4)**: scatter mxfp4 recv bytes
     DIRECTLY (no dequant/requant round-trip). Correct WITH gate_mode=interleave
     (gsm8k 0.92). The earlier a4w4 failure was purely the wrong gate_mode.
  2. **on-device route (SGLANG_MORI_MASKED_ROUTE=ondevice, default)**: replaced
     one_hot+cumsum+torch-scatter with the atomic route kernel
     `build_moe_route_maps_guarded` (new BOUNDS-GUARDED variant: skips
     e>=experts / slot>=max_m; the unguarded `build_route_maps` OOB-stores ->
     GPU memory-access-fault under cuda-graph capture). Invalid slots -> id E
     (skipped); masked_m = atomic.clamp(max=max_m); grouped built by GATHER
     `recv[rows_to_tokens]` instead of index_put scatter.

Decode A/B (cuda-graph, 2000in/500out, C=128):

| config | Output tok/s | TPOT | vs default |
|--------|-------------|------|-----------|
| default | 2426 | 43 ms | 1.0x |
| masked a8w4 (round-trip, torch route) | 536 | 215 ms | 4.5x slower |
| masked a4w4 (no round-trip, torch route) | 587 | 196 ms | 4.1x slower |
| **masked a4w4 + on-device route** | **1204** | **94 ms** | **2.0x slower** |

gsm8k 0.92 (limit100), unit 4.18e-6, cuda-graph 0 faults, all preserved.
Next lever: the a2 fp4 quant (was 24%) -- fuse into stage1 epilogue (emit fp4
a2 directly) or stage2 prologue; then combine/gather. Still fundamentally ahead
only via low-latency dispatch (no scatter at all).

---

# mori-EP decode slowdown — root cause (2026-07-03)

Why `tp8 + dp8 + mori-ep` is ~4× slower than the non-EP DP path on DSV4-Pro
(8×MI355X, 8k/1k). Investigated 2026-07-03 with decode traces. **The bottleneck
is NOT the mori a2a comm — it is that the EP MoE misses the tuned aiter kernel
and falls back to an expensive fp4 heuristic path.**

Code refs:
- sglang: `/sgl-workspace/sglang-upstream` branch `feat/dsv4-ep-tbo-prefill`
- aiter: `/sgl-workspace/aiter` (HEAD incl. PR #3856 opus a8w4 decode); flydsl `0.2.2`
- launch: `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh` (`MODE=dp` vs `MODE=mori-ep`)

---

## 1. Baselines (8k in / 1k out, conc256, unified script, flydsl 0.2.2)

| metric | DP (non-EP, `MODE=dp`) | mori-EP (`MODE=mori-ep`) | EP vs DP |
|---|---:|---:|---:|
| Total throughput | **30,994 tok/s** | **7,480 tok/s** | **−75.9% (DP 4.14×)** |
| Output throughput | 3,444 | 831 | −75.9% |
| Mean TTFT | 16,361 ms | 20,913 ms | +27.8% |
| Median TPOT | **58.2 ms** | **287.6 ms** | **+394% (~5×)** |
| duration (2048 req) | 609 s | 2,523 s | 4.14× |

The gap is overwhelmingly **decode** (TPOT ~5×), not prefill (TTFT ~+28%).
`--enable-prefill-delayer` was ruled OUT as the cause: EP decode batches ran at
FULL occupancy (32 req/rank = conc256, steady) — the delayer only helps when
decode is *drained*, which isn't happening here.

## 2. Decode trace method

Both engines: `--disable-cuda-graph` (eager → real per-kernel durations; cuda-graph
makes decode kernel `dur` garbage, see `TRACE_PROFILING.md §2`), **GPU-only** profile
(`POST /start_profile {"activities":["GPU"],"with_stack":false,"record_shapes":false}`
— otherwise python-stack recording bloats the trace to 2GB+ and truncates). Same
wave: ISL 8192 / OSL 2000 / conc128, wait for pure decode, profile ~6s, rank 0.
Normalize by per-(MoE-layer×step) count (DP 2867, EP 1403; ratio 2.04 ≈ decode
throughput ratio). GPU busy ~92–97% both (not idle-waiting on comm).

## 2b. ⚠️ Decode-trace analysis METHODOLOGY — the pitfall that produced wrong numbers (READ FIRST)

Several earlier sections (§5q, §5r, §5x) reported **wrong** per-step/per-layer decode numbers (e.g.
"DP decode MoE = 380 µs/layer", "EP does *less* work than DP") because of a bad measurement method.
§5y corrected them. Record the method so it isn't repeated.

**WHY the early analysis was wrong (compounding errors):**
1. **Prefill contamination (the big one).** The profiler window captured continuous batching =
   prefill chunks **interleaved** with decode steps. `mfma_moe1/2` fire for BOTH prefill (large M,
   slow) and decode (small M, fast). Summing **all** MoE-kernel durations across the window and
   dividing by the *decode*-step count attributes prefill work to decode → inflates the "per decode
   step" MoE massively. This is what made DP look like 380 µs/layer when the true decode value is
   77 µs/layer.
2. **Bad step-count normalization.** Detecting `nstep` as "count of a per-layer decode kernel ÷ 61"
   gave `nstep=1` for a window that actually held several prefill chunks + one decode step → dividing
   a prefill+decode sum by 1 = garbage.
3. **cuda-graph spin/overlap contamination (DP).** DP's RCCL collectives spin-wait on a serialized
   stream (§5r), so sum-of-durations overcounts DP comm ~40× and bleeds wait into other buckets. A
   cuda-graph "sum of kernel durations" is NOT a valid step time for DP.
4. **Cross-config bucket mis-attribution.** DP and EP decompose the MoE into DIFFERENT kernels
   (DP `opus_moe_stage2`; EP `mfma_moe2`; different tiles) that fall into different categorizer
   buckets → bucket-by-bucket EP-vs-DP is apples-to-oranges.

**THE CORRECT METHOD (what actually works — matches hand-reading the trace):**
1. Pick **one rank** (e.g. rank 7) — not an aggregate.
2. **Isolate the LAST decode step**: the final contiguous cluster of kernels, *after all prefills have
   drained* (pure decode path). Segment by idle gaps and take the **last** segment; verify it has
   exactly `num_MoE_layers` (61) of the per-layer stage-1 kernel = exactly one decode step.
3. Read **per-launch** durations of the kernels that fire **once per layer** in that step
   (stage1, stage2, dispatch, combine, allgather, reduce_scatter) — i.e. per-LAYER, not summed.
4. Compare EP vs DP **by kernel role** (stage1↔stage1, combine↔reduce_scatter), NOT by categorizer
   bucket, and NOT by whole-trace sums.
5. For DP comm specifically: cuda-graph durations are spin-inflated — read the comm kernel in the
   clean last-decode-step only, or use an **eager** trace for real serialized durations.

**Rule of thumb:** never divide a whole-trace kernel sum by a step count when prefill+decode are
mixed. Always clip a single, pure decode step first. Absolute per-step numbers from cuda-graph traces
are unreliable for anything on a spinning/serialized comm stream — use per-layer, last-step, per-role.

## 3. Per-(MoE-layer × step) decode GPU-time breakdown

| component | DP (non-EP) | mori-EP |
|---|---:|---:|
| fp4 weight **upscale** (`upscale_fp4x2_block32`) | **absent** | **1.16 ms (26.2%)** |
| per-group act **quant** (`dynamic_per_group_scaled_quant`, EP 32×32) | tiny | **1.74 ms (39.6%)** |
| MoE GEMM2 down | `opus_moe_stage2_a8w4_decode` **0.067 ms** | `mfma_moe2_cshuffle` **0.61 ms** |
| MoE GEMM1 up/gate | 0.11 ms | 0.41 ms |
| DP-attn gather (`allgather_vec`) | 1.03 ms (54%) | — |
| DP-attn combine (`reduce_scatter`) | 0.37 ms (19%) | — |
| **mori a2a dispatch+combine** (`EpDispatch/EpCombineIntraNode`) | — | **0.10 ms (only 2.4%)** |
| **per layer×step total** | **~1.94 ms** | **~4.41 ms (2.27×)** |

**mori a2a is only 2.4% of EP decode** — and is actually *cheaper* than DP's
all_gather+reduce_scatter (which is 73% of DP decode). The EP slowdown is
**~66% in fp4 upscale (26%) + fine per-group quant (40%)** plus a heavier
`mfma_moe2_cshuffle` GEMM.

## 4. ROOT CAUSE — EP MoE is a permanent aiter tuned-CSV MISS → heuristic fp4 fallback

Server log (EP), every layer:
```
[fused_moe] using 2stage default for
  ('gfx950', 256, 131072, 7168, 3072, 48, 5, Silu, bf16, fp8_e4m3, fp4_e2m1_x2, per_1x32, True, False)
[fused_moe] no tuned FlyDSL config for (...same...), using heuristic FlyDSL fallback
  (kn1='flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w4_bnt0_gui', kn2='flydsl_moe2_..._atomic')
```
So EP is **CSV-MISS → `cfg=None` → default/heuristic path** (NOT a tuned row that
picked a non-opus kernel). Two independent reasons the EP key can never match
`dsv4_fp8fp4_tuned_fmoe.csv`:

1. **token tier = 131072.** `aiter/fused_moe.py` `get_padded_M` pads EP's MoE M to
   `_PADDED_M_TIERS = [32768, 131072]` → **131072**. The CSV has NO row at token
   131072 for any shape (max tuned token = 32768); tier-fallback 131072→32768 then
   needs `expert=48` at 32768, but the expert=48 rows only reach token=16384. Miss.
2. **topk mismatch.** EP lookup key uses **topk=5** (`topk -= int(is_ep)`, 6→5;
   EP appends a masked fake-expert slot). The CSV's only `expert=48, inter=3072`
   rows are **topk=6**. Miss on topk regardless of M.

Contrast **DP** (hits the CSV): key `expert=384, inter=512, topk=6, token∈tuned tiers`
→ `flydsl_moe1_afp8` + **`opus_moe_stage2_a8w4` (PR #3856)** — native fp4, fp8
activation, **no upscale**.

### Why the heuristic path is the expensive mxfp4 one
mori init: `dispatch_dtype=<DispatchDtype.fp4: 'mxfp4_blockwise'>`,
`combine_dtype=fp8`. Activations arrive as **mxfp4**, so the expert GEMM runs the
mxfp4-activation path (`mxfp4_moe_sort` + fp4 weight `upscale_fp4x2` + 32×32
per-group quant + `mfma_moe2_cshuffle`). DP keeps **fp8 activations** (a8w4) → the
tuned `opus_moe_stage2_a8w4` kernel that consumes fp4 weights directly (no upscale).

This confirms the original intuition ("fp4 MoE tuning") — but the mechanism is a
**tuning MISS on the EP shape**, not "missing EP rows being the whole story", and
it is compounded by the mxfp4 dispatch dtype forcing the non-a8w4 kernel family.

## 5. How to route EP to a fast kernel (opus_a8w4) — must fix BOTH

- **(a) Activation dtype:** `opus_moe_stage2_a8w4` needs **fp8** activations. While
  mori dispatches **mxfp4** (`SGLANG_MORI_DISPATCH_DTYPE=auto`→mxfp4), the a8w4/opus
  path can never be selected. Test `SGLANG_MORI_DISPATCH_DTYPE=fp8` so the expert
  input is fp8 (a8w4). (Watch accuracy + dispatch-comm cost trade-off.)
- **(b) Tuned CSV coverage for the EP key:** even with fp8 activations, the lookup
  must hit a tuned row for `expert=48, inter=3072, topk=5` at the actual M tiers
  (incl. the padded 131072 / 32768). Run the aiter fmoe tuner for that shape (or add
  rows) so an `opus`/`flydsl` a8w4 kernel is chosen instead of `2stage default`.
  Note: the existing `expert=48` rows are `topk=6` and cap at token 16384 — they do
  NOT cover the EP runtime key.
- **(c) Padded-M waste (secondary):** `get_padded_M`→131072 means EP tunes/keys on a
  huge M even for decode. `MORI_MAX_DISPATCH_TOKENS_DECODE=256` bounds decode
  dispatch, but the tuning tier is still large. Worth checking whether decode-M
  actually lands on a smaller tier at runtime (the logged 131072 line is
  prefill-dominated; lru_cache logs each unique key once).

## 5b. Experiment: offline-tune the EP key (2026-07-03) — partial win

Ran the aiter offline tuner for the single EP key and re-tested:
```
# untuned key (captured from EP server log; get_padded_M rounds M -> 131072 tier):
131072,7168,3072,48,5,ActivationType.Silu,torch.bfloat16,torch.float8_e4m3fn,torch.float4_e2m1fn_x2,QuantType.per_1x32,1,0
cp .../dsv4_fp8fp4_tuned_fmoe.csv /workspace/ep_tune/tuned_fmoe.csv   # protect shipped csv
python3 csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py \
  -i untuned_ep.csv -o /workspace/ep_tune/tuned_fmoe.csv -o2 profile_ep.csv --last
# then serve with:  AITER_CONFIG_FMOE=/workspace/ep_tune/tuned_fmoe.csv MODE=mori-ep ...
```
- Tuner picked **native a8w4 flydsl** for the EP key: `flydsl_moe1_afp8_wfp4_bf16_t128x256x256`
  + `flydsl_moe2_afp8_wfp4_bf16_t64x256x128_atomic_xcd4_sbm128` (fallback `cktile_a8w4_bm64`).
  ~5 min tune, 676 CK candidates over 8 GPUs, err 0.0%.
- **Runtime pickup CONFIRMED:** with `AITER_CONFIG_FMOE` set, the log flips from
  "no tuned FlyDSL config → heuristic fallback" to
  `using 2stage (kernelName1='flydsl_moe1_afp8...', kernelName2='flydsl_moe2_afp8...')`.
- **Perf (conc256 8k/1k):** total **7,480 → 8,775 tok/s (+17%)**, median **TPOT 288 → 237 ms
  (−18%)**. Correct direction, but **EP is still ~3.5× slower than DP (30,994 / TPOT 58ms)**.

**Why tuning alone doesn't close the gap — the padded-M waste.** `get_padded_M` rounds the
EP MoE M to the top tier **131072**, and mori's fixed per-rank dispatch buffer means the
expert GEMM processes a ~131072-token padded batch **even at decode** (≈256 real tokens).
The tuned kernel is the fastest kernel *for that huge shape*, but it is still doing
131072-token work for a handful of real tokens. To actually approach DP, must **shrink the
effective decode M** (bound mori decode dispatch / avoid padding to 131072) AND/OR fix the
mxfp4-dispatch upscale (§5a). Kernel tuning is necessary-but-not-sufficient.

Note on `AITER_ONLINE_TUNE=1`: it works mechanically (writes the key to `untuned_fmoe.csv`
and runs `gemm_moe_tune.py` inline via `os.system`), but it blocks a serving forward while
tuning the huge M=131072 shape → high watchdog-timeout risk. Prefer the **offline** tuner.

## 5c. Experiment: SGLANG_MORI_DISPATCH_DTYPE=fp8 (2026-07-03) — NEUTRAL

Hypothesis: mori's mxfp4 dispatch forces the `upscale_fp4x2` path; dispatching fp8
activations would skip it. Tested `SGLANG_MORI_DISPATCH_DTYPE=fp8` (+ tuned config).
(unified script edited to honor the override: `${SGLANG_MORI_DISPATCH_DTYPE:-auto}`.)
Confirmed `dispatch_dtype=<DispatchDtype.fp8:'float8_blockwise'>` in the MORI init log,
and fused_moe still hits the tuned flydsl a8w4 kernel.

| conc256 8k/1k | heuristic | tuned (mxfp4 disp) | **tuned + fp8 disp** |
|---|---:|---:|---:|
| Total tok/s | 7,480 | 8,775 | **8,740** |
| Median TPOT | 288 ms | 237 ms | **237.6 ms** |

**Verdict: NEUTRAL (~0%).** ⇒ the dominant decode cost is NOT the activation dispatch
format. The `upscale_fp4x2` + per-group quant is driven by the **fp4 EXPERT WEIGHT**
dequant on the **padded M=131072 batch** every decode step — independent of whether
activations arrive as mxfp4 or fp8. So the two remaining real levers are both about the
**padded-M waste**, not dispatch dtype:
- **shrink the effective decode M** (mori pads decode to a huge fixed buffer → 131072
  tier; ~256 real tokens do 131072-token expert work). This is the #1 lever now.
- (weight-dequant of fp4 experts per step on that huge M is the cost `upscale_fp4x2`
  represents; only a smaller M or a decode kernel that avoids re-dequant helps).

## 5d. ROOT of the padded M=131072 + deepep-mode test (2026-07-03)

Traced why the EP MoE key is fixed at token=131072 (prefill AND decode):
- `MoriEPDispatcher` inits a **single intra-node mori op** with
  `num_max_dispatch_tokens_per_rank = SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK
  (=16384) `; the recv/MoE capacity = **16384 × world(8) = 131072**. `get_padded_M`
  tiers are `[32768, 131072]`, so any M≥32768 rounds to **131072**. fused_moe is keyed/
  sized on this capacity, so **decode (~256 real tokens) runs a 131072-token padded MoE**.
- `DeepEPMode.AUTO.resolve()` → prefill=normal / decode=low_latency *in theory*, but this
  **DSV4 intra-node mori integration builds ONE op** (log: single `[MORI init] ...
  mode=INTRA_NODE num_max_dispatch_tokens_per_rank=16384`, no separate LL buffer). So
  `--deepep-mode auto` does NOT give decode a small buffer here.

**Empirical (conc256 8k/1k, all with tuned config):**
| lever | Total tok/s | Median TPOT |
|---|---:|---:|
| tuned, `deepep=normal`, mxfp4 disp | 8,775 | 237 ms |
| tuned, `deepep=normal`, **fp8 disp** | 8,740 | 237.6 ms |
| tuned, **`deepep=auto`** | 8,777 | 236.8 ms |

⇒ **fp8-dispatch and deepep=auto are both NEUTRAL.** The padded M=131072 is fixed by the
single 16384 mori buffer and can't be lowered without (a) breaking prefill (which needs
`MORI_MAX_DISPATCH_TOKENS_PREFILL=8192` → num_max_dispatch ≥ 8192 → M ≥ 65536 → still the
131072 tier), or (b) a proper **separate small-buffer low-latency decode path** for the
DSV4 intra-node mori integration (does not exist today), or (c) passing the **actual**
(compacted) token count to fused_moe instead of the buffer capacity.

**Net status of the mori-EP decode investigation:** the only realized win is **tuning the
EP fmoe key (+17%, 7,480→8,775)**. dispatch-dtype and deepep-mode don't move it. Closing
the remaining ~3.5× gap to DP requires a decode-sized MoE M (framework/mori-integration
work), which is out of scope of config/tuning knobs. Given DSV4's productive path is DP
(non-EP), mori-EP is not recommended for decode-heavy serving regardless.

## 5e. 3-way MoE decode breakdown: DP vs EP-heuristic vs EP-tuned (2026-07-06)

Eager GPU-only decode traces (conc128, 8k/2000), normalized **per MoE-layer × step**
(step proxy = `mfma_moe1` count). Confirms where tuning helped and what remains.

| component (ms/exec) | DP (non-EP) | EP heuristic (pre-tune) | EP TUNED (current) |
|---|---:|---:|---:|
| fp4 weight **upscale** (`upscale_fp4x2`) | — | 0.578 | **0.578 (unchanged)** |
| MoE gemm2 down | 0.036 | 0.303 | 0.426 |
| act **quant** (`dynamic_per_group_scaled_quant`) | 0.011 | 0.881 | **0.250 (−72%)** |
| MoE gemm1 up/gate | 0.057 | 0.207 | 0.142 |
| MoE sort | — | 0.030 | 0.023 |
| EP combine (a2a) | — | 0.038 | 0.073 |
| **MoE total / exec** | **0.108 ms** | **1.998 ms** | **1.419 ms** |
| (DP comm gather+RS) | 0.710 | — | — |
| MoE share of decode GPU time | 11% | 91% | 85% |
| eager decode gen tok/s/rank (rr=16) | 136 | 60 | **80** |

**Findings:**
- **EP-tuned MoE ≈ 13× DP MoE** (1.42 vs 0.108 ms/exec). Pre-tune was 18.5×; **tuning cut
  MoE −29%**, almost entirely from **act_quant 0.881→0.250 ms (−72%)** + gemm1 −31% (the
  flydsl a8w4 kernel fuses the quant). Eager decode 60→80 tok/s/rank (+33%).
- **`fp4_weight_upscale` is UNCHANGED (0.578 ms) and is now the single biggest MoE item
  (35%).** It is fp4 EXPERT-WEIGHT re-dequant, independent of kernel/dispatch — bound only
  to the **padded M=131072** (re-expands fp4 weights over 131072 rows every decode step).
- DP's MoE is tiny (0.108 ms, 11%) because DP runs the MoE on the *actual* gathered tokens
  (small at decode); DP decode is instead **comm-bound** (gather+RS 0.71 ms, 73%). EP is
  the opposite: MoE-bound (85%), inflated by padded-M.
- **Takeaway:** tuning is the only realized win (+17% e2e); the residual ~13× MoE gap is
  the padded-M waste (§5d). Removing the 0.578 ms/exec upscale needs a decode-sized M, not
  a kernel/dispatch knob.

## 5f. Pre-permute contrast: DP has NO pre-quant; EP does a dequant→requant round-trip (2026-07-06)

Traced the `upscale_fp4x2_block32_kernel` (the 0.578 ms/exec item in §5e). It is NOT
weight-side (correcting §5c/§5e wording) — it dequants the **mori-dispatched fp4
ACTIVATIONS** back to bf16. Call chain:
- Triton kernel: `srt/layers/moe/rocm_moe_utils.py:193` `upscale_fp4x2_block32_kernel`
- wrapper `rocm_moe_utils.py:277` `upscale_mxfp4(hidden_state(M,packed_N) fp4x2, scale) -> (M,7168) bf16`; grid=(M, 7168/256) → cost ∝ **M×hidden**, and M = padded **131072**.
- call site `srt/layers/moe/moe_runner/aiter.py:348` (mori branch).

**The two pre-permute paths (why DP is cheap, EP is not):**
- **DP** = `pre_permute_standard_to_aiter` (`aiter.py:218`, `standard→aiter`): passes
  `hidden_states` **bf16 unchanged**, `a1_scale=None`, straight into `fused_moe`. The
  activation quant happens **inside fused_moe, fused into moe_sort** (`fused_mx_quant_moe_sort`),
  on the **actual** token count → cheap (DP trace: ~65 ms total, incl. a small
  `dynamic_per_group_scaled_quant Li32×Li128`).
- **EP** = `_pre_permute_deepep_to_aiter` (`aiter.py:281`, mori branch): mori **quantizes
  the activation to mxfp4 for the a2a** (bandwidth), sets `a1_scale=hidden_states_scale`.
  Then because aiter's W4A4 clamped-SwiGLU/INTERLEAVE path can't consume `fp4x2`, it
  `upscale_mxfp4` **dequants fp4→bf16** and drops the scale (`a1_scale=None`) → `fused_moe`
  **re-quantizes bf16→fp8** (the big standalone `dynamic_per_group_scaled_quant Li32×Li32`).

So it's **not** that DP "skips" a needed quant. DP quantizes **once** (fused, cheap, real M).
EP does a **quantize(for dispatch) → dequant(upscale) → re-quantize** round-trip that is
EP-exclusive, and both the extra upscale and the extra requant are amplified by padded
M=131072 → EP act-quant 2449 ms (heuristic) / 944 ms (tuned) vs DP ~65 ms.

**Why fp8 dispatch (§5c) didn't remove it:** with fp8 dispatch, `is_fp4_dispatch=False` →
the sibling branch `aiter.py:337` fires `upscale()` (fp8→bf16) instead — same round-trip,
same cost. The round-trip is inherent to "a2a quantizes activation, but the W4A4
swiglu-interleave fused_moe wants bf16/fp8, not the dispatched packed format".

**Untested lever to skip the round-trip:** the branch is gated by `swiglu_interleave =
quant_info.swiglu_limit>0 and SGLANG_USE_AITER_MOE_GU_ITLV(true)` (`aiter.py:333`). Setting
`SGLANG_USE_AITER_MOE_GU_ITLV=false` would skip the `:348` branch (and `:337/:352`), letting
fused_moe consume the dispatched activation directly — IF aiter has a matching kernel and
DSV4's clamped-SwiGLU stays correct (risk: accuracy / no fp4x2-activation kernel → error).
Still doesn't fix the padded-M inflation of the GEMMs themselves.

### Why EP's post-upscale bf16 still doesn't take DP's fused quant+sort (2026-07-06)
Follow-up Q: after `upscale`, EP is bf16 with `a1_scale=None`, same as DP — so why the big
**standalone** `dynamic_per_group_scaled_quant` instead of DP's fused `fused_mx_quant_moe_sort`?
Answer: both call the SAME `aiter.ops.quant.fused_dynamic_mx_quant_moe_sort` (`quant.py:998`),
which chooses fused-vs-split **purely by M**:
```python
token_num_quant_moe_sort_switch = [8*256/topk,   # stage1 ≈ 341 for topk=6
                                   8*1024/topk]   # stage2
use_fused = (is_stage1 and M <= switch[0]) or (not is_stage1 and M <= switch[1]*eff_topk)
if use_fused:  fused_dynamic_mx_quant_moe_sort_hip(...)   # ONE fused kernel = fused_mx_quant_moe_sort
else:          dynamic_per_group_scaled_quant(...); mxfp4_moe_sort_hip(...)  # split: standalone quant + sort
```
- **DP** decode: real M (~128–256 global) ≤ 341 → **fused** (small `fused_mx_quant_moe_sort`, ~30 ms).
- **EP** decode: **padded M=131072 ≫ 341** → **split path** → standalone `dynamic_per_group_scaled_quant`
  (915 ms) + `mxfp4_moe_sort` (51 ms). The split path is designed to "win at large M" (real
  prefill); EP's M is *fake*-large (padding), so it pays the split cost for ~hundreds of real tokens.

⇒ It is **not** the bf16-ness — it's the **M threshold `8*256/topk ≈ 341`**: EP's padded
M=131072 trips the fused→split switch. So the padded M=131072 is the single root that causes
ALL THREE EP-only costs: (1) tuned-kernel miss (§4), (2) the upscale round-trip's requant, and
(3) this fused→split flip. Shrinking decode M to the real token count would remove/shrink all
three at once (upscale + standalone quant + the padded GEMMs).

## 5g. Where M=131072 comes from + how to shrink decode effective M (2026-07-06)

**M source:** `moe_runner/aiter.py:294` `hidden_states = dispatch_output.hidden_states`;
`M = hidden_states.shape[0]` = the mori **normal** dispatch's **fixed** recv buffer =
`num_max_dispatch_tokens_per_rank(16384) × world(8) = 131072`, independent of the actual
received token count. The real counts (`num_recv_tokens_per_expert`) are known but the buffer
is padded to capacity.

**Cost is grid-over-padded, not real compute:** `upscale_fp4x2_block32_kernel`
(`rocm_moe_utils.py:193`) DOES early-exit per block (`if pid_m >= recv_token_num: return`), so
the *compute* is bounded to real tokens; but the wrapper launches `grid=(M=131072, hidden/256)`
≈ 3.7M program instances, most of which just load+compare+return → that launch/early-exit
overhead is the 2186 ms. (The moe GEMMs are already bounded by `num_valid_ids`/`sorted_ids`.)

**Levers to shrink decode effective M (ranked):**
1. **`SGLANG_MORI_MOE_MAX_INPUT_TOKENS`** (existing; `aiter.py:312`) truncates
   `hidden_states[:cap]` → smaller M/grid; safe (combine reads only `[0, totalRecvTokenNum)`)
   **iff cap ≥ actual recv**. But it's a single static value that must fit PREFILL's large recv
   (~8192/rank×topk ≈ 49k), so it can't be decode-small on a mixed server. (Note the script's
   `MORI_MOE_MAX_INPUT_TOKENS_DECODE=2048` is a **mori C++ env, NOT read by sglang** — confirmed
   grep-empty — so it does nothing on the sglang side today.)
2. **Size the upscale/quant grid to a host-known decode upper bound** (targeted, no GPU sync):
   a rank's recv ≤ `global_tokens × topk`, and at decode `global_tokens` is host-known (batch
   size) → cap ≈ `decode_tokens×topk` (e.g. conc256: 256×6=1536 ≪ 131072). Keep the kernel's
   GPU early-exit for exactness; only shrink the launched grid. Medium, sglang-side. **← chosen (§5h)**
3. **Proper low-latency decode buffer** (§5d): separate small `[experts, decode_cap, hidden]`.
   Bigger integration; not wired for DSV4 intra-node mori today.
4. **Lower `num_max_dispatch_tokens_per_rank`**: blocked — prefill needs ≥8192 → M≥65536 → still
   the 131072 `get_padded_M` tier.

## 5h. Lever-2 prototype (dynamic decode-M cap) — ATTEMPTED, CRASHED, reverted (2026-07-06)

Prototyped lever 2 by reusing the existing truncation (`moe_runner/aiter.py:312`,
`hidden_states[:cap]`) with a **dynamic** cap = `min(M, global_tokens×topk + num_local_experts×256)`
(`global_tokens` from host-side `get_dp_global_num_tokens()`), env-gated `SGLANG_MORI_MOE_DYNAMIC_CAP=1`.

- **Mechanically worked:** decode fused_moe key dropped **131072 → 16384/32768** (prefill stays
  131072). So the cap *does* collapse decode M.
- **But it CRASHED** during gsm8k (a rank died with no Python traceback = C++/HIP abort; other
  ranks then failed on the DP mlp-sync `all_gather` with gloo "connection closed by peer").
- **Root of the crash:** the truncation shrinks `hidden_states/topk_ids/topk_weights/a1_scale`
  but **NOT `num_local_tokens` (`num_recv_tokens_per_expert`)**, which fused_moe/moe_sort still
  consume. When `cap < padded totalRecvTokenNum`, the sort/quant/GEMM index tokens past the
  truncated buffer → illegal memory access → abort. The comment "combine only reads
  `[0, totalRecvTokenNum)`" is about *combine*, not the stage1 quant/sort which trusts
  `num_local_tokens`. And a host-safe cap can't be guaranteed: mori's per-expert / dispatch
  padding can push actual `totalRecvTokenNum` beyond `global_tokens×topk + margin`, so any static
  host bound risks `cap < totalRecv` on some batch (gsm8k's variable lengths hit it).

**Reverted** the change (`moe_runner/aiter.py` back to original). Same reason a grid-cap-only
variant fails: the upscale kernel's early-exit needs `grid_m ≥ actual_recv`, which is also a
GPU-resident count.

**Corrected direction for shrinking decode M (needs the ACTUAL recv count, not a host guess):**
1. **Clamp `num_local_tokens` consistently with the truncation** so the whole stage1
   (sort/quant/GEMM) agrees on the smaller M — requires re-deriving per-expert counts to fit the
   cap (non-trivial; must stay ≥ real tokens).
2. **GPU-count-aware sizing:** size the grid / M from the mori `totalRecvTokenNum` scalar (a
   one-time `.item()` sync per layer — adds latency, maybe acceptable given decode is
   launch/comm-bound), or a persistent kernel that reads the GPU count.
3. **Proper low-latency decode buffer** (§5d) — the clean fix; separate small
   `[experts, decode_cap, hidden]` for decode. Needs LL wiring for the DSV4 intra-node mori path.

Net: the padded-M waste is real and is the last big lever, but it cannot be removed by a
host-side static cap (correctness). It requires GPU-count-aware M or an LL decode buffer =
framework/mori-integration work. Config/tuning knobs are exhausted; **tuning the EP fmoe key
(§5b, +17%) remains the only safe realized win.**

## 5i. mori DOES return the actual recv count — but cuda-graph blocks host-side use (2026-07-06)

Follow-up Q: doesn't mori report the real received-token count we could size M by? **Yes:**
- `mori/ops/dispatch_combine.py:617-633`: `out=from_gpu_ptr((max_recv=131072, hidden))` (padded
  capacity) and **`total_recv = from_gpu_ptr(total_ptr, (1,), ...)`** — the 5th `dispatch()`
  return = **actual total recv tokens** (a `(1,)` GPU tensor). sglang holds it as
  `dispatch_output.num_recv_tokens_per_expert` (`packed_recv_count`).
- The `upscale_fp4x2_block32_kernel` **already uses it** GPU-side (`recv_token_num` early-exit),
  so *compute* is already bounded to real tokens. Only the launched **grid** (131072) and the
  **fused_moe tuning-key M** are the padded value.

**Why we still can't shrink host-side M with it:** `total_recv` is a GPU scalar; using it for a
host-side slice / grid dim / tuning key needs `.item()` (GPU→host sync). **Decode runs under
cuda graph** (`cuda graph: True`), where a `.item()` during capture is illegal and dynamic
shapes are incompatible with the static graph. That is why sglang keeps the static padded M,
and why the §5h host-guess truncation was doomed. ⇒ the graph-safe way to exploit "recv is small
at decode" is a **static per-cuda-graph-bs decode shape** (bound = `graph_bs×topk`, host-known at
capture) = the **low-latency decode buffer** (§5d option 3), not the runtime GPU count.

## 5j. Low-latency (AsyncLL) decode buffer — evaluation + prototype plan (2026-07-06)

Goal: give decode a **small static per-rank buffer** (graph-safe) instead of the padded
131072, via mori's low-latency (AsyncLL) path.

**Architecture (exists):**
- `MoriEPDispatcher.__init__` (`moriep.py:1042-1051`) builds BOTH `_low_latency_dispatcher`
  (if `deepep_mode.enable_low_latency()`) and `_normal_dispatcher` (if `enable_normal()`);
  AUTO enables both. Constructed at `fused_moe_triton/layer.py:113` with `deepep_mode=get_deepep_mode()`.
- `_get_impl()` (`moriep.py:1127`) routes per-batch: `resolve(get_is_extend_in_batch())` →
  prefill→NORMAL, decode→LOW_LATENCY. `decode_cuda_graph_runner.py:791` sets
  `is_extend_in_batch=False`, so **decode is supposed to pick the LL dispatcher**.
- Single-node OK: `init_mori_op:241-244` — world≤8 defaults INTRA_NODE, but
  `async_mode = deepep_mode.enable_low_latency() or enable_sdma` → **LOW_LATENCY (AsyncLL)**.
- LL `dispatch_a` **asserts kernel_type is AsyncLL** (`moriep.py:825`).

**Blockers found:**
1. **LL didn't engage empirically.** §5d `--deepep-mode auto` decode stayed INTRA_NODE (M=131072),
   no AsyncLL `[MORI init]` seen — so decode did NOT route to `_low_latency_dispatcher` despite
   the routing above. Root not yet pinned (deepep_mode propagation to the dispatcher, lazy
   mori_op, or a DSV4-specific gate). **Gating experiment for the prototype.**
2. **Shared buffer size.** Both impls take one `num_max_dispatch_tokens_per_rank`
   (`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=16384`). For LL to actually shrink M it needs
   its OWN small decode cap (e.g. 256–512); with 16384 the AsyncLL per-expert buffer
   (`[num_local_experts, cap, hidden]`) would be huge. Needs per-impl parametrization (LL small,
   normal 8192+). Also AsyncLL kernel is only selected when the LL mori_op is built with a
   low-latency deepep_mode (mode=LOW_LATENCY), independent of `INTER_KERNEL_SWITCH_THRESHOLD`.
3. **DSV4 AsyncLL correctness untested.** The `deepep_ll→aiter` pre-permute path + DSV4 forward
   must run the AsyncLL decode output correctly (gsm8k gate).

**Prototype plan (ranked steps):**
1. (gating) Instrument/confirm WHY decode doesn't hit `_low_latency_dispatcher` under
   `--deepep-mode auto` (log `resolved_deepep_mode` + `get_is_extend_in_batch()` inside
   `_get_impl`; check `get_deepep_mode()` value). Fix routing so decode→LL.
2. Parametrize a **separate small LL decode cap** (new env, e.g. `SGLANG_MORI_LL_MAX_DISPATCH_TOKENS`,
   default ~512) for the `_low_latency_dispatcher`'s mori_op only; keep normal at 8192+.
3. gsm8k correctness gate on the AsyncLL decode path (DSV4).
4. Measure decode TPOT/throughput vs the 8,775 tuned baseline; expect the upscale/quant/GEMM to
   collapse from M=131072 → M≈(cap×num_local_experts) ~ real tokens.

**Effort:** medium-large, correctness-sensitive, multi-file (moriep dispatcher construction +
per-impl sizing + validation). Higher risk than tuning; it is the only path that removes the
padded-M waste in a cuda-graph-safe way.

### Prototype attempt results (2026-07-06) — blocker chain pinned
Traced why LL never engages and how far it gets:
1. **`server_args.py:5625` force-downgrades `auto→normal` for MORI EP** (`"auto set
   deepep_mode=normal for MORI EP"`) → the `_low_latency_dispatcher` is never built
   (`_get_impl` debug: `deepep_mode=NORMAL, has_ll=False`, even at decode `is_extend=False`).
   Gated that downgrade behind `SGLANG_MORI_ALLOW_AUTO_LL=1` (opt-in).
2. With the bypass, `deepep_mode='auto'` survives and the **LL mori_op builds single-node
   (`mode=<EpMode.LOW_LATENCY>`)** — so LL *is* reachable single-node. **But it crashes at init:**
   `ValueError: Fp8BlockwiseQuant currently only supports IntraNode/IntraNodeLL combine`.
3. Root: `get_ep_dispatch_configs` maps `EpMode.LOW_LATENCY → AsyncLL` (inter-node RDMA kernel),
   which does NOT support DSV4's fp8 combine single-node. mori **does** have an
   **`IntraNodeLL`** kernel (`dispatch_combine.py:142,577` `ep_intranode`) that supports fp8
   combine + is single-node low-latency — but the LL impl `dispatch_a` **asserts
   `kernel_type is AsyncLL`** (`moriep.py:825`), i.e. the LL path is hardwired for AsyncLL.

**Remaining fix chain to make LL decode work single-node (each needs validation):**
(a) map single-node (world≤8) `LOW_LATENCY → IntraNodeLL` in `get_ep_dispatch_configs`/`init_mori_op`;
(b) relax the LL `dispatch_a`/`dispatch_b` AsyncLL-only assert to accept IntraNodeLL (and verify
its dispatch_b output shape/API matches — LL impl is written for AsyncLL);
(c) give the LL impl a **separate small decode cap** (num_max_dispatch ~256–512) so M shrinks;
(d) gsm8k gate + decode perf. ⇒ this is genuine mori-integration dev (several iterations,
crash-prone), not a config toggle. Env-gated hooks left in place (default off, no behavior
change): `SGLANG_MORI_ALLOW_AUTO_LL`, `SGLANG_MORI_DEBUG_IMPL`.

## 5k. ✅ FIX LANDED — small-cap IntraNode decode mori_op (2026-07-06): +151%, gsm8k 0.95

The IntraNodeLL/AsyncLL detour (§5j) turned out unnecessary. Key realization: the **normal
(IntraNode) dispatcher already uses synchronous `dispatch()` which supports fp8 combine and any
buffer size** — so decode just needs a **second IntraNode mori_op with a small per-rank cap**,
routed by `is_extend`. No AsyncLL, no server_args bypass, no deepep-mode change.

**Implementation (moriep.py, ~37 lines, env-gated `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS`, default 0=off):**
- `_MoriEPDispatcherImplBase`: add `num_max_dispatch_tokens_per_rank_decode` (env) + `_mori_op_decode`.
- `mori_op` property: when cap>0 AND `not get_is_extend_in_batch()` → build/return a 2nd
  `init_mori_op(..., decode_cap, deepep_mode=NORMAL, instance_id+4096, ...)` (distinct cache/op);
  else the existing big op. Both are IntraNode (sync, fp8-combine OK, cuda-graph-safe static shape).
- Correctness is exact: recv ≤ Σ(send) ≤ decode_cap×world, and decode send/rank (≈conc/dp) ≪ cap,
  so no valid tokens dropped (unlike §5h's host-guess truncation which crashed).

**Result — mori-EP conc256 8k/1k (decode_cap=512, + tuned fmoe key, mem0.9):**
| config | Total tok/s | Median TPOT | % of DP (30,994) |
|---|---:|---:|---:|
| heuristic (baseline) | 7,480 | 288 ms | 24% |
| + tuned fmoe key (§5b) | 8,775 | 237 ms | 28% |
| **+ decode small-cap op** | **22,027** | **77.7 ms** | **71%** |
- Decode fused_moe M: **131072 → 4096** (512×8); TPOT **237→77.7 ms (−67%)**; throughput
  **+151% over tuned / +194% over heuristic**. gsm8k **0.95/0.95** (correct). Boots with both
  ops (`num_max_dispatch=16384` prefill + `512` decode), no crash. TTFT slightly up (~+10%,
  prefill still on the big op + faster decode shifts contention; secondary).
- **This is the padded-M fix** (§5d/§5g/§5i): collapsing decode M closes most of the EP↔DP gap
  (24%→71% of DP). Prefill unchanged (still big op + tuned kernel).

**Follow-ups:** (1) the decode key `(token=4096, expert=48, topk=5)` is still "no tuned config →
heuristic" — tuning it (§5b-style) could add more; (2) sweep decode_cap (256/512/1024) + validate
at conc512; (3) TTFT check; (4) upstream-worthiness: it's a clean, opt-in, correctness-preserving
knob. **Realized wins now: tuned fmoe key (+17%) AND decode small-cap op (+151%) → mori-EP from
24%→71% of the DP path.**

## 5l. decode_cap sweep + decode-key scope (2026-07-06)

Swept `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS` (with the §5k dual-op + tuned prefill key).
**Decode fused_moe M = `nextPow2(cap×world=cap×8)` and is conc-INDEPENDENT** (fixed recv
buffer), so the decode tuning key is fully determined by the cap.

| cap | decode M (key token) | conc256 tok/s | conc256 TPOT | conc512 tok/s | conc512 TPOT | gsm8k |
|---:|---:|---:|---:|---:|---:|---:|
| 256  | **2048** | **23,869** | **69.4 ms** | **30,339** | 98.9 ms | 0.975 |
| 512  | 4096 | 22,027 | 77.7 ms | (~) | — | 0.95 |
| 1024 | 8192 | 20,632 | 86.0 ms | 27,591 | 114.4 ms | — |

(DP reference: conc256 30,994 / conc512 ~38,187.) So cap=256 conc256 = **77% of DP** (was 24%).

**Findings:**
- **Smaller cap ⇒ smaller M ⇒ faster** (monotonic), down to the safety floor `cap ≥ decode
  send/rank = conc/dp`. At conc512, send/rank=64, so **cap=256 is safe with ~2× headroom**
  (cap=128→M=1024 would be safe for conc≤1024 and likely a bit faster; untested).
- All correct (gsm8k 0.95–0.975), no crashes at conc512.
- **Decode-key tuning scope:** the key is `(token=nextPow2(cap×8), model_dim=7168, inter_dim=3072,
  expert=48, topk=5, per_1x32, a=fp8/w=fp4)`. If we standardize a cap → **ONE key** to tune
  (cap=256 → token=2048). To keep cap configurable → tune the small set
  {1024, 2048, 4096, 8192} for (expert=48, topk=5). All are ≤16384 (in the existing expert=48
  CSV token range) but currently MISS on topk (CSV rows are topk=6, EP key is is_ep topk=5) →
  they run "no tuned config → 2stage default" today. Tuning them (§5b offline tuner) would stack
  on top of the already-large cap win.

**Recommendation:** default `decode_cap=256` (M=2048, best measured, 2× safety headroom to
conc512); tune the single decode key `token=2048, expert=48, topk=5` next.

## 5m. Tuning the decode key (token=2048) — NEUTRAL (no stacked win) (2026-07-06)

Offline-tuned the decode key `token=2048, expert=48, topk=5` (flydsl a8w4 primary + cktile
fallback, appended to the tuned csv) and confirmed decode now hits it (`using 2stage
(flydsl_moe1..., flydsl_moe2_...atomic_persist_sbm128)` instead of "no tuned config").

| conc256 8k/1k (cap=256) | decode heuristic (2stage default) | decode key TUNED |
|---|---:|---:|
| Total tok/s | 23,869 | 23,192 |
| Median TPOT | 69.4 ms | 72.4 ms |
| conc512 tok/s | 30,339 | 29,731 |

**Verdict: NEUTRAL (within ~3% noise, if anything marginally worse).** At small decode M (2048)
the MoE is latency/memory-bound and the heuristic "2stage default" kernel is already as good as
the tuned flydsl kernel — tuning only mattered at the large prefill M (131072, §5b). So **decode-key
tuning is NOT worth it; the cap fix alone captures the win.**

**Final recipe (mori-EP decode):** `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=256` (the §5k dual-op) —
that is the win (24%→77% of DP @conc256). The tuned **prefill** key (131072) still helps prefill;
the decode key does not need tuning. Net realized: **7,480 → ~23,900 tok/s @conc256 (≈3.2×,
gsm8k 0.95–0.975)** from the decode small-cap op, opt-in and correctness-preserving.

## 5n. Remaining gap vs DP after the decode-cap fix = PREFILL (2026-07-07)

Decomposed mori-EP (cap=256) vs DP at conc256 8k/1k from server batch logs (per-step, so
robust to NP/conc differences):

| metric | DP | mori-EP (cap=256) | EP/DP |
|---|---:|---:|---:|
| **prefill compute** (full 8192-chunk, tok/s/rank, p50) | **7,070** | **5,025** | **0.71 (−29%)** |
| **decode** (gen tok/s/rank @ rr=32) | 854 | 688 | 0.81 (−19%) |
| client total tok/s | 30,994 | 23,869 | 0.77 |
| client mean TTFT | 16,361 ms | 27,958 ms | +71% |
| client median TPOT | 58.2 ms | 69.4 ms | +19% |

(prefill p50 distributions are tight — DP 7020–7139, EP 4982–5083, n≈300 each — so the
−29% prefill-compute gap is solid, not variance.)

**Conclusion: after the decode small-cap fix, the dominant remaining gap is PREFILL** — EP
prefill compute is ~29% slower per step, which amplifies to TTFT +71% under conc256 queueing.
Decode is now only a secondary ~19% gap (TPOT 69 vs 58 ms). i.e. the decode-M fix closed the
decode side; prefill is the new bottleneck.

**Prefill-gap hypotheses (untested):** (1) mori a2a dispatch/combine vs DP `all_gatherv`+
`reduce_scatterv`; (2) **padded-M in the PREFILL MoE** — prefill still uses the big op (M=131072)
while prefill actual recv/rank ≈ chunk×world×topk×(local/total) ≈ 8192×8×6×(48/384) ≈ 49k, so
~2.7× padding waste there too. A *medium* prefill cap (e.g. 8192 → recv 65536 ≥ ~49k actual)
would halve the prefill MoE grid — worth testing (same dual-op mechanism, is_extend=True side).
Next: prefill trace (attn vs MoE vs a2a) OR try a prefill cap to isolate (1) vs (2).

## 5o. ✅ Prefill gap is ALSO padded-M — right-size the prefill cap (2026-07-07): EP → 90% of DP

Tested hypothesis (2) from §5n by lowering the PREFILL cap. The prefill (big) op uses
`SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK` (=16384 in the mori-ep launch), giving recv
buffer 16384×8=131072 — but per-rank prefill send = the chunk (8192/rank for CHUNK=65536),
so worst-case recv = 8192×8 = 65536; the buffer was **2× over-provisioned**. Setting the cap
to **8192** (= chunk/rank) shrinks the prefill MoE M 131072→65536 (no tokens dropped — 65536 is
the absolute worst-case recv).

**Result (conc256 8k/1k, prefill cap=8192 + decode cap=256 + tuned prefill key):**
| metric | DP | EP decode-cap only | EP + prefill-cap=8192 | 
|---|---:|---:|---:|
| prefill compute tok/s/rank (p50) | 7,070 | 5,025 | **7,484 (> DP!)** |
| total tok/s | 30,994 | 23,869 | **27,875** |
| mean TTFT | 16,361 ms | 27,958 ms | **19,110 ms** |
| median TPOT | 58.2 ms | 69.4 ms | 64.0 ms |
| % of DP | 100% | 77% | **90%** |
- gsm8k **0.98/0.98** (correct). Prefill compute +49% (5025→7484), now *above* DP; TTFT −32%.
- **Confirms the prefill gap was padded-M too, NOT a2a comm.** Both prefill and decode suffered the
  same over-sized mori recv buffer.

**Combined recipe (mori-EP, DSV4, 8k/1k) — EP from 24% → 90% of DP:**
- `SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192` — **config only**, right-size the prefill cap
  to the per-rank chunk (the shipped 16384 is 2× over-provisioned for CHUNK=65536). Must be
  ≥ chunk/dp (=8192 here); scale with the chunk.
- `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=256` — the new decode small-cap op (the committed patch).
- Both are the same root fix (stop running the MoE over an over-padded recv buffer), applied to
  the prefill (existing knob, right-sized) and decode (new dual-op) sides.

**Follow-ups:** (1) sglang could auto-size `num_max_dispatch` from the per-rank chunk instead of a
fixed 16384; (2) remaining ~10% to DP is small (TTFT 19.1 vs 16.4, TPOT 64 vs 58) — likely mori
a2a per-step overhead vs DP gatherv/RS; (3) re-run a clean full A/B (NP_MULT=8) for publishable
absolutes.

## 5p. cuda-graph trace on ROCm now (2026-07-14) + capped-EP decode time distribution

Retried a **cuda-graph** decode trace (vs the eager method) on the updated sglang build.

**Two findings up front:**
1. **cuda-graph decode was BROKEN on this ROCm build** — the DSV4 compress-plan JIT kernel
   (`jit_kernel/dsv4/compress.py` → `sgl_kernel/utils.cuh::getSMVersion`) used CUDA-only APIs
   (`cudaDevAttrComputeCapabilityMajor/Minor`, `cudaDeviceGetAttribute`) not shimmed for HIP →
   ninja compile fails during `init_forward_metadata_in_graph` (graph-only path; eager avoids it).
   A sglang ROCm regression from the Jul-6→14 update, unrelated to EP. Fixed by hipifying
   `getSMVersion` (guarded `hipDeviceGetAttribute` + `hipDeviceAttributeComputeCapability*`).
   Also re-applied the recurring `cohere2_moe.py @strict` fix (reverted by the repo sync).
2. **cuda-graph kernel durations ARE real on this build** (min 3.3µs, p50 4.4µs, 100% ≥1µs — NOT
   the old ns-garbage in `TRACE_PROFILING §2`). BUT replays are opaque `hipGraphLaunch` (119 in the
   window), so the trace exposes **one representative captured step** (61 MoE-layer kernels), not a
   per-step sum. Good for a time-distribution; not for absolute per-window GPU busy.

**Capped-EP decode time distribution (per captured step = 61 MoE layers; cap 8192/256, cuda-graph):**
| component | ms/step | per-layer | note |
|---|---:|---:|---|
| MoE GEMM (mfma_moe1 10.5 + mfma_moe2 7.75) | 18.3 | 0.30 ms | dominant (~52%) |
| mori a2a (EpDispatchIntraNode 2.1 + EpCombineIntraNode 4.83) | 6.9 | 0.11 ms | ~20% |
| attn (paged_decode) + dense GEMM (Cijk/ck/fp8gemm) | ~8 | — | ~23% |

- **vs §5e (pre-cap eager):** MoE per-layer **1.42 → 0.30 ms (−79%)** — the cap fix confirmed at
  kernel level; **mori a2a unchanged at 0.11 ms/layer** (comm is M-independent), so it's now a
  bigger *fraction* (~20%) purely because MoE shrank. MoE still dominant.
- **vs B200:** B200 (DeepEP) concluded EP wins because all2all comm ≪ DP gather/scatter (EP ~521 vs
  no-EP ~3,749 µs/step). Our MI355X mori a2a is likewise cheap per-layer (0.11 ms), consistent — the
  MI355X EP problem was never comm, it was the padded-M MoE (now fixed). Remaining MI355X decode is
  MoE-GEMM-bound (~52%), not comm-bound. (kernel-level absolute a2a differs from B200 — different
  kernel family (mori IntraNode vs DeepEP) + our sum is over 61 layers of one captured step.)

Note: `getSMVersion` hipify is a working-tree fix needed for ANY cuda-graph run on this ROCm build
(separate from the EP feature; re-apply after repo sync).

## 5q. DP cuda-graph trace — B200-style EP-vs-DP comm contrast (2026-07-14)

Captured the matching **DP** decode trace with the *same* method as §5p (cuda-graph ON, GPU-only
profiler, sustained decode). Same getSMVersion + cohere2 fixes required to boot. Both traces are
opaque `hipGraphLaunch` replays, so per-step absolutes come from clipping the **densest contiguous
decode window** (idle-gap segmentation); DP window = 3,950 kernels, EP window = 3,611 kernels →
comparable amount of decode work, so the split % and window-wall are apples-to-apples.

**Densest decode-window breakdown (same categorizer both sides):**
| category | DP cuda-graph | EP capped cuda-graph |
|---|---:|---:|
| **COMM** (DP: RCCL allgatherv+reduce_scatter / EP: mori a2a) | **103.6 ms (79.0%)** | **6.9 ms (13.7%)** |
| GEMM/MoE | 14.9 ms (11.3%) | 29.0 ms (57.5%) |
| ATTN | 2.8 ms (2.1%) | 5.0 ms (10.0%) |
| QUANT | 2.3 ms (1.8%) | 3.6 ms (7.2%) |
| other (route/norm/rope/mem/misc) | ~7.6 ms | ~7.9 ms |
| **window wall** | **136.2 ms** | **50.6 ms** |

**B200-style headline — the bottleneck flips, exactly like B200:**
- **DP decode is COMM-BOUND**: ~**79%** of GPU time is RCCL `allgatherv + reduce_scatter`, and it all
  sits on the **main compute stream (tid=8)** → fully **serialized / exposed**, not overlapped. MoE
  GEMM is only ~11%.
- **EP (capped) decode is COMPUTE-BOUND**: MoE GEMM ~**57%**, mori a2a only ~**14%**.
- **Comm shrinks ~103.6 → 6.9 ms across comparable windows = ~15× cheaper**, matching/exceeding
  B200's ~7× all2all-vs-gather/scatter advantage (B200: EP ~521 vs no-EP ~3,749 µs/step). EP's
  all2all replaces DP's gather/scatter and flips the bottleneck from comm → MoE, same as B200.
- Comparable-window wall **136 → 51 ms (~2.7× faster)**, consistent with EP overtaking DP once the
  padded-M MoE cost was fixed (§5o).

**Caveat:** DP's RCCL kernel duration includes spin-wait (waiting for the slowest peer + for the
producer compute), so part of the 103.6 ms is sync/idle — but because it's serialized on the
critical stream it is exposed decode latency regardless. EP's mori kernels carry the same caveat, so
the contrast holds. (B200 measured comm the same way, so the ~7× vs ~15× are directly comparable.)

> ⚠️ **Read §5r before trusting the DP comm number here.** The 103.6 ms is ~90% RCCL spin-wait, not
> data movement (the transfer floor is ~12 ms). The real EP-vs-DP delta is MoE-GEMM, not comm — §5r
> explains why DP still wins ~10% despite this table.

## 5r. ⚠️ Why does DP still win ~10% if EP "saves 97 ms of comm"? — §5q over-counts DP comm (2026-07-14)

The §5q window said DP comm = 103.6 ms vs EP a2a = 6.9 ms. If that were *real exposed* comm, EP
would win by ~2×, not lose by 10%. The bench (DP ~10% faster, §1/§5o) is ground truth, so §5q's
comm number is the thing that's wrong. Interrogating the RCCL kernel-duration distribution shows why:

**DP RCCL (`ncclDevKernel`) decode kernels, n=248 in window:**
```
min 44.6µs | p10 395 | p50 745 | p90 1,334 | max 129,148 | mean 1,835µs | std/mean = 5.6
```
- The **min (44.6 µs) ≈ the pure-bandwidth estimate** for the decode allgather (B≈128–256 × 7168 ×
  bf16 = 1.8–3.7 MB ⇒ ~37–73 µs @ ~50 GB/s). So the *actual data movement* is only ~45–75 µs/collective.
- The mean is ~40× the min with a fat tail (std/mean 5.6, a 129 ms outlier) — the classic
  **barrier / spin-wait** signature, NOT bandwidth. RCCL runs on the **main compute stream**
  (tid=8, serialized), so while the collective spins waiting for the slowest peer / for the next
  microbatch's input, that idle spin is counted as "comm busy."
- ⇒ **Real movable DP comm ≈ 248 × ~50 µs ≈ 12 ms** in the window (~min-floor), not 103.6 ms.
  The other ~90 ms is spin/idle — an artifact of the **profiling burst not being
  throughput-saturated** (in saturated steady-state serving those barriers overlap across
  microbatches and the spin collapses). EP's single stream, by contrast, was ~100% busy
  (50.5 ms busy / 50.6 ms wall) with tight, real GEMM kernels — no spin padding.

**So EP does NOT save ~97 ms of comm.** Reconciling with real GPU work per comparable window:
| real work | DP | EP capped | Δ (EP−DP) |
|---|---:|---:|---:|
| comm (DP transfer-floor / EP mori a2a, both *real* kernel work) | ~12 ms | 6.9 ms | **−5 ms** |
| GEMM/MoE | 14.9 ms | 29.0 ms | **+14 ms** |
| attn + quant | 5.1 ms | 8.7 ms | **+3.6 ms** |

- EP's **real comm saving is small (~5 ms)** — mori a2a is cheap, but so is DP's *actual* transfer;
  DP's apparent comm was ~90% spin.
- EP **pays real extra compute**: MoE GEMM ~**2×** (15→29 ms) because each EP rank runs the
  capacity-padded dispatched tokens for its local experts (the cap shrank M but still pads more than
  DP's own-local-tokens-only MoE), plus the dequant→requant round-trip (§5f) and more quant.
- **Net: EP adds ~+14 ms real GEMM (+ quant) and only removes ~5 ms comm ⇒ EP is net slower ⇒ the
  ~10% bench gap.** The profiling is consistent with the bench once RCCL spin is removed; it never
  actually contradicted it.

**Takeaway / methodological fix:** GPU kernel *duration* is not a faithful comm cost for collectives
on a serialized stream — subtract the barrier/spin (use the min-floor or a byte-model) before
comparing. The honest MI355X story: EP's comm was already fine; EP loses on **MoE-GEMM compute**
(padded-capacity expert work), which is where any further EP win must come from — not comm.
(This also nuances §5q's B200 analogy: B200's ~7× was measured the same duration-way and likely
carries similar spin; the *directional* "all2all cheaper than gather/scatter" holds, but the
absolute multiplier is inflated on both platforms.)

## 5s. Why is B200 EP TTT so much faster than MI355X EP? — trace vs trace (2026-07-14)

Compared the B200 EP trace (`trace_hcOFF_msOFF_c256/`, EP8 + `deepep` + `megamoe`, 8k in / 2k out,
conc256, full 10 s steady-state continuous-batching run) against the MI355X EP trace (§5p/§5q/§5r),
same categorizer. **Bench anchor (both 8k/1k conc256, EP8):**
| | B200 EP (deepep+megamoe) | MI355X EP capped (mori) | MI355X DP |
|---|---:|---:|---:|
| total tok/s | **41,858** | 27,875 | 30,994 |
| tok/s / GPU | **5,232** | 3,484 | 3,874 |
| Med TPOT | **48.2 ms** | 64.0 ms | 58.2 ms |
| Med ITL | **29.8 ms** | — | — |
→ B200 EP is **~1.5× the per-GPU throughput** and **~25% lower TPOT** than MI355X EP.

**Whole-run GPU-time distribution (B200 EP, per-stream summed):**
| bucket | B200 EP | note |
|---|---:|---|
| **MoE GEMM** (`sm100_fp8_fp4_mega_moe`) | **51.0%** | ONE fused kernel, 20,069 launches |
| dense GEMM (`deep_gemm 1d1d` + nvjet + cublas) | 27.4% | attn projections / dense |
| attn (flashMLA sparse fp8 sm100) | 6.4% | |
| norm+rope (fused) | 4.4% | |
| quant (`per_token_group_quant`, fused) | 3.2% | |
| route (topk/router/pre_dispatch) | 2.5% | |
| **COMM** (all_gather 33ms + hc_head 23ms + all_reduce 6.9ms / **whole 10 s**) | **0.6%** | a2a is FUSED into mega_moe |

**The answer: it's MoE-GEMM throughput + kernel fusion, NOT communication.** Both platforms are
MoE-bound (B200 51%, MI355X ~52%), and on *both* comm is cheap (§5r) — so the TTT gap is compute:

1. **B200 fuses dispatch+GEMM+combine into one `mega_moe` kernel** (megamoe backend). It's a
   fixed-capacity fused grouped-GEMM: duration is a tight **~272 µs/layer** (p10 250 / p50 272 /
   p90 302, *not* bimodal → same padded shape decode & prefill; fires **61×/step = 61 MoE layers**,
   so **per-step MoE GEMM = 272 µs × 61 ≈ 16.7 ms**). MI355X mori runs the a2a as a
   **separate** dispatch+combine (§5p: 6.9 ms/step ≈ 0.11 ms/layer, ~14–20%) *plus* separate
   sort/quant, on a single stream — all serialized and exposed.
   > **✔ per-step MoE verified by direct per-batch sum (not division), 2026-07-14:** segmenting the
   > run into forward batches (via `_all_gather_kernel_inner`, 328 batches, each exactly 61
   > mega_moe launches) and summing mega_moe *within each decode batch* gives **decode-step MoE GEMM
   > = 16,696 µs (p50), mean 16,749** with a very tight spread (p10 16,522 / p90 16,882) → per-layer
   > **274.6 µs**. This is authoritative (326 independent samples, no averaging of the prefill tail;
   > only 2 prefill/mixed batches at ~20.2 ms exist). A separate window's "**MoE expert GEMM = 18,697
   > µs/step**" is ~12% higher because its **denominator was ~294 steps** (5,500,532 µs total ÷ 294 ≈
   > 18,709; actual is 328 → ÷328 ≈ 16,770), i.e. an undercounted step boundary (and/or that bucket
   > folded in `per_token_group_quant` ~807 µs/step + activation). The 274.6 µs/layer figure itself is
   > rock-solid either way, so the MoE-bound conclusion and the B200-vs-MI355X per-layer contrast are
   > unaffected.
   > **⚠️ correction (2026-07-14): "a2a" has TWO parts — don't say it's ~0.**
   > (a) The **inter-GPU wire transfer** (all2all over NVLink) IS genuinely fused into `mega_moe` —
   >     there is **no separate dispatch/combine kernel** in this trace (P2P `Memcpy DtoD` = only
   >     3 ms across the whole 10 s run). *This* is what the "COMM 0.6%" bucket measured.
   > (b) But the **token-routing PREP compute** is real and nonzero: `mega_moe_pre_dispatch`
   >     (153 µs/step) + `_router_triton` (269 µs/step) + `transpose_and_pack` quant-for-dispatch
   >     (142 µs/step) ≈ **564 µs/step**. A separate window's "comm (token routing) = 542 µs/step (EP)
   >     vs 4,266 (noEP)" bucket = exactly these prep kernels (I had scattered them into ROUTE/QUANT
   >     above, hence the misleading "a2a≈0" phrasing). noEP is 4,266 because DP replaces the fused
   >     a2a with real `all_gather + reduce_scatter` collectives → EP saves ~3,724 µs/step (the B200
   >     ~7× finding). Either way routing is ≪ the 16.7 ms/step MoE GEMM, so the MoE-bound conclusion
   >     stands; only the wording was wrong.
2. **Per-MoE-layer all-in cost:**
   | per MoE layer | B200 EP | MI355X EP capped |
   |---|---:|---:|
   | grouped MoE GEMM | ~272 µs (fused) | ~300 µs (mfma_moe1+moe2, §5p) |
   | a2a **wire transfer** | fused into GEMM (~0 separate) | +110 µs (separate mori dispatch+combine) |
   | routing prep (pre_dispatch+router+pack) | +~9 µs (542 µs/step ÷ 61) | included in mori path |
   | sort / quant | fused | + separate kernels |
   | **all-in** | **~281 µs** | **~450+ µs** |
   ≈ **1.6× per-layer MoE-path advantage** → matches the ~1.5× per-GPU throughput & TPOT 48 vs 64.
3. **Blackwell fp4 tensor cores + HBM3e**: `sm100_fp8_fp4` GEMM/attn kernels are simply faster per
   FLOP/byte; dense 1d1d GEMM averages ~15.8 µs, decode MLA attn ~20 µs — tiny.
4. **More fusion elsewhere**: fused q/k-norm+rope, fused per-token-group quant, fused silu-mul →
   fewer launches / less overhead than MI355X's fragmented aiter path.
5. **NOT overlap**: B200 overlap factor was only **1.07×** (busiest stream 58.6% occ) — so the win
   is per-kernel speed + fusion, *not* two-batch overlap (this run was `msOFF`).

**Bottom line (ties §5r + B200):** on both platforms comm is a non-issue; the MI355X EP deficit is
entirely the **MoE path** — (a) separate, un-fused a2a + sort/quant vs B200's single fused mega_moe,
and (b) slower per-FLOP MoE GEMM. Any further MI355X EP win must come from **fusing the mori a2a into
the MoE grouped-GEMM (a megamoe-equivalent)** and faster fp4/fp8 MoE kernels — not from comm.

## 5t. Roadmap — can we fuse mori dispatch/combine INTO the aiter MoE GEMM (a "mori-megamoe")? (2026-07-14)

Motivated by §5s: B200's win is a single fused `mega_moe` (dispatch+GEMM+combine in one kernel). Can
MI355X do the same with mori (a2a) + aiter (grouped-GEMM), given they're two separate libs? Explored
both repos (`/sgl-workspace/mori`, `/sgl-workspace/aiter`) + the sglang bridge.

**Verdict: the hardware primitive is NOT the blocker — the lib siloing is.**

### Good news: the megamoe primitive already exists on ROCm and mori already uses it
- mori has a symmetric heap + xGMI P2P: `hipDeviceEnablePeerAccess` + `hipIpcOpenMemHandle`
  (`src/application/memory/symmetric_memory.cpp`), device-side peer pointer via
  `SymmMemObj::GetAs(pe)` (`include/mori/application/application_device_types.hpp`).
- **mori combine `_p2p` already reads remote-GPU tokens *inside* the kernel**
  (`src/ops/dispatch_combine/intranode.hpp:637`), validated by `examples/shmem/test_p2p_direct_access.cpp`.
  → "read peer-GPU tokens inside a compute kernel" (the core megamoe move) is proven on MI355X.

### Bad news: mori and aiter are genuinely two worlds
| | mori | aiter |
|---|---|---|
| memory model | `SymmMemObjPtr` / symmetric heap | plain `torch.Tensor`, single-GPU VA |
| token layout | flat recv buffer + routing map (`dispDestTokIdMap`) | expert-sorted via `sorted_token_ids` into a **local** contiguous buffer |
| shared handle/args/headers | — none — | — none — |

Current sglang path is **not** zero-copy (multiple materializations):
`quant → mori dispatch (WarpCopy into symmetric recv buf) → optional upscale → aiter moe_sorting
(another scatter) → 2-stage GEMM → mori combine`. aiter `mfma_moe1` gathers rows by indexing a
**local** X buffer via `sorted_token_ids` — no concept of remote pointers / multi-GPU gather
(`aiter/ops/flydsl/kernels/moe_gemm_2stage.py`). No `mega_moe` equivalent exists in aiter.

### Two concrete blockers (not abstract difficulty)
1. **mori fp8 blockwise combine P2P-read is unimplemented** — `python/mori/ops/dispatch_combine.py:776`
   raises `"P2P read path not yet implemented"` for `Fp8BlockwiseQuant`. DSV4 decode **is** fp8
   combine → the combine-side fusion needs this filled in first.
2. **sglang uses `use_external_inp_buf=True` → combine runs `_nop2p`** (local staging copy, not remote
   read). Must move off this to enable any P2P/fused combine.

### Three-tier plan (shallow → deep)
- **Tier 1 — soft fusion (low risk, do first):**
  - Skip redundant upscale/quant when dispatch dtype == GEMM input dtype (already partial).
  - Use mori `dispatch_standard_moe` (expert-major `[E, M_max, H]`) + teach aiter to consume it →
    **eliminate the `moe_sorting` scatter pass**. Needs `ENABLE_STANDARD_MOE_ADAPT=ON`
    (default OFF in `mori/setup.py`) + aiter change. Still 2 launches, but one fewer full
    materialization.
  - Lean on **AsyncLL** to overlap comm behind GEMM (sglang already supports; overlap, not fusion).
- **Tier 2 — medium: P2P-aware aiter stage-1 (the real megamoe analog):**
  - Pass mori `dispatchOut` symmetric handle + routing map into aiter; extend `mfma_moe1` load path
    so each sorted row computes `(src_pe, src_token)` and reads `GetAs(src_pe)[token*H+k]` directly.
  - Then **mori dispatch skips WarpCopy entirely — only publishes routing metadata** → dispatch is
    fused into the stage-1 GEMM read (exactly what megamoe does). Combine stays separate unless
    stage-2 also does weighted P2P scatter (needs blocker #1 fixed).
- **Tier 3 — hard: a unified `mori-megamoe` HIP kernel** doing route + remote gather + MFMA +
  combine in one launch (port aiter FlyDSL/Opus tile configs into mori). Closest to B200 deep_gemm
  megamoe; highest engineering cost.

### ROI reality check (why this is NOT the top priority)
Per §5r/§5s, MI355X EP is **MoE-GEMM-bound (~52–57%), not comm-bound** (mori a2a ~14–20%, and even
that is spin-inflated). Fusion mainly recovers the comm + the intermediate materialize/sort/quant
passes — it does **not** close the raw per-FLOP MoE-GEMM gap to B200 (per-layer ~274 µs B200 fp4 vs
~450+ µs MI355X). **Recommendation:** do Tier 1 first (cheap, removes real materialization + overlaps
comm); Tier 2 is the true megamoe analog and is feasible (mori `_p2p` combine is the template) but
needs aiter-kernel work + blocker #1; Tier 3 only if matching B200's architecture is the goal. To
actually chase B200 TTT, invest in **faster fp4/fp8 MoE GEMM kernels**, not comm fusion.

Key files: mori `python/mori/ops/dispatch_combine.py`, `src/ops/dispatch_combine/intranode.hpp`,
`include/mori/application/application_device_types.hpp` (`SymmMemObj::GetAs`); aiter
`aiter/fused_moe.py`, `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`; sglang bridge
`srt/layers/moe/token_dispatcher/moriep.py`, `srt/layers/moe/moe_runner/aiter.py`.

## 5u. ✅ Tier-1 v1 LANDED — per-op decode dispatch dtype = bf16 (skip the round-trip) (2026-07-14)

Implemented the safest slice of §5t Tier-1: eliminate the decode fp4→bf16→fp8 activation round-trip
(§5f) by dispatching the **decode** small-cap op in **bf16** while **prefill keeps fp4** (bandwidth).
With bf16 dispatch the aiter bridge skips `upscale_mxfp4` (a1_scale becomes None) and `fused_moe`
does a single fused quant+sort at the capped M (256 ≤ 341 → the cheap fused path, exactly like DP).

**Code (`srt/layers/moe/token_dispatcher/moriep.py`):** made the dispatch dtype per-op.
- New knob `SGLANG_MORI_DECODE_DISPATCH_DTYPE=auto|bf16|fp8|fp4` (default `auto` = same as prefill →
  no behavior change unless set). Parsed in `_apply_dispatch_dtype_override`.
- `decode_dispatch_dtype` property + `_use_decode_op()` (mirrors the `mori_op` decode predicate:
  `decode_cap>0 and not is_extend`) + `active_dispatch_dtype` (used by both `dispatch_a` sites so the
  quant matches the recv-buffer layout the op was built for).
- `_build_mori_op(..., dispatch_dtype=None)`; the decode op is built with `decode_dispatch_dtype`, the
  prefill op with `self.dispatch_dtype`. Server log confirms **both** ops built: decode=bf16,
  prefill=fp4.

**A/B (8k in / 512 out, conc256, same build, mori-EP capped: prefill_cap 8192 + decode_cap 256):**
| metric | fp4 baseline | bf16 (global) | **per-op decode=bf16** |
|---|---:|---:|---:|
| Median TPOT (ms) | 82.77 | 82.12 | **81.49 (−1.5%)** |
| Median ITL (ms) | 47.65 | 46.53 | **46.32 (−2.8%)** |
| Mean TTFT (ms) | 18,988 | 19,327 (+1.8%) | **18,929 (≈baseline)** |
| Total tok/s | 36,449 | 36,450 | **36,844 (+1.1%)** |
| Output tok/s | 2,144 | 2,144 | **2,167 (+1.1%)** |
- **gsm8k (full, 5-shot, same build):** per-op **0.9295** vs fp4 baseline **0.9325** (±0.007) →
  statistically identical, **no regression** (bf16 decode is strictly higher-precision than fp4
  decode, so it can't hurt; this build's gsm8k baseline is ~0.93, not the older 0.95).

**Findings:**
- **Global bf16 is a wash** (decode −2.4% ITL but prefill TTFT +1.8% from the 4× a2a bandwidth on the
  large prefill volume). **Per-op is the clean capture**: decode gets the ITL/TPOT win, prefill keeps
  fp4 so TTFT is unchanged → net **+1.1% total throughput, −2.8% decode ITL, correctness-neutral**.
- **Magnitude confirms §5r/§5s**: post-cap (§5k) the round-trip was already cheap (M=256), so killing
  it is only ~a couple % — the real cost is still MoE GEMM. Tier-1(a) is a free, correct, small win;
  it is NOT the lever that closes the gap to DP/B200.

**Recommended recipe add:** `SGLANG_MORI_DECODE_DISPATCH_DTYPE=bf16` (on top of the §5o caps). Cheap,
correct, decode-only.

**Not yet done (rest of Tier-1):** the bigger piece — eliminating the aiter `moe_sorting` scatter via
mori `dispatch_standard_moe` expert-major layout (needs `ENABLE_STANDARD_MOE_ADAPT=ON` mori rebuild +
aiter change) — remains (§5t Tier-1 item 2). AsyncLL overlap (item 3) is still blocked by the fp8
combine incompatibility (§5j/§5k).

## 5v. Evaluation — Tier-1 item 2 (eliminate `moe_sorting` via expert-major dispatch): NOT worth it (2026-07-14)

Evaluated the concrete change (mori `dispatch_standard_moe` expert-major layout → skip aiter
`moe_sorting`). **Verdict: mis-classified as "Tier-1" — it's really Tier-2/3 effort for ~3–5% upside.
Recommend deferring.**

**Upside is small (measured, decode trace `trace_ep_capped`, per 61-layer step):**
- `opus_moe_sorting_entry` (MultiPhase 0.91 + ClearWork 0.25) + `mxfp4_moe_sort` 0.34 ≈ **1.5 ms/step**
  (~3% of the ~50 ms step); + permute `index_elementwise` ~1.0 ms → **at most ~5%** of decode.
- **DP pays the same** `opus_moe_sorting` (1.81 ms/step) → sorting is **inherent to aiter's grouped
  MoE, not an EP-specific cost**. Removing it in EP would be an EP-only *advantage* over DP of a few
  %, not a gap-closer.

**Why it's expensive (layouts are NOT compatible today):**
- aiter's grouped GEMM (`mfma_moe1*`/Opus) reads a **flat `[M,H]`** buffer and gathers rows via
  `sorted_token_ids` (packed `(topk_slot<<24)|row`) + `sorted_expert_ids` (per `block_size_M`=32
  tile) + `num_valid_ids`. It does **not** read an expert-major `[E, max_tpe, H]` buffer.
- mori `dispatch_standard_moe` produces expert-major `[E, max_tpe, H]` + per-expert counts, **but**:
  (1) it still does a **flat all2all + in-kernel convert** (doesn't save the transfer), (2) **no
  block padding**, (3) **no** aiter routing metadata, (4) **no expert-major scales** (convert copies
  hidden bytes only), (5) token index is the global send-pos, not the recv-row aiter wants.
- `moe_sorting` also does 10 jobs beyond reorder (block-pad, packed indices, sorted weights, EP
  `expert_mask` remap, `num_local_tokens` truncation, `moe_buf` zero, local_topk_ids, inter-stage
  quant+sort). A "metadata-only from expert-major" shim still leaves the GEMM gathering from flat
  `[M,H]` → you'd convert to expert-major then flatten back, **negating the win**.

**Real win requires a new aiter GEMM entry that reads `[E, max_tpe, H]` directly** — a kernel project,
not config. And it needs coordinated changes across **all three repos** + a mori rebuild with
`ENABLE_STANDARD_MOE_ADAPT=ON` (necessary, not sufficient):
- **mori** (rebuild): call `dispatch_standard_moe`; ideally fuse expert-major write into recv; add
  block padding + expert-major scale scatter; keep `packed_recv_src_info` for combine.
- **aiter**: new expert-major stage-1/2 GEMM entry (or teach FlyDSL `mfma_moe1` an expert-major X
  pointer mode) + `fused_moe(skip_moe_sorting=...)` API.
- **sglang**: moriep branch to stdmoe, extend `MoriEP*DispatchOutput`, rewire `AiterRunnerCore` to
  `fused_moe_2stages` with prebuilt/expert-major args + `combine_standard_moe`.

**Recommendation:** **do NOT pursue item 2 now.** Small upside (~3–5% decode, and it's a general MoE
cost), large cross-repo/kernel effort, mori rebuild. The higher-ROI lever remains **faster fp4/fp8
MoE grouped-GEMM kernels** (the actual B200 gap, §5s) — invest there. Revisit expert-major dispatch
only if/when aiter grows an expert-major grouped-GEMM entry point (then mori stdmoe becomes the
cheap enabler). Corrects §5t's optimistic "Tier-1 item 2" framing.

## 5w. Evaluation — how much headroom is left in the fp4/fp8 MoE GEMM kernel? (2026-07-14)

Roofline of the DSV4-Pro MoE GEMM (H=7168, moe_inter=3072, E=384, topk=6, EP8 → 48 local
experts/rank, weights **fp4** `wfp4` + mxfp4 e8m0 scale, act fp8). Measured on MI355X (~8 TB/s HBM3e).

**Decode MoE GEMM = the §5s bottleneck, but it is DEEPLY MEMORY-BOUND (weight-read), not compute:**
- Achieved (capped EP, `trace_ep_capped`): stage1 `mfma_moe1` 172 µs + stage2 `mfma_moe2` 127 µs =
  **299 µs/layer** (18.2 ms/step).
- Per-expert fp4 weights = W1[2·3072,7168] + W2[7168,3072] + scale ≈ **35 MB**; at conc256 nearly all
  48 local experts activate → **~1,685 MB weight read / layer / rank**.
- Tokens/expert at decode ≈ **4** (256 seqs × topk6 ÷ 384 experts). Compute/mem crossover is
  **M≈312 tokens/expert** → decode runs at **~1.3% of the compute-roofline arithmetic intensity** =
  firmly HBM-bound. **fp4-vs-fp8 *compute* speed is irrelevant for decode.**
- Weight-read floor @8 TB/s = **211 µs/layer** (12.8 ms/step). Achieved 299 µs = **~70% of peak HBM /
  1.42× above floor.**

**Derivations (assumptions explicit — both are computed, not measured):**

*① ~4 tokens/expert (decode):* pure routing arithmetic —
```
decode tokens/step (global, conc256) = 256   # steady-state: all 256 in-flight seqs decode 1 tok
  × topk 6                            = 1,536  expert-assignments/step
  ÷ 384 experts                       ≈ 4.0    tokens/expert (avg)
```
Assumes (a) 256 global decode tokens/step (conc256, DP8 → 32/rank, but MoE is EP-shared so it sees all
256), (b) **balanced routing** (real routing is skewed → some experts more, some fewer; 4 is the
mean). Per rank: 48 local experts × 4 ≈ 192 tokens dispatched in. 4 ≪ crossover 312 → memory-bound.

*② 211 µs/layer weight-read floor @8 TB/s:*
```
per expert (fp4 0.5B/elem + mxfp4 e8m0 scale 1B/32elem = 0.53125 B/elem):
  W1 gate+up [2·3072, 7168] = 44.04M elem × 0.53125 = 23.40 MB
  W2 down    [7168, 3072]   = 22.02M elem × 0.53125 = 11.70 MB   → 35.09 MB/expert
per layer/rank (all 48 local experts active at conc256): 48 × 35.09 = 1,685 MB
floor = 1.685e9 B ÷ 8.0e12 B/s = 210.6 µs/layer   (×61 = 12.8 ms/step)
```
Assumes (a) **weight-read-dominated** (decode activations ≈1.4 MB ≪ 1,685 MB → negligible; scale
included), (b) all 48 experts active (true at conc256), (c) **8 TB/s nameplate** — real achievable HBM
is ~80–90% of nameplate, so vs *achievable* peak (~6.8 TB/s → ~248 µs) the achieved 299 µs is ~**83%**,
i.e. even less true headroom (reinforcing the ~10–15% practical-kernel-headroom conclusion).

**Why B200's 274 µs (fused dispatch+GEMM+combine) does NOT violate this floor — the key subtlety:**
B200's `mega_moe` = 274 µs/layer reads the *same* fp4 weights (same model, EP8 → 48 local experts,
same ~1,685 MB) on the *same* HBM class (~8 TB/s) → **same ~211 µs floor**. 274 µs = 1.30× floor =
~77% BW eff — **above** the floor, fine.
- **You must NOT "subtract" the dispatch+combine cost from 274 to get a GEMM-only number.** On a
  memory-bound kernel the fused a2a is **overlapped with weight streaming** (the kernel loads remote
  peer tokens *while* it streams the 1,685 MB of weights it is already bandwidth-bound on) → the comm
  adds ≈0 *serial* time. If you naively subtracted an ~110 µs comm you'd get ~164 µs < 211 µs floor —
  which is exactly the impossible/contradictory number, proving the comm is **hidden, not additive.**
- So B200's 274 µs **is** essentially the weight-read-bound MoE time (comm tucked underneath); there is
  no sub-floor "GEMM-only". MI355X's 299 µs GEMM is at the *same* wall (70% BW), the difference is
  MI355X **exposes** the a2a as a *separate* +110 µs serial kernel (§5p) instead of hiding it.
- **⇒ The MI355X gap vs B200 on decode is not a slower GEMM (both ~70–77% of the same floor) — it's
  the *exposed vs overlapped* comm.** That is why fusion (§5t/§5u), not a faster kernel, is the decode
  lever; and it also caps the fusion win at "hide the ~110 µs a2a", not "make the GEMM faster".

**⇒ Decode MoE GEMM kernel headroom is SMALL (~10–15% realistic, ~30% theoretical):**
- The ~30% to the roofline floor is mostly eaten by skinny-GEMM inefficiency at M≈4/expert
  (wave-quantization, per-expert tile boundaries, mxfp4 scale reads) that tile tuning can't remove.
- The decode shape **is already tuned** (`tuned_fmoe.csv` has token=1..256 keys), and §5m found
  **decode-key tuning NEUTRAL** → the kernel is already near its practical BW limit.
- **Same HBM class as B200 (~8 TB/s) → same floor.** B200's §5s decode edge is **fusion** (no separate
  a2a/quant/sort) + slightly better BW efficiency, **not** a faster GEMM. So "write a faster decode
  MoE kernel" has limited upside.

**The real decode-MoE lever is ARITHMETIC INTENSITY, not the kernel.** Weight read is ~fixed (all
experts active); the only way to cut per-token cost is amortize those reads over MORE tokens/expert:
- **Higher concurrency / bigger decode batch** — throughput scales ~linearly with tokens/expert until
  M/expert→~312 (currently ~4 → up to ~75× arithmetic-intensity headroom before compute-bound). This
  is by far the biggest decode-MoE dial.
- **Speculative decoding / MTP** (DSV4 ships MTP) — verify K tokens/step → K× tokens/expert → same
  weight read amortized → better decode tok/s at fixed concurrency.

**Prefill MoE GEMM = compute-bound (large M)** — *here* fp4 tensor-core throughput + tile/tuned-CSV
choices genuinely matter, and it's the legit "faster fp4 kernel" target (and where Blackwell fp4
wins). But prefill is already ~90% of DP (§5o), so upside is bounded.

**Verdict / where to invest:**
1. **Decode MoE GEMM kernel itself: ~10–15% at most** (BW-eff tuning of the skinny shape) — low ROI,
   largely tapped (§5m). Not the lever.
2. **Arithmetic intensity (concurrency + MTP/spec-decode): highest decode ROI** — amortizes the fixed
   weight reads; this is the real decode-throughput dial, orthogonal to the kernel.
3. **Prefill fp4 GEMM tuning: modest**, already near DP.
- Net: the MoE GEMM is *not* a big compressible block on decode (it's at the memory wall); chase
  tokens/expert (batch/MTP) and fusion (§5u/§5t), not a faster decode kernel.

## 5x. DP vs EP MoE both sit at ~55–70% of the SAME HBM floor — historical gap was padded-M, not DP being "at the ceiling" (2026-07-14)

Q: B200 EP-MoE ≈ noEP-MoE, but MI355X (historically) EP-MoE ≫ DP-MoE — does that mean MI355X's DP
MoE is optimized to the HBM ceiling? **A: No.**

**Config:** MI355X DP = `tp8 + dp8 + enable_dp_attention, ep_size=1` → DP-attention + **TP-sharded MoE**
(each rank holds 1/8 of every expert). EP = mori, 48 full local experts/rank. **Both put ~1/8 of total
MoE weight per rank → the SAME ~211 µs/layer (12.8 ms/step) weight-read floor (§5w).**

**Measured decode MoE GEMM/step (same build, per-step):**
| MI355X decode | MoE GEMM/step | µs/layer | vs 211 µs floor | BW eff |
|---|---:|---:|---:|---:|
| weight-read floor @8 TB/s | 12.8 ms | 211 | 1.0× | 100% |
| **EP capped** (mfma_moe1 10.5 + mfma_moe2 7.75) | 18.3 ms | 299 | 1.43× | **~70%** |
| **DP TP-MoE** (mfma_moe1 13.9 + opus_stage2 5.82 + mfma_moe2 2.03 + reduce 1.44) | ~23 ms | ~380 | ~1.8× | **~55%** |

- **Neither is at the HBM ceiling** — both ~55–70% of the same floor (the rest lost to skinny-GEMM
  M≈4/expert inefficiency). Post-cap, **EP MoE is actually ≈/slightly faster than DP MoE**. (DP/EP use
  different kernel decompositions — DP `opus_moe_stage2` vs EP `mfma_moe2`, different tiles — so
  kernel-by-kernel isn't exact, but both land in the 55–70% band.)
- **The historical MI355X EP≫DP MoE gap was the padded-M bug**, not DP optimization: mori's
  over-provisioned recv buffer made EP run the MoE over M=131072 → 1.42 ms/layer (§5e). B200's
  deepep/megamoe sizes the recv buffer correctly → never had the pathology → B200 EP-MoE ≈ noEP-MoE.
  After our cap fix (§5k/§5o), MI355X EP-MoE dropped to 0.30 ms/layer ≈ DP → gap closed by fixing EP's
  padding, **not** because DP is at the ceiling.

## 5y. ✅ Decode cap was STILL 2× over-provisioned — right-size to CGBS/dp: EP 91%→98% of DP (2026-07-14)

**Correction to §5x first:** §5x's DP decode-MoE number (380 µs/layer) was WRONG — it used
whole-trace sums contaminated by prefill layers. Isolating the **last decode step** on rank-7 (the
clean decode path) gives the truth:
| per-layer (last decode step, rank7) | EP cap=256 | DP | EP/DP |
|---|---:|---:|---:|
| MoE stage1 (`mfma_moe1`) | 169 µs | 47 µs | 3.6× |
| MoE stage2 | 125 µs | 30 µs | 4.1× |
| **MoE total** | **294 µs** | **77 µs** | **3.8×** |
| dispatch / allgather | 28 | 9.9 | 2.8× |
| combine / reduce_scatter | 84 | 9.5 | 8.8× |
So EP decode was **3.8× DP on MoE + 5.7× on comm** — the decode gap is real and MoE-dominated
(not "EP does less work"; §5x's "EP≈faster" was the contamination artifact).

**Root cause = residual padded-M (again).** Decode MoE M = `nextPow2(decode_cap × world)`; `cap=256`
→ **M=2048**, but the real max decode batch is only `CGBS/dp = 1024/8 = 128` tokens/rank. The recv
buffer was 2× over-provisioned → EP ran the decode MoE + combine over 2× padded rows (and M=2048
also trips the fused→split quant threshold 341, §5f). §5l's sweep only tested caps ≥256 (upward:
256/512/1024) and never went lower.

**The cap floor is tied to `--cuda-graph-max-bs` (CGBS):** each rank's max decode batch = CGBS/dp,
and `cap ≥ CGBS/dp` (else cuda-graph capture fails with a shape mismatch, e.g. `tensor a(128) vs
b(64)`). So the tightest safe cap for CGBS=1024 is **128**. Going below needs CGBS+MAXRUN lowered
too (which caps concurrency).

**Sweep (8k/1k conc256, on top of bf16-decode + prefill cap 8192):**
| config | total tok/s | TPOT | ITL | gsm8k | vs DP |
|---|---:|---:|---:|---:|---:|
| DP | 30,643 | 55.94 | 36.81 | — | 100% |
| EP cap=256 (old) | 27,957 | 64.04 | 46.53 | (0.93 full) | 91.2% |
| **EP cap=128 (M=1024, CGBS=1024)** | **30,081** | 58.25 | 40.75 | 0.95 | **98.2%** |
| EP cap=64 (M=512, CGBS=512) | 30,200 | 57.87 | 40.24 | 0.935 | 98.6% |
| EP cap=32 (M=256, CGBS=256) | 30,097 | 57.73 | 39.84 | 0.92 | 98.2% |

- **cap=256→128 = the win: +7.6% tput, ITL −12%, EP 91%→98% of DP, FREE drop-in** (cap=128=CGBS/dp,
  no CGBS/concurrency change, gsm8k 0.95 intact).
- cap 64/32: only ~−0.9 ms extra ITL, require lowering CGBS/MAXRUN (concurrency ceiling ↓), and
  **gsm8k declines 0.95→0.935→0.92** (token drops near the tight cap). Not worth it.
- **New recommended recipe:** `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128` (= CGBS/dp) instead of
  256. Follow-up: auto-size the decode cap to `cuda_graph_max_bs / dp_size` (like the §5o prefill-cap
  suggestion) so it's never over-provisioned.

## 5z. Evaluation — Option 3: padding-aware (masked) aiter MoE GEMM = the permanent fix (2026-07-14)

Goal: make decode MoE compute scale with REAL per-expert tokens (not the padded recv buffer), like
B200 deep_gemm — so an over-sized buffer is free and no cap tuning is ever needed.

**Where the padded-M cost actually comes from (mechanism):**
- `moe_sorting` **does** honor `num_local_tokens` (sorts only real rows) — but the **allocations**
  (`sorted_ids`, `sorted_expert_ids`) and the **GEMM grid-Y** (`= sorted_expert_ids.shape[0]`) and the
  **tuning key** (`get_padded_M(token_num=hidden_states.shape[0])`) are all sized from the **buffer M**
  (`decode_cap × world`). cap=256 → ~519 M-blocks vs cap=128 → ~263 (~2×).
- The FlyDSL/Opus GEMM **early-exits invalid blocks** via `num_valid_ids[0]`, but the **CTAs are still
  launched** at the static max grid (cuda-graph freezes shapes at capture). Invalid blocks are
  **cheap-but-not-free** (scheduler/residency) → matches the ~14% ITL per 2× buffer (not 2×).
- So it's neither "free" (deep_gemm) nor "full 2× cost" — it's launch/grid overhead ∝ buffer M.

**Does aiter already have a masked/variable-count GEMM?**
- **YES, but gfx1250-only:** `aiter/ops/flydsl/grouped_moe_gfx1250.py` + `..._masked` kernels
  (`compile_moe_grouped_gemm{1,2}_{a8w4,mxfp4}_masked`, `_make_contiguous_psum_layout(masked_m=...)`,
  explicitly "CUDAGraph-safe") — a deep_gemm-analogous masked path with per-expert `masked_m`.
- **Our MI355X = gfx950**, whose production path uses the **sorted layout + `num_valid_ids` early-exit
  only** — NO per-expert `masked_m`. So the masked path is NOT active on our box.
- sglang already computes per-expert counts (`mori_op.local_expert_count`) but **only forwards the
  scalar `total_recv`** to aiter, not the per-expert vector.

**Effort / payoff:**
| path | change | effort | win |
|---|---|---|---|
| **partial** (grid/tuning key from real count, per-cap-bucket alloc, add `num_rows` to all quant) | bridge + `moe_kernels.py` | ~3–5 days | ~10–20% decode MoE — **≈ what Option-1 auto-cap already gets for free** |
| **full masked** (port gfx1250 `masked_m` grouped GEMM to gfx950 MFMA + wire `local_expert_count` + masked quant + tuning CSVs + graph validation) | aiter kernels + sglang | **2–4+ weeks** | buffer size irrelevant (B200 parity); removes cap tuning entirely; recovers residual (M=1024→~256) |

**Verdict:**
- The **partial** fix overlaps with Option-1 (auto-size cap, §5y) — not worth doing separately.
- The **full masked GEMM is the true "一勞永逸" fix**: it makes the over-padding problem *structurally
  vanish* (oversized buffer free, no env, no per-conc cap tuning) and matches B200. It's a **2–4 week
  kernel project**, but NOT from scratch — aiter's **gfx1250 masked kernels are a direct template**,
  deep_gemm's `fp8_m_grouped_gemm_nt_masked` is the reference pattern, and sglang already has
  `local_expert_count` to wire through.
- **Perf delta over Option-1 is small (~2–5% decode)** — the value is **deployment robustness**
  (no cap knob, buffer-size-independent), not a big speedup. Do Option-1 now; schedule Option-3 if
  eliminating all cap tuning / matching B200's buffer-agnostic behavior is a priority.

## 5z-b. Port scoping — FlyDSL is NOT arch-portable; build masked on gfx950 opus C++ instead (2026-07-14)

Scoped "use the gfx1250 masked kernel as the template." **Critical finding: you cannot retarget the
FlyDSL gfx1250 masked kernel to gfx950 — it's welded to RDNA4 hardware:**
| | gfx1250 masked (FlyDSL) | gfx950/CDNA4 (MI355X) |
|---|---|---|
| matrix engine | `wmma_scale_f32_16x16x128_f8f6f4` | `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4` |
| wave size | 32 (hardcoded) | 64 |
| global→LDS | TDM async tensor DMA | buffer_load + ds_write |
| multi-CTA | cluster launch + mcast | plain workgroup barriers |
- FlyDSL is an **MLIR DSL emitting ROCDL intrinsics directly** — no `arch=`/`target=` knob; kernels
  are hand-written per arch (`moe_gemm_2stage.py` gfx950 MFMA sorted vs `gemm_mxscale_gfx1250.py`
  WMMA+TDM+cluster masked). ⇒ porting the FlyDSL kernel = **near-rewrite, 4+ weeks**, and needs the
  external `flydsl` pkg to even emit `mfma_scale_*` for gfx950 (unverified).

**Better route — the real template is on our own box (gfx950 opus C++):** ~**2–3 weeks**.
- gfx950 ALREADY has the scaled MMA the masked kernel needs: `csrc/include/opus/opus.hpp`
  (`mfma_scale_f32_16x16x128_f8f6f4`, E8M0 block scales, wave64 handled).
- gfx950 ALREADY has an A8W4 **decode stage2** MoE kernel: `csrc/opus_moe/include/gfx950/a8w4/`.
- The masked_m *concept* + `contiguous_psum`/`_make_contiguous_psum_layout` helpers are **arch-neutral**
  (torch/prefix-sum) → reusable.
- sglang bridge **already forwards** `num_recv_tokens_per_expert` to `fused_moe` (and the gfx1250
  grouped hook at `aiter/fused_moe.py:515` just needs its gfx1250-only gate relaxed) → near-zero
  bridge change.
- **Missing pieces:** (1) a gfx950 **stage1** (gate/up) A8W4 decode kernel — doesn't exist yet
  (only stage2); (2) a masked/contiguous scheduler variant (stage2 currently consumes the sorted
  layout, not masked_m); (3) gfx950 E8M0 scale-preshuffle variant (MFMA_SCALE packing ≠ WMMA_SCALE).

**Milestones (smallest-correct-first):**
1. gfx950 A8W4 **stage1** decode kernel (mirror the existing stage2 via `opus.hpp` MFMA_SCALE); validate
   standalone vs torch ref on synthetic `[E, max_m, K]`.
2. **Sorted-layout end-to-end** (new stage1 + existing stage2 on `sorted_token_ids`/`num_valid_ids`) →
   a correct gfx950 A8W4 decode MoE before touching masked_m.
3. **Swap in masked_m**: `num_recv_tokens_per_expert` → `contiguous_psum` → dense `[contiguous_m,K]` +
   per-tile `blk_m < masked_m[e]` guard (mirror `gemm_mxscale_gfx1250.py:694-718`); grid static on
   `max_m` for cuda-graph. Then compute scales with real per-expert tokens (B200 parity).
- Files: aiter `csrc/opus_moe/include/gfx950/a8w4/` (new stage1 + masked scheduler),
  `aiter/ops/opus/moe_stage2_a8w4.py` (+meta), `aiter/fused_moe.py:515` (relax gate),
  `grouped_moe_gfx1250.py` (accept external masked_m / gfx950 branch), tuning CSV (gfx950 rows).
- **Early unknowns to de-risk first:** does `flydsl` expose `mfma_scale_*` for gfx950 (else C++-only);
  the E8M0 scale-word packing for MFMA_SCALE vs WMMA_SCALE.

### ✅ Unknowns RESOLVED (2026-07-14) — scope drops to ~1–2 wk, stays in FlyDSL
Both unknowns came back favorable, and they re-scope the port DOWN (no C++ opus route needed):
1. **flydsl DOES emit MFMA_SCALE for gfx950** — `flydsl/expr/rocdl/__init__.py::mfma_scale_f32_16x16x128_f8f6f4`
   + `cdna4.py::MFMA_Scale`; generated ops include `mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4`.
2. **The gfx950 MFMA_SCALE MoE compute kernel ALREADY EXISTS in FlyDSL** — `kernels/mixed_moe_gemm_2stage.py`
   emits `mfma_moe1_silu_mul_a{fp8}_w{fp4}` (line 219) and `mfma_moe2_a{fp8}_w{fp4}` (line 3086) — **these
   are literally the `mfma_moe1_silu_mul_afp8_wfp4` / `mfma_moe2_afp8_wfp4` kernels in our decode trace.**
   It's wave64, gfx950, uses `rocdl.mfma_scale_f32_16x16x128_f8f6f4`, and **already handles the a8w4
   E8M0 block-scale packing** (so unknown #2 is already solved — no WMMA↔MFMA scale-word reverse-eng).
3. **What that kernel LACKS = only the scheduler**: it uses the **sorted layout** (`arg_sorted_token_ids`
   + `arg_num_valid_ids`, `blk_valid = bx_m < num_valid_ids`), NOT masked_m. No `masked_m`/`contiguous_m`/
   `bincount` in it.

**Re-scoped plan (best route): graft the masked_m scheduler onto the EXISTING gfx950 FlyDSL kernel.**
- Port only the **scheduler** from `gemm_mxscale_gfx1250.py:694-718` / `grouped_moe_gfx1250.py`
  (per-expert `blk_m < masked_m[e]` guard + `contiguous_psum` dense layout — the psum helper is
  **arch-neutral torch/prefix-sum**, directly reusable) into `mixed_moe_gemm_2stage.py`. **Reuse its
  MFMA_SCALE compute inner loop, a8w4 scale packing, wave64 — all unchanged.**
- This ALSO kills a *second* padding source the sorted layout has: per-expert ceil-to-`block_m`(=32)
  padding (48 experts × 32 = 1536 rows for ~192 real tokens = 8× waste even with a right-sized recv
  buffer). masked_m processes exactly `masked_m[e]` rows.
- Stays entirely in FlyDSL Python (no C++ opus stage1 needed, no WMMA→MFMA rewrite). Est **~1–2 weeks**.
- Bridge: sglang already forwards `num_recv_tokens_per_expert`; add a gfx950 branch to the
  `aiter/fused_moe.py:515` grouped hook (relax the gfx1250-only gate) that feeds masked_m from it.
- Milestones unchanged in spirit: (1) add masked scheduler to `mfma_moe1`, validate vs torch on
  `[E,max_m,K]`; (2) add to `mfma_moe2` → masked stage1+stage2 e2e; (3) wire the bridge + gfx950
  tuning CSV rows; keep grid static on `max_m` for cuda-graph.

## 5z-c. Milestone-1 findings (masked gfx950 MoE GEMM port) — mapping done, kernel write pending (2026-07-14)

Mapped the port concretely. **Good news, refined scope, and a blocker.**

**What already exists (reusable):**
- **gfx950 MFMA_SCALE MoE compute + a8w4/mxfp4 E8M0 scale packing** — `kernels/mixed_moe_gemm_2stage.py`
  (`compile_mixed_moe_gemm1` @72, kernel body @374, launch @2831). It's the source of the trace's
  `mfma_moe1_silu_mul_afp8_wfp4` / `mfma_moe2_afp8_wfp4`. Its scheduler = **sorted layout**: grid-Y =
  `size_expert_ids_in` blocks; block `bx` → `bx_m = bx*sort_block_m` → `expert_ids[bx]`, gather via
  `sorted_token_ids`, guard `blk_valid = bx_m < num_valid_ids[0]` (@620-628).
- **Masked scaffolding is largely arch-neutral & already implemented** — the gfx1250 grouped path
  (route maps → scatter to `[E,max_m,K]` → `masked_m` (bincount) → `contiguous_psum` → GEMM →
  gather-reduce) runs on non-gfx1250 with the GEMM mocked (`_mock_grouped_gemm`). Masked guard pattern:
  `gemm_mxscale_gfx1250.py:694-718` (`blk_m < masked_m[e]`); layout: `grouped_moe_gfx1250.py`
  `_make_contiguous_psum_layout` (torch/prefix-sum, arch-neutral).
- **Validation harness**: `op_tests/test_flydsl_grouped_gemm_gfx1250.py` — `_torch_moe_ref` (reuses
  `torch_moe_stage1/2`), logits_diff<0.01 gate, a8w4/mxfp4, `AITER_FORCE_GFX1250=1` + mock to run
  scaffolding on any arch.

**Design (agreed): write a gfx950 masked `gemm1` that matches the `compile_moe_grouped_gemm1_a8w4_masked`
interface** — reuse `mixed_moe_gemm_2stage.py`'s MFMA_SCALE inner loop + scale packing, swap the sorted
scheduler for the masked one (static grid on `max_m`, per-tile `blk_m < masked_m[e]`, `[E,max_m,K]`
input via contiguous_psum). Bonus: kills the sorted layout's per-expert ceil-to-`block_m`(=32) padding.

**⚠️ BLOCKER found (Milestone-1 de-risk run on our gfx950 box):** the masked scaffolding is NOT fully
arch-neutral — the **weight-scale preshuffle is arch-specific**. `moe_shuffle_scale` folds to the
grouped-only **n32k4 e8m0** layout on gfx1250, but on gfx950 dispatches to `shuffle_scale` (asserts 2D)
→ the gfx950 masked GEMM needs its **own MFMA_SCALE-compatible scale layout** (which `mixed_moe`
already uses for the sorted path — so reuse THAT scale layout, not the gfx1250 n32k4 one).

**Remaining (multi-day kernel work, NOT started):**
1. Write `compile_moe_grouped_gemm1_a8w4_masked` **for gfx950** in a new `..._gfx950.py` (or branch in
   mixed_moe): masked scheduler + mixed_moe MFMA_SCALE compute + **mixed_moe's** scale layout.
2. gfx950 branch in `grouped_moe_gfx1250.py` (feed `masked_m` from `num_recv_tokens_per_expert`; use the
   gfx950 scale preshuffle, not n32k4).
3. Adapt the test harness scale-shuffle for gfx950; validate gemm1 vs `torch_moe_stage1` (logits_diff<0.01).
4. Then gemm2, then bridge wiring (`fused_moe.py:515` gate relax), then gfx950 tuning CSV.
- Realistic: this is the **~1–2 week kernel project** itself; mapping/design is done, implementation is next.

## 5z-d. Milestone-1 IMPLEMENTATION STATE — RESUME HERE (2026-07-14)

Building the gfx950 masked stage1 as a `grouped_masked_m` flag INSIDE
`/sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` (reuse all compute; branch
only scheduler + addressing). v1 = simple+correct, `[E,max_m,K]` scatter (contiguous_psum later).

**Design invariants (decided):**
- Reuse mixed_moe's MFMA_SCALE compute + **its** a8w4 E8M0 scale layout (`make_preshuffle_scale_layout`)
  — NOT the gfx1250 n32k4 layout (that was the blocker). Reuse `shuffle_weight(layout=(16,16))`.
- ABI-preserving: in masked mode, **repurpose the `arg_sorted_token_ids` kernel slot to carry
  `masked_m[E]`** (int32 per-expert live count). No new kernel arg.
- Block map (masked): `bx → (expert = bx // m_tiles_per_expert, local_row = (bx % m_tiles_per_expert)
  * sort_block_m)`; `m_tiles_per_expert = ceil(max_m/sort_block_m)` (compile-time const, cuda-graph safe).
  Guard `blk_valid = bx_m < masked_m[expert]`. Input/output/scale rows = `expert*max_m + bx_m + local`.
- stage1 output stays grouped `[E,max_m,inter]` (un-permute happens post-stage2), so NO token gather.

**DONE (edits landed, `grouped_masked_m=False` default path byte-identical, syntax-checked):**
1. compile sig: added `grouped_masked_m: bool=False, max_m: int=0` (+ `max_m>0` check) — near line 97.
2. `m_tiles_per_expert` const — after `sort_block_m = max(32, tile_m)` (~line 116).
3. **scheduler branch** (~line 633): masked computes expert/bx_m from `bx`, loads `masked_m` from
   `ptr_buffer_resource(arg_sorted_token_ids, experts*4)`, sets `blk_valid`/`exp_valid`.
4. **x-row branch** (~line 723): masked `x_row = expert*max_m + bx_m + row_local` (÷4 dword), else sorted gather.

**TODO (next session — edit→compile→validate loop):**
1. **scale-x load addressing** — find where A-scale (`sx_rsrc`, `layout_a_scale` @~435 uses `sorted_m`)
   is indexed by row in the K-loop; add masked branch: scale row = `expert*max_m + bx_m + local`. Must
   match how the Python entry lays out the `[E,max_m,K//32]` A-scale (per-1x32 mxfp8, mixed_moe layout).
2. **output-store row** — find the epilogue store (writes silu_mul result); masked row = `expert*max_m
   + bx_m + local` into grouped `[E,max_m,inter]` out (search the store after the epilogue, ~line 1500+).
   Also `arg_out_scale_sorted` if the stage1 out is quantized per-row for stage2.
3. **launcher** `launch_mixed_moe_gemm1` (~line 2860 now): masked grid persist-dim =
   `m_tiles_per_expert*experts` (not `size_expert_ids_in`); pass `masked_m` tensor into the
   `arg_sorted_token_ids` position; keep other args (expert_ids/num_valid/sorted_weights) as dummies.
4. **Python entry** (new, e.g. `flydsl_masked_moe_stage1` in `moe_kernels.py` or a `grouped_moe_gfx950.py`):
   build `masked_m` (bincount of recv per expert) + scatter x/scale into `[E,max_m,K]`/`[E,max_m,K//32]`
   (reuse the arch-neutral route/scatter from `grouped_moe_gfx1250.py` `_build_route_maps_naive`), call
   `compile_mixed_moe_gemm1(grouped_masked_m=True, max_m=..., ...)`.
5. **compile** a tiny config on gfx950 (E=8, max_m=64, model_dim=512, inter=512) — fix DSL errors.
6. **validate** vs `torch_moe_stage1` (adapt `op_tests/test_flydsl_grouped_gemm_gfx1250.py::_torch_moe_ref`
   stage1 half): logits_diff < 0.01. Gotcha: the harness `moe_shuffle_scale` is gfx1250-n32k4 — use
   the mixed_moe/gfx950 scale shuffle instead.

**Key file map:** kernel `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py` (stage1 compile @72, body @374,
launcher @~2860); masked ref pattern `gemm_mxscale_gfx1250.py:694-718`; scatter/route helpers
`grouped_moe_gfx1250.py`; validation `op_tests/test_flydsl_grouped_gemm_gfx1250.py`; sglang bridge
`srt/layers/moe/moe_runner/aiter.py` (`fused_moe.py:515` gate to relax later) + `token_dispatcher/moriep.py`
(`num_recv_tokens_per_expert` already surfaced). Landed Tier-1 commit: `c82ee195f`. getSMVersion +
cohere2 working-tree fixes still needed for any cuda-graph run.

### ✅ Milestone-1 COMPLETE — masked gfx950 stage1 compiles & validates (2026-07-14)
All 6 TODO items landed; the gfx950 masked (deep_gemm-style) **stage1** GEMM is correct end-to-end.
- **Kernel edits** (`mixed_moe_gemm_2stage.py`, `compile_mixed_moe_gemm1(grouped_masked_m, max_m)`):
  1. **scale-x addressing** — `a_scale_row = expert*max_m + bx_m` block-indexed (`//32`) into `layout_a_scale`;
     `stride_n0` is c_mn-independent (only depends on `c_k`), so the preshuffle stride is identical to the
     sorted path — only the row offset + buffer size (`sorted_m = experts*max_m` in masked) change.
  2. **output store** — masked `precompute_row` writes expert-major `[E,max_m,inter]`:
     `global_row = expert_idx*max_m + (bx_m+row_local)`, `row_valid = (bx_m+row_local) < masked_m[expert]`.
     No token scatter, no `lds_tid` read (its sorted-gather load is gated off in masked mode).
  3. **scheduler/grid** — `bx → (expert=bx//m_tiles_per_expert, local=(bx%mtpe)*sort_block_m)`;
     `masked_m[E]` read from the repurposed `arg_sorted_token_ids` slot. NO launcher change needed —
     grid is driven by the runtime `size_expert_ids_in = m_tiles_per_expert*experts` arg. `masked_tag`
     added to `module_name`/`cache_tag` so masked vs sorted JITs don't collide.
  - Validations: `max_m % 32 == 0` (scale 32-block align) AND `max_m % sort_block_m == 0` (tiles don't
    straddle expert boundaries). Default (`grouped_masked_m=False`) path is byte-identical.
- **Python entry:** `aiter/ops/flydsl/grouped_moe_gfx950.py::flydsl_masked_moe_stage1` (standalone thin
  launcher, reuses `_s1_args_fp4`; `masked_m→arg_sorted_token_ids`, dummies for expert_ids/sorted_weights,
  `num_valid_ids`=1-elem, out_scale empty). A-scale preshuffle = **`e8m0_shuffle`** (== `shuffle_scale`
  non-gui) over `[E*max_m, K//32]` — this IS the `make_preshuffle_scale_layout` host producer (same one the
  a4w4 test uses for the B-scale). `a1_scale=None` → `a_scale_one` (skips scale-x, for GEMM-only de-risk).
- **Validation:** `op_tests/test_flydsl_masked_moe_stage1_gfx950.py` — direct grouped stage1 ref
  (dequant the SAME quantised tensors; per-1x32 mxfp8 act + mxfp4 wt), compares only `masked_m[e]` rows.
  **PASS logits_diff ≈ 1.4e-6** (≪0.01 gate) across {a_scale_one T/F, max_m 64/96/128, tile_m 32/64,
  E 8/16, model/inter 512/768}. Both the `a_scale_one` (GEMM/scheduler/output only) and the real scale-x
  path pass, so scale-x addressing (TODO #1) is confirmed correct.
- **Not committed** (working tree only), same as the getSMVersion/cohere2 fixes.

### ✅ Milestone-2 COMPLETE — masked gfx950 stage2 + stage1→stage2 e2e (2026-07-14)
Grafted the SAME masked scheduler onto **`compile_mixed_moe_gemm2`** (`mfma_moe2_afp8_wfp4`). Masked stage2
reads the grouped `[E,max_m,inter]` stage1 output and writes grouped `[E,max_m,model_dim]` — NO token
scatter/combine (the weighted un-permute is a *later* combine pass, not part of the GEMM; this matches B200
deep_gemm where stage2 also emits grouped and combine is separate).
- **Kernel edits** (`compile_mixed_moe_gemm2(grouped_masked_m, max_m)`), mirroring stage1:
  1. **scheduler** (non-persistent branch): masked `expert=bx//m_tiles_per_expert`,
     `bx_m=(bx%mtpe)*tile_m`, `masked_m[E]` from repurposed `arg_sorted_token_ids`, `blk_valid=bx_m<masked_m`,
     `expert_b_base=expert*expert_b_stride`, `tile_has_tokens=const true`; skips sorted expert-id/first-tok loads.
  2. **x-row** = `expert*max_m + bx_m + row_local` (no token gather); **A-scale row** = same, `//32` block idx.
  3. **sx buffer** sized `experts*max_m` rows; **lds_tid** prologue gated off (masked precompute doesn't read it).
  4. **output store** — masked `precompute_row` writes grouped `[E,max_m,model_dim]`
     (`global_row=expert*max_m+bx_m+row_local`, `row_valid=local<masked_m`). Requires **`accumulate=False`**
     (plain store, unique rows), `persist_m>0` (non-persistent), `sort_block_m==tile_m`. No launcher change.
- **Python entry:** `grouped_moe_gfx950.py::flydsl_masked_moe_stage2` (reuses `_s2_args_fp4`; topk=1 so
  out/x buffers size to `E*max_m`; W2 = `shuffle_weight(16,16)`, scales via `e8m0_shuffle`).
- **Validation** (`test_flydsl_masked_moe_stage1_gfx950.py`, `--stage stage2|e2e|all`):
  - masked **stage2** vs grouped down-proj ref: **PASS logits_diff ≈ 1.4e-6** across {max_m 64/128, tile_m
    32/64, E 8/16, model/inter 512/768/1024}.
  - masked **e2e** (kernel stage1 → per-1x32 fp8 quant → kernel stage2) vs full grouped ref (silu→down):
    **PASS logits_diff ≈ 1.6e-6**.
  - sorted-path a4w4 **stage2** regression (atomic + per-slot, accumulate T/F): **100% PASS** (no default-path
    change).
- **Not committed** (working-tree only), same as getSMVersion/cohere2.

### ✅ Milestone-3 COMPLETE — full masked grouped MoE (route→scatter→s1→s2→combine) + fused_moe hook (2026-07-14)
The end-to-end aiter-side masked MoE now runs and matches `torch_moe`.
- **Full entry:** `grouped_moe_gfx950.py::flydsl_masked_moe_gfx950(hidden, w1_shuf, w2_shuf, w1_scale_shuf,
  w2_scale_shuf, topk_weight, topk_ids, ...)` → `[T, model_dim]`. Pipeline:
  1. **route maps** — reuse arch-neutral `grouped_moe_gfx1250._build_route_maps_naive` →
     `(topids_to_rows[T,topk] = expert*max_m+slot, rows_to_tokens[E*max_m], masked_m[E]=bincount)`.
  2. **scatter + quant** — gather `hidden` into expert-major `[E,max_m,K]` by `rows_to_tokens`, per-1x32
     mxfp8 quant (`_quant_per1x32_fp8`) + `e8m0_shuffle` A-scale.
  3. **masked stage1** → grouped `[E,max_m,inter]` (M1 kernel).
  4. per-1x32 fp8 quant of stage1 out.
  5. **masked stage2** → grouped `[E,max_m,model_dim]` (M2 kernel).
  6. **combine / un-permute** — `flydsl_moe_gather_reduce(s2, topids_to_rows, topk_weight)`:
     `out[t] = Σ_k w[t,k]·s2[topids_to_rows[t,k]]`. This is the ONLY un-permute in the whole masked path
     (stage1's grouped output feeds stage2 directly, no re-permute between stages) — the piece the masked
     GEMMs deliberately omit.
- **(a) fused_moe bridge hook:** `_maybe_grouped_gfx950_masked_moe` + call site at `fused_moe.py` right
  after the gfx1250 hook. **Env-gated `AITER_GFX950_MASKED_MOE` (default OFF → returns None → zero behavior
  change)**; supported surface = a8w4/per_1x32/g1u1/silu|swiglu/GGUU/no-bias, else falls through. Shuffles
  raw fp4 weights + e8m0 scales internally; `max_m` = `AITER_GFX950_MASKED_MAX_M` or derived from routing
  (host `.item()`, eager-only — the sglang bridge instead passes the static recv-buffer cap).
- **Validation** (`test_flydsl_masked_moe_stage1_gfx950.py`, `--stage full|hook|all`):
  - **full** (direct entry) vs `torch_moe` (a8w4 ref: fp8-roundtrip act → torch_moe_stage1 → fp8-roundtrip →
    torch_moe_stage2, doweight): **PASS logits_diff ≈ 5–6e-6** across {tokens 32/64/128, topk 2/4/6, E 8/16,
    dim 512/768/1024}.
  - **hook** (through `fused_moe` with env on): **PASS, identical 5.9e-6**.
  - **env-OFF default `fused_moe`** with fp4x2 weights: runs the normal CK 2-stage path (hook returns None) —
    **no regression**. Sorted-path a4w4 stage1/stage2 still 100%.
- **Not committed** (working-tree only).

### ✅ Milestone-4 (part) — flat-recv bridge primitive built + validated (2026-07-14)
Pinned the actual mori layout and built the exact primitive the sglang bridge calls.
- **mori layout (confirmed):** `MoriEPNormalDispatcher.dispatch_b` (`moriep.py:727`) returns
  `packed_recv_hidden` = a **FLAT `[M, K]`** recv buffer (M = recv capacity, already fp8/fp4-quantized by
  `dispatch_a`), plus `recv_scale [M, K//32]` e8m0, `recv_topk_ids` (expert per recv row), and
  `num_recv_tokens_per_expert`. It is NOT expert-major and NOT expert-sorted — the current aiter runner
  (`moe_runner/aiter.py:294`) runs moe_sorting on it. mori's own `combine` does the final topk-weighted
  un-permute back to tokens (so the MoE must emit **flat recv-order** output, un-weighted).
- **Primitive:** `grouped_moe_gfx950.py::flydsl_masked_moe_gfx950_recv(recv_hidden fp8, recv_scale,
  recv_expert_ids, w*_shuf, w*_scale_shuf, ...)` → flat `[M, model_dim]`. It:
  1. scatters flat recv rows into expert-major `[E, max_m, K]` **reusing the dispatched fp8 + e8m0 scale
     directly — NO dequant→requant round-trip** (this is the §5f cost that the mori path pays today; the
     masked path removes it because the masked GEMM consumes fp8 + per-1x32 scale natively);
  2. masked stage1 → per-1x32 fp8 quant → masked stage2 (grouped);
  3. un-permutes back to flat recv order (plain row copy, NO topk weight — mori `combine` applies it).
  `max_m` is a static per-expert cap (graph-safe; bridge sets it to the recv-buffer capacity / decode bound,
  never a GPU `.item()`).
- **Validation** (`--stage recv`): per-recv-row ref (each row through its expert, fp8-roundtrip a2):
  **PASS logits_diff ≈ 1.6e-6** across {E 8/16, M≈469/1762, max_m 128/192, tile_m 32/64, dim 512/768/1024}.

### ✅ Milestone-4 (part 2) — sglang mori-branch masked wiring LANDED (env-gated, default off) (2026-07-14)
Wired the masked path into the sglang mori-EP runner. **Confirmed layout compatibility:** DSV4 quark w4a4
(`quantization/quark/schemes/quark_w4a4_mxfp4_moe.py:220-241`) stores `w13_weight`/`w2_weight` as
`shuffle_weight((16,16))` and `w13/w2_weight_scale` as `e8m0_shuffle(view(E*N, -1))` — **exactly** the
masked entry's expected layout, so `quant_info` tensors pass through with no re-shuffle.
- **`moe_runner/aiter.py::_maybe_run_mori_masked`** (+ call at the top of `AiterRunnerCore.run`): gated by
  `SGLANG_MORI_MASKED_MOE=1` AND `SGLANG_MORI_MASKED_MAX_M>0` (per-expert cap), `is_mori`, PER_1X32, gfx950,
  fp8 recv. Derives per-row expert ids from the packed `num_recv_tokens_per_expert` (host List[int] →
  `repeat_interleave`, expert-contiguous recv assumption), calls `flydsl_masked_moe_gfx950_recv` with the
  fp8 recv buffer + scale + pass-through weights, returns flat `[M, model_dim]` recv-order (mori `combine`
  reduces to tokens, unchanged). try/except → returns None on any mismatch (falls through; `SGLANG_MORI_MASKED_DEBUG=1`
  to log). Default OFF = zero behavior change (py_compile clean).
- **pre_permute (`_pre_permute_deepep_to_aiter`) gate:** when the masked env is on + fp8 dispatch, SKIP the
  `upscale` fp8→bf16 round-trip (§5f) so `run()` receives fp8 + a1_scale (the masked GEMM consumes it
  natively). This is where the §5f round-trip win is realized.

**Milestone-4 (remaining) — on-server validation + perf (NOT run this session):**
- Bring up DSV4 mori-EP with `SGLANG_MORI_DISPATCH_DTYPE=fp8` (masked needs fp8 recv),
  `SGLANG_MORI_MASKED_MOE=1`, `SGLANG_MORI_MASKED_MAX_M=<per-expert cap>` (start ~128; = a safe static
  per-expert bound, NOT total recv M — setting it to M re-introduces E× padding). Verify with
  `SGLANG_MORI_MASKED_DEBUG=1` it doesn't fall through.
- **Open items to confirm on server:** (1) mori fp8 dispatch scale block size — masked assumes **per-1x32
  e8m0**; if mori dispatches `float8_blockwise` per-128, the scale layout won't match (needs a per-1x32
  dispatch or a scale-reblock). (2) recv buffer is expert-**contiguous/packed** (the count→expert-id
  derivation assumes it); if it's dispatch-order, derive ids from `recv_topk_ids` instead. (3) cuda-graph:
  `num_recv_tokens_per_expert` must be static under capture (the per-expert-id build + `.item()` sum are
  host ops — fine eager, need a graph-safe static variant for captured decode).
- Then gsm8k + decode TPOT/throughput vs the §5y cap baseline. Expected win: removes BOTH the residual
  per-expert ceil-to-block padding AND the fp4/fp8→bf16→fp8 activation round-trip (§5f); result is
  buffer-size-independent so the §5y decode-cap knob becomes unnecessary (the "一勞永逸" fix).
- (c) gfx950 masked decode-key tuning CSV: low priority (§5m decode-key tuning neutral; masked removes
  padded-M, fixed 32×256×256 tile expected fine — revisit only if a trace shows it suboptimal).
- Perf note: the primitive's scatter/un-permute are torch index-copies (correctness-first); fuse later
  (mori already delivers the packed recv buffer, so scatter = one index build; fp8+scale reused as-is; only
  the stage1→2 per-1x32 quant + e8m0 preshuffle remain fusable).

## 6b. Server validation checklist for the masked GEMM (READ THIS for e2e validation)

To validate the landed masked stage1/stage2 at the sglang server level:

**Read first:** §5z-d (impl state + wiring), §5y (recipe + the exact baseline numbers to beat), §2/§2b
(bench + decode-trace method), and this §6b.

**Pre-flight (working-tree fixes required for ANY cuda-graph run on this ROCm build):**
- `sglang-upstream/python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` — getSMVersion HIP shim (§5p).
- `sglang-upstream/python/sglang/srt/configs/cohere2_moe.py` — `@strict` no-op fix (§7).
- Landed Tier-1 commit on the mori-ep branch: `c82ee195f` (per-op decode bf16).

**Enable the masked path:** the aiter grouped hook is gated at `aiter/fused_moe.py:515`
(`_use_grouped_gemm_enabled()` = `AITER_USE_GROUPED_GEMM=1` or gfx1250). For gfx950 you must have (a)
relaxed that gate to route gfx950 → the new masked kernel, and (b) fed `masked_m` from
`num_recv_tokens_per_expert` (already surfaced by `moriep.py`). Confirm the server log shows the masked
grouped path is taken (not the sorted `mfma_moe1_..._sort` path).

**Launch recipe (8k/1k conc256, the standard EP point):**
```bash
cd /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/
AITER_USE_GROUPED_GEMM=1 \                 # (or the gfx950 masked gate you added)
SGLANG_MORI_DECODE_DISPATCH_DTYPE=bf16 \
SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128 \   # = CGBS/dp; masked should make this less critical
SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK=8192 \
AITER_CONFIG_FMOE=/workspace/ep_tune/tuned_fmoe.csv \
MODE=mori-ep PORT=8000 bash run_sgl_dsv4_unified.sh   # health: curl :8000/health
```

**Bench + correctness:**
```bash
WORKLOADS="8192:1024" CONCS="256" NP_MULT=2 WARM_MULT=1 \
  RESULT_DIR=/workspace/bench_results_dsv4_epdbg/masked bash sweep_dsv4_sglang_client.sh
lm_eval --model local-completions --model_args model=/dockerx/data/deepseek-ai/DeepSeek-V4-Pro/,base_url=http://localhost:8000/v1/completions,num_concurrent=128,max_retries=3,tokenized_requests=False --tasks gsm8k --num_fewshot 5
```

**Pass/compare targets (same build, 8k/1k conc256 — from §5y):**
| ref | total tok/s | Med TPOT | Med ITL | gsm8k(full) |
|---|---:|---:|---:|---:|
| DP | 30,643 | 55.94 | 36.81 | ~0.93 |
| EP cap=128 (current best) | 30,081 | 58.25 | 40.75 | 0.95 |
| **masked target** | **≥ 30,081, aim ~DP** | **≤ 58** | **→ ~37** | **≥ 0.93** |
- The masked win shows up as **decode ITL closing 40.8 → ~37** (masked processes real per-expert tokens,
  not the padded M=1024). Correctness gate: full gsm8k within noise of ~0.93 (no drop).
- Also capture a **last-decode-step rank-7 trace** (method §2b) and check per-layer MoE stage1/stage2
  dropped from ~294 µs toward DP's ~77 µs — that's the direct mechanism confirmation.

## 6c. ⚠️ Server-validation attempt (2026-07-14) — masked path ACTIVATES on DSV4 but recv-layout assumption was WRONG

Ran the §6b recipe on a live DSV4 mori-EP server (8×MI355X, eager `--disable-cuda-graph`,
`SGLANG_MORI_MASKED_MOE=1`, `SGLANG_MORI_MASKED_MAX_M=128`, `SGLANG_MORI_DECODE_DISPATCH_DTYPE=bf16`,
`SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128`). Wiring in `moe_runner/aiter.py::_maybe_run_mori_masked`
(+ `flydsl_masked_moe_gfx950_recv`, extended to accept bf16 recv + single per-1x32 quant).

**What worked (confirmed):**
- The masked branch **gates + activates correctly** (decode-only via `get_is_extend_in_batch()`; env-gated;
  fp8-scale-mismatch fall-through). Log: `[mori-masked] ACTIVE: E=48 M=1024 K=7168 inter=3072 max_m=128`.
- The masked stage1/stage2 gfx950 kernels **JIT-compile and RUN on real DSV4 shapes** (E=48 local, K=7168,
  inter=3072, M=1024) with **no crash / no illegal-access** — the kernels themselves are production-shape-safe.
- DSV4 quark w4a4 weights (`shuffle_weight(16,16)` + `e8m0_shuffle`) pass through with no re-shuffle (verified).
- Default-off path unchanged; server boots+serves normally.

**What was WRONG → garbage output ("Paris" then gibberish):** the recv→expert mapping. Dumped the actual
tensors (`SGLANG_MORI_MASKED_DUMP=1`):
```
recv_topk_ids shape=(1024, 6) dtype=int32   sample=[293,35,354,162,18,357, 293,35,354,162,18,357, ...]
num_recv_tokens_per_expert = Tensor, counts total_recv ~ 6/8   (per-LOCAL-expert)
```
⇒ the mori NORMAL recv buffer is **token-major** (`[M, hidden]`, one row per *received token*), and
`recv_topk_ids` is **`[M, topk=6]` GLOBAL expert ids** (values 0..383), NOT local, NOT expert-sorted. My
branch's `expert_id = repeat_interleave(arange(E), num_recv_tokens_per_expert)` (expert-contiguous
assumption) is **completely wrong** — that's why output was garbage. (The working sorted path feeds this
same `[M,6]` global `recv_topk_ids` to `moe_sorting`, which maps global→local via `expert_mask` and builds
one sorted (row,slot) entry per **local** expert.)

**Correct masked recv path (for next session):** it's the FULL routing, not a repack:
1. map `recv_topk_ids` global→local (local block `[ep_rank*E_local, +E_local)`; non-local slot → drop);
   the expert_map/mask lives in the MoE layer but is **not yet threaded into the runner masked branch**.
2. route each (recv-row, local-slot) → grouped `[E_local, max_m, K]` (like `flydsl_masked_moe_gfx950`'s
   `_build_route_maps_naive` on the LOCAL ids), masked s1/s2.
3. combine back to `[M, model_dim]` = weighted (`recv_topk_weights`) sum over each token's LOCAL experts,
   THEN mori `combine` sums the per-rank partials across ranks. **OPEN: confirm whether mori `combine`
   re-applies topk weights** (`_combine_core` calls `mori_op.combine(hs, None, topk_ids, **kwargs)` — need
   to check `_combine_kwargs` for weights). If combine weights, the MoE must NOT (pass weight=1 to
   gather_reduce); if not, MoE applies `recv_topk_weights` (doweight). Match whatever the sorted
   `fused_moe(mori)` path does (check `quant_info.doweight_stage1` for DSV4 + the mori combine kwargs).
- `max_m`=128 was fine capacity-wise (total_recv≈6–8 ≪128); the mapping, not the cap, was the bug.

**Also still open (from §6b):** (3) cuda-graph — the current bridge does host `.item()` + data-dependent
scatter (eager-only). A graph-safe decode needs the route-map + scatter as GPU kernels reading the GPU recv
counts (mori's fused route kernels or the gfx1250 `moe_route_maps`/`moe_scatter_copy` kernels are the
template). So even with the mapping fixed, perf (cuda-graph ITL) needs the GPU-side route/scatter.

**Status:** masked GEMM kernels are e2e-correct in aiter unit tests (M1–M3) AND run crash-free at DSV4 scale
on-server; the sglang bridge's recv→local-expert routing + combine-weight matching + graph-safe route/scatter
remain. Left `SGLANG_MORI_MASKED_MOE` default-OFF. The `_maybe_run_mori_masked` counts-cumsum expert
derivation is KNOWN-WRONG and must be replaced with the global→local routing above.

### 6c-ii. Second attempt (2026-07-14) — global→local routing + weighted combine: output coherent-ish but gsm8k=0.0
Rewrote `flydsl_masked_moe_gfx950_recv` to the CORRECT structure and re-validated on-server:
- **global→local map** `local = recv_topk_ids − base`, `base = argmax(expert_mask)` (first local index; per-rank
  block `[ep_rank*E_local, +E_local)`). Server log confirms base∈{0,48,96,144,...} per rank ✓.
- **full per-(recv-row, local-slot) routing** into `[E_local, max_m, K]` + masked s1/s2 + **WEIGHTED**
  `flydsl_moe_gather_reduce` (topk weight applied in-MoE; mori `combine` `_combine_kwargs={}` → sums partials,
  no re-weight — verified in moriep). Non-local/overflow slots → weight 0.
- **aiter unit test** (`--stage recv`, global ids + base + weights, bf16 & fp8): **PASS logits_diff ≈ 4e-6**.
- **On-server:** output improved from pure garbage → **partially coherent** (" Paris. It is a thing for that a
  bolt of …" then degrades), but **gsm8k(limit=40) = 0.0** ⇒ still a real correctness bug, not noise.

**Narrowed remaining bug = recv-row VALIDITY / flat-buffer LAYOUT (the last unknown):**
- I assumed valid tokens occupy a contiguous prefix `[0, total_recv)` (`total_recv=Σ num_recv_tokens_per_expert`).
  This is almost certainly WRONG. The mori flat recv `[M=1024, hidden]` (=decode_cap 128 × world 8) likely
  places valid tokens **per-source-rank or per-expert region**, NOT a dense prefix; padding rows carry
  `recv_topk_ids=0` (a VALID global id → indistinguishable from real expert-0 by ids alone).
- The working `fused_moe(mori)` determines validity elsewhere: `num_local_tokens=num_recv_tokens_per_expert`
  is forwarded and `moe_sorting(recv_topk_ids)` builds the sorted layout; `num_local_tokens` is used as
  `num_rows` for the quant (`fused_moe.py:807`). Need to read **`moe_sorting`** + mori dispatch to learn the
  EXACT valid-row predicate + flat layout (per-rank stride? per-expert stride? a separate valid mask/`-1`
  sentinel?). Warmup batches showed identical consecutive `recv_topk_ids` (dummy replicated tokens), which
  muddied the earlier read.
- Also re-confirm: is each recv row one-per-token (moe_sorting picks its local experts among the 6) — which is
  what my routing assumes — vs token×expert expanded. moe_sorting consuming `[M,6]` implies one-per-token, but
  verify against mori.

**Next session:** (1) read `aiter` `moe_sorting` + mori `dispatch_combine.py` to nail the flat recv valid-row
layout/predicate; (2) mark valid rows exactly like the sorted path (not `[0,total_recv)`); (3) re-run gsm8k
(target ≥0.93) then the graph-safe GPU route/scatter for perf. Kernels + routing math are correct
(unit-validated); only the recv-buffer indexing contract with mori remains. `SGLANG_MORI_MASKED_MOE` default-OFF.

### 6c-iii. Third attempt (2026-07-14, PAUSED) — validity FIXED (expert_mask + weight>0); real-row diff = fp8-vs-fp4 quant noise; gsm8k still TBD
Resolved the validity semantics (was the §6c-ii `[0,total_recv)` bug):
- **Definitive mechanism found:** `fused_moe.py::get_topk_valid_mask` = `expert_mask[topk_ids]` — a slot is
  valid iff its GLOBAL expert is local. NO `total_recv` prefix. `expert_mask` is 1 over the contiguous local
  block `[ep_rank*E_local, +E_local)`, so `base = argmax(expert_mask)`, `local = global - base`.
- **`flydsl_masked_moe_gfx950_recv` rewritten** (signature dropped `total_recv`): valid =
  `(local in [0,E)) AND (recv_topk_weights > 0)`. The **weight>0** term excludes the mori recv buffer's
  zero-inited PADDING rows (their `topk_ids=0` would otherwise route to local expert 0, consume `max_m` slots
  and drop real tokens). Real topk weights are strictly >0 (softmax). Unit test + a padding-stress case
  (`--stage recv`): **PASS logits_diff ≈ 4e-6** (bf16 & fp8). Sglang branch updated to derive `base` from
  `expert_mask` (no `.item()` on counts; removed the KNOWN-WRONG cumsum).
- **On-server DIFF diagnostic** (masked vs default `fused_moe`, same inputs, `SGLANG_MORI_MASKED_DUMP=1`
  returns the default output on the 1st call then masked): the giant `max~1e33/mean=inf` diffs are **ALL on
  PADDING rows** (`topk_ids=[0,0,0,0,0,0]`, `w=0`) where masked=0 (correct) and the DEFAULT emits garbage
  `~3e32` (uninit expert-0 on a zero row) — **both ignored by mori combine, harmless**. On **REAL rows** (e.g.
  row0 `topk_ids=[293,35,354,162,18,357]`, base=0 → only locals {35,18} kept) masked vs default differ only
  ~0.01 abs (sign flips on ~0 values): this is **fp8-vs-fp4 ACTIVATION quant noise** — masked runs a8w4
  (fp8 act, my per-1x32 quant), the DSV4 default runs **a4w4** (fp4 act, `q_dtype_a=fp4x2` since
  `AITER_BF16_FP8_MOE_BOUND=0` + Silu/SEPARATED). fp8 is higher precision, so masked should NOT degrade the
  model vs default.
- **Status: unresolved / paused.** Earlier greedy sniff still looked gibberishy, and the pre-fix gsm8k was
  0.0; the post-fix gsm8k run was **interrupted before completing** (no number yet). The masked math is
  unit-correct and the real-row on-server diff is only quant noise, so it *should* now pass — **must confirm
  with a full/limit gsm8k next session.**

### 6c-iv. ✅ ROOT CAUSE of gsm8k=0 ISOLATED (2026-07-15) — it's NOT masked; it's the eager decode small-cap op
Ran a controlled A/B matrix (gsm8k limit=40, DSV4 mori-EP). **Masked is confirmed correct at the compute
level; the e2e gsm8k=0 was a pre-existing environment bug, not the masked kernels.**

| config (all mori-EP, conc64) | decode dispatch | decode cap | cuda-graph | gsm8k | sniff |
|---|---|---|---|---:|---|
| STOCK baseline | mxfp4 | none(16384) | **ON** | **0.85** | coherent |
| plain eager | mxfp4 | none(16384) | OFF | **0.85** | coherent |
| eager + small-cap | mxfp4 | **128** | OFF | **0.0** | garbage |
| eager + small-cap + bf16 | bf16 | **128** | OFF | 0.0 | garbage |
| eager + small-cap + **masked ON** | bf16 | 128 | OFF | 0.0 (0.025) | garbage |

- **The broken factor = `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128` (the §5k/§5y decode small-cap dual-op)
  running in EAGER (`--disable-cuda-graph`).** It gives gsm8k=0 **even with masked OFF** (mxfp4, default) →
  pre-existing bug, only ever validated under cuda-graph (§5y). Eager itself is fine (0.85 without the cap).
- **Masked-kernel correctness (independently established):** (a) aiter unit tests 4e-6; (b) on-server
  **real-row (padding-excluded) diff vs the eager default = real_max 0.02, no row >0.05, all finite** — i.e.
  masked faithfully reproduces the default MoE (the 0.02 is just fp8-a8w4 vs fp4-a4w4 activation noise). So
  masked matched the (broken-by-cap) default exactly; both broken by the SAME cap bug, not by masked.
- **max_m overflow ruled out** (no `[masked-recv OVERFLOW]` even at max_m=512; counts ≪ cap at decode).
- **Second blocker for eager masked validation:** running masked WITHOUT the small-cap op (cap=0) →
  recv M=131072 (the normal op) → the Python/torch route-map+scatter over 131072 rows/layer is far too slow
  (prefill crawled to 3-12 tok/s, requests time out: "state was deleted in TokenizerManager"). The GEMM is
  fine (only real tokens via max_m); it's the eager torch scatter over the padded buffer that's slow — the
  exact thing the GPU-side route/scatter (graph-safe work) would fix.

**⇒ e2e gsm8k validation of masked is blocked by TWO pre-existing/env issues, both orthogonal to the masked
kernels:** (1) the decode small-cap op is broken in eager; (2) the big-buffer masked path needs a GPU-side
route/scatter (torch scatter over M=131072 is too slow in eager). Neither is a masked-correctness problem.

### 6c-v. Corrections to 6c-iv + a4w4 switch (2026-07-15) — masked still e2e-wrong; cap/eager were NOT the bug
Ran a cleaner A/B and **corrected two wrong conclusions from §6c-iv**:
- ❌ §6c-iv said "eager decode small-cap op is broken". **WRONG.** Clean control **stock eager + cap128 +
  mxfp4 (masked off) = gsm8k 0.90** (`run_sgl_dsv4_eager_plain.sh` + `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128`).
  The small-cap op works in eager. (You (user) were right that non-masked EP was ≥0.93.)
- The earlier "eager+cap128 = 0" runs were confounded by (a) `SGLANG_MORI_DECODE_DISPATCH_DTYPE=bf16` and/or
  (b) **`SGLANG_MORI_MASKED_DUMP=1` computing `fused_moe` TWICE per call → ~2× decode latency → request
  timeouts → gsm8k≈0** (NOT a correctness signal). Any gsm8k under DUMP is invalid.
- **Switched the masked path to a4w4** (the natural fit): mxfp4 decode dispatch (the WORKING config) gives
  **fp4x2 recv + per-1x32 e8m0 scale**, consumed DIRECTLY by the masked kernel with `a_dtype="fp4"` (no
  upscale/round-trip). `flydsl_masked_moe_gfx950_recv` now handles fp4/fp8/bf16 recv; fp4x2 scatter uses the
  uint8 byte view (fp4x2 can't be zero-filled/indexed). Unit test **PASS 3.9e-6** (fp4), 4.0e-6 (fp8/bf16).
  sglang branch accepts fp4x2 recv, `model_dim=K*2`, pre_permute upscale-skip covers fp4 (decode-only).
- **BUT masked is STILL e2e-wrong** (clean, NO DUMP): gsm8k **0.025**, sniff diverges after "Paris."
  (`'Paris. Paris is, and and in print English words monkey…'`). NOT weighting (`SGLANG_MORI_MASKED_NOWEIGHT=1`
  also garbage), NOT DUMP, NOT NaN (masked output finite). The 0.015 masked-vs-default "match" earlier was on
  **warmup dummy batches** (identical repeated `topk_ids`), NOT real varied decode tokens — so it never
  actually validated real-token correctness on-server.

**⇒ Current state:** masked kernels + routing pass all aiter unit tests (4e-6) AND run finite/crash-free at
DSV4 scale, but produce WRONG e2e decode output on real mori inputs. The unit tests don't reproduce the bug,
so it's a **real-input interpretation mismatch** between my synthetic recv tensors and mori's actual
recv (`recv_topk_ids`/`recv_topk_weights`/`hidden_states_scale`/`hidden_states` layout, or the
expert_mask→base, or how the default a4w4 actually consumes them). Baseline sanity: eager mxfp4 cap128
masked-OFF = 0.90 (harness + config are healthy).

### 6c-vi. Clean real-token diff = 0.02 (masked correct) BUT a reproducible SCRIPT-level anomaly blocks e2e (2026-07-15)
Did the §6c-v clean diagnostics. Two solid results + one unresolved anomaly:

**(A) Masked kernel/routing is CORRECT on real tokens.** One-shot masked-vs-default diff on genuinely-real
*varied* decode batches (distinct topk_ids ≥3, not warmup dummies): **real_max 0.012–0.039, real_mean ~0.01–0.02,
0 rows >0.05, all finite** — i.e. masked (a4w4, fp4 recv) reproduces the default a4w4 MoE within fp4 quant
noise. Confirmed the a4w4 switch works: masked activates with `recv_dtype=float4_e2m1fn_x2`, K=3584(=7168/2).

**(B) The e2e gsm8k=0 does NOT track the masked CODE — it tracks the masked SCRIPT.** Controlled A/B (all
eager, cap128, mxfp4/auto decode dispatch, gsm8k limit40, conc64):
| launch | masked | gsm8k | runs |
|---|---|---:|---|
| `run_sgl_dsv4_eager_plain.sh` + cap128 | OFF | **0.85 / 0.90** | 2× good |
| eager_plain + cap128 + `DECODE_DISPATCH_DTYPE=auto` | OFF | **0.875 / 0.925** | same server 2× good |
| `run_sgl_dsv4_masked.sh` (`MASKED_MOE=0`) | OFF | **0.0 / 0.0 / 0.0** | 3× deterministically bad |
| `run_sgl_dsv4_masked.sh` (`MASKED_MOE=1`, mxfp4) | ON | **0.0 / 0.025** | 2× bad |
- **Same server repeats are stable** (eager_plain 0.875→0.925; masked-script 0.0→0.0) → NOT runtime-flaky;
  it's **per-launch deterministic** and correlated with the *script*, not the run.
- **The two scripts' resolved envs differ ONLY by the gated `SGLANG_MORI_MASKED_MOE/MAX_M/DEBUG` (+ the
  masked script's `DECODE_DISPATCH_DTYPE`/`DECODE_MAX_DISPATCH_TOKENS`, which I override to match).** Verified:
  `get_bool_env_var("SGLANG_MORI_MASKED_MOE","false")=="0" → False` (masked genuinely off); those envs are
  read ONLY in `moe_runner/aiter.py` and gated behind `MASKED_MOE` → **provably no effect on the default path
  when off.** grep-confirmed no other reader in sglang.
- ⇒ **Paradox:** masked-script-masked-OFF ≡ eager_plain by config+code, yet 0.0 vs 0.9, reproducibly. So the
  e2e failure is an **environmental/launch anomaly of the masked script**, NOT masked correctness (which (A)
  shows is fine). Root of the anomaly UNRESOLVED (exhausted env/code analysis; not flaky; not the gated envs
  by inspection). Prime remaining suspects to check fresh: (i) an env in the masked block interacting at mori
  init in a way not visible in the gate (e.g. `DECODE_DISPATCH_DTYPE` being *present* changing op-build order
  even when =auto — though eager_plain+auto was GOOD, so unlikely); (ii) a stale working-tree/JIT-cache state
  that differs between launches; (iii) something about the masked script file itself (recreate it fresh from
  eager_plain + `MASKED_MOE=1` only, and bisect the env block one var at a time).

**Net:** masked GEMM + a4w4 routing validated numerically e2e (real-token diff 0.02 vs default); the on-server
gsm8k blocker is a reproducible masked-*script* anomaly orthogonal to masked correctness. `SGLANG_MORI_MASKED_MOE`
default-OFF; GPUs idle.

### 6c-vii. ✅ Bisect result — the "script anomaly" was a TEST BUG; masked-ON IS genuinely the cause (2026-07-15)
Ran the §6c-vi bisect. **Root of the "masked-script vs eager_plain" paradox: `run_sgl_dsv4_masked.sh`
hardcoded `export SGLANG_MORI_MASKED_MOE=1` (not `${:-}`), which OVERRODE my command-line
`SGLANG_MORI_MASKED_MOE=0`.** So every "masked-OFF-script" run was actually **masked-ON**. The A/B in §6c-vi
was invalid. (Fixed the script to `${SGLANG_MORI_MASKED_MOE:-1}` so future toggling works.)
- Proof: `eager_plain + cap128 + auto + ALL masked envs (MASKED_MOE=0)` = **gsm8k 0.9** (genuinely off,
  eager_plain has no hardcoded override) → the masked ENVS are harmless; it was masked being ON.
- ⇒ **Corrected conclusion: masked-ON genuinely breaks e2e (gsm8k 0.0 / garbage single-request output).**
  This supersedes §6c-v/§6c-vi's "not the masked code" framing.

**What's RULED OUT as the masked bug (this session):** max_m overflow (max_m=1024 == recv M, overflow
impossible → still 0.0); topk weighting (`NOWEIGHT` also garbage); DUMP 2×-timeout (clean no-DUMP also 0);
request timeout (single greedy request RETURNS garbage text, not empty/error → correctness, not latency);
per-launch nondeterminism (same server stable; masked-ON stable 0, eager_plain stable 0.9).

**The paradox that remains:** the one-shot REALDIFF on the FIRST varied decode batch showed masked ≈ default
at **0.02** (real rows), yet e2e is 0.0. The all-calls REALDIFF then showed later batches with real_max
**2–10 and ~3e38** on weight>0+finite rows — BUT these are likely mori PADDING rows carrying STALE nonzero
weights/ids (my `weight>0` filter can't exclude them; the DEFAULT also emits ~3e38 there). So the per-row
diff is **polluted by padding** and can't cleanly confirm/refute masked correctness on the rows that matter.

### 6c-viii. ✅ PRIMARY BUG FIXED (padding prefix) + secondary ~15% diff localized (2026-07-15)
The post-combine/on-off localization (via in-branch masked-vs-default REALDIFF, restricted to normal-magnitude
real rows) found the real bug and a residual:

**PRIMARY BUG (FIXED): masked processed PADDING rows the default excludes.** Direct evidence: on rows where
`default(ref)==0`, masked was nonzero; those rows are exactly the ones with `row_index >= sum(num_local_tokens)`.
So **valid recv rows = the PREFIX `[0, total_recv)`, `total_recv = sum(num_recv_tokens_per_expert)`** (dump:
`num_local_tokens=[6]` → rows ≥6 are padding → default zeros them). My `weight>0` validity was WRONG —
mori padding rows carry STALE nonzero weights, so weight-filtering processed garbage rows.
- **Fix:** `flydsl_masked_moe_gfx950_recv` valid = `(local in [0,E)) & (row < total_recv)` (added `total_recv`
  param); sglang branch passes `total_recv = int(num_local_tokens.sum())`. Unit test updated (padding =
  rows≥total_recv with STALE nonzero weights) → **PASS 3.9e-6 (fp4/fp8/bf16)**. On-server confirmed:
  `refzero&masked!=0 rows=[]` (padding over-compute gone). **KEEP this fix.**
- (This vindicates §6c-ii's `[0,total_recv)` prefix idea; §6c-iii wrongly replaced it with `weight>0`.)

**RESIDUAL (still gsm8k 0): a ~15% masked-vs-default diff on genuine real rows** (masked vs default differ
~0.2–0.4 on rows with |ref|~1–2, i.e. ~10–25% per-element, systematic not noise). Sniff still garbage. Since
masked matches a FP32 torch ref at 4e-6 in the unit test (K=512), this is a real-scale / precision-path
difference vs the shipping a4w4 kernel. **Top suspect:** masked runs **stage1 a4w4 (fp4 act) → per-1x32 FP8
quant of s1 → stage2 a8w4 (FP8 a2)**, whereas the default DSV4 path is **a4w4 throughout (stage2 consumes FP4
a2)**. Different intermediate precision per stage → systematic ~15% that compounds over 61 layers → garbage.
Other suspects: masked stage2 a2-scale layout vs default; the default using a different kernel family (opus vs
flydsl mixed_moe) with its own rounding; DSV4's large routed weights (sum≈2.5, some >1) amplifying small errs.

**RESUME HERE (next session):**
1. **Match the default's per-stage dtype:** make the masked recv primitive quantize s1→**FP4** (a4w4 stage2)
   instead of fp8, to mirror the shipping a4w4 path exactly (add a per-1x32 fp4 quant for a2; the a4w4 sorted
   test `test_flydsl_moe_a4w4.py` shows the layout). Re-run the REALDIFF — if the ~15% collapses to fp4-noise,
   that was it; then gsm8k.
2. If not: localize stage1-only vs stage2-only (dump masked s1 vs a torch a4w4 stage1 ref on the REAL grouped
   input; then stage2) to see which stage introduces the 15%.
3. Cross-check which kernel the DEFAULT a4w4 decode uses (opus_a8w4 vs flydsl) — if default is a8w4 (fp8 act)
   not a4w4, then masked should ALSO be a8w4 (fp8 recv), but mori fp8 dispatch is per-1x128 (incompatible) →
   would need a per-1x32 fp8 requant of the fp4 recv. Determine DSV4's actual decode a-dtype from a server log
   (`[fused_moe] using ...` line) to pick a4w4 vs a8w4 for masked.

**Infra note:** `run_sgl_dsv4_masked.sh` MASKED_MOE now overridable; masked is genuinely toggled via
`SGLANG_MORI_MASKED_MOE` on `run_sgl_dsv4_eager_plain.sh` (no hardcode). Known-good baseline (masked OFF, eager,
cap128, mxfp4) = gsm8k ~0.85–0.90. GPUs idle.

**(archived) §6c-vii RESUME plan (superseded by the 6c-viii fix + residual above):**
1. **Compare POST-COMBINE token output** (masked vs default), which is padding-free. Capture, for ONE real
   prompt at conc=1, the final combined `[num_tokens, hidden]` with masked ON vs OFF (toggle via
   `SGLANG_MORI_MASKED_MOE`, now that the script is fixed) — e.g. dump the MoE layer's *post-combine* residual
   contribution, or just diff generated logits/token-ids greedily. If they diverge on token 1-2, the masked
   MoE is wrong for real tokens (then it's a real-input kernel/routing bug the synthetic unit test misses —
   inspect recv `hidden_states_scale` layout vs `e8m0_shuffle` expectation, and the a4w4 kernel at K=7168).
2. **OR** identify mori's TRUE valid-row set (which recv rows combine actually reads — from the dispatch
   handle / origin map) and restrict the diff to those, removing the padding pollution, to get the honest
   masked-vs-default on real rows.
3. Sanity: masked aiter UNIT tests still pass 4e-6 on synthetic data, so the bug is real-input-specific
   (recv tensor interpretation) OR a large-shape (K=7168) kernel edge case not covered by the K=512 unit test.

Note: fixed `run_sgl_dsv4_masked.sh` MASKED_MOE to be overridable. `SGLANG_MORI_MASKED_MOE` default-OFF in
sglang. GPUs idle.

**(archived) §6c-vi RESUME plan (its premise "not the masked code" was wrong — see 6c-vii):**
0. **Bisect the masked-script anomaly (§6c-vi):** start from the KNOWN-GOOD `eager_plain + cap128` launch and
   add the masked env block ONE var at a time (`MASKED_MOE=1`, then `MASKED_MAX_M`, ...), gsm8k each, to find
   which single addition flips 0.9→0. That directly identifies the culprit (or proves it's masked-ON itself,
   in which case (A)'s 0.02 diff must be re-examined on MORE batches / deeper into generation). Recreate
   `run_sgl_dsv4_masked.sh` fresh (cp eager_plain + append only `SGLANG_MORI_MASKED_MOE=1`) to rule out a
   stale-file effect. Clear JIT cache + working tree between launches.
1. If masked-ON is genuinely the culprit despite the 0.02 diff: the diff snapshots may miss deep-context /
   long-sequence divergence — instrument accumulated worst-diff over a full gsm8k question (many steps),
   real-rows-only, to catch late divergence.

**(archived) §6c-v RESUME plan (superseded by 6c-vi bisect above):**
1. Instrument the branch to compute both masked `out` and in-branch `ref` (default a4w4) but ONLY on batches
   that are (a) NOT warmup (varied topk_ids, e.g. many distinct rows) and (b) restricted to genuinely-real
   rows (exclude padding: not just weight>0 — also verify the row's token is a real recv token). Report the
   per-real-row diff there. Do this at low concurrency (conc=1, a couple prompts) to avoid the 2×-fused_moe
   timeout (or compute ref only every Nth call). If real-token diff is large → the masked MoE/routing is
   wrong on real inputs (dig into recv tensor interpretation). If small → the bug is outside the MoE values
   (combine ordering / which rows / dtype).
2. Prime suspects to check against the WORKING default path directly (read `_moe_sorting_impl`/opus for the
   mori a4w4 path): (a) does the default apply topk weight in stage2 or not for mori? match it exactly;
   (b) is `recv_topk_weights` normalized the same way I assume; (c) is the recv `hidden_states_scale` the
   per-1x32 e8m0 my e8m0_shuffle expects, or a different (already-shuffled?) layout; (d) is `expert_mask`
   base contiguous as assumed.
3. Cross-check: temporarily route the masked branch to build the grouped input then call the SHIPPING
   `fused_moe` per-expert (bypassing my kernels) to see if the wrapper/routing is right vs the kernels.

**(archived) §6c-iii RESUME plan superseded by §6c-iv, itself corrected by §6c-v above:**
- **Path A (fastest to a number): fix the eager decode small-cap op bug.** It's pre-existing (repro: stock
  mori-EP + `SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128` + `--disable-cuda-graph` → gsm8k 0, masked OFF).
  Investigate the §5k dual-op decode routing under eager (moriep `mori_op` property / `_use_decode_op` /
  `get_is_extend_in_batch()` behavior when not capturing; likely the 2nd small IntraNode op mis-dispatches or
  the per-op decode-dtype/buffer is mis-wired off cuda-graph). Once cap=128 works eager → M=1024 → masked
  torch scatter is fast → run gsm8k (expect ~0.85 since masked==default at 0.02).
- **Path B (the real perf fix anyway): GPU-side graph-safe route/scatter** so masked runs under cuda-graph
  (where decode is healthy, 0.85) — removes both blockers at once and gives real perf. Bigger (port the
  gfx1250 `moe_route_maps`/`moe_scatter_copy` kernels or write masked route/scatter reading GPU recv counts).
- Either way, masked-kernel correctness is DONE (unit 4e-6 + on-server real-row 0.02). Only the
  env/perf plumbing to *exercise* it e2e remains. `SGLANG_MORI_MASKED_MOE` default-OFF; GPUs left idle.
- Repro scripts: `useful-scripts/benchmarking/dsv4/run_sgl_dsv4_masked.sh` (masked, envs overridable),
  `run_sgl_dsv4_eager_plain.sh` (stock + `--disable-cuda-graph`, the 0.85 eager baseline).

**(archived) earlier RESUME plan (superseded by 6c-iv):**
1. Relaunch (recipe below) and **run gsm8k** (`--limit 200` then full). If ≥0.93 → correctness DONE; if still
   ~0 → the remaining suspect is a **few catastrophically-wrong REAL rows** (not padding) — instrument the
   DIFF to print real-row (weight>0) diffs only, excluding padding, and find which experts/rows blow up
   (candidates: `max_m=128` overflow on a hot local expert at MTP/spec decode → dropped tokens; or a
   token routed to >1 local expert being mis-summed).
2. If correctness OK: measure decode ITL/throughput vs §5y (30,081 / ITL 40.75). Note eager-only for now.
3. Then graph-safe GPU route/scatter (the torch `.item()`/dynamic scatter is eager-only) for cuda-graph perf.

**Launch recipe used (eager, default-OFF unless envs set):**
```bash
# script: useful-scripts/benchmarking/dsv4/run_sgl_dsv4_masked.sh  (= mori-ep + these envs + --disable-cuda-graph)
#   SGLANG_MORI_MASKED_MOE=1  SGLANG_MORI_MASKED_MAX_M=128
#   SGLANG_MORI_DECODE_DISPATCH_DTYPE=bf16  SGLANG_MORI_DECODE_MAX_DISPATCH_TOKENS=128
#   SGLANG_MORI_MASKED_DEBUG=1   (SGLANG_MORI_MASKED_DUMP=1 for the masked-vs-default DIFF)
ps aux | grep "[l]aunch_server --model-path" | awk '{print $2}' | xargs -r kill -9   # NB: never `pkill -f sglang.launch_server` (matches its own cmdline, kills the shell)
find /sgl-workspace/aiter/aiter/jit/build \( -name lock -o -name 'lock_*' \) -delete
cd /dockerx/home/wunhuang/tmp/useful-scripts/benchmarking/dsv4/ && nohup bash run_sgl_dsv4_masked.sh > /tmp/masked_srv.log 2>&1 & disown   # ready ~90-120s ("Uvicorn running")
```
**Code touchpoints (all working-tree, default-OFF):** aiter `ops/flydsl/grouped_moe_gfx950.py`
(`flydsl_masked_moe_gfx950_recv` = the bridge primitive; kernels in `kernels/mixed_moe_gemm_2stage.py`);
sglang `srt/layers/moe/moe_runner/aiter.py` (`_maybe_run_mori_masked` + call in `AiterRunnerCore.run`,
+ pre_permute upscale-skip gate); aiter unit test `op_tests/test_flydsl_masked_moe_stage1_gfx950.py`
(`--stage recv|full|stage1|stage2|e2e|hook|all`). GPUs left idle; server stopped.

## 6. Quick reproduce

```bash
# DP baseline
MODE=dp PORT=8000 bash useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
# mori-EP
MODE=mori-ep PORT=8000 bash useful-scripts/benchmarking/dsv4/run_sgl_dsv4_unified.sh
# bench (both): sglang-oai client, 8192:1024, conc256, NP_MULT=8 WARM_MULT=2
# decode trace: --disable-cuda-graph + GPU-only /start_profile (see §2)
```

## 7. Pre-flight gotchas hit on 2026-07-03 (base changed after aiter update)
- **flydsl mismatch:** updated aiter pins `flydsl==0.2.2` but env had **0.2.0** →
  `AttributeError: 'ArithValue' object has no attribute 'ir_value'` in
  `flydsl moe_gemm1` (`layout_utils.py:150`) → all ranks die at warmup. Fix:
  `pip install flydsl==0.2.2`. (Installed `amd-aiter` wheel metadata still tags the
  old commit / pins 0.2.0 — cosmetic; the source tree at `/sgl-workspace/aiter`
  needs 0.2.2.)
- **stale aiter JIT baton locks:** killing a server mid-JIT-compile leaves
  `aiter/jit/build/lock_module_*` and nested `module_*/build/lock` → the next server
  deadlocks "waiting for baton release" (no active compiler, no file progress).
  Fix: `find /sgl-workspace/aiter/aiter/jit/build \( -name lock -o -name 'lock_*' \) -delete`.
- **profiler truncation:** eager traces with default `with_stack=True` produce 2GB+
  files dominated by `python_function` events; kill-before-flush truncates them AND
  the GPU `kernel` events are in the lost tail. Always capture GPU-only + poll trace
  size until stable before killing.
