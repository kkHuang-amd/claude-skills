# FlyDSL Kernel-Debug Toolkit (learned from ROCm/FlyDSL official skills)

**Purpose.** We have hit a *kernel-internal* blocker (the fixed-slot dispatch cross-PE
rendezvous deadlock in dual-mode, see `DUAL_MODE_DISPATCH_DIAGNOSIS.md`). To move past
"the kernel owner has to look at this", we need our own device-side debugging capability.
This doc distills FlyDSL's **official** kernel-debug methodology (their `.claude/skills` +
`docs`) and then **applies it concretely** to our dispatch deadlock with a line-anchored
`fx.printf` instrumentation plan.

Everything referenced here is already **cloned locally** on branch `mega_moe_v1`, so read the
source, not GitHub HTML:

| What | Local path |
| --- | --- |
| Official debug skills | `/sgl-workspace/FlyDSL/.claude/skills/{debug-flydsl-kernel,oob-detection,capture-kernel-trace,kernel-trace-analysis}/SKILL.md` |
| Authoring / tile skills | `/sgl-workspace/FlyDSL/.claude/skills/{flydsl-kernel-authoring,flydsl-tile-programming,lds-optimization,gemm-optimization,prefetch-data-load}/SKILL.md` |
| Trace analyzers (scripts) | `/sgl-workspace/FlyDSL/.claude/skills/kernel-trace-analysis/scripts/{hotspot_analyzer.py,pmc_l2_analyzer.py}` |
| Design reference (our kernel) | `/sgl-workspace/FlyDSL/docs/moe_stage1_mega.md` (§1 sync protocol, §3.3, **§8 co-residency/CUDAGraph**) |
| Authoring / testing guides | `/sgl-workspace/FlyDSL/docs/{kernel_authoring_guide.md,testing_benchmarking_guide.md,kernel_tuning_guide.md,layout_system_guide.md}` |
| Runnable examples | `/sgl-workspace/FlyDSL/examples/0{1..5}-*.py` + `examples/notebooks/*.ipynb` |

> Our own prior playbook (`../FLYDSL_KERNEL_AUTHORING.md`) is the *safe-editing / escalation*
> layer; this doc is the *device-observability / root-causing* layer that sits under it.

---

## 1. The toolkit — symptom → tool

FlyDSL's `debug-flydsl-kernel` skill classifies by symptom. Mapping to the tools:

| Symptom | First move | Primary tool |
| --- | --- | --- |
| "My fix didn't work" | **Clear caches** (see §2, do this ALWAYS) | `rm -rf ~/.flydsl /tmp/flydsl*` + `*.cache_clear()` |
| Wrong values / NaN / zeros | host-side buffer inspection, all-1s isolation | `torch.cuda.synchronize()` + prints; sentinel fills |
| Suspected OOB / silent corruption | **static interval analysis first**, then runtime guard | `oob-detection` skill + `fx.printf` guard |
| **GPU hang / deadlock** (← us) | localize the spin, print epoch/flags | **device-side `fx.printf`** (§3) |
| Perf regression / stalls | ATT trace, then hotspot analysis | `rocprofv3` + `hotspot_analyzer.py` (§5) |
| Cache/HBM behavior | PMC counters (separate pass) | `rocprofv3 -i pmc_*.yaml` + `pmc_l2_analyzer.py` |
| "what IR did it emit" | dump MLIR | `FLYDSL_DUMP_DIR` (default `~/.flydsl/debug`) |

The single most useful primitive for our class of bug is **device-side `fx.printf`**.

---

## 2. Rule 0 — clear the cache (every time you change kernel code)

FlyDSL aggressively caches compiled HSACO. Stale cache is the #1 cause of "I edited the
kernel and nothing changed":

```bash
rm -rf ~/.flydsl /tmp/flydsl*
```

If a Python `@functools.lru_cache`'d compile wrapper is in play (e.g.
`compile_fused_moe_gemm1`), also call `<fn>.cache_clear()` in the harness. **Do this before
every re-run while instrumenting**, otherwise your `fx.printf` may simply not be in the
binary that runs.

---

## 3. Device-side `fx.printf` — the deadlock microscope

### 3.1 API (confirmed from `python/flydsl/expr/primitive.py:1308` + `tests/unit/test_rocir_print.py`)

```python
import flydsl as fx
fx.printf("gb1={} epoch={} rank={}", gb1_i64, epoch_i32, rank_i32)
```

- Placeholders are **`{}`** (not `%d`). Accepts `i32/i64/f32` SSA values **and** Python
  literals/strings/types (literals are folded into the string at trace time).
- Lowers `fly.print` → `gpu.printf`. Output goes to stderr of each rank's process.
- **Keep it narrow.** Guard by lane/block so only *one* thread prints, e.g. inside
  `if gwid == 0:` / `if lane == 0:` — otherwise 8 ranks × thousands of waves drown the signal.
- It is a real device call: it perturbs timing (can mask/again a race), and it is **not**
  safe inside a CUDAGraph capture (writes host-visible state). So instrument in an **eager**
  micro-harness run, not under graph replay (see §4).

### 3.2 FlyDSL gotchas that bite while instrumenting (from `debug-flydsl-kernel` §6)

- `range_constexpr(N)` for compile-time-unrolled loops (`i` is a Python int); `range(...)`
  becomes an MLIR loop (`i` is an `ArithValue`, can't index Python lists).
- **No Python `if` on runtime GPU values** unless you intend an `scf.IfOp`; and
  **never** `const_expr(lane == 0)` — `lane`/`warp_id`/`thread_id` are runtime SSA (the
  compiler knows the *range*, not the current lane).
- **Divergent barrier = deadlock.** `fx.barrier()` needs *all* threads in the workgroup;
  if some take a different runtime branch, it hangs. So put a `fx.printf` *before* a barrier,
  never only on one side of a divergent `if` that also contains the barrier.
- `buffer_load` offset units are in `dtype` elements (bytes/4 for i32-viewing fp8).

---

## 4. Applied: root-causing our fixed-slot dispatch deadlock

### 4.1 Crystallized root cause (what the design doc + code confirm)

`docs/moe_stage1_mega.md` §1.1/§3.3 + `kernels/mega_moe/dispatch.py` show the fixed-slot
block0 sync is a **symmetric all-to-all rendezvous across all 8 EP ranks**, plus a
CUDAGraph-safe monotonic epoch. There are exactly **four spin points** where a hang can
live (fixed-slot path, `emit_dispatch_prologue`):

| # | `dispatch.py` line | Spin | Waits for |
| --- | --- | --- | --- |
| S1 | `:207` | `int64_wait_until_equals(a_gb1, tg2)` | all **local** blocks arrived (needs **co-residency**, §8) |
| S2 | `:219` | `int32_wait_until_equals(rnum_remote, 0)` | peer's `recv_num[me]` slot **drained from previous launch** |
| S3 | `:223` | `int32_wait_until_greater_than(rn_src, 0)` | **each peer** posted its recv-count signal |
| S4 | `:289` | `int32_wait_until_greater_than(a_meta, e0)` | block0 published `meta_flag` (all other blocks wait) |

Coordination buffers (disp indices, fixed-slot): `a_gb1=dp(4)`, `a_meta=dp(18)`,
`a_trecv=dp(21)`, `a_dctr=dp(22)`, `a_rnum=dp(23)`, `p_rnum=dp(24)` (peer/symmetric).
The compact path has the analogous cross-PE `done2` handshake at `:399–419`
(`int32_wait_until_equals(a_cd, epoch_i32)` at `:418`).

**Why dual-mode deadlocks.** S2/S3 are a *rendezvous*: rank A writes peer B's `recv_num[A]`
and waits for B's signal in A's own `recv_num[B]`. This only completes if **all 8 ranks are
executing the *same* dispatch kernel at the same time**. Under DP-attention a per-rank
`forward_mode` divergence (some ranks pick decode/fixed-slot, others pick prefill/compact)
means the two groups touch **different** flag slots and can *never* rendezvous → S3 (or the
compact `:418`) spins forever. Separating the decode flags (`recv_num_dec` vs `recv_num`)
made this **worse**: even a momentary divergence becomes unrecoverable because the groups no
longer share the slot they'd need to meet on. This is exactly why the `forward_mode`
cross-rank fix was necessary — and why it's fragile: it only holds if *every* rank agrees on
the mode for *every* launch. (§8 co-residency is the S1 cousin: if `gx·gy > cu_num` some
block never launches and S1 hangs; verify `gx·gy ≤ cu_num` at construction.)

### 4.2 The instrumentation plan (line-anchored)

Goal: when the server/harness hangs, the **last line each rank printed** tells us (a) which
mode that rank chose, (b) which spin it died on, (c) the exact peer/value mismatch. Add these
`fx.printf`s to `kernels/mega_moe/dispatch.py` (fixed-slot path), all guarded to one thread:

```python
# S1 banner — right after epoch is known (~:209), block0, single lane:
if gwid == 0:
    if lane == 0:
        fx.printf("[FZ rank={} epoch={} gb1={} tg2={}] arrived; entering xPE\n",
                  fz_rank, epoch_i32, gb1now, tg2)                      # after S1 :207-209

    # S2/S3 are inside `for dpe in range(lane, fz_npes, 64)` / `for spe in ...`.
    # `lane` is runtime => this is an MLIR loop, so `dpe`/`spe` are SSA values (ArithValue),
    # NOT Python ints — pass them to printf directly (do NOT wrap in arith.constant).
    # For npes<=64 the loop is naturally narrow (only lanes 0..npes-1 run one iter each).
    for dpe in range(lane, fz_npes, 64):
        fx.printf("[FZ rank={} epoch={}] S2 wait rnum==0 dpe={}\n",
                  fz_rank, epoch_i32, dpe)                              # before :219
        ...
    for spe in range(lane, fz_npes, 64):
        fx.printf("[FZ rank={} epoch={}] S3 wait signal spe={}\n",
                  fz_rank, epoch_i32, spe)                             # before :223

# S4 — consumer wait (:288), single thread:
if tid == 0:
    fx.printf("[FZ rank={} epoch={}] S4 wait meta>{}\n", fz_rank, epoch_i32, e0_i32)  # before :289
```

> `fz_rank`, `epoch_i32` (`:209`), `gb1now` (`:205`), `tg2` (`:206`), `e0_i32` (`:197`) are all in scope
> at these points. The compact path's analogue is `:399–419` (banner after `:395`, `done2` wait at `:418`).

Read the output like this:

- **All ranks stop after S1, before S3** → confirms the rendezvous: cross-check the epoch and
  mode each rank printed. If epochs differ or a rank never printed the `[FZ ...]` banner at
  all (it's in the *compact* path instead) → **mode divergence** = the root cause.
- **One rank stuck at S3 `spe=k`, and rank k never printed anything** → rank k is in a
  different kernel/mode (or crashed) → same conclusion, now with the culprit rank id.
- **Stuck at S1** → co-residency / a block never launched (`gx·gy > cu_num`, or the graph
  replayed a grid that doesn't fit) → §8 issue, not the rendezvous.
- **Stuck at S4 only** → block0 died between S1 and publish (look at block0's last line).

### 4.3 Reproduce in isolation (DONE — this method proved the root cause, 2026-07-20)

The micro-harness (`tests/kernels/test_mega_moe.py`, 8-rank torchrun) is the controlled
environment — no scheduler, no CUDAGraph, deterministic token counts. We added a
`--hang-diverge N` mode that forces **rank0 → prefill(compact)** while **ranks 1..7 →
decode(fixed-slot)** *simultaneously* at iter N, then `torch.cuda.synchronize()`.

```bash
rm -rf ~/.flydsl /tmp/flydsl*        # Rule 0 (only needed after editing the kernel)
# baseline (lockstep, all ranks agree) -> PASS, banners show FZ r0..r7 ep=N then CP r0..r7 ep=M:
PYTHONPATH=/sgl-workspace/FlyDSL MORI_SHMEM_HEAP_SIZE=40G torchrun --standalone --nproc_per_node=8 \
  tests/kernels/test_mega_moe.py --network v4_pro --quant a8w4 --tokens 512 --mtpr 8192 \
  --hang-iters 8 --hang-decode-mtpr 512 --hang-decode-tokens 64 2>&1 | tee /tmp/mega_dbg.log
# forced divergence at iter 2 -> DEADLOCK (add --hang-diverge 2). Then, from another shell:
grep -aoE '\[MEGA-DBG (FZ|CP) r[0-9]+ ep=[0-9]+\][^\\]*' /tmp/mega_diverge.log | tail -40
for p in $(pgrep -f test_mega_moe.py); do py-spy dump --pid $p 2>/dev/null | grep -A6 MainThread; done
pkill -9 -f test_mega_moe.py    # release the wedged GPU (recovers clean, no reset needed)
```

**Observed at the hang (the smoking gun):** ranks 1–7 froze at `[MEGA-DBG FZ r{1..7} ep=4]
post-S2 enter-S3` (no `xPE-done`); rank0 froze at `[MEGA-DBG CP r0 ep=5] enter-xPE1` (no
`xPE1-done`); all 8 `py-spy` at `synchronize → _run_full_e2e`. Because GPU `printf` may not
flush from a hung kernel, the **host** prints are the primary proof: every rank prints
`RETURNED (async launch ok)` but **none** prints `SURVIVED`.

**Conclusion:** a single-rank mode divergence is sufficient to deadlock — it is not a subtle
memory race. Since the lockstep baseline never hangs, a server hang means the *scheduler* is
still introducing divergence → fix in the integration layer (`mega_moe_flydsl.py` mode
selection must be provably cross-rank identical), OR make the two dispatch schemes
rendezvous-compatible in the kernel.

> Cleanup: the `dispatch.py` `fx.printf` was **reverted** after the repro (it would spam every
> compact-only prefill on the shipping path). The `--hang-diverge` harness flag is kept
> (test-only, default `-1`).

### 4.4 Escalation-grade evidence to hand the kernel owner (captured)

`DUAL_MODE_DISPATCH_DIAGNOSIS.md` §5 now carries: the exact spin lines (`:223` S3 fixed-slot /
`:418` compact cross-PE#1), the per-rank mode+epoch at hang, the `py-spy` stacks, and the
violated invariant — "S2/S3 (and compact `:418`) require all `npes` ranks in the same dispatch
scheme with equal epoch". That is a precise, reproducible bug report, not "it hangs somewhere".

---

## 5. Perf debugging (for later — when it runs but is slow)

Not our current blocker, but part of the capability. From `capture-kernel-trace` +
`kernel-trace-analysis`:

```bash
# 1) discover kernel names
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=./ \
  rocprofv3 --stats --kernel-trace -f csv -o /tmp/discover -- python <script>
# 2) ATT trace (source-mapped) with an input.yaml (kernel_include_regex, att_target_cu=1,
#    kernel_iteration_range "[1,[2-4]]" to skip warmup, att_buffer_size 0x6000000)
FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1 PYTHONPATH=./ rocprofv3 -i input.yaml -- python <script>
# 3) analyze the downloaded ui_output_agent_* dir
python .claude/skills/kernel-trace-analysis/scripts/hotspot_analyzer.py <trace_dir>
```

- `FLYDSL_DEBUG_ENABLE_DEBUG_INFO=1` gives source→ISA mapping (DWARF in HSACO).
- **PMC (cache/HBM) is a *separate* pass** and must be **≤ ~4 TCC counters per job**;
  packing many counters forces multi-pass which has triggered **GPU hangs**. Split jobs.
- Interpretation cheats: L2 hit `TCC_HIT/(TCC_HIT+TCC_MISS)`; 32B fraction
  `TCC_EA0_RDREQ_32B/TCC_EA0_RDREQ` (~0% = full 64B lines).

---

## 6. Capability checklist (our "kernel-dev readiness")

- [x] Know where the official skills/docs/examples live (local, `mega_moe_v1`).
- [x] Know the design of the kernel we're modifying (`moe_stage1_mega.md`, esp. §1/§3.3/§8).
- [x] Can add device-side `fx.printf` (correct API, guarded, cache-cleared, eager not graphed).
- [x] Can localize a deadlock to a specific spin line + read per-rank output.
- [x] Understand the FlyDSL gotchas (range_constexpr, divergent barrier, const_expr, buffer_load units).
- [ ] Can capture + read an ATT trace end-to-end on this box (do once on a *working* kernel to build muscle).
- [ ] Can run the static-OOB interval analysis on one real access in `gemm1.py`/`epilogue.py`.

**Next concrete step:** run §4.2/§4.3 to convert the dual-mode deadlock into per-rank spin
evidence, then either (a) fix the integration-layer mode selection if divergence is
scheduler-introduced, or (b) hand the kernel owner the precise spin + invariant.
