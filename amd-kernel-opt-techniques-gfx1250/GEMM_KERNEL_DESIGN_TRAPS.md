# gfx1250 GEMM / MoE kernel design: measured limits, traps, and design misses

Source: proxy-MoE kernel redesign on h13-17, 2026-10-28 (`proxy-moe/MOE_KERNEL_DESIGN_h13-17_1028.md`, microbenchmarks in
`proxy-moe/scripts/*.hip`, ATT in `/shared_nfs/kk/results/proxy_12l/p1_att_h13-17_1028/`). One partition = 96 CUs, wave32.
Numbers are in shader cycles; clocks under load are power-capped (~1.0-1.4 GHz), so never reason in "idle-clock" microseconds.

## 1. Measured hardware numbers (use these, not spec-derived guesses)

| quantity | value | how measured |
|---|---|---|
| `v_wmma_scale_f32_32x16x128_f4` | **issue 8 cyc per SIMD, latency ~27 cyc** -> >= 4 independent WMMAs in flight per SIMD | `wmma_fp4_rate.hip`, register operands |
| WMMA with per-lane VGPR scales | same 8.1-8.3 cyc (no penalty vs SGPR scales) | same tool, VS=1/2 |
| fp4 x fp4 peak | 4 x 131072 / 8 = 65.5k FLOP/clk/CU = 6.3 MFLOP/clk per partition (~7 PFLOP/s at 1.15 GHz) | derived from the above |
| VGPR file | **~1024 per lane per SIMD**: <= ~512 VGPR/wave keeps 2 waves/SIMD; `__attribute__((amdgpu_waves_per_eu(1,1)))` lets one wave use ~1000 without spills (without it the compiler spilled at 667) | 354-VGPR kernel runs 2 waves/SIMD concurrently |
| LDS read | 153 B/clk/CU (2 WG x 4 waves), 217 (2 x 8 waves) | gluon `lds_bw_gfx1250.py` |
| global_load_b128 -> VGPR, per-CU port | ~150 B/clk/CU (12-48 CUs active) | `l2_vgpr_bw.hip scale` |
| **L2 aggregate** | **~8.5 KB/clk per partition = ~88 B/clk/CU when all 96 CUs pull** | same, 96 CUs |
| load-to-use latency | ~320-345 cyc (L2 / last-level cache); HBM latency not isolated | dependent pointer chase |
| loads in flight needed | >= 8 waves x 8 b128 loads per CU to approach the rate | `l2_vgpr_bw.hip` U/waves sweep |
| identical addresses from different CUs | NOT merged by L2 (no gain vs private) | `cluster_mcast_bw.hip` "shared" |
| `cluster_load_b128` multicast | cluster 2: 1.75x, cluster 4: ~2x delivered per CU; data correct | `cluster_mcast_bw.hip` |
| multicast under drift | small drift (< ~500 cyc per 8 iters) absorbed; large drift without a barrier -> **half of private**; a cluster barrier every 8 iters costs 10-15% in lockstep | same, `s_sleep` jitter |
| HBM | ~8 TB/s -> machine balance ~900 fp4 flop per HBM byte at 1.15 GHz | spec + measured peak |

Builtins verified on gfx1250 (asm checked): `__builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a v16i32, b v8i32, short modC,
c v16f32, 0, 0, scaleA, 0, 0, scaleB, bool, bool)`, `__builtin_amdgcn_cluster_load_b32/_b64/_b128(GAS ptr, 0, mask)`,
`__builtin_amdgcn_global_load_async_to_lds_b64/_b128(GAS src, LAS dst, 0, 0)` + `__builtin_amdgcn_s_wait_asynccnt(n)`,
`clock64()` = `s_get_shader_cycles_u64` (shader cycles), `wall_clock64()` = realtime message. Cluster launch: `hipLaunchKernelEx`
+ `hipLaunchAttributeClusterDimension` with `{1,N,1}`.

## 2. Traps hit (each cost a full iteration)

1. **`__syncthreads()` inside a pipelined loop drains every load.** It is a fence + barrier; the compiler emits
   `s_wait_loadcnt 0x0` before it, so register prefetch of the next k-step is waited out every step (prototype: 2642 -> 1869
   cyc/step after the fix). Use `__builtin_amdgcn_s_barrier()`; for async global->LDS use `s_wait_asynccnt` per wave + barrier.
2. **Spills drain prefetch.** Scratch reloads use the same `loadcnt` counter, so each reload forces `s_wait_loadcnt 0x0` and
   waits for all in-flight global loads. A 132 B spill (24 scratch ops) made the token-double-buffer version slower than the
   version without it; ATT: `s_wait_loadcnt` 49% of the hot loop. Always check `.private_segment_fixed_size: 0` for the hot kernel.
3. **Coalesce per instruction, not per lane.** A fragment stored as 64 contiguous bytes per lane (4 x b128 per lane) makes each
   b128 instruction touch 2 KB sparsely; storing piece u at `u*512 + lane*16` (each instruction = 512 contiguous bytes)
   was 1602 -> 1419 cyc/step. You own the layout in a prototype; verify with a reference that uses the same load function.
4. **64-bit `%` in address math** compiles to a long software sequence (hundreds of cycles); a microbenchmark measured the divide,
   not the memory (first L2 numbers 3-4x too low). Use power-of-2 masks / 32-bit math; grep the asm for division sequences.
5. **Read-all-then-compute serializes LDS and WMMA** when the waves of a WG are barrier-locked: 4 waves read 36 KB together
   (~200 cyc at 153-217 B/clk) and then all do WMMAs (~260 cyc). Double-buffer fragments in VGPRs (read k+1 during k).
6. **1 wave per SIMD + barriers every step = no wave-level latency hiding.** All latency must be covered by prefetch depth,
   which the VGPR budget caps (512 acc + 4 x 68 weight buffers + 2 x 72 token buffers ~ 1000).
7. **Short-kernel clock estimates lie.** Realtime-derived MHz on 0.03-1 ms kernels was inconsistent; report cycles from
   `clock64()` per wave and derive clock only from long runs (per-CU busy cycles / elapsed ms).
8. **Microbenchmark footprint.** A "512 MB" pointer-chase ring only touched ~17 MB in 4096 hops: it measured a cache, not HBM.
9. **Per-CU vs aggregate.** Bandwidth measured with few CUs (~150 B/clk/CU) overstates what each CU gets at full occupancy (~88).
10. **aiter grouped MoE cluster auto-config:** CSV `cluster_m=-1` + `cluster_n=4` becomes cluster_m=4 for stage 1 = a 16-WG
    cluster with a cluster barrier every `num_buffers` k-steps, even when weight multicast is inactive (one M-tile per expert).
    Observed bimodal per-process timing (fast mode = same cycles, higher clock; slow mode = +30% cycles). Cause still open.

11. **`__builtin_amdgcn_s_barrier()` is not a memory barrier for the compiler.** LLVM moved the WMMAs that consume an LDS stage
    (and the `s_wait_dscnt` they imply) past the next step's barrier and async global->LDS writes -> other waves overwrote the stage
    while reads were in flight (0.02-0.09% wrong outputs, varying per run; it hid in one config and showed in another). Before a
    "stage free" barrier: `asm volatile("s_wait_dscnt 0x0" ::: "memory")`, and wrap `s_barrier` in `asm volatile("" ::: "memory")`.
    There is no `__builtin_amdgcn_s_wait_dscnt` in this toolchain. Same race class as aiter's `s_wait_dscnt(0)` fix.
12. **L2 prefetch (`__builtin_amdgcn_global_prefetch`) of streamed weights made things slower** (P1: +11-15% at 1 wave/SIMD,
    +70% at 2 waves/SIMD), so do not assume prefetch hides HBM latency; measure.
13. **HBM streaming, measured (M2b):** async global->LDS reaches ~8.8 TB/s with only 4 waves and 96 KB in flight per CU (loaded
    HBM latency ~1000-1200 cyc); plain register loads reach <= 5.2 TB/s at 4 waves/CU and 7.1 at 16 for the same pattern.
    Per-wave in-flight loads are capped at 63 (6-bit loadcnt / asynccnt). Note a slow GEMM prototype was NOT HBM-limited:
    weights-only streaming (~1.1 Mcyc) and compute-only (~1.2 Mcyc) each fit, the sum did not overlap (3.6 Mcyc). Measure
    each component alone before blaming bandwidth.

14. **Branches inside a pipelined loop kill register prefetch.** Per-step guards (`if (k + PD < KS) load(...)`, real/dummy
    issue, `if (k < KS)`) split the body into many basic blocks; the waitcnt pass merges pending-load state at each join and
    waits conservatively (`s_wait_loadcnt 0x3..0x0` where `0x28` was possible) -> prefetch drained every step. Clamp indices
    (`min(k + PD, KS - 1)`), make the unroll divide the trip count, and check the hot loop is 1-2 basic blocks (`grep -c '^.LBB'`)
    with the expected `s_wait_loadcnt` values. Fixing it cut the weight-wait share from ~45% to 14% (ATT).
15. **ATT `code.json` disassembly ignores `s_set_vgpr_msb`**: VGPR numbers > 255 appear modulo 256 (fake overlaps between
    accumulators and load targets). Read register allocation from the compiler `.s` (it prints `/*v[626:641]*/`).

16. **Two waves of one WG loading the same global data do not share it for free.** In a 2x2 wave grid the two M-waves each
    loaded the same weight fragments; L2 traffic doubled and the kernel was L2-bound (656 B/WMMA -> 2.6 Mcyc predicted, 2.9 measured).
    Count each wave's global loads as L2 traffic unless a cache hit is measured.
17. **Multicast that wins in a pure streaming microbenchmark can lose inside a GEMM** (P1: 2.38 -> 2.55 Mcyc, 2x2: 2.93 -> 5.08);
    cluster members drift with compute. A/B it in the real kernel.
18. **HIP TDM works** (see `proxy-moe/MOE_KERNEL_DESIGN_h13-17_1028.md` "P1 v8-v10" for the verified descriptor layout; probe
    `proxy-moe/scripts/tdmprobe.hip`). Row pitch goes in `tensor_dim0_stride`. Tokens via TDM beat per-lane async loads by 12-18%.
19. **Loop cycles are not kernel time.** A prototype with 2.3 Mcyc of loop ran 2.5-3.0 ms (~0.8-0.9 GHz implied): prologue,
    epilogue stores and the last WG round matter; time the whole kernel against the baseline in the same session.

20. **Time kernels at steady state.** 1 warmup + 3 timed launches (~10 ms) gave run-to-run clocks of 0.88-1.34 GHz and ms that
    did not follow cycle counts; 10 warmup + 50 timed launches (the op test uses 101 iterations) is stable to ~+-3%. Compare
    candidates only under the same timing method, alternating with the baseline in one session.
21. **Split barriers hide barrier time.** With 4 lockstepped waves at 1 wave/SIMD, `__builtin_amdgcn_s_barrier_signal(-1)` right
    after the data-ready waits, half of the step's WMMAs, then `__builtin_amdgcn_s_barrier_wait(-1)` cut 13-19% of loop cycles
    (barrier was 21% in ATT). Retire this wave's LDS reads (`s_wait_dscnt 0`) BEFORE the signal so stage reuse stays race-free.

22. **FlyDSL JIT cache ignores env-read module globals.** A kernel variant selected by `os.environ` at import (e.g. a new
    `AITER_TDM_*` switch) compiles once and every later run reuses that binary, even with the env flipped -> identical timings.
    Set `FLYDSL_RUNTIME_ENABLE_CACHE=0` (or make the switch a real kernel argument) for A/B runs.
23. **rocprofv3 on any torch/FlyDSL process needs `PYTHONPATH=proxy-moe/scripts/rocprof_env_fix`** (else `Option
    'spirv-expand-step' registered more than once`); see MOE_PREFILL_f19-14_1028.md.

24. **Global atomics are too slow to fuse a MoE top-k reduce** (T=8192, top-16, H=16384): `global_atomic_add_f32` 11.7 ms,
    `global_atomic_pk_add_bf16` 3.6 ms vs per-route bf16 write + gather-reduce 2.3 ms. Benchmark atomics at full size first.

## 3. Design-estimate misses (what the roofline forgot)

1. **HBM streaming of the weights.** Gate-up weights (4.29 GB per layer per rank) take 0.62 Mcyc at 1.15 GHz, about equal to the
   WMMA bound (0.70). The first design table only had WMMA + LDS (+ L2 later) and called the redesign "3.4x". Always include HBM.
2. **Wrong peak.** Using an fp8 x fp4 spec-derived 2350 TFLOP/s instead of the measured fp4 peak put the machine balance at
   ~300 flop/B instead of ~900, i.e. "compute-bound" when it is balanced. Measure the instruction rate first.
3. **"> 256 VGPR -> 1 wave/SIMD" was wrong** (budget is ~512 per wave for 2 waves/SIMD).
4. **L2 is the scarcest resource at full occupancy** (88 B/clk/CU < LDS 153-217): moving traffic from LDS to direct global loads
   only helps if the L2 side shrinks too (bigger WG tiles, multicast).
5. **"swap-AB" was not a padding fix:** the WMMA already puts tokens on its 16-wide side (min padding 16 rows).
6. **Bit-exact correctness was easy, speed was not:** the prototype matched the reference on the first run but was 1.5-2.7x slower
   than the kernel it was meant to replace; the bounds table predicted 2-3.4x faster.

## 4. Checklist before writing a GEMM / MoE prototype

- Bounds table per design with ALL of: WMMA (measured rate), LDS bytes/WMMA, L2 bytes/WMMA vs ~88 B/clk/CU, HBM bytes vs ~8 TB/s,
  in cycles at the serving clock. The kernel cannot beat the max; expect ~1.3-2x above it on a first prototype.
- Latency coverage: bytes in flight per CU >= latency x rate (L2 ~330 cyc; HBM larger) -> prefetch depth -> VGPR/LDS cost.
- VGPR budget written out (acc + operand buffers + prefetch + addresses) vs 512 (2 waves/SIMD) or ~1000 (1 wave/SIMD).
- After each build: `.vgpr_count`, `.private_segment_fixed_size == 0`, grep the hot loop for `s_wait_loadcnt 0x0`, `scratch_`,
  `__syncthreads`-generated waits, division sequences.
- Measure with ablations (drop weight loads / token loads / both) and ATT (`rocprofv3 --att --att-target-cu 1
  --att-shader-engine-mask 0x1 --kernel-include-regex <name>`; `proxy-moe/scripts/att/att_hot_loop_gfx1250.py` and
  `att_wait_breakdown.py`) before changing the design.
