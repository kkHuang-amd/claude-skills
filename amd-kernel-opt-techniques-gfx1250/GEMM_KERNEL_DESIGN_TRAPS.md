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
