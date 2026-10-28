---
name: amd-kernel-opt-techniques-gfx1250
description: >-
  Measured techniques and traps for kernel and serving optimization on AMD gfx1250 (MI450/MI455,
  wave32, WMMA, ROCm 10.x, torch 2.12), learned while porting the proxy-MoE (Laguna) MI355X
  optimizations: fp32 GEMV vs fp32 WMMA (v_wmma_f32_16x16x4_f32 lane layout), FlyDSL 0.3.4 traps,
  ISA resource checks for spills, power-capped clocks under serving load, RCCL CUMEM/IPC/DDA
  behaviour and hangs, aiter custom all-reduce under CUDA graphs, stale aiter JIT modules after a
  source switch, and making torch.profiler see GPU kernels in sglang. Use when optimizing a
  small-M GEMM/GEMV/router or any kernel on gfx1250, when a gfx1250 serving A/B or profile looks
  wrong, or when TP8 collectives on gfx1250 are slow or hang. Pair with
  gfx1250/mi400-optimization-skills (process) and amd-kernel-opt-techniques (gfx950 techniques).
---

# AMD kernel optimization on gfx1250: measured techniques and traps

Source: MI455 proxy-MoE work, node h13-17, 2026-10-23. Full log with every number:
`proxy-moe/MI455_RUN_h13-17.md`. Arch differences: `gfx1250/ARCH_DIFF_FOR_KERNEL_OPT.md`.
Process to follow: `gfx1250/mi400-optimization-skills` (profile first, one factor at a time,
correctness before performance). Kernel sourcing: check ROCm/FlyDSL and ROCm/aiter main for a
gfx1250 kernel before writing one.

Node facts that change conclusions (h13-17, B0):
- 8 GPUs are 4 sockets x 2 partitions; amd-smi reports socket power, throttle and temperature
  only on even GPUs; odd GPUs read N/A, clock included.
- 96 CUs per GPU.
- Measured HBM read ceiling per partition: 8.09 / 7.92 TB/s (4 GiB persistent Triton read, grid 8/CU, block 2048 int32, 4 warps; `proxy-moe/scripts/hbm_read_ceiling.py`). Theoretical ~10.1 TB/s (amd-smi 20.3 TB/s is per SOCKET). The old 5.47 TB/s (32 MiB buffer) was not a ceiling.

## 1. Small-M fp32 GEMV / router: VALU for M <= 2, fp32 WMMA above

Case: Laguna router, logits[M,512] = x[M,16384](bf16->fp32) @ W[512,16384]^T (fp32, 32 MiB).
- hipBLASLt `F.linear` fp32 picks `Cijk_Alik_Bljk_S_B_..MT32x32x16`: 207-238 us at every M
  1-64 (0.16 TB/s); MI355X ran the same op in ~14 us. Flat time vs M = not bandwidth-bound.
- FlyDSL VALU GEMV (wave32, 8 waves/block, a wave owns E experts x one K chunk, wave
  `shuffle_xor` reduce, fixed-order K-chunk sum): 9.3-13.4 us at M 1-2, but 232 us at M=64.
- Attribution at M=64 (ISA: 96 VGPR, 0 spills; larger experts-per-wave did not help) points to
  VALU instruction count, not spills or x re-reads. Two earlier guesses (unroll, x traffic) were
  wrong; check ISA before believing either.
- fp32 WMMA (`v_wmma_f32_16x16x4_f32`, rows = tokens, cols = 16 experts/wave, K split 64):
  14-23 us at M 2-32, 36.8 us at M=64. Same bytes, far fewer instructions.
- Dispatch used: M <= 2 VALU, 3..64 WMMA, else `F.linear` (prefill).
- Accuracy vs fp64: VALU 1.2e-7, WMMA 3e-7, hipBLASLt 2.4e-6. The "mismatch" at M=32 was the
  reference's error.
- End to end (1K1K, TP8): c=2 ITL -34%, c=64 -11%.
- Code: `proxy-moe/kernels/router_gemv_gfx1250.py`; sglang `SGLANG_OPT_LAGUNA_ROUTER_FLYDSL`.

## 2. fp32 WMMA lane layout (wave32, measured with one-hot probes)

`rocdl.wmma_f32_16x16x4_f32(vec8f32, a: vec2f32, b: vec2f32, c: vec8f32)` from the raw MLIR
binding `flydsl._mlir.dialects.rocdl`; FlyDSL's cdna5 helpers wrap only scaled/fp8 WMMA.
- A (16 x 4): lane l holds row `l % 16`, slot s holds `k = 2*(l//16) + s`.
- B (4 x 16): lane l holds col `l % 16`, slot s holds `k = 2*(l//16) + s`.
- C (16 x 16): element (m, n) in lane `n + 16*(m//8)`, VGPR `m % 8`.
- A and B share the k assignment, so any physical-k permutation works if both sides use it. A
  lane can load 8 contiguous k (2 x 16 B fp32) and feed 4 WMMAs with slots (8h+2i, 8h+2i+1).
- Probe: `proxy-moe/kernels/wmma_f32_layout_probe.py`.

## 3. FlyDSL 0.3.4.1 traps (installed version; upstream main API differs)

- `gpu.thread_id` / `block_id` are index-typed; mixing them with `fx.Int32` arithmetic makes
  `fx.slice(view, (None, off))` SEGFAULT in `_infer_int_tuple_type`. Wrap them in `fx.Int32(...)`.
- `range(fx.Int64(a), fx.Int64(b), fx.Int64(1), init=[...])` is the runtime loop. A plain int
  bound unrolls and silently ignores `init`. With ONE carried value, the state and the results
  come back bare, not as a list; normalize before indexing.
- Buffer loads/stores top out at 128 bits: a 256-bit `BufferCopy` fails in LLVM ("Do not know
  how to expand this operator's operand"). Split into 16 B pieces.
- Good templates for this API: aiter `gemm_a8w8_splitk_reduce_gfx1250.py` (buffer views),
  `causal_conv1d_update.py` (element-offset vector IO).
- ISA resources: `FLYDSL_DUMP_IR=1 FLYDSL_DUMP_DIR=<d> FLYDSL_RUNTIME_ENABLE_CACHE=0`, then
  FlyDSL repo `.claude/skills/isa-resource-diff/scripts/isa_resource_table.py summarize <d>`.
  The 512-VGPR cap plus spills turned 46 us into 2074 us (M x experts-per-wave too large).

## 4. Benchmark at serving clocks, not idle clocks

At 1K1K c=64 decode every socket is PPT-limited about 60% of the time (thermal 0%). GFX clock is
0.98-1.12 GHz vs 2.35 GHz idle (max 2.4). PPT0 1400 W and PPT1 1900 W are already at their max.
- Idle-clock microbenchmarks overstate compute-bound kernels by up to about 2x; validate in the
  server.
- Energy matters: a change that removes spin time (a faster AR) raised the PPT-throttled share
  from 59% to 65% and lowered clocks 8-12%.
- Sampler: `proxy-moe/scripts/gpu_power_sample.py sample2/summarize2` (throttle share =
  delta `*_accumulated` / delta `accumulation_counter`).

## 5. RCCL on gfx1250 TP8

- sglang sets `NCCL_CUMEM_ENABLE=0` unless symm-mem is on. RCCL then uses `P2P/IPC`: a 64 KB
  all-reduce takes 129 us (2.30.7) / 55 us (2.32.3), vs 11-12 us with `P2P/CUMEM`; the process
  also hangs at teardown. Set `NCCL_CUMEM_ENABLE=1` and verify the transport with `NCCL_DEBUG=INFO`.
- With CUMEM=1 the DDA path (`ddaAllReduceFlatLL`) deadlocks in the first warmup prefill about
  half the time (2.30.7 3/5, 2.32.3 4/7). `RCCL_DDA_LL_ONESHOT_THRESHOLD=0` does not fix it;
  `RCCL_DDA_ENABLE=0` does (0/6), at 34 us / 69 us for 64 KB / 2 MB. Repro and evidence:
  `proxy-moe/RCCL_DDA_HANG_REPRO.md`.
- A standalone torchrun replay of the same collective sequence did not hang; the trigger needs
  the server.
- The hang toolchain that worked: `py-spy dump --native` (host stuck in stream sync),
  `NCCL_DEBUG_SUBSYS=COLL` (last collective identical on all ranks), and `rocgdb -batch -ex 'info
  threads'` per scheduler (which kernel's waves are resident).
- Cross-rank AR attribution: align kernels by host timestamp (`proxy-moe/scripts/ar_align.py`),
  not by call index; profiler starts differ per rank.

## 6. aiter custom all-reduce at TP8 under CUDA graphs

- After a source switch, delete stale `aiter/jit/*.so`. `AITER_REBUILD=1` + `get_module` did not
  rebuild every module; a stale `module_custom_all_reduce_gfx1250.so` still carried an old
  `world size > 4` check (`strings` found it in the `.so`, not in the source).
- Under sglang graph capture the aiter all-GATHER took the registered path (`all_gather_reg`)
  while aiter forces copy-in for all-reduce on gfx1250. Result: `flush_graph_buffers` count=2
  -> old-arch IPC meta op on a gfx1250 handle -> `ValueError: cannot create std::vector larger
  than max_size()` (IPC), or a page fault in `ag_gfx1250_naive_vec` (VMM). The symm_mem probe
  failed and fell back to IPC.
- Fix: sglang uses `all_gather_unreg` during capture on gfx1250 (HaiShaw/sglang `ac677aad64`).
- Measured (TP8 graph): 64 KB 7.3-13.5 us, 2 MB 144 us. Small messages win, large lose, so
  cap it with `AITER_CUSTOM_AR_MAX_SIZE=1048576`. Server c=2 ITL 7.27 -> 6.68 ms.
- The sglang log line `[AR] All-reduce call path: NCCL (custom AR disabled)` printed although the
  profile showed `aiter::ar_ll_gfx1250`. Confirm collectives by kernel name in a trace.

## 7. Seeing GPU kernels at all

torch 2.12 on ROCm 10.1: sglang schedulers inherit `ROCPROFILER_REGISTER_LIBRARY` and
torch.profiler records no GPU kernels (only CPU events, no error). Fix: sglang PR #44292 (on
HaiShaw branch as `064be752f8`). For other multi-process tools (aiter op tests), put
`proxy-moe/scripts/rocprof_env_fix` on `PYTHONPATH` (a `sitecustomize.py` with the same logic).
`rocprofv3 --attach` also failed on the server tree. Profile at c=2 needs `PROFILE_DELAY` of
about 15 s, or the short bench ends before capture.

## 8. GEMM / MoE kernel design: measured limits and traps

Full write-up: `GEMM_KERNEL_DESIGN_TRAPS.md` (read before designing or prototyping a WMMA GEMM / MoE kernel). Short form:
- fp4 WMMA 32x16x128: 8 cyc issue / ~27 cyc latency per SIMD; VGPR file ~1024/lane/SIMD (<= ~512 per wave keeps 2 waves/SIMD).
- L2 aggregate ~88 B/clk/CU at full occupancy (< LDS 153-217, < per-CU port ~150); L2 does not merge identical requests;
  `cluster_load_b128` multicast 1.75-2x, collapses under large drift without a cluster barrier.
- Bounds tables must include WMMA + LDS + L2 + HBM (weights streaming was forgotten once and the "3.4x" design ran 1.5x slower).
- `__syncthreads()` in a pipelined loop waits out all prefetch (use `__builtin_amdgcn_s_barrier()`); any spill drains loadcnt;
  coalesce per instruction (512 B per b128 per wave), not per lane.

## Related

- `gfx1250/mi400-optimization-skills`: process, gates, report template.
- `amd-kernel-opt-techniques`: gfx950 techniques; re-measure before porting.
- `perf-bottleneck-attribution`: enabling condition != outcome; check the ISA before guessing.
