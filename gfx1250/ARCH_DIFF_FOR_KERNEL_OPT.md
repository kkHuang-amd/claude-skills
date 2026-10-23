Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# gfx950 (MI355X) vs gfx1250 (MI455 / "MI400" in aiter) for kernel optimization

Written 2026-10-23 for the proxy MoE port (hidden 16384, 512 experts top-16 + shared,
MXFP4 W4A4 experts, BF16 GQA attention, FP8 KV, TP8/EP8 with all-reduce).

Source keys used below:
- [FLY] FlyDSL `.claude/skills/llvm/references/arch-gfx1250.md` (LLVM-option and ISA notes, raw.githubusercontent).
- [NODE] read-only queries on this node: `rocminfo`, `amd-smi static -g 0`, `/opt/rocm/llvm/bin/llvm-mc` (AMD LLVM 24.0.0git). 8x gfx1250, ASIC Revision 1 (= B0).
- [AIT] `/sgl-workspace/aiter` @ `2bb549102` (2026-10-10).
- [REPO] this skill dir: `SKILL.md`, `STATUS.md`, `EXPERIMENT_LOG.md` (E-numbers), `topk_v2_gfx1250/`, `gfx1250-quickreduce-optimization/`, `cuda-cluster-launch-kernels/`.
- [PUB] AMD public MI355X specs. Not re-measured in this repo.

## 1. Parameter table

| Parameter | gfx950 (MI355X) | gfx1250 (this node) | Source |
|---|---|---|---|
| Wave size | 64 | **32 only**. `+wavefrontsize64` silently emits an EMPTY object with rc=0. | [FLY], [NODE] |
| Matrix unit | MFMA (`v_mfma_*`), plus scaled `v_mfma_scale_f32_{16x16x128,32x32x64}_f8f6f4` | **WMMA only**. There is no MFMA, and both `FeatureMAIInsts` and `FeatureGFX90AInsts` are absent. | [FLY], [PUB] |
| Dense WMMA | n/a | `wmma.*` in f64, f32, f16, bf16, fp8, bf8 and iu8, including 16x16x128 (fp8). Also `swmmac.f32.16x16x64.{f16,bf16}`. | [FLY] |
| Scaled MX WMMA | n/a | `wmma.scale{,16}.f32.16x16x128.f8f6f4` (14 args). Also `v_wmma_scale_f32_32x16x128_f4` and `v_wmma_scale16_f32_32x16x128_f4` (fp4 x fp4, both assemble for gfx1250). | [FLY], [NODE] llvm-mc |
| AGPRs | 256 (unified 512 with arch VGPRs) | **none**. The `"a"` asm constraint fails, and `amdgpu-agpr-alloc` is a no-op. | [FLY] |
| VGPRs per wave | 256 arch (512 unified) | **1024** addressable, allocation granule 16. aiter kernels pin accumulators at v512+. | [FLY], [AIT] `mxfp4_preshuffle_gfx1250_tdm.py` |
| Max waves/EU, waves/CU | 8, 32 | 16, 64 (rocminfo: Max Waves Per CU 64, 4 SIMDs/CU) | [FLY], [NODE] |
| LDS per CU/WG | 160 KB | **320 KB** (rocminfo GROUP segment 320 KB; aiter `chip_info` 320*1024) | [FLY], [NODE], [AIT] |
| CU count | 256 | **96** on this node (rocminfo, amd-smi NUM_COMPUTE_UNITS). aiter notes that gfx1250 "also reports 256" on some systems, so the count depends on SKU or partition. Read it at runtime. | [NODE], [AIT] chip_info.py:204 |
| Shader engines / clock | unknown (not measured here) | 6 SEs, max 2400 MHz | [NODE] |
| L1 / L2 | 32 KB L1 per CU, 4 MB L2 per XCD, 256 MB Infinity Cache | 32 KB L1 per CU. amd-smi reports **L2 4096 KB shared by ~97 CUs**, plus extra 16-256 KB L1-level SIMD caches shared by 2-9 CUs. The MALL / last-level cache size is **unknown**. | [PUB], [NODE] |
| Cache line | 128 B (L2) | rocminfo reports 64 B. Treat as unverified. | [PUB], [NODE] |
| HBM capacity / BW | 288 GB / 8 TB/s | **288 GB / 20.3 TB/s** (amd-smi MAX_BANDWIDTH 20275 GB/s, VRAM 294912 MB). The older R1 box had ~432 GB per GPU. Achieved bandwidth is **unknown** (not measured). | [PUB], [NODE], [REPO] SKILL.md |
| Async global->LDS | `buffer_load ... lds` (direct-to-LDS) | `global.load.async.to.lds.{b8,b32,b64,b128}` plus cluster forms, waited with `s_wait_asynccnt` | [FLY] |
| TDM (tensor DMA) | none | `tensor.load.to.lds` / `tensor.store.from.lds` (4 descriptor groups), waited with `s_wait_tensorcnt` | [FLY] |
| LDS transpose read | `ds_read_tr*` | `ds_load_tr{4.b64,6.b96,8.b64,16.b128}`. Each arch hard-errors on the other's intrinsic. | [FLY] |
| Waitcnt | packed `s_waitcnt` | split counters `s_wait_{loadcnt,storecnt,dscnt,kmcnt,...}`. Legacy `s.waitcnt` makes ISel **crash** ("Cannot select"). | [FLY] |
| Workgroup clusters | none | cluster launch (`{1,N,1}` only), split cluster barrier, `cluster_load_b32` multicast global->VGPR. **No remote-LDS (DSMEM)**: `map_shared_rank` returns null. | [FLY], [REPO] topk_v2_gfx1250 |
| Prefetch | n/a | `global_prefetch_b8 scope:SCOPE_SE`. It is silently dropped if the LLVM build lacks the ISel pattern. | [FLY] |
| Silicon stepping | n/a | A0 vs B0+. aiter ASM kernels are **B0+ only** (`require_gfx1250_asm`). This node is B0. | [AIT] asm_guard.py, [NODE] |

### Does gfx1250 lack `V_WMMA_SCALE_F32_32X16X128_F4`?

Conclusion: **the claim in `SKILL.md` ("gfx1250 lacks the fp4-activation scaled WMMA") is not supported, and current evidence contradicts it for B0 silicon.** Evidence:

1. `llvm-mc -mcpu=gfx1250` assembles `v_wmma_scale_f32_32x16x128_f4` (encoding `...0x35,0xcc...`), `v_wmma_f32_32x16x128_f4` and `v_wmma_scale16_f32_32x16x128_f4`. [NODE]
2. The FlyDSL arch doc lists `wmma.scale{,16}.f32.16x16x128.f8f6f4`, and fp4 is one of the f8f6f4 formats. It does not mention the 32x16x128_f4 opcode, so it neither confirms nor denies it. [FLY]
3. aiter calls `rocdl.wmma_scale_f32_32x16x128_f4` in its production gfx1250 MoE kernel when `a_is_fp4` (`aiter/ops/flydsl/kernels/mxfp4_preshuffle_gfx1250_tdm.py:1098`, WMMA_N=32 for fp4). Grouped a4w4 tests have existed since `76ea4ac32` (2026-06-15). `fused_moe.resolve_activation_dtype` now **defaults gfx1250 to fp4x2 activations** unless `AITER_FORCE_A8W4=1`. [AIT]
4. The original evidence was confounded. In E3 the `SQC (inst)` faults also occurred for triton **a8w4 and a16wfp4** kernels, which do not use an fp4-activation WMMA. That box ran the June aiter fork and an unknown stepping; an A0-erratum explanation is possible but **unknown**. [REPO] EXPERIMENT_LOG E0/E3

Action: before relying on a4w4 here, run the aiter grouped a4w4 op test (`op_tests/test_flydsl_grouped_gemm_gfx1250.py`) once the GPUs are free. Do not run it during the benchmark. Treat `SKILL.md` §§1-3 as historical: they describe the June toolchain.

## 2. Implications for porting gfx950 optimizations

General rules:
- Re-derive every lane-count constant. 64 to 32 changes warp reductions (DPP/`ds_swizzle` and `__shfl` widths), lanes per row in a tile, top-k per-lane presort sizes, and wave-level ballot masks (`uint32` instead of `uint64`).
- Re-tune occupancy from scratch. `waves_per_eu=2` is 1/4 of peak on CDNA but 1/8 on gfx1250. `amdgpu-num-vgpr` is not doubled on gfx1250. Do not halve CDNA values, and do not reuse `512 // (arch+acc)` math. Any `waves_per_eu` max above 16 is silently dropped. [FLY]
- Every gfx950 `waves_per_eu`, tile and `kpack` config is invalid. gfx1250 does not support triton `kpack`. [AIT] triton README

By kernel class:
- **Small-M GEMV / MoE decode (M <= 16).** WMMA has a fixed M=16 (fp4 x fp4 is 32x16x128), so M=1..4 decode wastes most of the matrix unit, just as on gfx950. The bound is weight bandwidth. Peak bandwidth is about 2.5x MI355X (20.3 vs 8 TB/s [NODE]/[PUB]) with fewer CUs here (96 vs 256), so each CU must sustain about 6-7x more bytes/s.
  - This raises the value of deep async pipelines (TDM / `global.load.async.to.lds`) and of many loads in flight per CU (64 waves/CU, 1024 VGPRs).
  - The gfx950 "fewer VMEM instructions for the same bytes" tricks (packed e8m0 scale loads, shuffle_scale layouts) still apply. The scale layouts differ, though: GEMM MX-scale tiles are `(32,8)` on gfx950 and `(16,4)` on gfx1250 [AIT triton README], and MoE B-scales are n32k4 on gfx1250 [REPO CHANGES.md].
  - The aiter gfx1250 grouped MoE switches from per-expert masked_m to contiguous-M at `token_num > 16` [REPO HANDOVER_kernel_fixes]. For EP8 with 64 local experts, decode is in the masked regime.
- **MXFP4 GEMM.** Replace `v_mfma_scale_*_f8f6f4` (16x16x128 / 32x32x64) with `wmma_scale_f32_16x16x128_f8f6f4` (fp8 or mixed) or `wmma_scale_f32_32x16x128_f4` (fp4 x fp4).
  - Accumulators live in VGPRs, not AGPRs, so plan the 1024-VGPR budget.
  - Weights need gfx1250-specific preshuffles: a (16,16) shuffle for the grouped a8w4 kernel. fused_moe on gfx1250 also **requires GateMode.INTERLEAVE** (GUGU row-interleave), not the 16-row block shuffle [AIT fused_moe.py:1161]. gfx950 preshuffled checkpoints and layouts do not carry over.
  - Dense fp4 GEMM on gfx1250 goes through asm `gemm_a4w4` / `gemm_a8w4_mxfp8` (B0+) or gluon `gemm_afp4wfp4`, not the gfx950 triton kernels.
- **Quant kernels** (mxfp4/mxfp8 dynamic quant, scale shuffles). These are elementwise and bandwidth-bound; the main change is the wave32 reduction width for the 1x32 amax.
  - A 32-element group is exactly one wave32 row. QuickReduce loads one scale per 8-thread group and broadcasts it with a wave32 shuffle [REPO quickreduce].
  - `dynamic_mxfp4_quant`, `dynamic_mxfp8_quant` and `dynamic_per_token_quant_fp8_i8` worked on gfx1250 even in June [REPO E3]. aiter now has a gfx1250 A-scale layout writer (`scale_layout_m32k4`) for the asm GEMM [AIT quant.py:441].
- **Top-k / routing** (512 experts, top-16). Wave-level top-k written for 64 lanes needs a 32-lane redesign. 512 candidates is 16 per lane on wave32 instead of 8 on wave64.
  - For very large rows, the workgroup cluster + `cluster_load_b32` multicast pattern gave 1.13-1.74x on DSv4 TopK [REPO topk_v2 REFERENCE].
  - For 512 experts a single workgroup is enough, so clusters are probably unnecessary (inference, not measured).
- **All-reduce (TP8).** aiter has a dedicated `module_custom_all_reduce_gfx1250` [AIT custom_all_reduce.py:242]. The generic CAR signal-registration block is skipped on gfx1250 because of a deadlock in its vmm_exchange rendezvous [AIT communication.py:68].
  - QuickReduce PullQ4 (lossy INT4) beat CAR at 16 MiB and up on TP2 and 8 MiB and up on TP4 [REPO quickreduce]. Decode all-reduce of 16384 x bf16 x small batch is far below those sizes, so it stays latency-bound. TP8 numbers do not exist yet (unknown).
  - The gfx950 "hide independent work inside the cross-GPU wait" monokernel idea is untested on gfx1250.
- **Attention** (BF16 GQA, FP8 KV).
  - **Never use triton `tl.dot(fp8, fp8)` with K >= 128 on gfx1250**: it returns ~1e34 garbage, while K=64 and bf16 are fine [REPO E43, `artifacts/BUGREPORT_triton_fp8_dot_largek_gfx1250.md`]. For an FP8 KV cache, upcast K/V to bf16 inside the kernel and keep q in bf16. Do not downcast q to fp8. **Not reproduced on B0 (2026-10-23, h13-17, triton 3.8.0):** e4m3/e5m2 `tl.dot` with K = 64/128/256/512 matches the fp32 reference (rel err ~3e-5); test `artifacts/fp8dot_check.py`. The bug is likely A0-only (or an older Triton). Re-check before relying on this on other silicon.
  - Keeping softmax `p` in fp32 for the P·V dot ("FIX A") turned out to be accuracy-neutral on the current stack [REPO STATUS].
  - aiter has gfx1250 variants of unified_attention, mla and pa kernels, plus gluon attention configs [AIT].
- **LDS staging.** 320 KB of LDS (2x gfx950) allows deeper multi-stage pipelines and larger tiles.
  - TDM loads whole tiles with one descriptor and replaces the per-lane `buffer_load ... lds` loops. Transpose reads use `ds_load_tr*`.
  - A gfx950 dead end, LDS staging of a shared operand, may need re-measuring because the capacity and the async engine changed.
  - TDM is not automatically a win: the TopK TDM prototype lost to the plain path [REPO topk_v2 REFERENCE].
- **Persistent kernels.** Size grids from the runtime CU count (96 here, not 256).
  - Cluster barriers are available. HIP `this_cluster().sync()` **hangs with 1024-thread workgroups**; use `barrier_arrive()` plus `barrier_wait()` instead [REPO topk_v2 SKILL].
  - PDL is a no-op on ROCm [REPO cuda-cluster SKILL].
  - Never return early from a branch while peers will reach a cluster barrier.

## 3. Known gfx1250 bugs, workarounds and env knobs (this repo)

| Issue | Workaround | Pointer |
|---|---|---|
| triton `tl.dot(fp8,fp8)` gives garbage for K >= 128 (A0 / older triton; OK on B0 + triton 3.8) | upcast to bf16, or tile K <= 64 | EXPERIMENT_LOG E43, artifacts/BUGREPORT_* |
| June-fork aiter triton fp4 GEMMs (`gemm_afp4wfp4`, `gemm_a8wfp4`, `gemm_a16wfp4`) page-fault | in June: bf16-dequant linear. Now: asm `gemm_a4w4` or gluon. Re-verify. | SKILL §3/§4.2, E3 |
| A GPU fault wedges the GPU for the next process (hang at the first `torch.ones`) | kill and reap the dead process, wait. Suppress core dumps, which reach ~260 GB and fill the disk. | SKILL §7, E0/E7 |
| Grouped MoE split_k>1 raw+finalize gives illegal-address | June fork: `AITER_GROUPED_FORCE_SPLIT_K1=1` (local patch, **absent from current aiter**) | SKILL §4.6, E12 |
| Grouped a8w4 needs (16,16)-shuffled weights and n32k4 B-scales, otherwise GSM8K = 0 | `SGLANG_MOE_SHUFFLE_GFX1250` (default true) | STATUS "3 code changes", CHANGES.md B |
| aiter bisect "fix" (`bit_length`) does an OOB read of layout_buffer and crashes DSv4 | keep `ceil(log2)` | STATUS (E38/E40) |
| aiter `fused_qk_rmsnorm` JIT build failure (June) | Triton RMSNorm | SKILL §4.4 |
| FlyDSL grouped a8w4 IndexError (missing swiglu_limit) | patched | SKILL §4.5 |
| Legacy `s_waitcnt` intrinsic gives an ISel crash | split wait counters. Do not "fix" FlyDSL's guard. | [FLY] |
| `+wavefrontsize64`, or features in the target id (`gfx1250:xnack+`) | silently ignored or empty output with rc=0. Never set them. | [FLY] |
| HIP cluster `{N,1,1}` rejected, `this_cluster().sync()` hangs | use `{1,N,1}` and split arrive/wait | topk_v2_gfx1250/REFERENCE.md |
| fused small-batch cluster TopK (`map_shared_rank`) faults | route to the streaming v2 path | cuda-cluster-launch-kernels/SKILL.md |
| aiter asm kernels on A0 silicon | they raise. Check `is_gfx1250_asm_supported()`. | [AIT] asm_guard.py |

Other aiter MoE knobs on gfx1250 [AIT]:
- `AITER_FORCE_A8W4` (fp8 activations instead of fp4x2)
- `AITER_DISABLE_GROUPED_A8W4`
- `AITER_USE_GROUPED_GEMM` (auto-on for gfx1250)
- `AITER_FORCE_GFX1250`
- `AITER_GROUPED_DEBUG`
- `AITER_FLYDSL_COMPACT_SCALE_MIN_TOKENS`
- `AITER_FLYDSL_DISABLE_GEMM1_REQUANT`
- `AITER_SITUV2_A4W4` / `AITER_SITUV2_A8W4`
- `AITER_MOE_A8W4_BYPASS_QUANT`

## 4. MoE + MXFP4 kernel paths on gfx1250

**aiter `fused_moe`.**
- On gfx1250, `grouped_gemm_gfx1250_a8w4` is tried first. Despite the name it handles both a4w4 and a8w4 through `a_is_fp4`. It is only reachable with `GateMode.INTERLEAVE`.
- Kernels:
  - `aiter/ops/flydsl/grouped_moe_gfx1250.py`
  - `kernels/mxfp4_preshuffle_gfx1250_tdm.py` (TDM pipeline; the a4w4 path uses `wmma_scale_f32_32x16x128_f4`)
  - `kernels/mega_moe_gfx1250/` (mega-MoE with EP TDM dispatch and fp8/fp4 combine quant)
- Default activation is fp4x2 (a4w4); `AITER_FORCE_A8W4=1` switches to MXFP8. Production status:
  - a8w4: validated end to end on DSR1 (GSM8K 0.95) and DSv4 (0.925-0.94) [REPO].
  - a4w4: upstream added a4w4 prefill v2 and tuned DSR1 configs (`332b34fba`, 2026-10-02), but it is **not validated in this repo**.
  - EP: mega_moe has EP support. A comment notes that the MXMOE a4w4 port "does not support expert-parallel yet" (fused_moe.py:1451). Check which path EP8 hits.

**Gluon / Triton (aiter).**
- Gluon configs exist for gfx1250 MoE `a4w4` and `a8w4`, GEMM `gemm_afp4wfp4[_preshuffled]` and `gemm_afp8wfp8_preshuffled`, and quant `mxfp4` / `mxfp8`.
- Triton configs exist for MoE `a8w4` and many gemm_* entries (`configs/gfx1250/`).
- The June fork's triton fp4 GEMMs faulted. Current status is unverified.

**ASM (B0+ only).**
- `gemm_a4w4` (MXFP4/NVFP4 F4GEMM; K%32, N%16; ignores bias/alpha/beta)
- `gemm_a4w4o8` (MXFP8 output)
- `gemm_a8w4_mxfp8` (K%128, N%16)

**FlyDSL dense.** The June dense `compile_a8w4_gemm(grouped_masked_m=False)` faulted and was dropped [REPO E5].

**Fallbacks.**
- a8w4 (`AITER_FORCE_A8W4=1`): validated accuracy. The a8w4 vs a4w4 numeric difference was shown accuracy-neutral [REPO E30].
- bf16-dequant linear plus `F.linear`: slow, and the memory cost is large.

**SGLang.** `quark_w4a4_mxfp4_moe.py` has gfx1250 branches that apply the n32k4 B-scale, the GUGU interleave and the (16,16) shuffle. `aiter_mxfp4_w4a8_moe.py` also has a gfx1250 branch.
