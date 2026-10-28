Owner node: ctheliosmc-rck-g02-h13-17.rck.dcgpu

# gfx1250 workgroup cluster / multicast notes for the MoE kernel design (2026-10-28)

Sources: (S) claude-skills, (C) local / upstream code read on h13-17, (W) web search by a subagent (URLs below, not individually
re-verified), (M) our measurements (`MOE_PREFILL_h13-17_1028.md`, "Bimodality probe" / "Clock probe").

## What the hardware gives you

- Cluster launch, max 16 WGs (W: llvm-project#230186). Kernel attr `amdgpu-cluster-dims`; HIP `hipLaunchAttributeClusterDimension`,
  grid must be a multiple of the cluster (else error / hang, W: rocm-libraries#13882). HIP worked only as `{1,N,1}` in topk_v2 bring-up (S).
- A cluster likely lives inside one shader engine (`cluster` syncscope = `SCOPE_SE` in AMDGPUUsage; inference, W). No published
  co-residency / XCD placement guarantee found.
- Split cluster barrier (`s_cluster_barrier`, id -3 = WG barrier + one-wave signal + all-wave wait). Barrier completion is NOT memory
  ordering; add release/acquire (W: rocm-libraries#12857). HIP `this_cluster().sync()` hangs with 1024-thread WGs; use arrive/wait (S).
- Multicast: TDM `tensor_load_to_lds` WG mask (D# group1 dword0 [15:0]) and `cluster_load_b32` (global -> VGPR of each selected WG,
  mask in M0) (S topk_v2 REFERENCE, W). Data returns only to WGs that issued the matching request (W: FlyDSL#1233, ISA 10.7).
- **No DSMEM**: no ds_* access to a peer's LDS (`map_shared_rank` null; S ARCH_DIFF, W). Cross-WG data exchange = global memory,
  multicast, barriers only.
- HW bug: multicast mask 0xffff can hang the GPU; LLVM narrows to 0x00ff / 0xff00 (W: llvm-project#233363). Keep masks <= 8 WGs wide.

## What makes multicast actually pay (and fail)

- Multicast is not guaranteed: members must request the same tile at about the same time; drift -> silent fallback to separate
  requests (S mi400-tdm-data-movement). Staggered K start per WG kills it (W: rocm-libraries#11290). aiter#5274: a gather descriptor
  left `workgroup_mask=0` -> every peer re-fetched A.
- Mixed evidence elsewhere: FlyDSL 2x2 microbench ~1.15x (W: FlyDSL#1200); MI455X 2x2 +1..11% from 512 rows (W: tokenspeed#2719);
  TensileLite: 23/29 multicast comparisons slower (W: rocm-libraries#14338). TLX grouped GEMM needed all groups M % 1024 == 0 and
  program counts that keep cluster members on matching loop boundaries (W: facebookexperimental/triton#3643).
- **Our measurement (M):** EP8 stage1 256^3 with cluster_n=4 (auto cluster_m=4 -> 4x4 = 16 WGs, see below):
  fast mode = same cycles as no-cluster but ~1110 vs ~990 MHz (less data-movement energy under the power cap -> higher clock);
  slow mode = +30% cycles at lower power (stalls). Mode is fixed per process; cause still open.

## How our grouped MoE kernel uses clusters (C, `mxfp4_preshuffle_gfx1250_tdm.py` + `grouped_gemm_mxfp4.py`)

- `cluster_n` > 1: A (tokens) multicast across the N row of the cluster (`a_mcast_mask`).
- `cluster_m` (1/2/4, `AITER_FLYDSL_MXFP4_CLUSTER_M` overrides; CSV -1 -> **4 for stage1 when cluster_n == 4**, else 1):
  B (weights) multicast down a column only for M-tiles of the SAME expert (`use_b_mcast = full_cluster & group_size > 1`).
- Cluster barrier only when cluster_m > 1: once after the prologue and every `num_buffers` k-steps (`kt+1 % num_buffers == 0`),
  if `full_cluster`. `AITER_FLYDSL_DISABLE_CLUSTER_SYNC=1` removes them. Product limit cluster_m * cluster_n <= 16.
- At EP8/EP1 T=8192 an expert has ~256 rows = 1 M-tile of 256 -> B multicast is almost never active, yet the 16-WG barrier runs
  -> one slow WG stalls 16. Plausible slow-mode mechanism (hypothesis, test with the two env knobs above).

## Design implications for the new MoE kernel (Track A / B)

1. Energy per byte is first order: under the power cap fewer bytes moved -> higher clock (M). Count HBM + L2->LDS + LDS->VGPR bytes
   per WMMA in the roofline, not just cycles.
2. Weight multicast only works if cluster members process the same expert at the same k: schedule clusters along M within one
   expert (needs rows/expert >= cluster_m * tile_m), or for small rows/expert use swap-AB so the shared operand is the token tile.
   Pick cluster_m from rows/expert at runtime instead of a fixed 4; never pay a cluster barrier when group_size <= 1.
3. Keep cluster members in lockstep cheaply: identical loop bounds per cluster (persistent scheduler hands out work per cluster,
   not per WG), no per-WG K stagger, barrier interval tied to buffer reuse only.
4. Track A (weights global -> VGPR): **`__builtin_amdgcn_cluster_load_b128` / `_b64` exist** (verified asm, M3 2026-10-28) ->
   one 16 B/lane weight fragment per instruction. Measured (M3): multicast CS=2 delivers 1.75x, CS=4 ~2x per CU vs private
   (L2 aggregate ~82 B/clk/CU), data correct; plain loads of identical addresses are NOT merged by L2.
   Large drift without a barrier collapses multicast to half of private; small drift (< ~500 cyc per 8 iters) is absorbed.
5. No DSMEM: split-K / fused gate-up+down reductions across WGs must go through global memory (or redesign so one WG owns the sum).
6. Always A/B multicast vs separate loads on the real shapes, several processes each (mode is per process), and report cycles + clock.

## Pointers

- S: `gfx1250/mi400-optimization-skills/mi400-tdm-data-movement/` (multicast, decision-rules, anti-patterns),
  `mi400-gemm-optimization/references/decision-rules.md` (step 5), `gfx1250/ARCH_DIFF_FOR_KERNEL_OPT.md` (lines ~32-36, 71-87),
  `gfx1250/topk_v2_gfx1250/REFERENCE.md` (cluster_load_b32 + split barrier, 1.13-1.74x on DSv4 TopK), `gfx1250/cuda-cluster-launch-kernels/`.
- C: FlyDSL `python/flydsl/expr/rocdl/cluster.py` (`cluster_signal_once_per_wg`, `cluster_wait`, `cluster_barrier`,
  `compute_cluster_position`), `kernels/common/gfx1250_cluster.py` (`compute_mcast_masks`).
- W: llvm.org/docs/AMDGPUUsage.html; github.com/llvm/llvm-project/pull/{230186,233363}; github.com/ROCm/rocm-libraries/pull/{12857,
  11290,13882,14338}; github.com/ROCm/FlyDSL/pull/{1200,1233}; github.com/ROCm/aiter/pull/{5274,6136,6663};
  github.com/facebookexperimental/triton/pull/3643; github.com/lightseekorg/tokenspeed/pull/2719;
  chipsandcheese.com/p/scrying-the-amd-gfx1250-llvm-tea.
