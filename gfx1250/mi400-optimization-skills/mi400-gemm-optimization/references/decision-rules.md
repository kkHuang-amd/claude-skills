# GEMM Decision Rules

## 1. Establish the regime
Record M, N, K, dtype, batch/grouping, transposition/layout, epilogue, and frequency. Separate small-K/output-sensitive cases from compute-dense cases.

## 2. Diagnose access before instruction choice
- Neighboring lanes touching far-apart B addresses: prioritize coalescing/layout/shared staging.
- Wider loads increase transactions efficiency but materially increase VGPR pressure: compare with cooperative LDS staging.
- Reuse exists across nearby output columns/rows: test N/M tiling, but cap the sweep when occupancy falls or a new wave tail appears.

## 3. Choose compute organization
- Scalar/FMA or packed dot path remains instruction/register-port limited for a dense tile: evaluate WMMA.
- Porting an MI300 MFMA kernel: retune lane mapping, tile sizes, pipeline and scheduling for MI400/GFX12 wave32; do not perform instruction-only substitution.

## 4. Choose staging path
- Register global-load + LDS-store overhead is material: compare direct/async global-to-LDS and TDM.
- Address calculation for multidimensional tiles is material and reuse justifies LDS: evaluate TDM.
- Current tile compute leaves enough independent work: double-buffer and issue next-tile movement early, then use selective waits/split barriers correctly.

## 5. Consider macro-tiles/multicast
- Multiple waves reuse overlapping A/B data: test cooperative macro-tile staging.
- Multiple clustered workgroups request identical data: test multicast, including cluster-barrier frequency and fallback behavior under drift.

## 6. Optimize output
When K is small or output-store share is visible, compare direct scattered stores, store clauses/write combining, LDS remap to coalesced stores, and async LDS-to-global store where applicable.

## 7. Cache/prefetch
Use temporal hints from observed reuse. Use L2 prefetch only when future data is valid/predictable and the kernel has compute to overlap; memory-bound kernels can worsen from extra traffic.
