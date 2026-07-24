# Analysis record — FlyDSL MegaMoE stage1 decode cost (NO action needed)

Status: **analysis record, not a filed issue.** Superseded the earlier "mtpr padding" issue
draft — the micro-harness disproved that hypothesis. Recorded 2026-07-16.
Repo of interest: `ROCm/FlyDSL` `mega_moe_v1`, `kernels/moe/mega_moe.py` / `mega_moe_gemm1.py`.

## TL;DR
MegaMoE decode stage1 is **already near-optimal**. There is **no mtpr padding tax** to remove
(compact mode is mtpr-independent), and megav1 is **2.2x faster than the uncapped aiter baseline**
at mtpr=8192. The only residual is the compact-vs-non-compact mode overhead (~11.6%), which is not
worth a risky megakernel rewrite. Earlier "stage1 is 2x slower" was an unfair comparison
(vs dp's opus, which has no dispatch and is a different, hardware-tuned kernel).

## What we measured

### 1. Server per-layer decode (conc128, 8k/2000, cuda-graph, rank0)
`moe_gemm1` (fused dispatch+gemm1): mtpr=8192 → 222.9us; mtpr=512 → 175.1us. Initially read as a
"21% mtpr padding tax". **This was a confound** (see #2).

### 2. Micro-harness (tests/kernels/test_mega_moe.py, v4_pro a8w4, tokens=32, `--mtpr` override)
megav1 = full stage1+stage2 E2E (ms), 8 ranks:

| mtpr | mode | megav1 | uncapped aiter baseline |
|---:|---|---:|---:|
| 1024 | non-compact | 0.371 | 0.452 |
| 2048 | compact | 0.414 | 0.526 |
| 8192 | compact | **0.415** | 0.923 |

- **Within compact, mtpr is irrelevant** (2048→8192: 0.414→0.415, flat). So the server 512→8192
  delta was NOT padding — it was the **compact/non-compact mode switch** (auto-selected at
  `_max_buf = experts*mtpr*max(model_dim, 2*inter) >= 3GB`, i.e. mtpr>=2048 for this shape).
- compact costs **+11.6%** over non-compact (the extra cross-PE dispatch round), independent of mtpr.
- **megav1 (compact) is 2.2x faster than the uncapped aiter baseline** at mtpr=8192 (0.415 vs 0.923);
  the aiter baseline is the one that scales badly with mtpr (0.45→0.53→0.92), not megav1.

## Root cause of the compact overhead (for reference, not action)
compact (`kernels/moe/mega_moe_gemm1.py`, the NAIVE-COMPACT prologue) runs a **count-first pass +
prefix-sum + a 2nd cross-PE barrier round** to build a dense `num_valid` (avoids the
epr*mtpr fixed-slot buffer, which at mtpr=8192 would be ~22GB). non-compact uses fixed slots + 1
round but needs the big buffer, so it's only usable at small mtpr. Both prologue passes are
`cur_tok`-bounded (real tokens); nothing scales with mtpr. The +11.6% is purely the extra
cross-PE round's fixed latency, which dominates at decode (few tokens, tiny GEMM).

## Why no low-risk optimization here
- No mtpr padding to remove (compact is mtpr-independent).
- The ~11.6% compact overhead needs either (a) decode using a small-cap non-compact path — requires
  decoupling buffer cap from mtpr (the two-instance attempt hung: separate comb_op/shmem/a2a state
  on the shared mori heap deadlocks under continuous batching), or (b) restructuring the compact
  2-round cross-PE dispatch (deadlock/corruption risk). Not worth ~11% for that risk.
- megamoe is already competitive/faster than the fair EP baseline (mori-EP) and ~13% behind dp
  (non-EP), where the gap is dp's opus a8w4 (no dispatch, hardware-tuned), not a stage1 defect.

## Useful artifact kept
`tests/kernels/test_mega_moe.py --mtpr N` — micro-harness to reproduce large-mtpr + few-token
(decode-in-serving) and isolate compact/mtpr effects with a torch reference. Keep.
