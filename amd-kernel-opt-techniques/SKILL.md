---
name: amd-kernel-opt-techniques
description: >-
  Model-independent techniques for small-batch (decode, M <= 16) GPU kernels on AMD gfx950 (MI355X),
  each with measured before/after numbers: wave-level top-k with ordered u32 keys and per-lane presort,
  cutting VMEM load instructions for the same bytes (packed / paired e8m0 scale loads, shuffle_scale
  layout tricks), union/histogram by LDS scatter instead of scanning, hiding independent work inside a
  cross-GPU wait (persistent / monokernel launches), and a list of measured dead ends (stream-K tail
  split, LDS staging of a shared operand, L2-cached reads after an acquire, prefetch before a grid sync).
  Use when optimizing MoE / GEMV / router / top-k kernels at small M, when a kernel streams weights below
  the HBM ceiling, when ATT shows buffer_load issue stalls, or when deciding whether a technique from a
  monokernel applies to an ordinary kernel. Pair with perf-bottleneck-attribution (how to measure).
---

# AMD small-batch kernel techniques (gfx950)

Source: the Laguna (proxy MoE) FFN monokernel work, 2026-10-22, node crsuse2-m2m-231. Every number
below is from `proxy-moe/MONO_FFN_crsuse2-m2m-231.md` (Results table, same date) unless noted. Code:
github.com/kkHuang-amd/sgl-amd-mono branch `feat/laguna-ffn-mono`, `sgl_amd_mono/laguna/gfx950/`
(FlyDSL; all of it pushed at 98ec598). None of the techniques depends on the model; most do not depend on the kernel being a
monokernel (marked where they do).

## Workflow that found them (read before applying any)

1. **Floor first.** Measure the streaming ceiling (a plain read kernel: 6.9 TB/s on MI355X with 256
   CTAs x 4 waves, 16 KB a CTA an iteration) and a *weights-only* variant of your kernel (same weight
   addresses, everything else removed, weights consumed by an xor). The gap between them is what the
   rest of the kernel costs.
2. **Add back one thing at a time** to the weights-only probe (activation loads, MFMA, route weights,
   ...). The item that brings the time back is the cost — not the item a profiler stalls on.
3. **ATT the loop**: per-instruction-class stall + duration over the hot loop
   (`tools/att_loop_stalls.py <ui_output_dir>`; rocprofv3 `advanced_thread_trace`, one CU). If
   `buffer_load` *issue* dominates and `s_waitcnt vmcnt` does not, the loop is VMEM-issue-bound
   (technique 2), and adding bytes in flight (prefetch depth, BATCH) will not help.
4. **For fused / persistent kernels, measure the critical path, not the stage**: per-CTA
   `s_memrealtime` marks at stage boundaries in a chained CUDA graph (as served), then confirm a
   stage is on the path by **injecting a delay** after it: the layer must slow by about the delay.
   Two "stage got faster, layer did not" results here turned out to be (a) a stage really off the
   path and (b) a measurement bug (next point).
5. **Kernel cache keys must carry every switch that changes code.** A JIT cache keyed by source text
   but not by env flags silently reuses the first binary across processes: an env A/B then compares
   a binary with itself. Verify an A/B changed the binary (a probe that must move the time).

## Techniques

### 1. Wave top-k: ordered u32 keys, per-lane presort, 2 reductions a round

Problem: top-16 of 512 in one wave (8 values a lane) done as 16 rounds of 3 butterflies (max key,
lowest index of the max, the winner's weight): 7.0-7.7 us, serial, on the critical path.

Recipe (from the GLM-5.3 plugin's router, `glm53/gfx950/kernel.py`):
- key -> order-preserving u32: `kb = f.bitcast(i32); u = kb >= 0 ? kb ^ 0x80000000 : ~kb`.
- Each lane sorts its 8 (key desc, id asc on ties) with a fixed 19-comparator network (Batcher
  odd-even merge, 8 inputs; verified exhaustively by the 0-1 principle).
- Round r: `m = wave_umax(head)`; `win = 1023 - wave_umax(head == m ? 1023 - id : 0)` (lowest id on
  ties); the winning lane shifts its list (sentinel key 0). `writelane(win, r)` collects picks.
- Weights once at the end: `ds_bpermute` the activation from lane `id % 64`, select `id / 64`.
- Keep the renormalization sum in pick order (`readlane` r = 0..15, sequential add) if the result
  must stay bit-identical to the old kernel.

Result: ids identical incl. order, weights bit-identical; router kernel 13.5 -> 9.8 us (M1), top-k
section 7.0 -> 2.4 us; fused layer -4.3..-5.8 us (TP8 76.7 -> 72.2 us at M1). Applies to any
in-wave top-k (MoE routers, sampling). Code: `router_tp.py` (`TOPK_FAST`).

### 2. Same bytes, fewer VMEM instructions (small-M GEMV / MoE are often issue-bound)

Signature: ATT shows `buffer_load` issue at ~50 % of the loop (TP4 M8 MoE down-projection: 52 %),
changing loads in flight does nothing (BATCH 1..4 equal), a weights-only probe is much faster. A
lane at TP4 issued 16 loads an expert for 160 B: 4 weight dwordx4, 4 single-dword weight scales,
4 activation dwordx4, 4 single-dword activation scales.

Recipes (all bit-identical):
- **Pack scales you write yourself as bytes, a lane's groups adjacent** so one dword load covers
  them (MXFP4 activation / intermediate scales written by your own quant step): store
  `buffer_store(Int8(scale), base + pos(group))` with `pos` = the lane's K groups next to each other.
  MID scales: DOWN 183 -> 161 us (TP4 M8), 95 -> 90 (TP8). x scales: small (0-4 us).
- **aiter `shuffle_scale` layout already pairs groups**: byte index
  `((((r32*(cols/8) + c8)*4 + e)*16 + b)*2 + d)*2 + a` (row = 32 r32 + 16 a + b, col = 8 c8 + 4 d + e):
  one dword holds rows (r, r+16) x groups (c, c+4). If a lane's K steps are j and j+1 = groups c, c+4,
  load the dword once and take bytes `a` and `2 + a`. W2: DOWN 160 -> 158 us = the weights-only floor;
  W13 (gate and up rows share `a` when inter/16 is even): UG 324 -> 317 us (TP4 M8).
- **Row-major scales (quark [N, K/32])**: a lane's K steps at bytes `4 j + sub` form one contiguous
  run: a single dword-vector load (x2 / x4), byte `sub` of dword j. Shared expert alone -10 %.
- What does NOT fix it: L2-cached loads of the shared operand after an agent-scope acquire (the
  acquire alone cost 10-14 us), staging the shared operand in LDS (4 variants, all slower, no spills;
  every wave reading LDS + barriers cost more than the saved traffic).

### 3. Union / histogram by scatter, not scan

Problem: each expert thread scanned all `tokens x 16` top-k entries in LDS (2 reads each: 512 serial
LDS reads at M16): 15 us at M16.
Recipe: zero a flag table [expert][token] (each thread its row, b128 stores), barrier, each top-k
entry thread stores `k + 1` at [e][t] (an expert once a token: no conflict, no atomics), barrier,
each expert thread reads its row (1-4 b128) and gets the weight through `k`. Reuse an LDS buffer that
is idle at that point. Result: layer -18..-19 us at M16, -4 us at M8 (TP8 443.8 -> 425.4).
Applies to MoE align/sort kernels.

### 4. Hide independent work inside a cross-GPU wait (monokernel / persistent only)

A persistent launch has idle windows (waiting for peers' data). Work that does not depend on what is
awaited can move there: the routed-expert activation quant (needs only the attention output) moved
from after the routing to the router's logit-exchange window, on CTAs that are otherwise idle there.
The downstream wait for its flags fell 5.5 -> 0.4 us; layer -3.5..-6.2 us at every M. Device-scope
flag hand-offs between CTAs cost ~2-5 us each on MI355X — fewer hand-offs on the critical path pays.

### 5. Smaller items (earlier the same day)

- Nontemporal loads (`cache_modifier=2`) for weights streamed once: -6..-7 % at M 4..16.
- `rocdl.sched_barrier(0)` after each block of MFMAs stops the compiler moving the next block's
  loads/waits across them: MoE up/gate M16 503 -> 456 us.
- Seam all-reduce from 2 rows on: two-shot (reduce-scatter to the column owner, all-gather the bf16
  sum) beats one-shot once xGMI bytes dominate (M16 33 -> 21 us).

## Measured dead ends (save the time)

- Balancing a part-full last round of tasks: a split round costs ~10 us fixed latency (partial
  hand-off, two epilogues, waits), more than it recovers; a contiguous task deal costs ~5 % at full
  rounds (all CTAs streaming far-apart memory). Fix the round quantum another way.
- Prefetching the next stage's weights before a grid sync: within noise (one batch of latency).
- More loads in flight when the loop is issue-bound (BATCH, prefetch depth 2-3 spills registers).

## Related

- `perf-bottleneck-attribution` — the diagnosis rules these results follow (patterns 1-5).
- `flydsl` — FlyDSL language reference and gotchas.
- `tools/att_loop_stalls.py` — ATT loop aggregation (rocprofv3 `ui_output_*` directory input).
