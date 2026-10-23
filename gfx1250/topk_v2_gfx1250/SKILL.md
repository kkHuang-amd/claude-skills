---
name: gfx1250-cooperative-kernel
description: Bring up, debug, validate, and tune HIP cooperative-launch kernels on AMD gfx1250, especially SGLang DeepSeek-V4 TopK v2. Use when working with gfx1250 workgroup clusters, split cluster barriers, cluster_load multicast, cluster launch attributes, or TopK cluster-size tuning.
---

# gfx1250 Cooperative Kernels

Use `/sgl-workspace/sglang` as the repository unless the user specifies
another checkout. Read [REFERENCE.md](REFERENCE.md) before changing cluster
communication, synchronization, workspace layout, or dispatch.

## Non-negotiable facts

- gfx1250 has cluster launch, split cluster barriers, and masked global-to-VGPR
  or global-to-LDS multicast.
- It does not expose usable arbitrary peer-LDS addressing on the validated
  toolchain. `map_shared_rank` returns null because
  `__builtin_amdgcn_map_shared_rank` is absent.
- HIP `this_cluster().sync()` hangs with 1,024-thread workgroups. Use the
  validated `barrier_arrive()` / `barrier_wait()` sequence.
- `cluster_load_b32` reads global memory and multicasts the value to matching
  lanes in selected workgroups. It is not remote LDS.
- Keep `SGLANG_GFX1250_TOPK_CLUSTER` default-off until the requested production
  acceptance criteria are met.

Do not revive the remote-LDS, TDM-descriptor, or per-candidate global-atomic
prototypes unless the user explicitly asks to re-evaluate them.

## TopK implementation pattern

1. Build a 12-bit local histogram in each workgroup's LDS.
2. Stage one histogram per rank to global workspace.
3. Release-arrive and wait at cluster scope.
4. Use `__builtin_amdgcn_cluster_load_b32` to multicast peer bins into
   registers; each workgroup computes the same reduced histogram and threshold.
5. Collect greater-than and equal candidates locally.
6. Stage one count pair per rank, synchronize, and calculate deterministic
   rank prefixes. Emit greater-than candidates without per-candidate global
   atomics.
7. Stage bounded tie candidates globally; rank 0 performs exact tie selection.
8. Synchronize before any rank-dependent page transformation or kernel exit
   that could invalidate shared state.

The 12-bit histogram is required. A 10-bit histogram passed random tests but
lost high-valued candidates when duplicate-heavy threshold bins exceeded the
2,048-entry tie buffer.

## Bring-up workflow

1. Confirm gfx1250 and HIP version.
2. Run the split-barrier and register-multicast probes:

   ```bash
   bash /sgl-workspace/topk_v2_gfx1250/scripts/run_cluster_probes.sh
   ```

3. Make the smallest kernel change behind the default-off environment flag.
4. Run the duplicate-heavy test first.
5. Run the complete TopK v2 module with the flag on and unset.
6. Benchmark each required shape in a fresh process. Averages must not hide a
   regressing row.
7. Check edited files for diagnostics and run `git diff --check`.

Use:

```bash
bash /sgl-workspace/topk_v2_gfx1250/scripts/validate_topk_cluster.sh
```

## Cluster-width tuning

Cluster width means cooperating workgroups per TopK row. It is unrelated to
tensor-parallel degree.

Current widths:

- Batch up to 60, sequence at most 131,072: 8 workgroups.
- Batch up to 60, sequence above 131,072: 15 workgroups.
- Batch above 60, sequence at most 131,072: 4 workgroups.
- Batch above 60, sequence above 131,072: 2 workgroups.

Tune against the actual `(rows, local C4 sequence length, topk, output mode)`
seen by one GPU. TP1/TP2/TP4 need separate kernel constants only if those local
shapes differ. In the current DSV4 path TP duplicates the TopK geometry across
ranks, so one tuning applies to all three.

When tuning:

- Keep correctness enabled for every candidate.
- Compare flag-on against flag-unset, not against an obsolete single-CTA
  baseline.
- Use fresh processes because clocks and JIT state materially skew sequential
  measurements.
- Require every acceptance row to be faster.
- Recheck page-table, raw-index, dual-output, ragged, topk=2048, and
  duplicate-heavy cases after changing width or synchronization.

## Completion report

Report:

- Device architecture and HIP version.
- Primitive probe results.
- Exact correctness commands and pass counts.
- Every baseline/cluster timing row and speedup.
- Chosen widths and dispatch boundaries.
- Whether the feature remains default-off.
