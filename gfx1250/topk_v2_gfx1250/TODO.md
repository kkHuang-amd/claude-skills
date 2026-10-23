# gfx1250 Cooperative Kernel Follow-ups

## A/B validation

Compare the current TopK v2 paths with identical server and benchmark settings:

- A: `SGLANG_GFX1250_TOPK_CLUSTER=0`
- B: `SGLANG_GFX1250_TOPK_CLUSTER=1`
- Tensor parallel sizes: TP1, TP2, TP4
- Accuracy gate: GSM8K, 1,319 examples, five-shot, temperature 0
- Serving workload: random 8,192 input / 1,024 output tokens, 256 prompts,
  concurrency 64, 128 warmups, seed 42

For each TP size:

1. Start A, wait for health, run GSM8K, then run serving benchmark.
2. Stop A and ensure its workers exit.
3. Start B with otherwise identical environment and arguments.
4. Run the same GSM8K and serving commands.
5. Compare GSM8K accuracy before interpreting performance.
6. Retain complete server, evaluation, and benchmark logs.

Do not compare A and B concurrently because they would contend for GPUs.

## AgentX-style long-context serving

- [ ] Adapt the InferenceX MI355X HiCache settings to DeepSeek-V4-Flash TP4.
- [ ] Diagnose the reproducible TP4 131K memory fault reported in
      `ncclDevKernel_Generic_4`; see `AGENTX_HICACHE.md`.
- [ ] Run cluster-off/on at 131K input, concurrency 8.
- [ ] Run cluster-off/on at 262K input, concurrency 16.
- [ ] Record cold-cache and, separately, repeated warm-HiCache results.
- [ ] Confirm the TopK cluster route is actually selected before attributing
      serving differences to it.
- [ ] Run `long_context_accuracy.py` for both cluster-off and cluster-on at
      131K and 262K contexts; require identical code recovery.

See `AGENTX_HICACHE.md` for the command and rationale.

## Future: AOT DSA TopK cluster route

Prototype a gfx1250-only cooperative-launch route for
`python/sglang/kernels/aot/include/hip/dsa_topk_coop.cuh`.

Goals:

- Target the existing multi-block, four-launch long-row path first.
- Use `hipLaunchKernelEx` with `{1, N, 1}` cluster dimensions.
- Reuse split cluster barriers, global histogram staging,
  `cluster_load_b32` register multicast, rank-prefix output placement, and
  rank-0 exact refinement.
- Preserve the one-block path for short rows.
- Keep the new route default-off behind a separate experimental flag.
- Benchmark against PR #41687's best AOT path directly; do not reuse TopK v2
  speedup claims.

Acceptance:

- Exact existing AOT TopK correctness and tie tolerance.
- No regression in short-row serving shapes.
- Every selected long-row benchmark must beat the existing multi-launch path.
- TP1, TP2, and TP4 validation must use observed per-rank row geometry.
