# Next-session prompt (MI355X node)

Paste everything between the BEGIN/END markers as the first message of a new
chat. Notes for the human are below the END marker.

----------------------------- BEGIN PROMPT -----------------------------

## Operating rules — active from your first tool call

Read `/workspace/claude-skills/cap-tool-output/SKILL.md` and
`reduce-conversation-usage/SKILL.md` and apply both for this whole session.
Cap every command whose output you cannot predict: `| cut -c1-200` and
`| head -c 2000`. Count before you list (`wc -l`, `rg -c`, `test -f`). Never
search from `/`. Bounded git only (`git diff --stat -- <path>` before any
diff). Long jobs: full log to a file, only markers to the chat, filter
**inside** the command. Batch independent tool calls into one message. Keep
durable state in the project docs, not the chat.

Domain skills live in `/workspace/claude-skills/`. Two are relevant here:
`perf-bottleneck-attribution` (read it — this task is exactly its subject) and
`sglang-prefill-coalescer`. Read only the `description:` line of the others.

## Context — do not reconstruct it from chat history

Source of truth, in this order:

1. `agentx/mi355x/CONTINUE_HERE.md` — this node's state and the ranked
   directions. **Start here.**
2. `agentx/exchange/FINDINGS.md` — read the **INDEX at the top only**
   (live answer, retraction ledger, method rules). The file is ~1,600 lines and
   append-only; most of it is superseded.
3. `agentx/exchange/mi355x-decode-trace.md` §14-17 for the per-kernel detail.
4. `agentx/exchange/README.md` for the cross-node protocol.

**`git pull` before doing anything.** The B200 node is a separate machine that
shares nothing but this repo, and it pushes frequently.

One-line summary so you can sanity-check what you read: the AgentX c128 pdi=24
decode-step gap MI355X-vs-B200 is **74.0 vs 32.68 ms**, fully attributed and
agreed by both nodes. Diagnosis is closed. This session is **optimisation**.

## The task

Work directions **A and C in parallel**, with **B as a background
calculation**. All three are local; nothing is blocked on B200.

### A. A/B `--load-balance-method total_tokens` (up to ~10 ms, 24 % of the gap)

Both nodes' logs show the balancer levels *request count* while cost follows
*KV tokens*: MI355X `running-req` 1.38x vs `#full token` 2.27x (B200: 1.29x vs
2.45x). The cause is the router concentrating long conversations for prefix
reuse — **verified**, `router.log` in the reference run carries `cache_aware`.

- Launcher: `/workspace/InferenceX/benchmarks/single_node/agentic/dsv4_fp4_mi355x_sglang_mtp.sh`
  - **line 241** hardcodes `--load-balance-method total_requests` ← the change
  - **line ~368** is the router's `--policy cache_aware` (separate process,
    gated on `USE_SGLANG_ROUTER=true`, so it never appears in the server's
    `sglang_command.txt`)
  - pdi defaults to **10** in this file; the aligned runs pass
    `PREFILL_DECODE_INTERVAL=24`. Always confirm from the run's
    `sglang_command.txt`, never from script intent.
- Reference arm to compare against:
  `/workspace/results/megamoe-eplb-c128-b200aligned/` (complete c128 run, with
  `server.log`, `router.log` and the agg json).
- **Expect a TTFT regression.** This fights prefix reuse; it is an ITL↔TTFT
  trade of the same family as the pdi knob, not a free win. Report both, plus
  cache hit rate.
- Verify the outcome, not the flag: re-run `python3 analysis/kv_skew.py
  <new server.log>` and check `#full token` skew actually fell. A flag that
  silently does nothing has already cost this project a day.

### C. Eliminate the 3.85 ms of unfused copy kernels (9 %)

B200 spends 0.03 ms here; MI355X spends 3.85 across seven kernels, per step at
bs=10 (`analysis/kernel_dump.py` reproduces the list):

| ms/step | calls | kernel |
|---:|---:|---|
| 0.803 + 0.625 | 67 + 66 | `at::native::elementwise_kernel_manual_unroll` (direct_copy, two instantiations) |
| 0.757 | 183 | `_fill_padded_rows_kernel` — MoE row padding |
| 0.533 | 122 | `__amd_rocclr_fillBufferAligned` — hipMemset, 2/layer; look for a buffer that could be persistent |
| 0.392 | 92 | `vectorized_elementwise` bf16→fp32 copy |
| 0.363 | 69 | `index_elementwise_kernel` |
| 0.271 | 61 | `_swa_scatter_kernel` |
| 0.143 | 31 | `_fill_compress_tail_kernel` |

These are ordinary fusion/elision work with no cross-node dependency. Attribute
each to its call site first — remember kernel identity **cannot** be resolved
inside a cuda-graph-replayed window; use an `EXTEND` (eager) step and the GPU
event's own `External id` against `cpu_op`.

### B. Achieved bandwidth on the MLA decode kernel (background)

7.47 ms direct (18 %) **and** it is the multiplier on A's imbalance cost, so it
is the only item that pays twice — but **do not start by rewriting it.** Nobody
on either node has a roofline number. `record_shapes` is a dead end here (all
34,486 dim-carrying events are `aten::*` cpu_ops; zero attention/MoE ops), so
derive KV bytes per call from the DSv4 config plus per-rank `#full token`, and
compare against MI355X HBM peak. That decides whether 4.06x is closeable or
whether the kernel is already at the wall.

Two variants, selected by batch size — keep them separate:
`_paged_decode_split_kernel` (bs 9-10, 117.5-163.5 µs/call) and
`_paged_decode_fused_kernel` (bs 14-20, 165.3-330.7). B200's comparable is
18.7-47.5 µs/call. The published 4.06x is the **split** variant at bs=10.

## Ruled out — do not revisit, each has evidence

- **`prepare`'s CU count / `pcu1`+`qcu28` tuning.** `megamoe_prepare_compact`
  is a cross-rank **wait** (r = −0.942); a `wait_i32_until_equals` spin does not
  parallelise, and its idle CUs are idle *because the rank is waiting*.
- **Stream overlap / co-scheduling.** B200 measured true full serialisation at
  **3 % end to end**. MI355X has no room in the wait anyway.
- **`gemm`.** Equal (9.81 vs 9.36) once B200's SM-starvation artefact was removed.

## Node traps — every one of these has bitten at least once

- **VRAM reclaim is plateau-then-cliff.** After a kill it sits flat at
  29-53 GB/GPU with zero KFD holders for 10-20 min, then drops to baseline in
  one step. Never extrapolate a creep rate; never launch on top of a plateau.
  Per-GPU reset is unsupported here.
- **The launcher orphans everything** — `launch_server`, `sglang::*`, tokenizer
  workers, aiperf — holding ~290 GB/GPU *and* the dist-init port. After cleanup
  verify all three: process count, VRAM, port listeners.
- **Intermittent EPLB rebalance deadlock, ~1 run in 3.** `returned=` frozen,
  `errors=0`, `/metrics` still 200, schedulers alive. Judge liveness only by
  `returned=`/`done=` moving. A straight retry clears it.
- **Never `kill` a PID the tool reports for a backgrounded shell without
  listing its children first.** This has wedged every shell in the session
  twice (`echo` stops completing; it looks like a dead node and is not).
  Recovery is a terminal restart, which only the human can do. Also:
  `pgrep -f` / `rg` match your own command line.
- **Do not `git checkout` the launcher.** It was already dirty before the pdi
  edit (+81 lines of MegaMoE support). Backup:
  `/shared_nfs/kk/pr35619/mi355x_mtp.sh.bak.1228`.
- A fixed settle before `/start_profile` captures mid-ramp. Trigger on
  per-request `#full token` plateauing (≥ ~130k), not a constant.

## Working agreement

- **Commits:** one per *message to B200*, with tooling and appendices folded
  into the message they serve; use `--amend` for same-turn follow-ups. Do not
  force-push or rewrite pushed history — B200's clone builds on it.
- Update `CONTINUE_HERE.md` in the same commit as the work it describes.
- When you publish a number, publish **step wall, its `bs`, and the capture
  window's KV working set** together. Two of the three were missing from every
  early capture and that is what made three separate numbers wrong.
- Ask before launching anything that occupies the GPUs for more than a few
  minutes.

Acknowledge in a few lines with (a) what `git pull` brought in, (b) which
direction you are starting, and (c) the first command you intend to run. Then
wait.

------------------------------ END PROMPT ------------------------------

## Notes for the human (not part of the prompt)

- Written 2026-09-17, against `87b4c5a` + B200's `b088d2d`.
- If B200 has pushed materially new findings by the time you paste this, the
  new session's `git pull` will surface them and `FINDINGS.md`'s INDEX is
  maintained by both nodes, so the prompt should still be accurate.
- The prompt deliberately does **not** restate the 41.07 ms decomposition; it
  points at `CONTINUE_HERE.md` so there is exactly one copy to keep current.
