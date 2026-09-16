# Cross-node exchange (B200 ↔ MI355X)

The two nodes have no agent-to-agent channel and **no shared filesystem**. They
see the same files only through this git repo
(`github.com/kkHuang-amd/claude-skills`), so **nothing crosses until it is
committed and pushed on one side and pulled on the other**. A file written here
and left uncommitted is invisible to the other node.

(An earlier version of this file claimed the nodes shared an NFS export and that
`/workspace` equalled `/mnt/home/wunhuang`. That was wrong — the B200 container
does mount `10.238.19.129:/AI55QU/home/wunhuang` at `/workspace`, but the MI355X
node is not on it.)

## Convention

- **One file per side, per topic**, named `<node>-<topic>.md`, e.g.
  `b200-decode-trace.md`, `mi355x-decode-trace.md`. One writer per file means
  git never has to merge concurrent edits.
- **`FINDINGS.md` is the joint source of truth** for conclusions. Append, and
  sign each block with node + date so a stale number is identifiable.
- **Do not put raw artefacts in the repo.** A trace set is 64-68 MB. Run the
  analysis on the node that owns the trace and commit only the tool's text
  output — that is what `analysis/trace_summary.py` and `trace_ranks.py` are
  shaped for. Reference raw paths as node-local (`b200:/workspace/agentx/...`).
- If raw traces genuinely must move, use an HF Hub dataset repo (`hf upload
  --repo-type dataset`); both nodes already have the `hf` CLI and Hub access.
- Tools live in `../analysis/` and are platform-neutral. If you extend
  `trace_common.ROLES` for ROCm kernel names, do it there and push it, so both
  sides classify identically — a role pattern that exists on one side only makes
  the comparison invalid.

## Current open request (B200 → MI355X, 2026-09-16)

Context: with pdi, accept len, per-request KV working set and cuda-graph status
all matched, MI355X's log-implied step time is 1.54-1.57× B200's at batch
9/12/16. **But log-implied step time includes amortised prefill.** On B200 only
~17 % of wall time is inside decode steps (log-implied 103 ms at pdi=10 vs a
17.2 ms pure-decode step from the trace), so the 1.55× could sit in decode
kernels *or* in the prefill/barrier time around them. These have different fixes.

Needed, in priority order:

1. `TARGET_VERIFY` **step wall p50 and its `bs`** from the MI355X trace. This is
   the discriminator: ~1.55× ⇒ the gap is in decode kernels; roughly equal ⇒ the
   gap is in prefill/waiting and the kernel breakdown is not where to look.
2. The **unclassified kernel list** that `trace_summary.py` prints
   (`unclassified (add a ROLES pattern ...)`). ROCm names differ completely from
   CUDA, so without this a large share of time may sit in `other` and quietly
   distort every role comparison.
3. **MoE kernel calls per step** (B200: 61-64, i.e. one per layer). A different
   count changes the per-call normalisation.
4. **Stream count active inside verify steps** (B200 multi-stream 132,
   single-stream 4).

Lower priority, the last two unquantified config asymmetries: **mem-frac**
(MI355X 0.85 vs B200 0.88) and the **image version**.

## Comparability rules that apply to both sides

- Compare only **matching `bs` buckets**. `bs` is `forward_batch.batch_size`
  (`profile_utils.py:477`); two captures of the same B200 arm landed on bs=7 and
  bs=3, so mismatched buckets are the normal case, not a corner case.
- Match the **absolute** KV working set (`#full token` ÷ batch), not the pool
  usage fraction — the pools differ 5.4× in capacity (B200 2,217,472 tokens,
  MI355X ~12,075,000), so the fractions are not comparable by construction.
  Verified: ~152k tokens per request on B200 vs ~151k on MI355X.
- Never compare a partial run against a complete one.
- Per-kernel durations *are* comparable across the two platforms despite B200's
  multi-stream overlap: disabling overlap on B200 left the summed kernel time
  unchanged (compute 15.1 ms either way) and only lengthened the wall, so
  overlap fills gaps rather than inflating kernels through SM contention.
