# Kimi-K3 — CONTINUE HERE

Single entry point for resuming this project. Read this file plus
`DEV_RULES.md` and nothing else until you know which task you are doing.

Source of record: `HANDOFF_2026-08-21.md` (state), `SUMMARY.md` (detail).
State last updated 2026-08-21. Paths re-verified 2026-08-24 — see §0.

---

## 0. Environment preflight — RUN THIS FIRST

The K3 worktrees and run artifacts are **container-local**, not on shared
storage. As of 2026-08-24 they were **absent** from the `/sgl-workspace`
container this file was last edited in (`/workspace` did not exist at all).

Before trusting anything below, verify:

```bash
for wt in /sgl-workspace/sglang-k3-triton37 /sgl-workspace/aiter-k3-triton37 \
          /sgl-workspace/aiter-atom-current; do
  printf "%-40s " "$wt"
  [ -d "$wt" ] && git -C "$wt" log -1 --format='%h %d' || echo MISSING
done
```

- **All present** → continue to §1.
- **Missing** → you are in a fresh container. Do not improvise: follow
  `start_prompt.md` §2 (restore canvases) and §4 (restore code branches)
  first, then come back here.

---

## 1. Status

Two **independent, default-off** decode candidates are implemented in SGLang.
Production is **unchanged**.

```text
KDA input-projection MXFP4    passes at exact M32 and M64   1.020x / 1.428x
MLA shared-input PTPC         passes at exact M64           34.320 -> 29.920 us (1.1471x)
                              M32 loses -> excluded
```

The corrected MLA shared-input path quantizes normalized hidden once for both
QKV-A and g_proj, and retains SGLang's fused output gate.

---

## 2. Next action

Measure **each candidate's real TP8 capacity separately**, then run matched
common-client C2/C64 and complete prefill/decode retraces — **without bundling
the candidates**.

For MLA shared PTPC specifically: require the combined QKV-A + gate boundary
and the total C64 replay to move in the predicted direction **before** running
endpoint correctness gates.

---

## 3. Hard constraints

```text
Do not commit, push, discard, reset, or rewrite either worktree without an
  explicit request from the user.
Do not promote the combined dense prepared policy as one bundle
  (it costs 3,273,054,208 B = 3.048269 GiB/GPU).
Do not conflate the three uncommitted SGLang changesets (pre-existing latent
  MXFP4 / KDA input-projection / MLA shared-PTPC) when editing one of them.
Graph-external analysis is deferred — do not reopen it unasked.
```

Full working rules: `DEV_RULES.md`.

---

## 4. Worktrees (as recorded 2026-08-21 — re-verify via §0)

```text
/sgl-workspace/sglang-k3-triton37   perf/k3_opts_0812         455b744a  9 dirty files
/sgl-workspace/aiter-k3-triton37    integration/k3-core-only  b56d27be  clean
/sgl-workspace/aiter-atom-current   main                      dc4bdf1c  5 dirty files
```

Per-file breakdown and what each uncommitted file belongs to:
`HANDOFF_2026-08-21.md` §Worktrees.

---

## 5. Preserve (do not delete)

```text
/workspace/kimi-k3-runs/common-oai-sglang-atom-traces-2026-08-22/route-validation/
/workspace/kimi-k3-runs/matched-route-current-kernel-2026-08-23/
```

Both were absent on 2026-08-24. If this container has no `/workspace`, they
were never restored here — do not assume they were deleted.

---

## 6. Where the detail lives — open only what you need

```text
HANDOFF_2026-08-21.md                    full state, worktrees, script toolkit
  aiter-optimization-tracker/KDA_INPROJ_MXFP4_M32_M64_2026-08-23.md
  aiter-optimization-tracker/MLA_SHARED_PTPC_M64_2026-08-23.md
SUMMARY.md   section only, via: rg -n '^## ' SUMMARY.md && sed -n 'A,Bp' SUMMARY.md
  -> "Remaining optimization work"   open route directions
  -> "Decisions already made"        what is already settled
start_prompt.md   §2/§4 restore procedures, §5 runtime config
DEV_RULES.md      working / canvas / artifact-retention rules
```

Reusable tooling (when present):
`/workspace/useful-scripts/benchmarking/kimi-k3/` — start at its `README.md`.

Durable Cursor rules after an image swap:
`bash <skills>/kimi-k3/install_cursor_rules.sh /sgl-workspace`

---

## 7. Open route directions (from SUMMARY "Remaining optimization work")

```text
Highest-value unresolved:  fixed tiny-kernel chains
                           copies/materialization
                           attention residual and KDA launch boundaries
                           route / sort / quant handoff
Credible directions:       one-CTA LDS E896 sorter
                           stage1 ABI consuming route metadata and
                             token-major scale directly
Also pending:              finish analysis of retained B300 normal versus
                             single-stream/no-PDL summaries
```

---

## Maintenance

Update §1/§2/§4 after each accepted or rejected experiment, same as
`SUMMARY.md` and the canonical canvas (see `DEV_RULES.md` §1). Keep this file
under ~3 KB — detail belongs in the linked reports, not here.
