# Docker image swap — what to re-integrate, and everything measured before it

Written 2026-09-03 06:40Z, immediately before the image change. Results from the
session are in `ITL_GAP_FINDINGS.md` §11–§15; this file is only about surviving
the swap and rebuilding the tree.

## 0. READ THIS FIRST: `/sgl-workspace` does not survive

```
/workspace       /dev/sdb2   ext4      838G   <- PERSISTS
/shared_nfs      /dev/md0    xfs        56T   <- PERSISTS (model weights)
/                overlay                      <- the image
/sgl-workspace   (on overlay)                 <- DESTROYED by the swap
```

`/sgl-workspace/sglang` and `/sgl-workspace/aiter` are on the container overlay,
so both repos **and every uncommitted change in them** are gone after the swap.
Everything needed has therefore been exported to
**`/workspace/handoff-20260903-image-swap/`**:

| file | what it is |
|---|---|
| `0001-use-a-bounded-prefill-logits-buffer-*.patch` | our local `33979a814b`, i.e. our version of PR #37660. **Prefer fetching the PR instead** — see §2. |
| `aiter-tracked.patch` | `git diff` of the 9 modified aiter files |
| `aiter-staged.patch` | `git diff --cached` (one file, `csrc/kernels/dsv4_rotate_quant.cu`) |
| `aiter-untracked.tgz` | the 6 untracked aiter files (`flydsl_cache/` excluded on purpose) |
| `sglang-other-sessions-uncommitted.tgz` | the 19 files belonging to other sessions, backed up as a courtesy |
| `sglang_HEAD.txt`, `aiter_HEAD.txt`, `*_status.txt` | exact SHAs and dirty state at swap time |

An older, independent backup also exists at
`/workspace/tree-backup-20260902-072044/` (full dirty tarballs of both repos).

The scripts and tools all live under `/workspace/claude-skills/agentx/`, and
`/workspace/results/` holds every arm's artifacts, so **none of the measurement
work is at risk** — only the two source trees.

## 1. State at swap time

```
sglang  33979a814b  use a bounded prefill logits buffer and process oversized batches in row chunks
        83310485e1  DSV4 FP4 C4 indexer (#37353) — hand-applied working-tree integration
        52e1c24744  [Diffusion] Fuse FLUX.2 token concatenation and NVFP4 quantization (#37141)
        + 19 uncommitted files belonging to OTHER SESSIONS (not ours)

aiter   c16d44b93  fix(fmoe): retune GLM-5 FP8 decode kernels (#4811)
        + 9 modified, 1 staged, 6 untracked — ALL UNCOMMITTED, all ours/FP4-related
```

Every number in `DATA_AND_ANALYSIS_20260902.md` §1 was measured on exactly this
tree. State the sglang SHA with any comparison made after the swap, because the
new image's base will differ.

## 2. Re-integrating sglang PR #37660 — fetch it, do not replay our patch

The new image already carries the FP4 indexer (#37353, upstream commit
`f8cbf000f4a5`), so our `83310485e1` hand-applied integration is **no longer
needed and must not be applied**. Only the OOR fix is missing.

GitHub is reachable from the container (verified: `git ls-remote` succeeded), and
PR #37660 is still **open** (approved by `1am9trash`, not merged). Its refs as of
now:

```
refs/pull/37660/head   ee92dc805837b5f36099c299bf50cb62e3067ca1
refs/pull/37660/merge  2df5932cae79471ce0fce7a04d396ee771384c7b
```

The head is two `Merge branch 'main'` commits on top of the one functional
commit. **Cherry-pick the functional commit only:**

```bash
cd /sgl-workspace/sglang
git fetch https://github.com/sgl-project/sglang.git refs/pull/37660/head:pr37660
git cherry-pick e5d8c6dc82594989cccdee7f71e741da220cde9c
```

`e5d8c6dc8259` is "use a bounded prefill logits buffer and process oversized
batches in row chunks", and it sits directly on top of `f8cbf000f4a5`
(#37353) on that branch — the same base the new image has. So it should apply
cleanly, unlike our patch, which was generated against a hand-applied #37353.

**Why not our own patch:** `fp4_indexer_hip.py` is byte-identical between our
`33979a814b` and the PR, but `indexer.py` differs by 149/123 lines. Most of that
is base drift, but part of it is a deviation we had to make and should not carry
forward — see below.

### The one deviation we made, and whether it is still needed

Upstream calls `self.flashinfer_topk_transform(...)`, which did not exist in our
tree, so we substituted `topk_transform_512_flashinfer_unfused(...)` with an
inline comment (§4b). **Checked against the PR branch: `flashinfer_topk_transform`
is still not defined there either** (`git grep -c 'def flashinfer_topk_transform'
pr37660` = 0). So this is an upstream loose end, not a local one, and the same
substitution will likely be needed again in the new image. Verify after the
cherry-pick that the flashinfer branch resolves to a function that exists.

### Verify the fix landed

```bash
rg -c 'logits_rows_per_chunk' \
  python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py \
  python/sglang/srt/layers/attention/dsv4/indexer.py
rg -o 'SGLANG_DSV4_FP4_LOGITS_BUDGET_MB.{0,40}' \
  python/sglang/kernels/ops/attention/dsv4/fp4_indexer_hip.py   # default "2048" MB
python -c "import sglang.srt.layers.attention.dsv4.indexer, \
sglang.kernels.ops.attention.dsv4.fp4_indexer_hip; print('imports ok')"
```

Also confirm the FP4 indexer flag still resolves:
`--enable-deepseek-v4-fp4-indexer`.

## 3. Re-integrating the aiter changes

All uncommitted, all FP4-related. HEAD was `c16d44b93` (#4811).

```bash
cd /sgl-workspace/aiter
git log -1 --format='%h %s'          # compare against c16d44b93 first
git apply --3way /workspace/handoff-20260903-image-swap/aiter-tracked.patch
git apply --3way /workspace/handoff-20260903-image-swap/aiter-staged.patch
tar xzf /workspace/handoff-20260903-image-swap/aiter-untracked.tgz -C .
```

Modified (9): `configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv`,
`dsv4_fp8fp4_untuned_fmoe.csv`, `fused_moe.py`,
`ops/flydsl/kernels/mqa_logits/pa_mqa_logits_fp4.py`,
`pa_mqa_logits_fp4_prefill.py`, `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`,
`csrc/cpp_itfs/torch_utils.py`, `csrc/kernels/dsv4_rotate_quant.cu` (staged),
`op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py`.

Untracked (6): `eptune48_tuned.csv`, `eptune48_untuned.csv`,
`configs/profile_fmoe.csv`, `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ep_tune.py`,
`op_tests/test_moe_mxfp8_passthrough.py`, `xxx`.

`aiter/jit/flydsl_cache/` was **deliberately excluded** — it is a JIT build cache,
it will be regenerated, and it is large. Expect the first FP4 arm after the swap
to pay JIT compile time.

**If the new image's aiter is at a different HEAD, `--3way` may conflict.** The
FP4-critical files are `pa_mqa_logits_fp4_prefill.py` and `pa_mqa_logits_fp4.py`
(the flydsl MQA-logits kernels the FP4 indexer calls). Resolve those two first;
the CSV config files are tuning tables and conflict harmlessly.

## 4. After rebuilding: re-baseline before trusting any comparison

The image change moves the base of *everything* — sglang, aiter, ROCm, torch. So:

1. **Every FP4 number is stale twice over** — once from `33979a814b` changing the
   FP4 scoring path, now again from the image. §1's table is a record of the old
   image, full stop.
2. **The non-FP4 arms are stale too now.** Before the swap, `dptbo-c128` was still
   a valid baseline; after it, it is not.
3. **Re-run one arm as a bridge.** The cheapest meaningful one is `dptbo-c128`
   (`dptbo_c128.sh`) or `interval20-c128`, because both have a
   pre-swap partner and either pins how much the image alone moved. Until that
   exists, do not quote a cross-swap delta.
4. Keep `TREE_SHA_AT_START.txt` / `TREE_CHECKSUMS_AT_START.txt` in every arm
   script — the md5s will all differ now, which is expected, not tampering.

## 5. Everything measured in this session (details in ITL_GAP_FINDINGS.md)

- **§11 hicache smoke passed.** The launcher needs `KV_OFFLOADING=dram` +
  `KV_OFFLOAD_BACKEND=hicache` + positive `TOTAL_CPU_DRAM_GB`;
  `KV_OFFLOADING=hicache` is invalid and exits 1 before the model loads. The rust
  `DeepseekV4C4IndexerScale` worry was unfounded — no `.rs` references exist and
  no storage backend is configured.
- **§12 the c192 pass criterion was invalid.** `11.70 % → 8.6 %` were both
  whole-log figures contaminated by a ~37 min cold-cache warmup. Windowed:
  c96 5.59 %, c128 5.81 %, c160 5.96 %, c192 8.15 %. Always window it.
- **§13 hicache fixed the c192 collapse.** +58.36 % tok/s (22,994 → 36,414),
  TTFT −84.74 % (59.71 → 9.11 s), ITL flat. Queue p90 78 → 7, decode batch
  15 → 23. Only 1.4 pp of the recovered hits are direct CPU reads; the larger
  part is the GPU tier hitting more.
- **§14 c256: 39,284 tok/s but we lose to ATOM's c256 on all three axes**
  (−12.2 % tok/s, +19.7 % ITL, +17.3 % TTFT). Not the engine's ceiling: the CPU
  tier hit **99.98 % full**, so `HICACHE_RATIO=3.0` is the constraint there.
  Overall cache hit flat at 0.951 — only the tier split moved.
- **§15 c128 + interval 20 + hicache FAILED the pair** (ITL 57.98 ms passes,
  TTFT 13.77 s does not). **hicache is a no-op at c128** — CPU-tier hit 0.5 pp,
  device pool 73 %, KV occupancy 0.27, nothing to evict so nothing to recover.
  And the two levers are **orthogonal**: interval 20's TTFT is policy-imposed
  deferral (`#queue-req` p90 = 9), not prefill demand.

### Tooling fixed this session — carry these forward

- `arm_report.py` read `gpu_cache_hit_rate` and printed it as "cache", which made
  c256 look like a cache collapse (0.765) when overall was flat at 0.951. Now
  uses `overall_cache_hit_rate` and prints the tier split plus both pool
  utilisations, with `<!> CPU TIER FULL -- raise HICACHE_RATIO` above 98 %.
- `summary_table.py` gained `GPU-tier hit` / `CPU-tier hit` columns and the three
  new arms. Regenerate and re-sync into `DATA_AND_ANALYSIS §1` after any arm.
- Arm scripts gained `KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'`. Without
  it `process_agentic_result.py:89` exits 1 **after a fully successful
  benchmark** and writes no result JSON. §13 has the re-aggregate-only command.
- New **VRAM gate** in `hicache_fp4_int20_c128.sh`: the process-based idle check
  cannot see another container's job, and the node twice showed ~112–118 GB held
  across all 8 GPUs with zero visible processes. Starting an arm then would
  silently shrink the KV pool and break the memory matching. Copy that gate into
  every new arm script.

## 6. Work queue, unchanged by the swap except for ordering

1. **P0 — the fp8-path OOR fix, now the most urgent item.** `hicache-fp4-int20-c128`
   ran with free VRAM p10 **0.08 GB**, min **0.01 GB** and **10 late Triton
   device loads** — the exact §3 abort signature that killed
   `fp4-dptbo-c64-reclaim0`. PR #37660 covers the **FP4** path only; the fp8 path
   keeps the unbounded `torch.empty(total_tokens, max_seq_len)` at
   `dsv4/indexer.py:160` in `_aiter_fp8_paged_mqa_logits`. Mirror the bounded
   buffer there, and pre-load the Triton specialisations at engine init
   (`unified_kv_kernels/runtime.py:298,332`; `BLOCK` has ≤ ~11 power-of-2 values
   × `HAS_COMPRESS` × `compress_ratio` ∈ {0,4,128}). Upstream-shaped, and worth
   offering back as a follow-up to #37660.
2. **P1 — interval 15 at c128, hicache OFF.** The box is bounded by int 10
   (78.39 ms, 8.50 s) and int 20 (57.98 ms, 13.77 s); ATOM's (61.5 ms, 10.9 s)
   is strictly inside it, so a middle value can satisfy both axes. Then 12.
   hicache off per §15, which keeps it a clean single-variable series.
3. **P2 — c256 at `HICACHE_RATIO` 5–6.** The only place the tier is provably the
   constraint. Host DRAM 3,023 GB with 1,442.6 GB pinned at ratio 3, so it fits.
4. **P3** — FP4-off replicate of the §15 arm, only if P1 passes and clean
   attribution is then wanted.
5. **P4** — re-measure the FP4 curve. Mandatory now anyway, see §4 above.
6. **P5** — c224, purely to fill the curve. Low value.

### Settled — do not spend an arm re-litigating

TBO stays on. Do not profile a decode step. Do not re-run the chunk-size
diagnostic. The c192 collapse is not eviction and not memory pressure. ATOM's
`run 33074134043` is unreachable (ROCm blocks all classic PATs). Plus two new
ones: **hicache only pays where GPU KV pool occupancy is high** (96 % at c192,
100 % at c256, 73 % at c128 = nothing), and **hicache and
`--prefill-decode-interval` are orthogonal** — do not combine them expecting the
effects to add.

### Node traps that are still live after the swap

- **The node is shared.** Arm scripts wait for three consecutive idle checks and
  **refuse** rather than kill. Keep that. A blind kill preamble destroyed another
  session's c96 arm on 09-02. Check `ps -eo pid,lstart,args` before touching
  anything, and remember a job in another container's PID namespace is invisible
  while its GPU allocation is not.
- **Never `pkill -f` a pattern matching your own command line.**
- **A dead server keeps answering `GET /metrics` with 200** and aiperf then waits
  forever. Verify an arm early, not at the end.
- **`ITL p90` is per-request TPOT**, p90 across requests — not the p90 of token
  gaps.
- **Do not edit a running bash script.** bash reads it by byte offset; an edit
  mid-run corrupts execution. `hicache_fp4_int20_c128.sh` still carries one stale
  echo label (`=== single variable vs fp4-dptbo-c192? ===`) for exactly this
  reason — cosmetic only, safe to fix now that the arm has finished.
